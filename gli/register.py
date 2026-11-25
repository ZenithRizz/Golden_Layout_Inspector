"""Registration: camera frame -> golden layout (work) coordinates (F5.1/F5.2).

1. Detect the ArUco markers printed around the target.  Their corner
   positions are known in print pixels (design.json), so a homography maps
   the camera frame onto the layout.
2. Warp the frame into the work-scale layout image ("rectify").
3. Refine with ECC (sub-pixel affine alignment of rectified vs golden), which
   removes the residual error the difference detector cannot tolerate.

Detections in work px map back to camera px with the inverse homography for
the live overlay.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .layout import Design

ARUCO = cv2.aruco.DICT_4X4_50


@dataclass
class Registration:
    locked: bool
    H: np.ndarray | None          # frame px -> work px (includes ECC refinement)
    n_markers: int
    reproj_px: float              # marker corner reprojection error (work px)
    ecc: float                    # ECC correlation (0..1), -1 if not run
    quality: float                # 0..1 summary used as mapping confidence
    reason: str = ""


class Registrar:
    def __init__(self, design: Design, golden_bgr: np.ndarray, min_markers: int = 2,
                 max_reproj: float = 2.0, ecc_iters: int = 40):
        self.d = design
        self.golden = golden_bgr
        self.golden_gray = cv2.cvtColor(golden_bgr, cv2.COLOR_BGR2GRAY)
        self.size = design.work_size
        self.P2W = design.print_to_work_matrix()
        self.det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO),
                                           cv2.aruco.DetectorParameters())
        self.min_markers = min_markers
        self.max_reproj = max_reproj
        self.ecc_iters = ecc_iters
        # Golden at 1/2 scale for fast ECC
        self.gold_small = cv2.resize(self.golden_gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        self.last_ok_H = None
        self._cache = None   # (marker H, refined H, ecc)

    def markers(self, frame_gray):
        corners, ids, _ = self.det.detectMarkers(frame_gray)
        src, dst = [], []
        if ids is not None:
            for c, i in zip(corners, ids.ravel()):
                key = str(int(i))
                if key in self.d.markers:
                    src.extend(c.reshape(4, 2))
                    dst.extend(self.d.markers[key])
        return np.array(src, np.float32), np.array(dst, np.float32), 0 if ids is None else len(src) // 4

    def register(self, frame_bgr, refine=True) -> Registration:
        gray = frame_bgr if frame_bgr.ndim == 2 else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        src, dst_print, n = self.markers(gray)
        if n < self.min_markers:
            return Registration(False, None, n, float("inf"), -1, 0.0,
                                f"Align die: fiducials not found ({n}/4 markers visible)")
        dst = cv2.perspectiveTransform(dst_print.reshape(-1, 1, 2), self.P2W).reshape(-1, 2)
        H, _ = cv2.findHomography(src, dst, 0)
        if H is None:
            return Registration(False, None, n, float("inf"), -1, 0.0, "Homography failed")
        proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        err = float(np.sqrt(((proj - dst) ** 2).sum(1)).mean())
        ecc_val = -1.0
        if refine:
            c = self._cache
            if c is not None and self._moved(c[0], H) < 0.6:
                H, ecc_val = c[1], c[2]          # pose unchanged: reuse refinement
            else:
                Hm = H
                H, ecc_val = self._ecc(frame_bgr, H)
                self._cache = (Hm, H, ecc_val)
        q = min(1.0, n / 4) * (1.0 if err <= self.max_reproj else max(0.0, 1 - (err - self.max_reproj) / 4))
        if ecc_val >= 0:
            q *= float(np.clip((ecc_val - 0.3) / 0.5, 0.3, 1.0))
        locked = err <= self.max_reproj * 2 and (ecc_val < 0 or ecc_val > 0.35)
        reason = "" if locked else ("Layout mismatch: check selected design or target" if ecc_val >= 0 and ecc_val <= 0.35
                                    else f"Registration error {err:.1f}px too high")
        if locked:
            self.last_ok_H = H
        return Registration(locked, H, n, err, ecc_val, round(q, 3), reason)

    def _moved(self, H1, H2):
        w, h = self.size
        pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        a = cv2.perspectiveTransform(pts, np.linalg.inv(H1))
        b = cv2.perspectiveTransform(pts, np.linalg.inv(H2))
        return float(np.abs(a - b).max())

    def _ecc(self, frame_bgr, H):
        """Refine H with an affine ECC alignment on half-scale images."""
        w, h = self.size
        rect = cv2.warpPerspective(frame_bgr, H, (w, h), flags=cv2.INTER_LINEAR)
        rg = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(rg, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            cc, warp = cv2.findTransformECC(self.gold_small, small, warp, cv2.MOTION_AFFINE,
                                            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.ecc_iters, 1e-5),
                                            None, 3)
        except cv2.error:
            return H, 0.0
        # warp maps golden(small) -> rect(small).  Convert to full scale and compose:
        A = np.vstack([warp, [0, 0, 1]]).astype(np.float64)
        S = np.diag([2.0, 2.0, 1.0])
        A_full = S @ A @ np.linalg.inv(S)          # golden px -> rect px
        H_ref = np.linalg.inv(A_full) @ H           # frame -> rect -> golden
        return H_ref, float(cc)

    def rectify(self, frame_bgr, H):
        w, h = self.size
        return cv2.warpPerspective(frame_bgr, H, (w, h), flags=cv2.INTER_LINEAR)

    def valid_mask(self, frame_shape, H, border=10):
        """Work-px mask of the area actually seen by the camera (minus the target frame line)."""
        w, h = self.size
        ones = np.full(frame_shape[:2], 255, np.uint8)
        m = cv2.warpPerspective(ones, H, (w, h), flags=cv2.INTER_NEAREST)
        m = (cv2.erode(m, np.ones((7, 7), np.uint8)) > 0).astype(np.uint8)
        m[:border], m[-border:], m[:, :border], m[:, -border:] = 0, 0, 0, 0
        return m

    @staticmethod
    def to_frame(H, pts_work):
        Hi = np.linalg.inv(H)
        return cv2.perspectiveTransform(np.asarray(pts_work, np.float32).reshape(-1, 1, 2), Hi).reshape(-1, 2)
