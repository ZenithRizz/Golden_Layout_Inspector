"""Synthetic defect engine (F1).

Defects are injected into the *layer masks* of the golden layout, so they are
physically consistent (a bridge adds metal on the right layer, an open
removes it), or painted on top in the style of the physical demo targets
(marker ink, correction fluid, dust).  Every defect carries an exact
ground-truth box, class and layer, so no real defect image is ever needed.
"""
from __future__ import annotations

import cv2
import numpy as np

from .layout import composite, CUT_LAYERS, RENDER_STYLE

DEFECT_CLASSES = ["BRIDGE", "OPEN", "PARTICLE", "MISSING_VIA"]


def _march(M, p, d, lmax):
    """Distance along direction d from p until M is hit (None if not within lmax)."""
    h, w = M.shape
    x, y = p
    for k in range(1, lmax + 1):
        xx, yy = x + d[0] * k, y + d[1] * k
        if not (0 <= xx < w and 0 <= yy < h):
            return None
        if M[yy, xx]:
            return k
    return None


def _run(M, p, d, lmax=200):
    h, w = M.shape
    x, y = p
    k = 0
    while k < lmax:
        xx, yy = x + d[0] * (k + 1), y + d[1] * (k + 1)
        if not (0 <= xx < w and 0 <= yy < h) or not M[yy, xx]:
            break
        k += 1
    return k


_ORDER = [n for n, _, _ in RENDER_STYLE]


def _covered_by_higher(masks, layer, x0, y0, x1, y1):
    """Fraction of the box covered by any layer drawn above `layer` (excluding cut layers)."""
    i = _ORDER.index(layer)
    acc = None
    for name in _ORDER[i + 1:]:
        if name in CUT_LAYERS or name not in masks:
            continue
        sub = masks[name][max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] > 0
        if sub.size == 0:
            return 1.0
        acc = sub if acc is None else (acc | sub)
    return 0.0 if acc is None else float(acc.mean())


class Injector:
    def __init__(self, masks: dict, ppu: float, rng: np.random.Generator):
        self.m = {k: v.copy() for k, v in masks.items()}
        self.ppu = ppu
        self.rng = rng
        self.paint = []   # deferred paint ops applied after compositing
        self.gts = []

    # ------------------------------------------------------------------
    def bridge(self):
        rng = self.rng
        layer = rng.choice(["m1", "m2", "m1", "m2", "poly"])
        M = self.m[layer] > 0
        if M.sum() < 50:
            return False
        h, w = M.shape
        lmax = int(3.5 * self.ppu)
        for _ in range(300):
            p = (int(rng.integers(2, w - 2)), int(rng.integers(2, h - 2)))
            if M[p[1], p[0]]:
                continue
            dirs = [(1, 0), (0, 1)] if rng.random() < 0.5 else [(0, 1), (1, 0)]
            for d in dirs:
                a = _march(M, p, d, lmax)
                b = _march(M, p, (-d[0], -d[1]), lmax)
                if a is None or b is None or a + b < 3:
                    continue
                p0 = (p[0] - d[0] * (b + 1), p[1] - d[1] * (b + 1))
                p1 = (p[0] + d[0] * (a + 1), p[1] + d[1] * (a + 1))
                t = int(rng.integers(3, 6))
                bx0, bx1 = min(p0[0], p1[0]) - t, max(p0[0], p1[0]) + t
                by0, by1 = min(p0[1], p1[1]) - t, max(p0[1], p1[1]) + t
                if _covered_by_higher(self.m, layer, bx0, by0, bx1, by1) > 0.3:
                    continue
                style = "mask" if rng.random() < 0.6 else "ink"
                if style == "mask":
                    cv2.line(self.m[layer], p0, p1, 255, t)
                else:
                    col = [(170, 60, 20), (40, 40, 40), (150, 40, 150), (30, 30, 160)][rng.integers(4)]
                    self.paint.append(("line", p0, p1, t + 1, col, 0.9))
                x0, x1 = min(p0[0], p1[0]) - t, max(p0[0], p1[0]) + t
                y0, y1 = min(p0[1], p1[1]) - t, max(p0[1], p1[1]) + t
                self.gts.append({"cls": "BRIDGE", "layer": layer, "style": style,
                                 "box": [x0, y0, x1 - x0, y1 - y0]})
                return True
        return False

    def open(self):
        rng = self.rng
        layer = rng.choice(["m1", "m2", "m1", "m2", "poly"])
        M = (self.m[layer] > 0).astype(np.uint8)
        if M.sum() < 50:
            return False
        dt = cv2.distanceTransform(M, cv2.DIST_L2, 3)
        ys, xs = np.where((dt >= 1) & (dt <= 3.5))
        if len(xs) == 0:
            return False
        for _ in range(60):
            i = int(rng.integers(len(xs)))
            p = (int(xs[i]), int(ys[i]))
            hr = _run(M, p, (1, 0)) + _run(M, p, (-1, 0))
            vr = _run(M, p, (0, 1)) + _run(M, p, (0, -1))
            if max(hr, vr) < 3 * min(hr, vr) or max(hr, vr) < 12:
                continue  # not clearly a line
            wcut = int(rng.integers(3, 7))
            if hr > vr:   # horizontal line -> vertical cut
                up, dn = _run(M, p, (0, -1)), _run(M, p, (0, 1))
                x0, x1 = p[0] - wcut // 2, p[0] + (wcut + 1) // 2
                y0, y1 = p[1] - up - 1, p[1] + dn + 2
            else:
                lf, rt = _run(M, p, (-1, 0)), _run(M, p, (1, 0))
                y0, y1 = p[1] - wcut // 2, p[1] + (wcut + 1) // 2
                x0, x1 = p[0] - lf - 1, p[0] + rt + 2
            if _covered_by_higher(self.m, layer, x0, y0, x1, y1) > 0.25:
                continue
            style = "mask" if rng.random() < 0.6 else "whiteout"
            if style == "mask":
                self.m[layer][y0:y1, x0:x1] = 0
            else:
                self.paint.append(("rect", (x0 - 1, y0 - 1), (x1 + 1, y1 + 1), -1, (250, 250, 250), 1.0))
            self.gts.append({"cls": "OPEN", "layer": layer, "style": style,
                             "box": [x0 - 1, y0 - 1, x1 - x0 + 2, y1 - y0 + 2]})
            return True
        return False

    def particle(self):
        rng = self.rng
        h, w = next(iter(self.m.values())).shape
        r = float(rng.uniform(2.5, 12.0))
        cx, cy = rng.uniform(r + 2, w - r - 2), rng.uniform(r + 2, h - r - 2)
        n = int(rng.integers(6, 11))
        ang = np.sort(rng.uniform(0, 2 * np.pi, n))
        rad = r * rng.uniform(0.55, 1.3, n)
        pts = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], 1).astype(np.int32)
        col = [(30, 30, 30), (60, 70, 90), (90, 90, 90), (40, 60, 110)][rng.integers(4)]
        self.paint.append(("poly", pts, None, -1, col, float(rng.uniform(0.75, 1.0))))
        x, y, bw, bh = cv2.boundingRect(pts)
        self.gts.append({"cls": "PARTICLE", "layer": "foreign", "style": "particle",
                         "box": [x - 1, y - 1, bw + 2, bh + 2]})
        return True

    def missing_via(self):
        rng = self.rng
        layer = rng.choice(list(CUT_LAYERS))
        M = (self.m[layer] > 0).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(M, 8)
        if n <= 1:
            return False
        i = int(rng.integers(1, n))
        x, y, bw, bh, _ = stats[i]
        if bw > 12 or bh > 12:
            return False
        style = "mask" if rng.random() < 0.7 else "whiteout"
        if style == "mask":
            self.m[layer][lab == i] = 0
        else:
            self.paint.append(("rect", (x - 1, y - 1), (x + bw, y + bh), -1, (250, 250, 250), 1.0))
        self.gts.append({"cls": "MISSING_VIA", "layer": layer, "style": style,
                         "box": [int(x) - 2, int(y) - 2, int(bw) + 4, int(bh) + 4]})
        return True

    def add(self, cls):
        return {"BRIDGE": self.bridge, "OPEN": self.open, "PARTICLE": self.particle,
                "MISSING_VIA": self.missing_via}[cls]()

    def render(self, background=(245, 245, 245)):
        img = composite(self.m, background)
        for op, a, b, t, col, alpha in self.paint:
            over = img.copy()
            if op == "line":
                cv2.line(over, a, b, col, t, cv2.LINE_AA)
            elif op == "rect":
                cv2.rectangle(over, a, b, col, t)
            elif op == "poly":
                cv2.fillPoly(over, [a], col, cv2.LINE_AA)
            img = cv2.addWeighted(over, alpha, img, 1 - alpha, 0)
        return img


def camera_sim(img, rng, strength=1.0, misreg_px=0.6):
    """Degrade a clean render to look like a webcam / USB microscope capture.

    misreg_px: residual misregistration after homography + ECC refinement.
    """
    h, w = img.shape[:2]
    ang = np.degrees(rng.uniform(-1, 1) * misreg_px / (w / 2))
    sc = 1 + rng.uniform(-1, 1) * misreg_px / w
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    M[:, 2] += rng.uniform(-misreg_px, misreg_px, 2)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out = out.astype(np.float32)
    out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.4, 1.4) * max(strength, 0.3))
    gain = rng.uniform(0.75, 1.15) * rng.uniform(0.92, 1.08, 3)
    out = out * gain + rng.uniform(-20, 15)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    plane = 1 + rng.uniform(-0.15, 0.15) * (xx / w - 0.5) + rng.uniform(-0.15, 0.15) * (yy / h - 0.5)
    out = out * plane[..., None]
    g = rng.uniform(0.85, 1.2)
    out = 255 * (np.clip(out, 0, 255) / 255) ** g
    out = out + rng.normal(0, rng.uniform(1, 6), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    q = int(rng.integers(55, 96))
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR), M


def _contrast(a, b, box):
    x, y, w, h = box
    x0, y0 = max(x, 0), max(y, 0)
    pa, pb = a[y0:y + h, x0:x + w].astype(np.int16), b[y0:y + h, x0:x + w].astype(np.int16)
    return int(np.abs(pa - pb).max()) if pa.size else 0


def make_defective(masks, ppu, rng, n_defects=1, classes=DEFECT_CLASSES, min_sep=24,
                   min_contrast=70, golden=None):
    """Inject up to n_defects *optically visible* defects.

    A defect hidden under other layers (e.g. an open in m1 beneath m2, or a via
    stacked on another cut) has no optical signature, so no camera-based method
    could see it.  Those are rejected (contrast < min_contrast) so labels stay
    honest.  Returns None if the tile should be regenerated.
    """
    golden = composite(masks) if golden is None else golden
    inj = Injector(masks, ppu, rng)
    tries = 0
    while len(inj.gts) < n_defects and tries < n_defects * 10:
        tries += 1
        before = len(inj.gts)
        snap_m = {k: v.copy() for k, v in inj.m.items()}
        snap_p = list(inj.paint)
        cls = classes[int(rng.integers(len(classes)))]
        if not inj.add(cls) or len(inj.gts) == before:
            continue
        b = inj.gts[-1]["box"]
        if _contrast(inj.render(), golden, b) < min_contrast:
            inj.m, inj.paint = snap_m, snap_p   # invisible: undo and try again
            inj.gts.pop()
            continue
        c = np.array([b[0] + b[2] / 2, b[1] + b[3] / 2])
        for g in inj.gts[:-1]:
            gb = g["box"]
            if np.linalg.norm(c - [gb[0] + gb[2] / 2, gb[1] + gb[3] / 2]) < min_sep:
                return None
    return inj
