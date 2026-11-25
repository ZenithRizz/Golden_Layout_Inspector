"""Frame sources: USB camera, Pi camera, image files, and a virtual camera.

The virtual camera renders the printed target with injected defects and a
random perspective view, so the whole station can be developed, tested and
demoed without hardware.  It also exposes the ground truth for evaluation.
"""
from __future__ import annotations

import os
import threading
import time

import cv2
import numpy as np

from .layout import Design
from .synth import make_defective, camera_sim


class OpenCVCamera:
    kind = "usb"

    def __init__(self, index=0, width=1280, height=720, fps=30):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2) if os.name != "nt" else cv2.VideoCapture(index)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"camera {index} not available")

    def read(self):
        ok, f = self.cap.read()
        return f if ok else None

    def close(self):
        self.cap.release()


class PiCamera:
    kind = "picamera2"

    def __init__(self, width=1280, height=720):
        from picamera2 import Picamera2
        self.cam = Picamera2()
        cfg = self.cam.create_video_configuration(main={"size": (width, height), "format": "RGB888"})
        self.cam.configure(cfg)
        self.cam.start()

    def read(self):
        return self.cam.capture_array()[..., :3].copy()

    def close(self):
        self.cam.stop()


class HttpCamera:
    """Phone camera via an IP-camera app (e.g. 'IP Webcam' on Android) on the same device or LAN.

    Polls the app's single-JPEG endpoint in a background thread; no internet is involved.
    spec examples: http://127.0.0.1:8080              (no login)
                   http://user:pass@127.0.0.1:8080    (IP Webcam 'Login/password' set)
    Credentials can instead come from env vars GLI_CAM_USER / GLI_CAM_PASS
    (use these if the password contains @ : / characters).
    """
    kind = "http"

    def __init__(self, url, timeout=3.0):
        import base64
        import urllib.request
        from urllib.parse import urlsplit, urlunsplit, unquote
        self._ur = urllib.request
        parts = urlsplit(url.strip())
        user = unquote(parts.username) if parts.username else os.environ.get("GLI_CAM_USER")
        pwd = unquote(parts.password) if parts.password else os.environ.get("GLI_CAM_PASS", "")
        host = parts.hostname + (f":{parts.port}" if parts.port else "")
        path = parts.path.rstrip("/")
        if not path.lower().endswith((".jpg", ".jpeg")):
            path += "/shot.jpg"
        self.url = urlunsplit((parts.scheme, host, path, parts.query, ""))
        self.display = self.url                       # never show the password
        self.headers = {}
        if user:
            tok = base64.b64encode(f"{user}:{pwd}".encode()).decode()
            self.headers["Authorization"] = "Basic " + tok
        self.timeout = timeout
        self.frame, self.err = None, ""
        self._stop = False
        self._fetch()                       # fail fast if the app isn't running
        if self.frame is None:
            hint = " Wrong login/password?" if "401" in self.err else " Is the IP Webcam server started?"
            raise RuntimeError(f"no image from {self.url} ({self.err}).{hint}")
        threading.Thread(target=self._loop, daemon=True).start()

    def _fetch(self):
        try:
            req = self._ur.Request(self.url, headers=self.headers)
            with self._ur.urlopen(req, timeout=self.timeout) as r:
                data = r.read()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                self.frame, self.err = img, ""
        except Exception as e:
            self.err = str(e)

    def _loop(self):
        while not self._stop:
            self._fetch()
            time.sleep(0.01)

    def read(self):
        return None if self.frame is None or self.err else self.frame

    def close(self):
        self._stop = True


class FileSource:
    kind = "file"

    def __init__(self, path):
        self.frame = cv2.imread(path)
        if self.frame is None:
            raise RuntimeError(f"cannot read image {path}")

    def read(self):
        return self.frame.copy()

    def close(self):
        pass


class VirtualCamera:
    """Simulated camera looking at the printed target (with defects)."""
    kind = "virtual"

    def __init__(self, design_dir, n_defects=3, seed=None, size=(1600, 1200), jitter=True):
        self.d = Design.load(design_dir)
        self.masks = dict(np.load(os.path.join(design_dir, "masks_work.npz")))
        self.print_img = cv2.imread(os.path.join(design_dir, "target_print.png"))
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.jitter = jitter
        self.n_defects = n_defects
        self.lock = threading.Lock()
        self.new_die()

    def new_die(self, n_defects=None, classes=None):
        """Render a new 'die' (target) with fresh random defects."""
        n = self.n_defects if n_defects is None else n_defects
        golden = None
        for _ in range(20):
            if n == 0:
                from .layout import composite
                img = composite(self.masks)
                gts = []
                break
            kw = {} if classes is None else {"classes": classes}
            inj = make_defective(self.masks, self.d.work_ppu, self.rng, n, min_sep=40, **kw)
            if inj is not None and len(inj.gts) == n:
                img, gts = inj.render(), inj.gts
                break
        else:
            img, gts = inj.render(), inj.gts
        # paste the defective layout into the print-scale target
        b = self.d.print_border_px
        tgt = self.print_img.copy()
        h, w = tgt.shape[0] - 2 * b, tgt.shape[1] - 2 * b
        up = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        # keep paper white where the layout render is background
        tgt[b:b + h, b:b + w] = up
        with self.lock:
            self.target = tgt
            self.gts = gts
            self._pose()
        return gts

    def _pose(self):
        """Random camera pose: whole target visible with mild perspective."""
        H_t, W_t = self.target.shape[:2]
        fw, fh = self.size
        s = min(fw / W_t, fh / H_t) * self.rng.uniform(0.86, 0.95)
        cx, cy = fw / 2 + self.rng.uniform(-20, 20), fh / 2 + self.rng.uniform(-15, 15)
        ang = np.deg2rad(self.rng.uniform(-4, 4))
        src = np.float32([[0, 0], [W_t, 0], [W_t, H_t], [0, H_t]])
        pts = np.float32([[-W_t / 2, -H_t / 2], [W_t / 2, -H_t / 2], [W_t / 2, H_t / 2], [-W_t / 2, H_t / 2]]) * s
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        dst = pts @ R.T + [cx, cy]
        dst += self.rng.uniform(-12, 12, dst.shape)   # perspective
        self.Hcam = cv2.getPerspectiveTransform(src, dst.astype(np.float32))

    def read(self):
        with self.lock:
            tgt, Hc = self.target, self.Hcam
            if self.jitter:
                J = np.eye(3)
                J[:2, 2] = self.rng.normal(0, 0.15, 2)
                Hc = J @ Hc
        fw, fh = self.size
        frame = cv2.warpPerspective(tgt, Hc, (fw, fh), flags=cv2.INTER_AREA, borderValue=(90, 95, 100))
        frame, _ = camera_sim(frame, self.rng, strength=0.5, misreg_px=0.0)
        time.sleep(0.01)
        return frame

    def close(self):
        pass


class Rotated:
    """Rotate frames from any source by 90/180/270 degrees (env GLI_CAM_ROTATE).
    Phone IP-camera apps send the sensor's landscape image even when the phone is held upright."""
    _CODES = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}

    def __init__(self, src, degrees):
        self.src, self.code = src, self._CODES[degrees]
        self.kind = getattr(src, "kind", "camera")
        self.display = getattr(src, "display", None)

    def read(self):
        f = self.src.read()
        return None if f is None else cv2.rotate(f, self.code)

    def close(self):
        self.src.close()

    def __getattr__(self, name):              # e.g. VirtualCamera.new_die, .gts
        return getattr(self.src, name)


def open_source(spec: str, design_dir: str | None = None):
    src = _open_source(spec, design_dir)
    rot = int(os.environ.get("GLI_CAM_ROTATE", "0") or 0) % 360
    if rot in (90, 180, 270) and not isinstance(src, FileSource):
        return Rotated(src, rot)
    return src


def _open_source(spec: str, design_dir: str | None = None):
    """spec: 'virtual', 'picam', 'usb:0', '0', 'http://127.0.0.1:8080' (phone IP camera), or an image path."""
    if spec == "virtual":
        return VirtualCamera(design_dir)
    if spec.startswith("http://") or spec.startswith("https://"):
        return HttpCamera(spec)
    if spec == "picam":
        return PiCamera()
    if spec.startswith("usb:"):
        return OpenCVCamera(int(spec.split(":")[1]))
    if spec.isdigit():
        return OpenCVCamera(int(spec))
    return FileSource(spec)
