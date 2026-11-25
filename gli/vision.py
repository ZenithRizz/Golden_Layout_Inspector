"""Vision pipeline: golden-reference differencing + defect classifier.

Stage 1 (CPU, OpenCV): compare the rectified camera image with the golden
layout.  A min-over-small-shifts absolute difference tolerates 1-2 px of
registration error, so only real deviations survive.  Connected components
of the thresholded difference are defect *candidates*.

Stage 2 (NPU via HailoRT, or CPU via ONNX Runtime): a small CNN classifies
each candidate crop as NONE (noise/misregistration) or a defect class.  The
crop has 3 channels: captured gray, golden gray, |difference|.

Candidates the classifier rejects but whose difference is strong are kept
as UNKNOWN anomalies so nothing big is ever silently dropped.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

CLASSES = ["NONE", "BRIDGE", "OPEN", "PARTICLE", "MISSING_VIA"]
CROP = 64


@dataclass
class VisionConfig:
    shift: int = 1              # +/- px tolerance for residual misregistration (after ECC)
    blur: float = 1.0           # gaussian sigma before differencing
    thr: int = 35               # difference threshold (0-255)
    min_area: int = 4           # px at work scale
    min_score: float = 26.0     # mean |diff| inside a blob; removes resampling artefacts
    max_area: int = 6000
    merge_px: int = 3           # dilation used to merge nearby blobs
    max_candidates: int = 40
    anomaly_score: float = 110  # mean diff above which a rejected blob becomes UNKNOWN
    anomaly_area: int = 60
    min_conf: float = 0.35      # below this a defect class is not reported


@dataclass
class Detection:
    id: str
    cls: str
    conf: float
    box: list                   # [x, y, w, h] in work px
    score: float                # mean |diff| inside the blob
    area: int
    probs: dict = field(default_factory=dict)


def to_gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalise_to(cap_gray, gold_gray):
    """Least-squares gain/offset so capture brightness matches the golden image."""
    c = cap_gray[::4, ::4].astype(np.float32).ravel()
    g = gold_gray[::4, ::4].astype(np.float32).ravel()
    A = np.stack([c, np.ones_like(c)], 1)
    (a, b), *_ = np.linalg.lstsq(A, g, rcond=None)
    if not np.isfinite(a) or a <= 0.05:
        a, b = 1.0, 0.0
    return np.clip(cap_gray.astype(np.float32) * a + b, 0, 255).astype(np.uint8)


def _shift_tolerant_absdiff(cap, gold, s):
    # min over |dx|,|dy|<=s of |cap - shift(gold)|  ==  distance of cap outside [local min, local max] of gold
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * s + 1, 2 * s + 1))
    gmax = cv2.dilate(gold, k).astype(np.int16)
    gmin = cv2.erode(gold, k).astype(np.int16)
    capi = cap.astype(np.int16)
    return np.clip(np.maximum(capi - gmax, gmin - capi), 0, 255).astype(np.uint8)


def diff_map(cap_img, gold_img, cfg: VisionConfig):
    """Return (difference map uint8, brightness-normalised capture gray).

    Colour inputs are compared per channel (max over channels), which keeps
    layers apart (an m1 bridge over m2 is visible even if gray levels match).
    """
    if cap_img.ndim == 3 and gold_img.ndim == 3:
        chans_c, chans_g = cv2.split(cap_img), cv2.split(gold_img)
    else:
        chans_c, chans_g = [to_gray(cap_img)], [to_gray(gold_img)]
    d = None
    for c, g in zip(chans_c, chans_g):
        cn = normalise_to(c, g)
        if cfg.blur > 0:
            cn = cv2.GaussianBlur(cn, (0, 0), cfg.blur)
            g = cv2.GaussianBlur(g, (0, 0), cfg.blur)
        dc = _shift_tolerant_absdiff(cn, g, cfg.shift)
        d = dc if d is None else np.maximum(d, dc)
    cap_gray_n = normalise_to(to_gray(cap_img), to_gray(gold_img))
    return d, cap_gray_n


def find_candidates(dmap, cfg: VisionConfig, valid_mask=None):
    m = (dmap >= cfg.thr).astype(np.uint8)
    if valid_mask is not None:
        m &= valid_mask
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    if cfg.merge_px > 0:
        m = cv2.dilate(m, np.ones((cfg.merge_px, cfg.merge_px), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < cfg.min_area or area > cfg.max_area:
            continue
        blob = lab[y:y + h, x:x + w] == i
        score = float(dmap[y:y + h, x:x + w][blob].mean())
        if score < cfg.min_score:
            continue
        out.append(([int(x), int(y), int(w), int(h)], score, int(area)))
    out.sort(key=lambda t: -t[1] * t[2])
    return out[: cfg.max_candidates]


def crop_triplet(cap_gray, gold_gray, dmap, box, size=CROP):
    """3-channel crop centred on the box: captured, golden, difference (uint8 HWC)."""
    x, y, w, h = box
    side = int(max(size * 0.75, 1.6 * max(w, h)))
    cx, cy = x + w / 2, y + h / 2
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    H, W = cap_gray.shape

    def cut(img):
        pad = side
        p = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        c = p[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]
        return cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)

    return np.dstack([cut(cap_gray), cut(gold_gray), cut(dmap)])


# ---------------------------------------------------------------------------
# Classifier backends
# ---------------------------------------------------------------------------
def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class OnnxClassifier:
    """CPU backend (ONNX Runtime).  Also the CPU-only baseline for the report."""
    name = "cpu-onnx"

    def __init__(self, path, threads=2):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape
        self.static_batch = isinstance(shape[0], int)

    def __call__(self, crops_hwc: np.ndarray) -> np.ndarray:
        x = crops_hwc.astype(np.float32).transpose(0, 3, 1, 2)  # NCHW, 0-255
        if self.static_batch:
            return np.concatenate([self.sess.run(None, {self.inp: x[i:i + 1]})[0] for i in range(len(x))])
        return self.sess.run(None, {self.inp: x})[0]


class CvDnnClassifier:
    """CPU backend using OpenCV's DNN module (no ONNX Runtime needed - used on Android/Termux).
    Verified identical to ONNX Runtime on this model (max |diff| ~2e-6)."""
    name = "cpu-opencv-dnn"

    def __init__(self, path, threads=4):
        cv2.setNumThreads(threads)
        self.net = cv2.dnn.readNetFromONNX(path)

    def __call__(self, crops_hwc: np.ndarray) -> np.ndarray:
        self.net.setInput(crops_hwc.astype(np.float32).transpose(0, 3, 1, 2))
        return self.net.forward()


class HailoClassifier:
    """NPU backend (HailoRT).  Runs a .hef compiled by scripts/compile_hef.py.

    Uses the hailo_platform InferVStreams API (HailoRT 4.x on Raspberry Pi 5).
    The device stays open for the life of the station.
    """
    name = "hailo-npu"

    def __init__(self, hef_path):
        import hailo_platform as hpf
        self.hpf = hpf
        self.hef = hpf.HEF(hef_path)
        self.vdev = hpf.VDevice()
        params = hpf.ConfigureParams.create_from_hef(self.hef, interface=hpf.HailoStreamInterface.PCIe)
        self.ng = self.vdev.configure(self.hef, params)[0]
        self.ng_params = self.ng.create_params()
        self.in_info = self.hef.get_input_vstream_infos()[0]
        self.out_info = self.hef.get_output_vstream_infos()[0]
        self.in_params = hpf.InputVStreamParams.make_from_network_group(
            self.ng, quantized=False, format_type=hpf.FormatType.FLOAT32)
        self.out_params = hpf.OutputVStreamParams.make_from_network_group(
            self.ng, quantized=False, format_type=hpf.FormatType.FLOAT32)
        self._act = self.ng.activate(self.ng_params)
        self._act.__enter__()
        self._pipe = hpf.InferVStreams(self.ng, self.in_params, self.out_params)
        self._pipe.__enter__()

    def __call__(self, crops_hwc: np.ndarray) -> np.ndarray:
        x = crops_hwc.astype(np.float32)          # NHWC, 0-255 (normalisation is inside the model)
        shp = tuple(self.in_info.shape)
        if len(shp) == 3 and shp[0] == 3:          # unusual NCHW HEF
            x = x.transpose(0, 3, 1, 2)
        res = self._pipe.infer({self.in_info.name: x})
        out = np.asarray(res[self.out_info.name]).reshape(len(x), -1)
        return out

    def close(self):
        for obj in (getattr(self, "_pipe", None), getattr(self, "_act", None)):
            try:
                obj.__exit__(None, None, None)
            except Exception:
                pass
        try:
            self.vdev.release()
        except Exception:
            pass


class DiffOnlyClassifier:
    """Fallback when no model is available: every strong blob is UNKNOWN."""
    name = "diff-only"

    def __call__(self, crops_hwc):
        z = np.zeros((len(crops_hwc), len(CLASSES)), np.float32)
        z[:, 0] = -1.0
        return z


def load_classifier(onnx_path=None, hef_path=None, prefer="hailo"):
    errors = []
    if prefer == "hailo" and hef_path:
        try:
            return HailoClassifier(hef_path), errors
        except Exception as e:  # device missing, driver, version mismatch
            errors.append(f"Hailo NPU unavailable ({e.__class__.__name__}: {e}); using CPU")
    if onnx_path:
        try:
            return OnnxClassifier(onnx_path), errors
        except ImportError:
            pass                                  # no onnxruntime (e.g. Termux): use OpenCV DNN
        except Exception as e:
            errors.append(f"ONNX Runtime failed ({e}); trying OpenCV DNN")
        try:
            return CvDnnClassifier(onnx_path), errors
        except Exception as e:
            errors.append(f"ONNX model unavailable ({e}); using diff-only")
    return DiffOnlyClassifier(), errors


# ---------------------------------------------------------------------------
def classify_candidates(cap_gray, ref_gray, dmap, cands, classifier, cfg: VisionConfig):
    """Classify candidate blobs.  ref_gray is what the area *should* look like
    (the golden layout, or neighbouring cells in reference-free mode)."""
    if cands:
        crops = np.stack([crop_triplet(cap_gray, ref_gray, dmap, b) for b, _, _ in cands])
        probs = softmax(np.asarray(classifier(crops), np.float32))
    else:
        probs = np.zeros((0, len(CLASSES)), np.float32)
    dets = []
    for (box, score, area), p in zip(cands, probs):
        k = int(np.argmax(p))
        kd = int(np.argmax(p[1:])) + 1
        if k == 0:
            if score >= cfg.anomaly_score and area >= cfg.anomaly_area:
                cls, conf = "UNKNOWN", float(1.0 - p[0])
            else:
                continue
        else:
            cls, conf = CLASSES[kd], float(p[kd])
            if conf < cfg.min_conf:
                cls = "UNKNOWN"
        dets.append(Detection(id=f"d{len(dets) + 1:02d}", cls=cls, conf=round(conf, 3), box=box,
                              score=round(score, 1), area=area,
                              probs={c: round(float(v), 3) for c, v in zip(CLASSES, p)}))
    return dets


def detect(cap_img, gold_img, classifier, cfg: VisionConfig, valid_mask=None, timings=None):
    """cap_img/gold_img: rectified capture and golden image (BGR or gray, same size)."""
    t0 = time.perf_counter()
    dmap, cap_n = diff_map(cap_img, gold_img, cfg)
    gold_gray = to_gray(gold_img)
    cands = find_candidates(dmap, cfg, valid_mask)
    t1 = time.perf_counter()
    dets = classify_candidates(cap_n, gold_gray, dmap, cands, classifier, cfg)
    t2 = time.perf_counter()
    if timings is not None:
        timings["diff_ms"] = (t1 - t0) * 1000
        timings["classify_ms"] = (t2 - t1) * 1000
        timings["candidates"] = len(cands)
    return dets, dmap
