"""Reference-free inspection for dies we have no layout for (uploaded photos).

Memory arrays repeat.  Industrial array inspection uses this: each cell is
compared with its neighbours one period away ("cell-to-cell" comparison).  A
real defect differs from its neighbours on *both* sides, whereas the copy
("ghost") it creates in a shifted comparison is clean on at least one side, so
the minimum over the neighbour comparisons keeps only the defect.

Steps
  1. estimate the repeat period in x and y from the image's autocorrelation;
  2. compare the image with copies shifted by +-1 period (min over neighbours);
  3. keep only areas that are actually periodic (the array), so non-repeating
     periphery doesn't light up;
  4. reuse the station's candidate finder and CNN classifier, with the median of
     the neighbouring cells as the "reference" image.

Without a layout we can find and classify anomalies, but not map them to
bitcells, nets or spare rows/columns, so every finding goes to engineer REVIEW.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .vision import VisionConfig, classify_candidates, find_candidates, to_gray


def _period_1d(ac, min_lag, max_lag):
    seg = ac[min_lag:max_lag]
    if len(seg) < 3:
        return None, 0.0
    # local maxima, then the smallest lag whose peak is close to the best (avoids picking 2x the period)
    peaks = [i for i in range(1, len(seg) - 1) if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]]
    if not peaks:
        return None, 0.0
    best = max(seg[i] for i in peaks)
    if best <= 0:
        return None, 0.0
    for i in peaks:
        if seg[i] >= 0.85 * best:
            return i + min_lag, float(seg[i])
    return None, 0.0


def estimate_period(gray, min_lag=5, max_frac=0.3):
    """Return ((px, strength_x), (py, strength_y)); strength is normalised autocorrelation (0..1)."""
    g = gray.astype(np.float32)
    g = g - cv2.GaussianBlur(g, (0, 0), 25)          # remove illumination gradients
    g -= g.mean()
    h, w = g.shape
    out = []
    for axis, n in ((1, w), (0, h)):
        F = np.fft.rfft(g, n=2 * n, axis=axis)
        ac = np.fft.irfft(F * np.conj(F), axis=axis)
        ac = ac[:, :n] if axis == 1 else ac[:n, :]
        prof = ac.mean(axis=1 - axis) if axis == 1 else ac.mean(axis=1)
        prof = prof / max(prof[0], 1e-6)
        # unbiased: divide by overlap length
        prof = prof * (n / np.maximum(n - np.arange(n), 1))
        out.append(_period_1d(prof, min_lag, max(int(n * max_frac), min_lag + 3)))
    return out[0], out[1]


def _shift(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_REPLICATE)


def neighbour_diff(img, px, py, blur=1.0):
    """Min over the +-1 period neighbours of |img - neighbour| (max over colour channels).

    Near the image edge only the neighbours that exist are used; a pixel needs at least two
    real neighbours to be judged.  Also returns the median-of-neighbours reference image.
    """
    src = cv2.GaussianBlur(img, (0, 0), blur) if blur > 0 else img
    h, w = src.shape[:2]
    shifts = []
    if px:
        shifts += [(px, 0), (-px, 0)]
    if py:
        shifts += [(0, py), (0, -py)]
    s = src.astype(np.int16)
    diffs, refs, oks = [], [], []
    ys, xs = np.mgrid[0:h, 0:w]
    for dx, dy in shifts:
        n = _shift(src, dx, dy)
        d = np.abs(s - n.astype(np.int16))
        d = d.max(axis=2) if d.ndim == 3 else d
        ok = (xs - dx >= 0) & (xs - dx < w) & (ys - dy >= 0) & (ys - dy < h)   # neighbour inside the image
        diffs.append(np.where(ok, d, 255))
        oks.append(ok)
        refs.append(_shift(img, dx, dy))
    D = np.stack(diffs)
    dmin = D.min(axis=0).astype(np.uint8)
    n_ok = np.sum(np.stack(oks), axis=0)
    valid = (n_ok >= 2).astype(np.uint8)
    dmin[valid == 0] = 0
    # reference: neighbour with the smallest difference (what this spot should look like)
    best = D.argmin(axis=0)
    R = np.stack(refs)
    ref = np.take_along_axis(R, best[None, ..., None] if R.ndim == 4 else best[None], axis=0)[0]
    return dmin, ref.astype(np.uint8), valid


def periodic_mask(gray, px, py, min_ncc=0.6, min_std=6.0):
    """Array area = textured AND locally correlated with its copy one period away.

    Local normalised cross-correlation (box filters over ~2 periods).  Plain background is
    excluded (no texture), random periphery is excluded (no correlation).
    """
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    win = int(np.clip(2 * max(px or 0, py or 0), 15, 81)) | 1
    k = (win, win)
    mu = cv2.blur(g, k)
    var = np.maximum(cv2.blur(g * g, k) - mu * mu, 0)
    sd = np.sqrt(var)
    best = np.zeros_like(g)
    shifts = ([(px, 0), (-px, 0)] if px else []) + ([(0, py), (0, -py)] if py else [])
    for dx, dy in shifts:
        n = _shift(g, dx, dy)
        mun = cv2.blur(n, k)
        sdn = np.sqrt(np.maximum(cv2.blur(n * n, k) - mun * mun, 0))
        cov = cv2.blur(g * n, k) - mu * mun
        ncc = cov / (sd * sdn + 1e-3)
        best = np.maximum(best, ncc)
    m = ((best > min_ncc) & (sd > min_std)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))   # fill holes left by defects
    # drop small islands (random periphery that happens to correlate): keep regions >= 3x3 cells
    cell = max(px or 1, 1) * max(py or 1, 1)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 9 * cell
    m = keep[lab].astype(np.uint8)
    # stay about one cell inside the array edge, where the window mixes array and periphery
    e = int(max(px or 0, py or 0, 9)) | 1
    return cv2.erode(m, np.ones((e, e), np.uint8))


def inspect_reference_free(img_bgr, classifier, cfg: VisionConfig | None = None, max_side=1400,
                           min_strength=0.25):
    """Return dict with dets, analysed image, reference image, dmap, mask, period info, timings,
    or {'error': ...} if no repeating array is found."""
    t0 = time.perf_counter()
    cfg = cfg or VisionConfig()
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    img = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else img_bgr.copy()
    gray = to_gray(img)
    (px, sx), (py, sy) = estimate_period(gray)
    if not px or sx < min_strength:
        px, sx = None, sx
    if not py or sy < min_strength:
        py, sy = None, sy
    if not px and not py:
        return {"error": "No repeating memory array found in this image. Reference-free mode needs a photo of an "
                         "array area (repeating cells), sharp and roughly straight. For the station's own SRAM, "
                         "use the printed target with its four corner markers."}
    t1 = time.perf_counter()
    dmin, ref, valid = neighbour_diff(img, px, py, blur=cfg.blur)
    mask = periodic_mask(gray, px, py) & valid
    cands = find_candidates(dmin, cfg, mask)
    t2 = time.perf_counter()
    dets = classify_candidates(gray, to_gray(ref), dmin, cands, classifier, cfg)
    t3 = time.perf_counter()
    return {
        "dets": dets, "image": img, "reference": ref, "dmap": dmin, "mask": mask, "scale": scale,
        "period": {"px": px, "py": py, "strength_x": round(sx, 3), "strength_y": round(sy, 3)},
        "array_fraction": round(float(mask.mean()), 3),
        "timings": {"period_ms": (t1 - t0) * 1000, "diff_ms": (t2 - t1) * 1000,
                    "classify_ms": (t3 - t2) * 1000, "candidates": len(cands)},
    }
