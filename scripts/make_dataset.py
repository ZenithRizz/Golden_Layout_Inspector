#!/usr/bin/env python3
"""Generate the classifier training set from the golden design (F1).

For each random tile of the golden layout: inject 0-3 visible defects, apply
camera degradation, run the *same* candidate detector the station uses, and
label every candidate crop by overlap with the injected ground truth.  This
keeps training and runtime distributions identical.

Output: data/<name>/crops.npz  (X uint8 [N,64,64,3], y int, meta json)
        data/<name>/stats.json (candidate recall per class, counts)
        data/<name>/preview.png
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gli.layout import Design, composite  # noqa: E402
from gli.synth import make_defective, camera_sim  # noqa: E402
from gli.vision import (VisionConfig, CLASSES, diff_map, find_candidates,  # noqa: E402
                        crop_triplet, to_gray)

TILE = 256


def _overlap(b, g, m=2):
    return b[0] < g[0] + g[2] + m and g[0] - m < b[0] + b[2] and b[1] < g[1] + g[3] + m and g[1] - m < b[1] + b[3]


def work(args):
    seed, n_tiles, masks_path, ppu = args
    rng = np.random.default_rng(seed)
    masks = dict(np.load(masks_path))
    H, W = masks["m1"].shape
    cfg = VisionConfig()
    X, Y, rec = [], [], {}
    done = 0
    while done < n_tiles:
        y0, x0 = int(rng.integers(0, H - TILE)), int(rng.integers(0, W - TILE))
        tile = {k: v[y0:y0 + TILE, x0:x0 + TILE] for k, v in masks.items()}
        if tile["m1"].mean() < 3 and tile["m2"].mean() < 3:
            continue
        gold = composite(tile)
        n = 0 if rng.random() < 0.12 else int(rng.integers(1, 4))
        if n:
            inj = make_defective(tile, ppu, rng, n, golden=gold)
            if inj is None:
                continue
            img, gts = inj.render(), inj.gts
        else:
            img, gts = gold.copy(), []
        cap, _ = camera_sim(img, rng, strength=float(rng.uniform(0.4, 1.0)))
        dmap, cap_n = diff_map(cap, gold, cfg)
        cands = find_candidates(dmap, cfg)
        gg = to_gray(gold)
        for g in gts:
            r = rec.setdefault(g["cls"], [0, 0])
            r[1] += 1
            r[0] += any(_overlap(b, g["box"]) for b, _, _ in cands)
        for b, score, area in cands:
            lab = 0
            best = 0
            for g in gts:
                if _overlap(b, g["box"]):
                    gb = g["box"]
                    inter = max(0, min(b[0] + b[2], gb[0] + gb[2]) - max(b[0], gb[0])) * \
                        max(0, min(b[1] + b[3], gb[1] + gb[3]) - max(b[1], gb[1]))
                    if inter >= best:
                        best, lab = inter, CLASSES.index(g["cls"])
            X.append(crop_triplet(cap_n, gg, dmap, b))
            Y.append(lab)
        done += 1
    return np.array(X, np.uint8), np.array(Y, np.int64), rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="designs/sram16x16_scn4m")
    ap.add_argument("--out", default="data/sram16x16_scn4m")
    ap.add_argument("--tiles", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    d = Design.load(a.design)
    os.makedirs(a.out, exist_ok=True)
    per = int(np.ceil(a.tiles / a.workers))
    jobs = [(a.seed + i, per, os.path.join(a.design, "masks_work.npz"), d.work_ppu) for i in range(a.workers)]
    with Pool(a.workers) as p:
        res = p.map(work, jobs)
    X = np.concatenate([r[0] for r in res])
    Y = np.concatenate([r[1] for r in res])
    rec = {}
    for r in res:
        for k, (h, t) in r[2].items():
            v = rec.setdefault(k, [0, 0])
            v[0] += h
            v[1] += t
    counts = {c: int((Y == i).sum()) for i, c in enumerate(CLASSES)}
    stats = {"tiles": a.tiles, "crops": int(len(Y)), "counts": counts,
             "candidate_recall": {k: round(h / max(t, 1), 3) for k, (h, t) in rec.items()},
             "defects": {k: t for k, (h, t) in rec.items()}, "classes": CLASSES}
    np.savez_compressed(os.path.join(a.out, "crops.npz"), X=X, y=Y)
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    # preview grid: 8 per class
    rows = []
    for i, c in enumerate(CLASSES):
        idx = np.where(Y == i)[0][:8]
        tiles = [np.hstack([X[j][..., 0], X[j][..., 1], X[j][..., 2]]) for j in idx]
        while len(tiles) < 8:
            tiles.append(np.zeros((64, 192), np.uint8))
        row = np.hstack(tiles)
        row = cv2.copyMakeBorder(row, 16, 2, 0, 0, cv2.BORDER_CONSTANT, value=255)
        cv2.putText(row, c, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1)
        rows.append(row)
    cv2.imwrite(os.path.join(a.out, "preview.png"), np.vstack(rows))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
