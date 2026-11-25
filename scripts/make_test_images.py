#!/usr/bin/env python3
"""Generate test images for the Upload button, with an answer key.

Two kinds of image, matching the station's two inspection modes:

  target_*.jpg   Photos of the PRINTED TARGET (our OpenRAM 16x16 SRAM, with the 4 corner markers).
                 Compared against the golden OpenRAM layout: full mapping to bitcells/nets and a
                 repair/scrap decision.
  die_*.jpg      Photos of memory arrays WITHOUT markers (stand-ins for "any die").
                 Reference-free cell-to-cell check: each cell is compared with its neighbours.

Each image is kept only if the station, run on it, finds every planted defect with the expected
outcome, so the answer key matches what you will see.  Ground-truth overlays are saved as
answers/*_answer.png.

  python scripts/make_test_images.py --out test_images
"""
import argparse
import json
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gli.layout import Design  # noqa: E402
from gli.station import Station  # noqa: E402
from gli.synth import camera_sim, make_defective  # noqa: E402


def hit(b, g, m=5):
    return b[0] < g[0] + g[2] + m and g[0] - m < b[0] + b[2] and b[1] < g[1] + g[3] + m and g[1] - m < b[1] + b[3]


def matches(dets, gts):
    """Return (all planted found, no extras, class correct for all, mapping of gt -> det index)."""
    used, pairs = set(), []
    for g in gts:
        j = next((i for i, d in enumerate(dets) if i not in used and hit(d["box"], g["box"])), None)
        if j is None:
            return False, False, False, []
        used.add(j)
        pairs.append(j)
    extras = len(dets) - len(used)
    cls_ok = all(dets[j]["cls"] == g["cls"] for g, j in zip(gts, pairs))
    return True, extras == 0, cls_ok, pairs


def draw_answers(img, boxes_labels, scale=1.0, H=None):
    out = img.copy()
    for (x, y, w, h), lab in boxes_labels:
        if H is not None:   # work px -> frame px
            pts = cv2.perspectiveTransform(np.float32([[x, y], [x + w, y + h]]).reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)
            (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
        else:
            x0, y0, x1, y1 = x * scale, y * scale, (x + w) * scale, (y + h) * scale
        c = (int((x0 + x1) / 2), int((y0 + y1) / 2))
        r = int(max(x1 - x0, y1 - y0) / 2 + 14)
        cv2.circle(out, c, r, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(out, lab, (c[0] + r + 4, c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, lab, (c[0] + r + 4, c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return out


def other_memory_die(rng, defects=True):
    """A memory array of a different design (cell 34x22 px) beside non-repeating periphery."""
    H, W = 900, 1300
    img = np.full((H, W, 3), (200, 210, 215), np.uint8)
    cw, ch = 34, 22
    for y in range(0, H, ch):
        for x in range(260, W, cw):
            cv2.rectangle(img, (x + 3, y + 2), (x + 14, y + 19), (70, 110, 170), -1)
            cv2.rectangle(img, (x + 18, y + 4), (x + 30, y + 9), (40, 150, 90), -1)
            cv2.line(img, (x + 24, y), (x + 24, y + ch), (150, 60, 150), 3)
            cv2.circle(img, (x + 9, y + 11), 2, (30, 30, 30), -1)
        cv2.line(img, (260, y + 14), (W, y + 14), (180, 120, 40), 2)
    for _ in range(60):
        x, y = rng.integers(0, 240), rng.integers(0, H - 30)
        cv2.rectangle(img, (int(x), int(y)), (int(x + rng.integers(8, 40)), int(y + rng.integers(4, 25))),
                      tuple(int(c) for c in rng.integers(40, 200, 3)), -1)
    gts = []
    if defects:
        # particle, bridge between neighbouring vertical lines, open (cut) in a horizontal line
        cx, cy = 260 + 34 * int(rng.integers(8, 26)) + 16, 22 * int(rng.integers(5, 35)) + 7
        cv2.circle(img, (cx, cy), 7, (25, 25, 25), -1)
        gts.append({"cls": "PARTICLE", "box": [cx - 8, cy - 8, 16, 16]})
        bx, by = 260 + 34 * int(rng.integers(8, 26)) + 24, 22 * int(rng.integers(5, 35)) + 4
        cv2.line(img, (bx, by), (bx + 12, by), (150, 60, 150), 4)
        gts.append({"cls": "BRIDGE", "box": [bx - 4, by - 4, 22, 10]})
        ox, oy = 260 + 34 * int(rng.integers(8, 26)) + 5, 22 * int(rng.integers(5, 35)) + 14
        cv2.rectangle(img, (ox, oy - 2), (ox + 8, oy + 2), (200, 210, 215), -1)
        gts.append({"cls": "OPEN", "box": [ox - 2, oy - 5, 14, 12]})
    return cv2.GaussianBlur(img, (0, 0), 0.8), gts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="designs/sram16x16_scn4m")
    ap.add_argument("--out", default="test_images")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "answers"), exist_ok=True)
    st = Station(a.design, "virtual", onnx="models/defect_cls.onnx", runs=tempfile.mkdtemp(prefix="gli_testimg_"))
    st._stop = True
    cam = st.src
    cam.jitter = False
    d = Design.load(a.design)
    key = {}

    def keep(name, img, rec, gts, answer_img, expected):
        p = os.path.join(a.out, name)
        cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(os.path.join(a.out, "answers", name.replace(".jpg", "_answer.png")), answer_img)
        rows = []
        for g in gts:
            j = next(i for i, dd in enumerate(rec["defects"]) if hit(dd["box"], g["box"]))
            mp, sv = rec["mapping"][j], rec["decision"]["defects"][j]
            rows.append({"class": g["cls"], "region": mp["region"], "nets": mp["nets"][:6],
                         "rows": sv["full_rows"] or mp["affected_rows"], "cols": sv["full_cols"] or mp["affected_cols"],
                         "severity": sv["severity"], "action": sv["expected_action"]})
        key[name] = {"mode": rec.get("mode", "layout (OpenRAM golden)"), "expected_die_decision": expected,
                     "station_result": rec["decision"]["die_decision"], "defects": rows,
                     "allocation": rec["decision"]["allocation"] if rec.get("mode") != "reference_free" else None}
        print(f"{name:34s} {expected:10s} {[r['class'] + '@' + r['region'] for r in rows]}")

    # ---- printed-target photos (OpenRAM layout comparison) -------------------------------
    wanted = [("target_repair.jpg", "REPAIR", 2, ["OPEN", "BRIDGE", "PARTICLE"]),
              ("target_scrap.jpg", "SCRAP", 2, ["PARTICLE", "BRIDGE"]),
              ("target_pass.jpg", "PASS", 0, None)]
    for name, want, n, classes in wanted:
        for s in range(400):
            cam.rng = np.random.default_rng(a.seed + s * 7 + n)
            gts = cam.new_die(n, classes=classes)
            if len(gts) != n:
                continue
            frame = cam.read()
            rec = st.inspect(frame=frame, die_id=name.split(".")[0].upper())
            if "error" in rec:
                continue
            found, clean, cls_ok, _ = matches(rec["defects"], gts)
            if found and clean and cls_ok and rec["decision"]["die_decision"] == want:
                if want == "SCRAP" and not any(x["severity"] == "KILLER" for x in rec["decision"]["defects"]):
                    continue
                if want == "REPAIR" and not any(r in ("BITCELL_ARRAY",) for r in (m["region"] for m in rec["mapping"])):
                    continue
                Hm = st.reg.register(frame).H
                ans = draw_answers(frame, [(g["box"], g["cls"]) for g in gts], H=Hm)
                keep(name, frame, rec, gts, ans, want)
                break
        else:
            print("could not make", name)

    # ---- die photos without markers (reference-free, cell-to-cell) ---------------------------
    masks = dict(np.load(os.path.join(a.design, "masks_work.npz")))
    arr = next(r for r in d.regions if r["region"] == "BITCELL_ARRAY")["rect"]
    x0, y1 = d.um_to_work(arr[0] + 2, arr[1] + 2)
    x1, y0 = d.um_to_work(arr[2] - 2, arr[3] - 2)
    x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
    rng = np.random.default_rng(a.seed)
    for s in range(200):
        tile = {k: v[y0:y1, x0:x1] for k, v in masks.items()}
        inj = make_defective(tile, d.work_ppu, rng, 3, classes=["BRIDGE", "PARTICLE", "OPEN"], min_sep=70)
        if inj is None or len(inj.gts) != 3:
            continue
        sc = 1.6
        img = cv2.resize(inj.render(), None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        img, _ = camera_sim(img, rng, 0.5, 0.0)
        gts = [{"cls": g["cls"], "box": [int(v * sc) for v in g["box"]]} for g in inj.gts]
        rec = st.inspect(frame=img, allow_reference_free=True, die_id="DIE_SRAM_ARRAY")
        if "error" in rec:
            continue
        k = rec["reference_free"]["scale"]
        gts_k = [{"cls": g["cls"], "box": [int(v * k) for v in g["box"]]} for g in gts]
        found, clean, cls_ok, _ = matches(rec["defects"], gts_k)
        if found and clean and cls_ok:
            keep("die_sram_array_no_markers.jpg", img, rec, gts_k,
                 draw_answers(img, [(g["box"], g["cls"]) for g in gts]), "RE_INSPECT")
            break
    for s in range(200):
        img, gts = other_memory_die(rng)
        rec = st.inspect(frame=img, allow_reference_free=True, die_id="DIE_OTHER_MEMORY")
        if "error" in rec:
            continue
        found, clean, cls_ok, _ = matches(rec["defects"], gts)
        if found and clean and cls_ok:
            keep("die_other_memory.jpg", img, rec, gts, draw_answers(img, [(g["box"], g["cls"]) for g in gts]), "RE_INSPECT")
            break
    clean_img, _ = other_memory_die(rng, defects=False)
    rec = st.inspect(frame=clean_img, allow_reference_free=True, die_id="DIE_CLEAN")
    if "error" not in rec and not rec["defects"]:
        keep("die_other_memory_clean.jpg", clean_img, rec, [], clean_img, "PASS")
    rnd = np.full((700, 900, 3), 180, np.uint8)
    for _ in range(300):
        x, y = rng.integers(0, 880), rng.integers(0, 680)
        cv2.rectangle(rnd, (int(x), int(y)), (int(x + rng.integers(5, 60)), int(y + rng.integers(5, 60))),
                      tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    cv2.imwrite(os.path.join(a.out, "not_a_memory_array.jpg"), rnd, [cv2.IMWRITE_JPEG_QUALITY, 92])
    key["not_a_memory_array.jpg"] = {"mode": "rejected", "expected": "error: No repeating memory array found"}
    json.dump(key, open(os.path.join(a.out, "answer_key.json"), "w"), indent=1)
    print("written to", a.out)
    os._exit(0)


if __name__ == "__main__":
    main()
