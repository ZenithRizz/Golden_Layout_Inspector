#!/usr/bin/env python3
"""Build a golden design package from an OpenRAM GDS.

Outputs (designs/<name>/):
  design.json        regions, bitcell grid, pin nets, marker positions, scales
  golden_work.png    golden layout at work scale (what inspection compares to)
  masks_work.npz     per-layer masks at work scale
  target_print.png   printable target with ArUco markers (print scale)
  target_A4.pdf      same, fitted to A4 for printing

Usage:
  python scripts/build_design.py --gds openram_out/sram_16x16/sram_16x16_scn4m.gds \
      --name sram16x16_scn4m --window bank
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gli.layout import (Design, extract_design, layer_polygons, render_masks,  # noqa: E402
                        composite, build_print_target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gds", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="designs")
    ap.add_argument("--window", choices=["bank", "full"], default="bank")
    ap.add_argument("--work-ppu", type=float, default=4.0, help="work-scale pixels per um")
    ap.add_argument("--print-ppu", type=float, default=9.0, help="print-scale pixels per um")
    ap.add_argument("--spare-rows", type=int, default=2)
    ap.add_argument("--spare-cols", type=int, default=2)
    ap.add_argument("--words", type=int, default=0)
    ap.add_argument("--word-size", type=int, default=0)
    a = ap.parse_args()

    folder = os.path.join(a.out, a.name)
    os.makedirs(folder, exist_ok=True)
    d = extract_design(a.gds, a.name, window=a.window, work_ppu=a.work_ppu, print_ppu=a.print_ppu,
                       spare_rows=a.spare_rows, spare_cols=a.spare_cols, words=a.words,
                       word_size=a.word_size)
    polys = layer_polygons(a.gds)

    masks = render_masks(polys, d, d.work_ppu)
    golden = composite(masks)
    cv2.imwrite(os.path.join(folder, "golden_work.png"), golden)
    np.savez_compressed(os.path.join(folder, "masks_work.npz"), **masks)

    tgt = build_print_target(d, polys, os.path.join(folder, "target_print.png"))
    d.save(folder)

    # A4 PDF (fit to page, keep aspect)
    try:
        from PIL import Image
        im = Image.fromarray(cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB))
        a4_w_in, a4_h_in = 8.27, 11.69
        dpi = max(im.width / (a4_w_in - 0.4), im.height / (a4_h_in - 0.4))
        im.save(os.path.join(folder, "target_A4.pdf"), "PDF", resolution=dpi)
        print(f"target_A4.pdf at {dpi:.0f} dpi -> {25.4 / dpi * d.print_ppu:.2f} mm per um on paper")
    except Exception as e:  # pillow missing is not fatal
        print("PDF export skipped:", e)

    w, h = d.work_size
    print(f"design {d.name}: {d.rows}x{d.cols} bitcells, pitch {d.pitch[0]:.2f}x{d.pitch[1]:.2f} um, "
          f"{len(d.pins)} pin shapes, {len(d.regions)} regions, work image {w}x{h}px")
    regs = sorted({r['region'] for r in d.regions})
    print("regions:", ", ".join(regs))


if __name__ == "__main__":
    main()
