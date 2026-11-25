"""Defect -> design mapping (F5.3).

Turns a detection box in work pixels into layout terms: micrometre box,
region (bitcell array, decoder, sense amp, ...), layers present, nets the
defect touches (bl_12, br_12, wl_5, vdd, ...), and the bitcell rows/columns.
"""
from __future__ import annotations

import numpy as np

from .layout import Design, region_at, LAYERS

LAYER_BY_NUM = {v: k for k, v in LAYERS.items()}


class Mapper:
    def __init__(self, design: Design, masks_work: dict):
        self.d = design
        self.masks = masks_work
        bc = design.bitcells
        self.bc_rect = np.array([b["rect"] for b in bc], float) if bc else np.zeros((0, 4))
        self.bc_rc = np.array([[b["r"], b["c"]] for b in bc], int) if bc else np.zeros((0, 2), int)
        pins = design.pins
        self.pin_rect = np.array([p["rect"] for p in pins], float) if pins else np.zeros((0, 4))
        self.pin_net = [p["net"] for p in pins]
        self.pin_layer = [LAYER_BY_NUM.get(p["layer"], str(p["layer"])) for p in pins]

    def box_to_um(self, box):
        x, y, w, h = box
        x0, y1 = self.d.work_to_um(x, y)
        x1, y0 = self.d.work_to_um(x + w, y + h)
        return [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]

    @staticmethod
    def _hits(rects, r, margin=0.0):
        if len(rects) == 0:
            return np.zeros(0, bool)
        return ((rects[:, 0] < r[2] + margin) & (r[0] - margin < rects[:, 2]) &
                (rects[:, 1] < r[3] + margin) & (r[1] - margin < rects[:, 3]))

    def map(self, box, reg_quality=1.0, partial=False):
        r = self.box_to_um(box)
        region = region_at(self.d, r)
        x, y, w, h = box
        layers = []
        for name, m in self.masks.items():
            sub = m[max(y - 1, 0):y + h + 1, max(x - 1, 0):x + w + 1]
            if sub.size and (sub > 0).mean() > 0.05:
                layers.append(name)
        rows, cols, cells, nets = [], [], [], []
        if region in ("BITCELL_ARRAY",) or self._hits(self.bc_rect, r).any():
            hit = self._hits(self.bc_rect, r)
            rc = self.bc_rc[hit]
            rows = sorted({int(v) for v in rc[:, 0]})
            cols = sorted({int(v) for v in rc[:, 1]})
            cells = sorted({(int(a), int(b)) for a, b in rc})
            ph = self._hits(self.pin_rect, r, margin=0.15)
            nets = sorted({self.pin_net[i] for i in np.where(ph)[0]}, key=_net_key)
        conf = float(reg_quality) * (0.6 if partial else 1.0)
        if region == "BITCELL_ARRAY" and not cells:
            conf *= 0.5
        return {
            "layout_bbox_um": r,
            "region": region,
            "layers": layers,
            "nets": nets,
            "affected_rows": rows,
            "affected_cols": cols,
            "affected_cells": [list(c) for c in cells],
            "mapping_confidence": round(conf, 3),
            "partial": bool(partial),
        }


def _net_key(n):
    order = {"bl": 0, "br": 1, "wl": 2, "q": 3, "qb": 4, "vdd": 5, "gnd": 6}
    head = n.split("_")[0]
    nums = [int(t) for t in n.split("_")[1:] if t.isdigit()]
    return (order.get(head, 9), nums)
