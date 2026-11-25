"""Walk a GDS hierarchy and list absolute bounding boxes of every instance."""
import sys
from collections import Counter
import gdstk
import numpy as np


def _xf(ref):
    a = np.deg2rad(ref.rotation or 0.0)
    m = np.eye(3)
    if ref.x_reflection:
        m = np.diag([1.0, -1.0, 1.0]) @ m
    r = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    m = r @ m
    t = np.eye(3)
    t[:2, 2] = ref.origin
    return t @ m


def walk(cell, T=None, path="", depth=0, maxdepth=6, out=None):
    T = np.eye(3) if T is None else T
    out = [] if out is None else out
    for ref in cell.references:
        sub = ref.cell
        m = _xf(ref)
        rep = ref.repetition
        offs = [(0.0, 0.0)]
        if rep is not None and rep.size > 0:
            offs = rep.offsets
        for off in offs:
            to = np.eye(3)
            to[:2, 2] = off
            g = T @ to @ m
            bb = sub.bounding_box()
            if bb is not None:
                pts = np.array([[bb[0][0], bb[0][1], 1], [bb[1][0], bb[1][1], 1],
                                [bb[0][0], bb[1][1], 1], [bb[1][0], bb[0][1], 1]]).T
                q = (g @ pts)[:2]
                out.append(dict(depth=depth, path=f"{path}/{sub.name}", name=sub.name,
                                lo=q.min(1).tolist(), hi=q.max(1).tolist(), T=g))
            if depth < maxdepth:
                walk(sub, g, f"{path}/{sub.name}", depth + 1, maxdepth, out)
    return out


if __name__ == "__main__":
    lib = gdstk.read_gds(sys.argv[1])
    top = lib.top_level()[0]
    refs = walk(top)
    cnt = Counter(r["name"] for r in refs)
    for n, c in cnt.most_common():
        if "contact" in n or "via" in n:
            continue
        ex = next(r for r in refs if r["name"] == n)
        print(f"{c:4d}  {n:55s} d={ex['depth']} lo={np.round(ex['lo'],2).tolist()} hi={np.round(ex['hi'],2).tolist()}")
