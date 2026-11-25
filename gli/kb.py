"""Memory fault knowledge base (F7): retrieval + deterministic template advice."""
from __future__ import annotations

import fnmatch
import json
import os

_KB = None


def load(path=None):
    global _KB
    if _KB is None or path:
        p = path or os.path.join(os.path.dirname(__file__), "kb.json")
        with open(p) as f:
            _KB = json.load(f)
    return _KB


def _any(patterns, nets):
    return any(fnmatch.fnmatch(n, p) for p in patterns for n in nets)


def _all(patterns, nets):
    return all(any(fnmatch.fnmatch(n, p) for n in nets) for p in patterns)


def _same_index_pair(nets):
    bl = {n.split("_")[1] for n in nets if n.startswith("bl_")}
    br = {n.split("_")[1] for n in nets if n.startswith("br_")}
    return bool(bl & br)


def score(entry, cls, region, nets, n_cols, low_conf):
    if entry.get("low_confidence") and not low_conf:
        return -1
    if low_conf and not entry.get("low_confidence"):
        return -1
    s = 0
    if "*" in entry["cls"] or cls in entry["cls"]:
        s += 2 if cls in entry["cls"] else 0
    else:
        return -1
    if "*" in entry["region"] or region in entry["region"]:
        s += 2 if region in entry["region"] else 0
    else:
        return -1
    if "nets_all" in entry:
        if not _all(entry["nets_all"], nets):
            return -1
        s += 3
    if "nets_any" in entry:
        if not _any(entry["nets_any"], nets):
            return -1
        s += 2
    if entry.get("same_index"):
        if not _same_index_pair(nets):
            return -1
        s += 1
    if entry.get("nets_multi_cols"):
        if n_cols < 2:
            return -1
        s += 2
    return s


def retrieve(cls, region, nets, n_cols=0, low_conf=False, k=3):
    kb = load()
    scored = [(score(e, cls, region, nets, n_cols, low_conf), e) for e in kb["entries"]]
    scored = [t for t in scored if t[0] >= 0]
    scored.sort(key=lambda t: -t[0])
    return [e for _, e in scored[:k]]


def _fmt(vals):
    if not vals:
        return "?"
    vals = list(vals)
    if len(vals) > 4:
        return f"{vals[0]}-{vals[-1]}"
    return ", ".join(str(v) for v in vals)


def template_advice(det, mp, sev, expected_action):
    low = sev["severity"] == "REVIEW" and sev.get("reason", "").startswith("Low ")
    entries = retrieve(det["cls"], mp["region"], mp["nets"], len(mp["affected_cols"]), low_conf=low)
    e = entries[0] if entries else load()["entries"][-1]
    rows = sev.get("full_rows") or mp["affected_rows"]
    cols = sev.get("full_cols") or mp["affected_cols"]
    text = e["template"].format(rows=_fmt(rows), cols=_fmt(cols))
    return {
        "defect_id": det["id"],
        "fault_model": e["fault_models"][0],
        "affected": affected_text(mp, sev),
        "confirm_test": e["tests"][0],
        "recommended_action": expected_action,
        "explanation": text,
        "confidence": "MEDIUM",
        "kb_id": e["id"],
    }


def affected_text(mp, sev):
    rows = sev.get("full_rows") or []
    cols = sev.get("full_cols") or []
    parts = []
    if cols:
        parts.append("column " + _fmt(cols) + " (all rows)")
    if rows:
        parts.append("row " + _fmt(rows) + " (all columns)")
    if not parts and mp["affected_cells"]:
        cells = mp["affected_cells"]
        if len(cells) <= 3:
            parts.append(", ".join(f"cell r{r}c{c}" for r, c in cells))
        else:
            parts.append(f"{len(cells)} cells, rows {_fmt(mp['affected_rows'])}, columns {_fmt(mp['affected_cols'])}")
    if not parts and mp["region"] == "UNMAPPED":
        parts.append("repeating array area (marked on the image; no layout to name the cells)")
    if not parts:
        parts.append(mp["region"].replace("_", " ").lower())
    return "; ".join(parts)
