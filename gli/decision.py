"""Deterministic decision guard (F8).

Severity and repairability are decided by fixed, testable rules so the
station never makes an unsafe call because of an LLM error.  Llama explains
and reasons (fault model, confirming test); these rules can veto its action.

Fault footprint: what a physical defect does electrically to the array.
  * OPEN on a bitline (bl_c/br_c)        -> whole column c fails
  * OPEN on a wordline (wl_r)            -> whole row r fails
  * BRIDGE touching a wordline           -> row(s) fail
  * BRIDGE between bitlines / to supply  -> column(s) fail
  * otherwise                            -> the bitcells under the defect
"""
from __future__ import annotations

import re

PERIPHERY = ("ROW_DECODER", "SENSE_AMP", "WRITE_DRIVER", "PRECHARGE", "COL_MUX", "CONTROL")
COSMETIC_REGIONS = ("OUTSIDE", "DUMMY_ARRAY")


def _idx(nets, prefix):
    out = set()
    for n in nets:
        m = re.fullmatch(prefix + r"_(\d+)", n)
        if m:
            out.add(int(m.group(1)))
    return out


def footprint(det, mp, rows_total, cols_total):
    """Set of (row, col) cells electrically affected by a defect in the array."""
    nets = mp["nets"]
    cells = {tuple(c) for c in mp["affected_cells"]}
    bls = _idx(nets, "bl") | _idx(nets, "br")
    wls = _idx(nets, "wl")
    cls = det["cls"]
    full_cols, full_rows = set(), set()
    if cls == "OPEN":
        full_cols |= bls
        full_rows |= wls
    elif cls in ("BRIDGE", "PARTICLE"):
        if wls:
            full_rows |= wls
        if bls and (len(bls) > 1 or {"vdd", "gnd"} & set(nets) or len(_idx(nets, "bl")) and len(_idx(nets, "br"))):
            full_cols |= bls
    for c in full_cols:
        cells |= {(r, c) for r in range(rows_total)}
    for r in full_rows:
        cells |= {(r, c) for c in range(cols_total)}
    return cells, sorted(full_rows), sorted(full_cols)


def severity(det, mp, low_conf=0.60, min_map_conf=0.70):
    nets = set(mp["nets"])
    if det["cls"] == "UNKNOWN":
        return "REVIEW", "Unrecognised anomaly: needs a human look"
    if mp["region"] == "UNMAPPED":
        return "REVIEW", "No layout for this die: impact on bitcells and nets can't be mapped; engineer review"
    if det["conf"] < low_conf:
        return "REVIEW", f"Low detection confidence ({det['conf']:.2f})"
    if mp["mapping_confidence"] < min_map_conf:
        return "REVIEW", f"Low mapping confidence ({mp['mapping_confidence']:.2f})"
    if det["cls"] in ("BRIDGE", "PARTICLE") and {"vdd", "gnd"} <= nets:
        return "KILLER", "Supply-to-ground short"
    if mp["region"] in PERIPHERY:
        return "KILLER", f"Defect in periphery ({mp['region']}) has no spare to repair it"
    if mp["region"] == "ROUTING":
        return "REVIEW", "Defect on inter-block routing: nets unknown, engineer check"
    if mp["region"] in COSMETIC_REGIONS:
        return "COSMETIC", f"No functional circuit affected ({mp['region']})"
    if mp["region"] == "BITCELL_ARRAY":
        return "REPAIRABLE", "Bitcell array defect: repairable with spare rows/columns if budget allows"
    return "REVIEW", "Unclassified region"


def allocate(cells, spare_rows, spare_cols):
    """Must-repair analysis + greedy cover.  Returns (ok, rows_used, cols_used)."""
    F = set(cells)
    rows_used, cols_used = set(), set()
    sr, sc = spare_rows, spare_cols
    changed = True
    while changed and F:
        changed = False
        by_row, by_col = {}, {}
        for r, c in F:
            by_row.setdefault(r, set()).add(c)
            by_col.setdefault(c, set()).add(r)
        for r, cs in by_row.items():
            if len(cs) > sc:            # more failing columns in this row than spare columns
                if sr == 0:
                    return False, sorted(rows_used), sorted(cols_used)
                rows_used.add(r)
                sr -= 1
                F = {x for x in F if x[0] != r}
                changed = True
                break
        if changed:
            continue
        for c, rs in by_col.items():
            if len(rs) > sr:
                if sc == 0:
                    return False, sorted(rows_used), sorted(cols_used)
                cols_used.add(c)
                sc -= 1
                F = {x for x in F if x[1] != c}
                changed = True
                break
    while F:
        by_row, by_col = {}, {}
        for r, c in F:
            by_row[r] = by_row.get(r, 0) + 1
            by_col[c] = by_col.get(c, 0) + 1
        br = max(by_row.items(), key=lambda kv: kv[1])
        bc = max(by_col.items(), key=lambda kv: kv[1])
        if bc[1] >= br[1] and sc > 0:
            cols_used.add(bc[0])
            sc -= 1
            F = {x for x in F if x[1] != bc[0]}
        elif sr > 0:
            rows_used.add(br[0])
            sr -= 1
            F = {x for x in F if x[0] != br[0]}
        elif sc > 0:
            cols_used.add(bc[0])
            sc -= 1
            F = {x for x in F if x[1] != bc[0]}
        else:
            return False, sorted(rows_used), sorted(cols_used)
    return True, sorted(rows_used), sorted(cols_used)


def decide(dets, maps, design, too_many=20):
    """Per-defect severity, die decision, spare allocation, expected actions."""
    out = []
    all_cells = set()
    for det, mp in zip(dets, maps):
        sev, why = severity(det, mp)
        cells, frows, fcols = (set(), [], [])
        if sev == "REPAIRABLE":
            cells, frows, fcols = footprint(det, mp, design.rows, design.cols)
            all_cells |= cells
        out.append({"defect_id": det["id"], "severity": sev, "reason": why,
                    "failing_cells": sorted([list(c) for c in cells]),
                    "full_rows": frows, "full_cols": fcols})
    ok, rows_used, cols_used = allocate(all_cells, design.spare_rows, design.spare_cols)
    sevs = [o["severity"] for o in out]
    if len(dets) > too_many:
        die, die_reason = "RE_INSPECT", f"{len(dets)} defects: gross contamination or wrong setup"
    elif "REVIEW" in sevs:
        die, die_reason = "RE_INSPECT", "At least one defect needs review"
    elif "KILLER" in sevs:
        die, die_reason = "SCRAP", "Killer defect present"
    elif "REPAIRABLE" in sevs and not ok:
        die, die_reason = "SCRAP", "Failing rows/columns exceed the spare budget"
    elif "REPAIRABLE" in sevs:
        die, die_reason = "REPAIR", f"Repairable with {len(rows_used)} spare row(s) and {len(cols_used)} spare column(s)"
    else:
        die, die_reason = "PASS", "No functional defects" if dets else "No defects found"
    for o in out:
        o["expected_action"] = expected_action(o, die, rows_used, cols_used)
    return {
        "die_decision": die,
        "die_reason": die_reason,
        "defects": out,
        "allocation": {"ok": ok, "spare_rows_used": rows_used, "spare_cols_used": cols_used,
                       "spare_rows": design.spare_rows, "spare_cols": design.spare_cols,
                       "virtual_budget": design.spares_virtual},
    }


def expected_action(o, die, rows_used, cols_used):
    sev = o["severity"]
    if sev == "REVIEW":
        return "REINSPECT"
    if sev == "KILLER":
        return "SCRAP"
    if sev == "COSMETIC":
        return "NO_ACTION"
    if die == "SCRAP":
        return "SCRAP"
    cells = {tuple(c) for c in o["failing_cells"]}
    by_row = any(r in rows_used for r, _ in cells)
    by_col = any(c in cols_used for _, c in cells)
    if by_row and by_col:
        return "REPAIR_ROW_AND_COL"
    if by_row:
        return "REPAIR_ROW"
    if by_col:
        return "REPAIR_COL"
    return "REINSPECT"
