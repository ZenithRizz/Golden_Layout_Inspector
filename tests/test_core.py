"""Unit tests for the safety-critical logic: decision rules, spare allocation,
mapping, KB coverage and advisor output validation.

Run:  python -m pytest -q
"""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from gli import kb  # noqa: E402
from gli.advisor import Advisor, build_record, extract_json, prompt_record, validate  # noqa: E402
from gli.decision import allocate, decide, footprint, severity  # noqa: E402

DESIGN_DIR = os.path.join(ROOT, "designs", "sram16x16_scn4m")


class D:  # minimal design stub
    rows, cols, spare_rows, spare_cols, spares_virtual = 16, 16, 2, 2, True


def det(cls="BRIDGE", conf=0.9, i="d01"):
    return {"id": i, "cls": cls, "conf": conf, "box": [0, 0, 5, 5]}


def mp(region="BITCELL_ARRAY", nets=(), rows=(3,), cols=(5,), conf=1.0):
    return {"region": region, "nets": list(nets), "affected_rows": list(rows), "affected_cols": list(cols),
            "affected_cells": [[r, c] for r in rows for c in cols], "mapping_confidence": conf,
            "layers": ["m1"], "layout_bbox_um": [0, 0, 1, 1], "partial": False}


# ---------------------------------------------------------------- severity
def test_unknown_is_review():
    assert severity(det("UNKNOWN"), mp())[0] == "REVIEW"


def test_low_confidence_is_review():
    assert severity(det(conf=0.5), mp())[0] == "REVIEW"
    assert severity(det(), mp(conf=0.5))[0] == "REVIEW"


def test_periphery_is_killer():
    for r in ("ROW_DECODER", "SENSE_AMP", "WRITE_DRIVER", "PRECHARGE", "CONTROL"):
        assert severity(det(), mp(region=r))[0] == "KILLER"


def test_power_short_is_killer_even_in_array():
    assert severity(det("BRIDGE"), mp(nets=["vdd", "gnd"]))[0] == "KILLER"


def test_dummy_and_outside_are_cosmetic():
    assert severity(det(), mp(region="DUMMY_ARRAY"))[0] == "COSMETIC"
    assert severity(det(), mp(region="OUTSIDE"))[0] == "COSMETIC"


def test_unmapped_is_review_with_clear_reason():
    sev, why = severity(det(), mp(region="UNMAPPED", conf=0.0))
    assert sev == "REVIEW" and why.startswith("No layout")


def test_unmapped_advice_is_specific_not_low_confidence():
    d, m = det("PARTICLE"), mp(region="UNMAPPED", rows=(), cols=(), conf=0.0)
    sv = decide([d], [m], D())["defects"][0]
    r = Advisor(None).advise(d, m, sv)
    assert r["kb_ids"][0] == "KB25" and r["final_action"] == "REINSPECT"


def test_array_is_repairable():
    assert severity(det(), mp())[0] == "REPAIRABLE"


# ---------------------------------------------------------------- footprint
def test_bitline_open_fails_whole_column():
    cells, rows, cols = footprint(det("OPEN"), mp(nets=["bl_5"]), 16, 16)
    assert cols == [5] and len(cells) == 16


def test_wordline_bridge_fails_whole_row():
    cells, rows, cols = footprint(det("BRIDGE"), mp(nets=["wl_3", "q_3_5"]), 16, 16)
    assert rows == [3] and len(cells) == 16


def test_bitline_pair_bridge_fails_column():
    cells, rows, cols = footprint(det("BRIDGE"), mp(nets=["bl_5", "br_5"]), 16, 16)
    assert cols == [5]


# ---------------------------------------------------------------- allocation
def test_allocation_single_cell():
    ok, r, c = allocate({(3, 5)}, 2, 2)
    assert ok and len(r) + len(c) == 1


def test_allocation_full_column_uses_spare_column():
    ok, r, c = allocate({(i, 7) for i in range(16)}, 2, 2)
    assert ok and c == [7] and r == []


def test_allocation_two_defects_same_row_share_spare():
    ok, r, c = allocate({(4, 1), (4, 9), (4, 12)}, 2, 2)
    assert ok and r == [4] and c == []


def test_allocation_over_budget_fails():
    cells = {(i, c) for c in (1, 2, 3) for i in range(16)}
    ok, _, _ = allocate(cells, 2, 2)
    assert not ok


def test_zero_budget_fails():
    ok, _, _ = allocate({(0, 0)}, 0, 0)
    assert not ok


def test_empty_is_ok():
    assert allocate(set(), 2, 2)[0]


# ---------------------------------------------------------------- die decision
def test_die_pass_when_no_defects():
    assert decide([], [], D())["die_decision"] == "PASS"


def test_die_scrap_on_killer():
    out = decide([det(), det(i="d02")], [mp(), mp(region="SENSE_AMP")], D())
    assert out["die_decision"] == "SCRAP"


def test_die_reinspect_beats_scrap():
    out = decide([det(), det("UNKNOWN", i="d02")], [mp(region="SENSE_AMP"), mp()], D())
    assert out["die_decision"] == "RE_INSPECT"


def test_die_repair_with_allocation():
    out = decide([det("OPEN")], [mp(nets=["bl_5"])], D())
    assert out["die_decision"] == "REPAIR"
    assert out["allocation"]["spare_cols_used"] == [5]
    assert out["defects"][0]["expected_action"] == "REPAIR_COL"


def test_die_scrap_when_budget_exceeded():
    ds = [det("OPEN", i=f"d{i}") for i in range(3)]
    ms = [mp(nets=[f"bl_{c}"], cols=(c,)) for c in (1, 4, 9)]
    out = decide(ds, ms, D())
    assert out["die_decision"] == "SCRAP"


def test_too_many_defects_reinspect():
    ds = [det(i=f"d{i}") for i in range(21)]
    ms = [mp(region="DUMMY_ARRAY") for _ in range(21)]
    assert decide(ds, ms, D())["die_decision"] == "RE_INSPECT"


# ---------------------------------------------------------------- KB coverage
@pytest.mark.parametrize("cls", ["BRIDGE", "OPEN", "PARTICLE", "MISSING_VIA", "UNKNOWN"])
@pytest.mark.parametrize("region", ["BITCELL_ARRAY", "ROW_DECODER", "SENSE_AMP", "WRITE_DRIVER", "PRECHARGE",
                                    "CONTROL", "DUMMY_ARRAY", "ROUTING", "OUTSIDE", "UNMAPPED"])
def test_kb_covers_every_class_region(cls, region):
    assert kb.retrieve(cls, region, []), f"no KB entry for {cls} in {region}"


def test_kb_low_conf_path():
    assert kb.retrieve("BRIDGE", "BITCELL_ARRAY", [], low_conf=True)[0]["id"] == "KB22"


def test_kb_specific_beats_generic():
    assert kb.retrieve("BRIDGE", "BITCELL_ARRAY", ["bl_3", "br_3"])[0]["id"] == "KB01"
    assert kb.retrieve("OPEN", "BITCELL_ARRAY", ["wl_7"])[0]["id"] == "KB08"


# ---------------------------------------------------------------- advisor validation
def _case(cls="OPEN", nets=("bl_5",), rows=(7,), cols=(5,), region="BITCELL_ARRAY"):
    d, m = det(cls), mp(region=region, nets=nets, rows=rows, cols=cols)
    sv = decide([d], [m], D())["defects"][0]
    entries = kb.retrieve(cls, region, list(nets), len(cols))
    return d, m, sv, build_record(d, m, sv, entries)


def test_record_gives_rule_action_and_exact_location():
    *_, rec = _case()
    assert rec["affects"] == "column 5 (all rows)"
    assert rec["action"] == "replace column 5 with a spare column"
    assert "7" not in rec["_allowed_numbers"]          # the row under the box is irrelevant for a bitline open
    assert all(not k.startswith("_") for k in prompt_record(rec))


def test_extract_json_from_chatter():
    assert extract_json('Sure! {"a": {"b": 1}} trailing') == {"a": {"b": 1}}
    assert extract_json("no json") is None


def test_validate_accepts_good_answer():
    *_, rec = _case()
    ans = {"fault_model": "bl_open", "confirm_test": "march c-",
           "why": "The bitline of column 5 is broken, so cells past the break cannot be read."}
    out, probs = validate(ans, rec)
    assert not probs and out["fault_model"] == "BL_OPEN" and out["confirm_test"] == "March C-"
    assert out["recommended_action"] == "REPAIR_COL" and out["affected"] == "column 5 (all rows)"
    assert out["explanation"].endswith("Confirm with March C-, then replace column 5 with a spare column.")


def test_validate_rejects_leaked_numbers():
    *_, rec = _case()
    out, probs = validate({"fault_model": "TF", "confirm_test": "March C-",
                           "why": "The bitline of row 7 is stuck-open."}, rec)
    assert out is None and any("numbers" in p for p in probs)


def test_validate_handles_the_laptop_outputs():
    # what Llama 3.2 1B actually produced on the laptop (keys adapted to the new schema)
    *_, rec = _case("PARTICLE", nets=(), rows=(), cols=(), region="SENSE_AMP")
    out, probs = validate({"fault_model": "SA", "confirm_test": "March C- Functional test",
                           "why": "Column read failures due to faulty sense amp. Confirm with March C- "
                                  "Functional test; the sense amp has no defects."}, rec)
    assert not probs, probs
    assert out["fault_model"] == "SA_FAULT" and out["confirm_test"] == "March C-"
    assert "no defects" not in out["explanation"] and out["explanation"].endswith("then scrap the die.")
    *_, rec = _case("BRIDGE", nets=("wl_11", "q_11_3"), rows=(11,), cols=(3,))
    out, probs = validate({"fault_model": "WL_BRIDGE", "confirm_test": "March C-",
                           "why": "Extra wordline shorts appear as row-wide stuck-at or address faults. "
                                  "Confirm with a March SS test; the shorts are likely stuck-at."}, rec)
    assert not probs and "March SS" not in out["explanation"]
    assert out["explanation"].endswith("replace row 11 with a spare row.")


def test_validate_rejects_echoed_input_and_action_only():
    *_, rec = _case()
    out, probs = validate(prompt_record(rec), rec)                     # model echoed its input
    assert out is None and any("missing" in p for p in probs)
    out, probs = validate({"fault_model": "BL_OPEN", "confirm_test": "March C-",
                           "why": "Scrap the die."}, rec)              # only an action, no cause
    assert out is None


def test_pick_is_forgiving_but_safe():
    from gli.advisor import _pick
    assert _pick("SA", ["SA_FAULT"]) == "SA_FAULT"
    assert _pick("march c- functional test", ["March C-", "Functional test"]) == "March C-"
    assert _pick("March", ["March C-", "March SS"]) is None            # ambiguous
    assert _pick("", ["AF"]) is None


class _Fake:
    name = "fake"

    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), 0

    def available(self):
        return True

    def generate(self, prompt):
        self.calls += 1
        o = self.outputs.pop(0)
        if isinstance(o, Exception):
            raise o
        return o, {"ttft_ms": 1}


# the runner echoes the prompt (incl. the pre-started JSON); the advisor keeps only the answer part
GOOD = 'BL_OPEN", "confirm_test": "March C-", "why": "The bitline of column 5 is open, so reads past the break fail."}'
BAD = 'TF", "confirm_test": "March C-", "why": "Row 7 fails."}'


def test_advisor_uses_llama_when_valid():
    d, m, sv, _ = _case()
    r = Advisor(_Fake([GOOD])).advise(d, m, sv)
    assert r["source"] == "llama" and r["attempts"] == 1 and r["final_action"] == "REPAIR_COL"
    assert r["advice"]["explanation"].startswith("The bitline of column 5 is open")


def test_advisor_retries_once_then_accepts():
    d, m, sv, _ = _case()
    fake = _Fake([BAD, GOOD])
    r = Advisor(fake).advise(d, m, sv)
    assert r["source"] == "llama" and fake.calls == 2 and r["problems"]


def test_advisor_falls_back_to_template():
    d, m, sv, _ = _case()
    r = Advisor(_Fake([BAD, BAD])).advise(d, m, sv)
    assert r["source"] == "template" and r["advice"]["fault_model"] == "BL_OPEN"
    assert r["advice"]["affected"] == "column 5 (all rows)"
    r = Advisor(_Fake([RuntimeError("runner crashed")])).advise(d, m, sv)
    assert r["source"] == "template" and "runner crashed" in r["problems"][0]


def test_prompt_is_compact():
    from gli.advisor import SYSTEM, EXAMPLE_IN, EXAMPLE_OUT, _chat
    *_, rec = _case()
    p = _chat(SYSTEM, json.dumps(prompt_record(rec), separators=(",", ":")), EXAMPLE_IN, EXAMPLE_OUT)
    assert len(p) < 1700        # roughly 350-400 Llama tokens (was ~2200 chars / 518 tokens)


def test_template_advisor_always_answers():
    d, m, sv, _ = _case()
    r = Advisor(None).advise(d, m, sv)
    assert r["source"] == "template" and r["advice"]["fault_model"] == "BL_OPEN"
    assert r["final_action"] == "REPAIR_COL" and not r["conflict"]


# ---------------------------------------------------------------- real design mapping
@pytest.mark.skipif(not os.path.exists(os.path.join(DESIGN_DIR, "design.json")), reason="design not built")
def test_mapping_real_design_bitcell_and_regions():
    from gli.layout import Design, region_at
    from gli.mapping import Mapper
    d = Design.load(DESIGN_DIR)
    masks = dict(np.load(os.path.join(DESIGN_DIR, "masks_work.npz")))
    m = Mapper(d, masks)
    b = next(b for b in d.bitcells if b["r"] == 7 and b["c"] == 4)
    cx, cy = (b["rect"][0] + b["rect"][2]) / 2, (b["rect"][1] + b["rect"][3]) / 2
    px, py = d.um_to_work(cx, cy)
    out = m.map([int(px) - 1, int(py) - 1, 2, 2])
    assert out["region"] == "BITCELL_ARRAY"
    assert out["affected_rows"] == [7] and out["affected_cols"] == [4]
    sa = next(r for r in d.regions if r["region"] == "SENSE_AMP")["rect"]
    assert region_at(d, [sa[0] + 1, sa[1] + 1, sa[0] + 2, sa[1] + 2]) == "SENSE_AMP"
    # bitline pin of column 4 is found as a net
    pin = next(p for p in d.pins if p["net"] == "bl_4" and p["r"] == 7)
    x0, y1 = d.um_to_work(pin["rect"][0], pin["rect"][1])
    x1, y0 = d.um_to_work(pin["rect"][2], pin["rect"][3])
    out = m.map([int(x0), int(y0) + 4, max(1, int(x1 - x0)), 3])
    assert "bl_4" in out["nets"]
