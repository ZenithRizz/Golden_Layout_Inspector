#!/usr/bin/env bash
# Ask the station's advisor (Llama via ExecuTorch) about one sample defect and print the validated answer.
set -euo pipefail
unset PYTHONPATH
cd "$(dirname "$0")/../.."
source .venv/bin/activate
python - <<'PY'
import json, os
from gli.advisor import Advisor, LlamaExecuTorchBackend
from gli.decision import decide
H = os.path.expanduser("~")
b = LlamaExecuTorchBackend(
    runner=os.environ.get("GLI_LLAMA_RUNNER", f"{H}/executorch/cmake-out/examples/models/llama/llama_main"),
    pte=os.environ.get("GLI_LLAMA_PTE", f"{H}/gli-models/llama/llama3_2_1b_instruct_8da4w_xnn.pte"),
    tokenizer=os.environ.get("GLI_LLAMA_TOKENIZER", f"{H}/gli-models/llama/tokenizer.model"))
print("runner/model found:", bool(b.available()))
class D: rows = 16; cols = 16; spare_rows = 2; spare_cols = 2; spares_virtual = True
cases = [
    ({"id": "d01", "cls": "OPEN", "conf": 0.93, "box": [0, 0, 5, 5]},
     {"region": "BITCELL_ARRAY", "nets": ["bl_5"], "affected_rows": [7], "affected_cols": [5]}),
    ({"id": "d02", "cls": "BRIDGE", "conf": 0.91, "box": [0, 0, 5, 5]},
     {"region": "BITCELL_ARRAY", "nets": ["wl_11", "q_11_3"], "affected_rows": [11], "affected_cols": [3]}),
    ({"id": "d03", "cls": "PARTICLE", "conf": 0.97, "box": [0, 0, 5, 5]},
     {"region": "SENSE_AMP", "nets": [], "affected_rows": [], "affected_cols": []}),
]
adv = Advisor(b)
for det, m in cases:
    m.update({"affected_cells": [[r, c] for r in m["affected_rows"] for c in m["affected_cols"]],
              "mapping_confidence": 1.0, "layers": ["m1", "m2"], "layout_bbox_um": [0, 0, 1, 1], "partial": False})
    sv = decide([det], [m], D())["defects"][0]
    r = adv.advise(det, m, sv)
    print(f"\n=== {det['cls']} in {m['region']}  -> source={r['source']}  latency={r['latency_ms']} ms")
    print(json.dumps(r["advice"], indent=1))
    if r.get("stats"): print("stats:", r["stats"])
    if r.get("problems"): print("problems:", r["problems"])
    if r["source"] != "llama" and r.get("raw"): print("raw model output:", r["raw"][-600:])
PY
