#!/usr/bin/env python3
"""Benchmark & evaluation harness (F11, F2.3, F6.3).

Runs N virtual dies (or a folder of real images) through the station and
writes per-stage timings for the two pipelines, detection metrics against
ground truth (virtual camera), and advisor metrics.

  python scripts/benchmark.py --dies 50                       # vision + template advisor
  python scripts/benchmark.py --dies 30 --llama-runner ... --llama-pte ... --llama-tokenizer ...
  python scripts/benchmark.py --images data/real_targets/     # real photos (no ground truth)

Outputs bench/<stamp>/: frames.csv, advisor.csv, summary.json, summary.md
"""
import argparse
import csv
import glob
import json
import os
import platform
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gli.station import Station  # noqa: E402
from gli.camera import VirtualCamera  # noqa: E402


def iou_hit(box, gt, m=3):
    return box[0] < gt[0] + gt[2] + m and gt[0] - m < box[0] + box[2] and box[1] < gt[1] + gt[3] + m and gt[1] - m < box[1] + box[3]


def pct(v, q):
    return round(float(np.percentile(v, q)), 1) if len(v) else None


def env_manifest(st):
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "cpu_count": os.cpu_count(), "vision_backend": st.classifier.name, "advisor_backend": st.advisor.mode}
    try:
        import onnxruntime
        info["onnxruntime"] = onnxruntime.__version__
    except Exception:
        pass
    info["opencv"] = cv2.__version__
    try:
        info["cpu_temp_c"] = int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000
    except Exception:
        pass
    try:
        import subprocess
        info["throttled"] = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="designs/sram16x16_scn4m")
    ap.add_argument("--onnx", default="models/defect_cls.onnx")
    ap.add_argument("--hef", default=None)
    ap.add_argument("--dies", type=int, default=40)
    ap.add_argument("--defects", type=int, default=3)
    ap.add_argument("--images", default=None, help="folder of real captures instead of virtual dies")
    ap.add_argument("--live-frames", type=int, default=60, help="frames for live-loop FPS measurement")
    ap.add_argument("--llama-runner"), ap.add_argument("--llama-pte"), ap.add_argument("--llama-tokenizer")
    ap.add_argument("--llama-threads", type=int, default=4)
    ap.add_argument("--advice-wait", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    llama = None
    if a.llama_runner and a.llama_pte and a.llama_tokenizer:
        llama = {"runner": a.llama_runner, "pte": a.llama_pte, "tokenizer": a.llama_tokenizer, "threads": a.llama_threads}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join("bench", stamp)
    os.makedirs(out, exist_ok=True)
    st = Station(a.design, "virtual" if not a.images else a.images and "virtual", onnx=a.onnx if os.path.exists(a.onnx) else None,
                 hef=a.hef, llama=llama, runs=os.path.join(out, "runs"))
    st._stop = True                      # benchmark drives frames itself
    time.sleep(0.3)
    if isinstance(st.src, VirtualCamera):
        st.src.rng = np.random.default_rng(a.seed)
        st.src.jitter = False

    # ---- live loop throughput (vision pipeline only)
    live_t = []
    st.src.new_die(a.defects)
    f = st.src.read()
    st.process(f, force_refine=True)
    for _ in range(a.live_frames):
        f = st.src.read()
        t0 = time.perf_counter()
        st.process(f)
        live_t.append((time.perf_counter() - t0) * 1000)

    # ---- inspections
    rows, adv_rows, recs = [], [], []
    tp = fp = fn = 0
    cls_ok = cls_n = 0
    per_cls = {}
    sources = [None] * a.dies
    if a.images:
        sources = sorted(glob.glob(os.path.join(a.images, "*.jpg")) + glob.glob(os.path.join(a.images, "*.png")))
    for i, src in enumerate(sources):
        if src is None:
            gts = st.src.new_die(a.defects)
            frame = st.src.read()
        else:
            gts, frame = [], cv2.imread(src)
        rec = st.inspect(frame=frame, die_id=f"BENCH-{i:03d}")
        if "error" in rec:
            rows.append({"die": i, "error": rec["error"]})
            continue
        recs.append(rec)
        t = rec["timings"]
        rows.append({"die": i, "n_gt": len(gts), "n_det": len(rec["defects"]), "decision": rec["decision"]["die_decision"],
                     **{k: round(v, 2) for k, v in t.items() if isinstance(v, (int, float))}})
        if src is None:
            matched = set()
            for d in rec["defects"]:
                hit = [j for j, g in enumerate(gts) if j not in matched and iou_hit(d["box"], g["box"])]
                if hit:
                    j = hit[0]
                    matched.add(j)
                    tp += 1
                    cls_n += 1
                    ok = d["cls"] == gts[j]["cls"]
                    cls_ok += ok
                    pc = per_cls.setdefault(gts[j]["cls"], {"found": 0, "correct": 0, "total": 0})
                    pc["found"] += 1
                    pc["correct"] += ok
                else:
                    fp += 1
            for j, g in enumerate(gts):
                per_cls.setdefault(g["cls"], {"found": 0, "correct": 0, "total": 0})["total"] += 1
                if j not in matched:
                    fn += 1
        # wait for advice
        exp = rec.get("advice_expected", 0)
        t_end = time.time() + a.advice_wait
        while exp and len(st.pending.get(rec["id"], {})) < exp and time.time() < t_end:
            time.sleep(0.2)
        for did, res in st.pending.get(rec["id"], {}).items():
            s = res.get("stats", {})
            adv = res.get("advice") or {}
            adv_rows.append({"die": i, "defect": did, "source": res.get("source"), "attempts": res.get("attempts"),
                             "latency_ms": res.get("latency_ms"), "conflict": res.get("conflict"),
                             "fault_model": adv.get("fault_model"), "confirm_test": adv.get("confirm_test"),
                             "kb_top": (res.get("kb_ids") or [None])[0],
                             "fault_in_kb_top": adv.get("fault_model") in (res.get("record", {}).get("fault_options") or [])[:2],
                             **{k: s.get(k) for k in ("prompt_tokens", "generated_tokens", "model_load_ms", "ttft_ms",
                                                      "prefill_tok_s", "decode_tok_s", "inference_ms", "wall_ms")},
                             "problems": "; ".join(res.get("problems", []))[:200]})
        print(f"die {i:3d}: {rec['decision']['die_decision']:10s} det={len(rec['defects'])} gt={len(gts)} "
              f"decision {rec['timings']['inspect_to_decision_ms']:.0f} ms")

    def dump(name, rr):
        if not rr:
            return
        keys = sorted({k for r in rr for k in r})
        with open(os.path.join(out, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rr)
    dump("frames.csv", rows)
    dump("advisor.csv", adv_rows)

    col = lambda k: [r[k] for r in rows if k in r]  # noqa: E731
    llama_rows = [r for r in adv_rows if r["source"] == "llama"]
    summary = {
        "environment": env_manifest(st),
        "vision_pipeline": {
            "live_frame_ms_p50": pct(live_t, 50), "live_frame_ms_p95": pct(live_t, 95),
            "live_fps": round(1000 / np.mean(live_t), 1) if live_t else None,
            "inspect_to_decision_ms_p50": pct(col("inspect_to_decision_ms"), 50),
            "inspect_to_decision_ms_p95": pct(col("inspect_to_decision_ms"), 95),
            "register_ms_p50": pct(col("register_ms"), 50), "diff_ms_p50": pct(col("diff_ms"), 50),
            "classify_ms_p50": pct(col("classify_ms"), 50),
        },
        "detection": None if a.images else {
            "dies": len(recs), "true_positive": tp, "false_positive": fp, "missed": fn,
            "recall": round(tp / max(tp + fn, 1), 3), "precision": round(tp / max(tp + fp, 1), 3),
            "class_accuracy_on_found": round(cls_ok / max(cls_n, 1), 3), "per_class": per_cls},
        "advisor_pipeline": {
            "calls": len(adv_rows), "llama_answers": len(llama_rows),
            "template_fallbacks": sum(r["source"] == "template" for r in adv_rows),
            "llama_valid_rate": round(len(llama_rows) / max(len(adv_rows), 1), 3) if llama else None,
            "conflicts_with_rules": sum(bool(r["conflict"]) for r in adv_rows),
            "ttft_ms_p50": pct([r["ttft_ms"] for r in llama_rows if r["ttft_ms"] is not None], 50),
            "decode_tok_s_p50": pct([r["decode_tok_s"] for r in llama_rows if r["decode_tok_s"] is not None], 50),
            "prefill_tok_s_p50": pct([r["prefill_tok_s"] for r in llama_rows if r["prefill_tok_s"] is not None], 50),
            "model_load_ms_p50": pct([r["model_load_ms"] for r in llama_rows if r["model_load_ms"] is not None], 50),
            "advice_latency_ms_p50": pct([r["latency_ms"] for r in adv_rows if r["latency_ms"] is not None], 50),
            "advice_latency_ms_p95": pct([r["latency_ms"] for r in adv_rows if r["latency_ms"] is not None], 95),
        },
    }
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    md = ["# Benchmark " + stamp, "", "## Environment", "```", json.dumps(summary["environment"], indent=1), "```"]
    for sec in ("vision_pipeline", "detection", "advisor_pipeline"):
        if summary[sec] is None:
            continue
        md += ["", f"## {sec.replace('_', ' ').title()}", "", "| metric | value |", "|---|---|"]
        md += [f"| {k} | {json.dumps(v) if isinstance(v, dict) else v} |" for k, v in summary[sec].items()]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(md) + "\n")
    print(json.dumps(summary, indent=1))
    print("written to", out)
    os._exit(0)


if __name__ == "__main__":
    main()
