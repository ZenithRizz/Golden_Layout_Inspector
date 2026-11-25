"""Llama fault & repair advisor (F6) - Llama 3.2 1B Instruct via ExecuTorch on the Arm CPU.

The advisor receives a *structured* defect record (class, confidence,
region, layers, nets, rows/cols, deterministic decision) plus the top KB
entries, and must answer with one JSON object using only KB vocabulary.

Backends
--------
* LlamaExecuTorchBackend: runs ExecuTorch's `llama_main` runner (built from
  the ExecuTorch repo, examples/models/llama) on a `.pte` exported with
  XNNPACK + 8da4w quantisation.  Timing stats come from the runner's
  PyTorchObserver line (TTFT, prefill and decode tok/s).
* TemplateBackend: deterministic KB text.  Used when Llama is unavailable,
  times out, or returns invalid output twice.

Every Llama answer is validated (schema, vocabulary, index consistency,
<= 60 words).  The deterministic decision guard always wins on conflict.
"""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time

from . import kb as kbmod

ACTIONS = ["NO_ACTION", "REPAIR_ROW", "REPAIR_COL", "REPAIR_ROW_AND_COL", "SCRAP", "REINSPECT"]
KEYS = ["defect_id", "fault_model", "affected", "confirm_test", "recommended_action", "explanation", "confidence"]
LLM_KEYS = ["fault_model", "confirm_test", "why"]

# Llama's job: pick the fault model and the confirming test (from KB candidates) and write ONE
# sentence on what fails and why.  The station appends the test and the rule's action itself, so
# a small model can never turn a repair into a scrap.  The answer is pre-started with
# '{"fault_model": "' so the model continues the JSON instead of echoing its input.
SYSTEM = (
    "You are an SRAM test engineer helping an inspection operator. You get one defect that is "
    "already mapped onto the memory layout. Answer with JSON only, keys in this order: "
    "fault_model (copy one item of fault_options exactly), confirm_test (copy one item of "
    "test_options exactly), why (one sentence, at most 25 words: what this defect breaks "
    "electrically and why). Use only facts from the input. Do not give the action."
)
# The worked example uses a block this macro does not have (a column mux), so the model can learn the
# format without being able to copy the sentence for a real defect.
EXAMPLE_IN = {"defect": "OPEN", "where": "column mux", "nets": [], "affects": "column mux",
              "action": "scrap the die", "fault_options": ["MUX_FAULT"],
              "test_options": ["MATS+", "Functional test"],
              "note": "Mux opens disconnect a column from its sense amplifier."}
EXAMPLE_OUT = {"fault_model": "MUX_FAULT", "confirm_test": "MATS+",
               "why": "A break in the column multiplexer disconnects its column from the sense amplifier, so reads return stale data."}
PREFILL = '{"fault_model": "'


def _chat(system, user, example_in=None, example_out=None, prefill=""):
    # Llama 3 chat template.  BOS is added by the runner.
    s = f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
    if example_in is not None:
        s += (f"<|start_header_id|>user<|end_header_id|>\n\n{json.dumps(example_in)}<|eot_id|>"
              f"<|start_header_id|>assistant<|end_header_id|>\n\n{json.dumps(example_out)}<|eot_id|>")
    s += f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return s + prefill


def _lines(vals):
    vals = list(vals)
    if not vals:
        return "?"
    if len(vals) > 4:
        return f"{vals[0]}-{vals[-1]}"
    return ", ".join(str(v) for v in vals)


def action_text(expected, mp, sev):
    if mp["region"] == "UNMAPPED":
        return "have an engineer review it against the die's design before deciding"
    rows = sev.get("full_rows") or mp["affected_rows"]
    cols = sev.get("full_cols") or mp["affected_cols"]
    return {
        "REPAIR_COL": f"replace column {_lines(cols)} with a spare column",
        "REPAIR_ROW": f"replace row {_lines(rows)} with a spare row",
        "REPAIR_ROW_AND_COL": "replace the affected row and column with spares",
        "SCRAP": "scrap the die",
        "REINSPECT": "re-inspect before deciding",
        "NO_ACTION": "no repair needed",
    }.get(expected, "re-inspect before deciding")


def build_record(det, mp, sev, kb_entries):
    """Compact prompt record.  Keys starting with '_' are for validation only (not sent)."""
    fms, tests = [], []
    for e in kb_entries:
        fms += [f for f in e["fault_models"] if f not in fms]
        tests += [t for t in e["tests"] if t not in tests]
    nets = mp["nets"][:8] + ([f"+{len(mp['nets']) - 8} more"] if len(mp["nets"]) > 8 else [])
    affects = kbmod.affected_text(mp, sev)
    act = action_text(sev["expected_action"], mp, sev)
    rec = {
        "defect": det["cls"], "where": mp["region"].replace("_", " ").lower(), "nets": nets,
        "affects": affects, "action": act,
        "fault_options": fms or ["UNKNOWN"], "test_options": tests or ["Visual re-inspection"],
        "note": kb_entries[0]["source"] if kb_entries else "",
    }
    allowed = set(re.findall(r"\d+", affects + " " + act + " " + " ".join(nets))) | {"0", "1"}
    rec["_defect_id"] = det["id"]
    rec["_expected_action"] = sev["expected_action"]
    rec["_allowed_numbers"] = sorted(allowed, key=int)
    return rec


def prompt_record(rec):
    return {k: v for k, v in rec.items() if not k.startswith("_")}


def extract_json(text):
    start = text.find("{")
    while start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _norm(x):
    return re.sub(r"[\s_\-]+", "", str(x)).lower()


def _pick(value, options):
    """Map the model's choice onto an allowed option.

    Accepts exact, case/spacing variants, abbreviations ('SA' -> 'SA_FAULT') and answers that
    run two options together ('March C- Functional test' -> 'March C-').
    """
    v = _norm(value)
    if not v:
        return None
    for o in options:
        if _norm(o) == v:
            return o
    starts = [o for o in options if v.startswith(_norm(o))]          # 'marchc-functionaltest'
    if starts:
        return max(starts, key=lambda o: len(_norm(o)))
    prefix = [o for o in options if _norm(o).startswith(v)]          # 'sa' -> 'safault'
    if len(prefix) == 1:
        return prefix[0]
    inside = [o for o in options if _norm(o) in v]
    if len(inside) == 1:
        return inside[0]
    return None


# sentences giving an action or a test are dropped: the station states the test and the rule's action itself
ACTION_WORDS = re.compile(r"\b(scrap|spare|replace|repair\w*|discard|confirm\w*|run|test(s|ed|ing)?)\b", re.I)
NONSENSE = re.compile(r"\bno (defects?|faults?|issue)\b|\bnot (a )?defect", re.I)


def _clean_why(why):
    """Keep the cause sentence(s); drop any sentence that gives an action (the rules own that)."""
    sents = [t.strip() for t in re.split(r"(?<=[.!?])\s+", " ".join(str(why).split())) if t.strip()]
    keep = [t for t in sents if not ACTION_WORDS.search(t)]
    out = " ".join(keep).strip()
    if out and out[-1] not in ".!?":
        out += "."
    return out


def compose(why, test, rec):
    act = rec["action"]
    return f"{why} Confirm with {test}, then {act}."


def validate(ans, rec, max_words=30):
    """Check a Llama answer against the record.  Return (advice dict, problems)."""
    if not isinstance(ans, dict):
        return None, ["no JSON object in the answer"]
    probs = [f"missing {k}" for k in LLM_KEYS if k not in ans]
    if probs:
        return None, probs
    fm = _pick(ans["fault_model"], rec["fault_options"])
    ct = _pick(ans["confirm_test"], rec["test_options"])
    if fm is None:
        probs.append(f"fault_model {ans['fault_model']!r} not in {rec['fault_options']}")
    if ct is None:
        probs.append(f"confirm_test {ans['confirm_test']!r} not in {rec['test_options']}")
    why = _clean_why(ans["why"])
    if not why:
        probs.append("no cause sentence (only an action)")
    bad = set(re.findall(r"\b\d+\b", why)) - set(rec["_allowed_numbers"])
    if bad:
        probs.append(f"mentions numbers not in the record: {sorted(bad, key=int)}")
    if NONSENSE.search(why):
        probs.append("says there is no defect")
    if probs:
        return None, probs
    words = why.split()
    if len(words) > max_words:
        why = " ".join(words[:max_words]).rstrip(",;") + "."
    return {
        "defect_id": rec["_defect_id"], "fault_model": fm, "affected": rec["affects"], "confirm_test": ct,
        "recommended_action": rec["_expected_action"], "explanation": compose(why, ct, rec),
        "confidence": "MEDIUM", "why": why,
    }, []


class TemplateBackend:
    name = "template"

    def available(self):
        return True


class LlamaExecuTorchBackend:
    """Calls ExecuTorch's llama_main runner (C++), one process per request."""
    name = "llama3.2-1b-executorch"

    def __init__(self, runner, pte, tokenizer, threads=4, seq_len=768, timeout=45, max_new_tokens=90):
        self.runner, self.pte, self.tok = runner, pte, tokenizer
        self.threads, self.seq_len, self.timeout = threads, seq_len, timeout
        self.max_new_tokens = max_new_tokens

    def available(self):
        return all(p and os.path.exists(p) for p in (self.runner, self.pte, self.tok)) or \
            (self.runner and shutil.which(self.runner) and os.path.exists(self.pte) and os.path.exists(self.tok))

    def generate(self, prompt):
        # llama_main's --seq_len caps prompt + generated tokens.  Llama 3 averages ~4.5 chars/token on
        # these prompts, so len/4 slightly over-estimates the prompt and leaves ~max_new_tokens to answer.
        seq_len = min(self.seq_len, len(prompt) // 4 + self.max_new_tokens)
        cmd = [self.runner, f"--model_path={self.pte}", f"--tokenizer_path={self.tok}",
               f"--prompt={prompt}", "--temperature=0", f"--seq_len={seq_len}",
               f"--cpu_threads={self.threads}"]
        t0 = time.perf_counter()
        timed_out = False
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            stdout, stderr, rc = p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:      # keep whatever was generated before the timeout
            timed_out = True
            dec = lambda b: b.decode(errors="replace") if isinstance(b, bytes) else (b or "")  # noqa: E731
            stdout, stderr, rc = dec(e.stdout), dec(e.stderr), -1
        wall = (time.perf_counter() - t0) * 1000
        stats = parse_observer(stdout + "\n" + stderr)
        stats["wall_ms"] = round(wall, 1)
        stats["seq_len"] = seq_len
        if timed_out:
            stats["timed_out"] = True
        text = stdout
        marker = "<|start_header_id|>assistant<|end_header_id|>"
        if marker in text:
            text = text.rsplit(marker, 1)[-1]
        text = text.split("PyTorchObserver")[0].replace("<|eot_id|>", "")
        if rc != 0 and not text.strip():
            raise RuntimeError(("timed out" if timed_out else f"llama_main exited {rc}") + f": {stderr[-300:]}")
        return text, stats


def parse_observer(out):
    m = re.search(r"PyTorchObserver\s*(\{.*?\})", out)
    if not m:
        return {}
    try:
        s = json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}
    r = {"prompt_tokens": s.get("prompt_tokens"), "generated_tokens": s.get("generated_tokens")}
    try:
        load = s["model_load_end_ms"] - s["model_load_start_ms"]
        ttft = s["first_token_ms"] - s["inference_start_ms"]
        prefill = s["prompt_eval_end_ms"] - s["inference_start_ms"]
        decode = s["inference_end_ms"] - s["prompt_eval_end_ms"]
        r.update({"model_load_ms": load, "ttft_ms": ttft,
                  "prefill_tok_s": round(s["prompt_tokens"] / max(prefill, 1) * 1000, 2),
                  "decode_tok_s": round(s["generated_tokens"] / max(decode, 1) * 1000, 2),
                  "inference_ms": s["inference_end_ms"] - s["inference_start_ms"]})
    except (KeyError, TypeError):
        pass
    return r


class Advisor:
    def __init__(self, backend=None):
        self.backend = backend if backend is not None and backend.available() else TemplateBackend()

    @property
    def mode(self):
        return self.backend.name

    def advise(self, det, mp, sev, retry_budget_ms=9000):
        low = sev["severity"] == "REVIEW" and sev.get("reason", "").startswith("Low ")
        entries = kbmod.retrieve(det["cls"], mp["region"], mp["nets"], len(mp["affected_cols"]), low_conf=low)
        rec = build_record(det, mp, sev, entries)
        template = kbmod.template_advice(det, mp, sev, sev["expected_action"])
        t0 = time.perf_counter()
        result = {"source": "template", "attempts": 0, "problems": [], "stats": {}}
        ans = None
        if not isinstance(self.backend, TemplateBackend):
            user = json.dumps(prompt_record(rec), separators=(",", ":"))
            prompt = _chat(SYSTEM, user, EXAMPLE_IN, EXAMPLE_OUT, PREFILL)
            for attempt in range(2):
                result["attempts"] = attempt + 1
                try:
                    text, stats = self.backend.generate(prompt)
                    result["stats"] = stats
                    result["raw"] = text[-1200:]
                except Exception as e:   # timeout, crash, missing file
                    result["problems"].append(f"{e.__class__.__name__}: {str(e)[:200]}")
                    break
                if not text.lstrip().startswith("{"):
                    text = PREFILL + text            # runner did not echo the prefill
                cand, probs = validate(extract_json(text), rec)
                if cand is not None:
                    ans = cand
                    result["source"] = "llama"
                    break
                result["problems"] += probs
                # one retry, only if the first call was quick enough to keep the operator waiting briefly
                if (time.perf_counter() - t0) * 1000 > retry_budget_ms:
                    break
                prompt = _chat(SYSTEM, user + "\nYour previous answer was rejected: " + "; ".join(probs[:2]) +
                               ". Answer again with JSON only.", EXAMPLE_IN, EXAMPLE_OUT, PREFILL)
        if ans is None:
            ans = {k: template[k] for k in KEYS}
            ans["affected"] = rec["affects"]
        result.update({
            "advice": ans,
            "kb_ids": [e["id"] for e in entries],
            "rule_action": sev["expected_action"],
            "final_action": sev["expected_action"],
            "conflict": False,   # Llama never chooses the action; inconsistent explanations are rejected instead
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "record": rec,
        })
        return result


class AdvisorWorker(threading.Thread):
    """Background queue: one advisor call at a time, never blocks the live feed."""

    def __init__(self, advisor: Advisor, on_result, max_queue=5):
        super().__init__(daemon=True)
        self.advisor = advisor
        self.on_result = on_result
        self.q = queue.Queue()
        self.max_queue = max_queue
        self.current_die = None

    def submit(self, inspection_id, items):
        """items: list of (det, mp, sev).  Cancels queued work from older inspections."""
        self.current_die = inspection_id
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        for it in items[: self.max_queue]:
            self.q.put((inspection_id, it))

    def run(self):
        while True:
            insp_id, (det, mp, sev) = self.q.get()
            if insp_id != self.current_die:
                continue
            try:
                res = self.advisor.advise(det, mp, sev)
            except Exception as e:  # never kill the worker
                res = {"source": "error", "problems": [str(e)], "advice": None}
            self.on_result(insp_id, det["id"], res)
