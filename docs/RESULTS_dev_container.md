# Benchmark results: development container (2 vCPU x86, NOT the Pi)

Re-run `scripts/benchmark.py` on the Raspberry Pi 5 for the report numbers.


## Environment
```
{
 "platform": "Linux-6.18.44-fc-v653-x86_64-with-glibc2.39",
 "machine": "x86_64",
 "python": "3.13.18",
 "cpu_count": 2,
 "vision_backend": "cpu-onnx",
 "advisor_backend": "template",
 "onnxruntime": "1.29.0",
 "opencv": "5.0.0"
}
```

## Vision Pipeline

| metric | value |
|---|---|
| live_frame_ms_p50 | 72.6 |
| live_frame_ms_p95 | 83.6 |
| live_fps | 13.8 |
| inspect_to_decision_ms_p50 | 193.5 |
| inspect_to_decision_ms_p95 | 417.0 |
| register_ms_p50 | 140.4 |
| diff_ms_p50 | 36.4 |
| classify_ms_p50 | 2.8 |

## Detection

| metric | value |
|---|---|
| dies | 30 |
| true_positive | 57 |
| false_positive | 0 |
| missed | 33 |
| recall | 0.633 |
| precision | 1.0 |
| class_accuracy_on_found | 0.982 |
| per_class | {"BRIDGE": {"found": 22, "correct": 21, "total": 30}, "OPEN": {"found": 11, "correct": 11, "total": 27}, "MISSING_VIA": {"found": 6, "correct": 6, "total": 15}, "PARTICLE": {"found": 18, "correct": 18, "total": 18}} |

## Advisor Pipeline

| metric | value |
|---|---|
| calls | 55 |
| llama_answers | 0 |
| template_fallbacks | 55 |
| llama_valid_rate | None |
| conflicts_with_rules | 0 |
| ttft_ms_p50 | None |
| decode_tok_s_p50 | None |
| prefill_tok_s_p50 | None |
| model_load_ms_p50 | None |
| advice_latency_ms_p50 | 0.0 |
| advice_latency_ms_p95 | 0.0 |
