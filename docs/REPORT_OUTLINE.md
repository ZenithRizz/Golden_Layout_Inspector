# Technical report: outline (fill in on Sat/Sun)

Target length 8–12 pages. Each section lists the numbers or figures that belong there and the command that produces them.

## 1. Problem and motivation (½ page)
- PS5 requirements, and why memory dies: India's new ATMP/OSAT plants (Micron Sanand,
  Kaynes, Tata Assam). Existing inspection says *where* a defect is, not *what it breaks*.
- Contribution, in three bullets:
  - design-aware inspection from an OpenRAM layout;
  - zero real defect images needed;
  - on-device Llama fault reasoning with a deterministic safety guard.

## 2. System overview (1 page)
- Block diagram (copy from `docs/ARCHITECTURE.md` §2), redrawn.
- **Two pipelines, documented separately:**
  - **Vision:** Arm CPU, ONNX Runtime. If a Hailo HAT is used, it runs on the NPU via HailoRT instead.
  - **Advisor:** Llama 3.2 1B via ExecuTorch (XNNPACK, KleidiAI) on the Arm CPU.
- Hardware: Pi 5 8 GB, active cooler, USB webcam; no network.

## 3. Golden design and synthetic data (1½ pages)
- OpenRAM config (16×16, scn4m_subm).
- Figure: golden render with regions and bitcell grid (`golden_check` image).
- Bitcell and net extraction: 256 cells, 3,584 pin shapes, 7 region types.
- Defect injection: the four classes, mask vs painted styles, visibility filter (why).
- Dataset stats from `data/sram16x16_scn4m/stats.json`: counts per class, **candidate recall per class**.
- Figure: `data/.../preview.png`.

## 4. Vision pipeline (1½ pages)
- Registration: ArUco homography + ECC refinement; reprojection error and ECC values.
- Golden-reference difference (equation for the ±1 px tolerant difference); thresholds.
- CNN: architecture table, 147k parameters, 3-channel input rationale, training setup.
- Results from `models/metrics.json`: confusion matrix and per-class precision/recall.
- Detection on virtual dies (`benchmark.py`): precision, recall, class accuracy.
- **Real printed targets:** results on the physical set, reported separately and honestly.

## 5. Mapping, decision rules, repair (1 page)
- Region priority, net mapping, fault footprint table (open bitline → column, …).
- Spare allocation algorithm; die decision table.
- 73 unit tests (list the categories).

## 6. Llama advisor on ExecuTorch (1½ pages)
- Export: 8da4w, group 32, 4-bit embeddings, KV cache + SDPA, XNNPACK; `.pte` size.
- Prompt design: structured record, KB vocabulary, one worked example, validation, retry, template fallback.
- **Metrics** from `benchmark.py` with Llama:
  - model load time, TTFT, prefill tok/s, decode tok/s, total advice latency (p50/p95);
  - valid-JSON rate, fallback rate, conflicts with rules.
- Advice-quality table: Llama fault model vs KB-top entry on N cases (agreement %).
- Two example outputs (screenshots).

## 7. Real-time performance on the device (1 page)
Table from `bench/<stamp>/summary.md` on the **Pi 5**:

| Metric | Value |
|---|---|
| Live FPS | |
| Register / diff / classify (ms) | |
| Inspect → decision (ms) | |
| Advice latency (s) | |
| CPU temperature, throttle flags | |

Include a laptop comparison row if useful.

## 8. Operator experience (½ page)
Screenshots of Live, Inspection (zoom + advisor) and Summary/repair map. Mention the audio alert, override with reason, and that the log and export are immutable.

## 9. Limitations and future work (½ page)
Copy from `ARCHITECTURE.md` §4, then add:
- a persistent Llama runner;
- SEM/microscope imaging;
- extracted routing nets;
- real redundancy from the compiler;
- ATE result reconciliation;
- LoRA fine-tune.

## 10. Licences, attribution, originality (¼ page)
OpenRAM BSD-3 (as a tool), "Built with Llama", ExecuTorch BSD; no AGPL or non-commercial data; all project code written by the team.
