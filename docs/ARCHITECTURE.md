# Golden-Layout Inspector: how it works

This document walks through the code module by module, so every team member can
explain any part of the system to a judge. Read it alongside the source files it names.

## 1. The idea in one paragraph

We inspect an SRAM macro against its **own layout**. OpenRAM generates the macro,
so we know every polygon, every bitcell and every net. The station:
1. aligns the camera image to that golden layout;
2. finds anything that differs;
3. classifies the difference with a small CNN;
4. maps it to the exact bitcell, row, column and net;
5. decides repair or scrap with fixed rules;
6. asks Llama 3.2 1B (running through ExecuTorch on the Arm CPU) to explain the
   likely electrical fault and the test that confirms it.

Training data is generated from the same layout, so **no real defect images are needed**.

## 2. Hardware / pipeline split

```
            ┌────────────────────── VISION PIPELINE (real time) ──────────────────────┐
 USB cam ─► │ capture ─► ArUco + ECC registration ─► rectify to layout ─► golden diff │
            │            (register.py)                (register.py)      (vision.py)   │
            │      ─► candidate crops ─► CNN classifier (ONNX Runtime on Arm CPU,     │
            │                            or HailoRT on the NPU if an AI HAT is fitted) │
            └──────────────────────────────────────┬──────────────────────────────────┘
                                                   │ structured defect records
                    mapping.py (bitcell/net) ─► decision.py (rules, spares)
                                                   │
            ┌──────────────── ADVISOR PIPELINE (background worker) ───────────────────┐
            │ kb.py retrieval ─► prompt ─► llama_main (ExecuTorch, XNNPACK/KleidiAI,  │
            │ Llama 3.2 1B Instruct, 4-bit) on the Arm CPU ─► JSON validation ─► UI   │
            └─────────────────────────────────────────────────────────────────────────┘
                                                   │
                         store.py (SQLite log, images, CSV/JSON, repair map) ─► web UI
```

Our build is **Raspberry Pi 5, CPU-only**: both pipelines run on the Arm Cortex-A76 cores.
The vision pipeline's classifier runs in ONNX Runtime; Llama runs in ExecuTorch.
`compile_hef.py` and `HailoClassifier` exist so the classifier can move to a Hailo NPU
without code changes. The two pipelines are timed separately (`benchmark.py`).

## 3. Module by module

| File | What it does | Things to be able to explain |
|---|---|---|
| `scripts/build_design.py`, `gli/layout.py` | Reads the OpenRAM GDS. Walks the instance hierarchy to find regions (bitcell array, decoder, sense amps, write drivers, precharge, control). Finds all 256 `cell_1rw` bitcells and gives each a row/column. Transforms each bitcell's pin shapes (`bl`, `br`, `wl`, `vdd`, `gnd`, `Q`, `Q_bar`) into absolute coordinates with net names like `bl_12`. Renders the golden image and a printable target with 4 ArUco markers. | Why the boundary layer (63) is used for cell rectangles: mirrored rows make bounding boxes alternate. What `work_ppu` (4 px/µm) and `print_ppu` mean. |
| `gli/synth.py` | Injects defects into the **layer masks** (bridge = add metal, open = cut metal, missing via = delete a cut) or paints them like the physical demo (ink, correction fluid, dust). Rejects defects with no optical contrast (e.g. an open hidden under another metal) so labels stay honest. `camera_sim` adds blur, noise, colour cast, JPEG and residual misalignment. | Why "invisible" defects are rejected. Why we train on crops produced by the *runtime* detector (same distribution at train and test time). |
| `scripts/make_dataset.py` | Random 256×256 layout tiles → inject 0–3 defects → camera sim → run the station's candidate detector → label each candidate crop by overlap with ground truth (`NONE` if it's noise). | Reports candidate recall per class; that number belongs in the report. |
| `gli/model.py`, `scripts/train_classifier.py` | 8-conv CNN (~147k parameters), input 3×64×64: captured gray, golden gray, \|difference\|. Normalisation is inside the graph so the model takes raw 0–255 values (NPU friendly). Exports a dynamic-batch ONNX (CPU) and a batch-1 ONNX (Hailo DFC). | Why 3 channels instead of RGB: the network sees what *should* be there next to what *is* there, so "extra material" (bridge) and "missing material" (open) are easy to tell apart. |
| `gli/register.py` | ArUco corners → homography to layout pixels. Then **ECC** (`cv2.findTransformECC`, affine) aligns the rectified image to the golden image to sub-pixel accuracy. The refinement is cached while the camera doesn't move. Registration quality becomes mapping confidence. | Why ECC is needed: the difference detector only tolerates ±1 px of error. |
| `gli/vision.py` | `diff_map`: per channel, brightness-normalise, then the distance of the capture outside the golden image's local [min, max] in a 3×3 window (equivalent to the minimum over ±1 px shifts). `find_candidates`: threshold, morphology, connected components. `detect`: classify crops. Anything the CNN calls `NONE` but with a strong difference becomes `UNKNOWN` (the brief's "anomalies"). | Thresholds live in `VisionConfig`. The classifier backends: `OnnxClassifier` (CPU), `HailoClassifier` (NPU), `DiffOnlyClassifier` (fallback). |
| `gli/refree.py` | **Reference-free mode** for uploaded photos of dies we have no layout for. Finds the array's repeat period from the autocorrelation, compares every cell with its neighbours one period away (min over neighbours, so only the true defect survives, not its shifted "ghosts"), keeps only truly repeating areas (local cross-correlation with the shifted copy), then reuses the same candidate finder and CNN, with the neighbouring cell as the reference. Findings can't be mapped to bitcells without a layout, so they all go to REVIEW. | Why min over neighbours removes ghosts; why the area mask uses correlation plus texture (plain background and random periphery are excluded). Test results: our array without markers 38/40 found; a different memory pattern 14/15, all correctly classified; non-array images are rejected with a message. |
| `gli/mapping.py` | Detection box → µm box → region (most consequential overlap wins) → layers present → nets whose pin shapes touch the box → bitcell rows/columns. | Bitcell indices are only claimed inside the array. |
| `gli/decision.py` | **Deterministic guard.** Per-defect severity: REVIEW (unknown, low confidence) → KILLER (periphery, vdd–gnd short) → COSMETIC (dummy/outside) → REPAIRABLE (array). The fault footprint turns a defect into failing cells (a bitline open fails the whole column, a wordline bridge fails the whole row). Spare allocation is must-repair analysis plus a greedy cover. Die decision: any REVIEW → RE_INSPECT, any KILLER or over budget → SCRAP, any REPAIRABLE → REPAIR, else PASS. | Why rules decide and Llama explains: the station must never pass a die because of an LLM error. Every rule has a unit test. |
| `gli/kb.json`, `gli/kb.py` | 22 team-curated entries: defect class × region × net pattern → fault models, March tests, action, template sentence. Retrieval picks the most specific matching entry. The template is also the fallback advice. | Fault models (SAF, TF, CF, SOF, BL_OPEN, WL_OPEN, AF, …) and tests (MATS+, March C-, March SS, checkerboard, retention, IDDQ). |
| `gli/advisor.py` | Builds a compact JSON record plus allowed vocabulary from the KB, wraps it in the Llama 3 chat template with one worked example, runs `llama_main` (ExecuTorch), extracts JSON, validates it (keys, vocabulary, no invented row/column numbers, ≤60 words), retries once, then falls back to the template. The worker thread runs one call at a time and cancels queued work when a new die is inspected. | Why the validator checks numbers: a 1B model can invent indices. TTFT/tok/s come from the runner's `PyTorchObserver` line. |
| `gli/store.py` | SQLite in WAL mode with three tables (inspections, advice, actions), JPEG images, CSV/JSON export, repair-map PNG. Overrides are appended, never edited. | Low disk (< 500 MB) → metadata only. |
| `gli/station.py` | Camera thread + processing thread (live overlay at the vision pipeline's frame rate). `inspect()` freezes a frame, forces a fresh ECC, maps, decides, saves, then queues advice. | Focus gate (Laplacian variance), registration gate, more than 20 defects → RE_INSPECT. |
| `gli/server.py`, `gli/static/*` | FastAPI on localhost. MJPEG live stream; Home, Live, Inspection (frozen frame, layout zoom with highlighted bitcells, advisor card, repair map) and Log views. Plays an audio alert on SCRAP/RE_INSPECT. | Severity is shown as text + icon + colour (accessibility). |
| `gli/camera.py` | USB (V4L2), Pi camera, file, and a **virtual camera** that renders the target with random defects and perspective. Used for development, tests, benchmarks with ground truth, and as a demo fallback. | |
| `scripts/benchmark.py` | Live-loop FPS, inspect-to-decision latency, detection precision/recall against virtual ground truth, Llama TTFT / prefill / decode tok/s / load time, fallback and conflict counts. Writes CSV + summary.md for the report. | |

## 4. Known limitations (state these honestly in the report)

- **Optical imaging only.** Real sky130/SCMOS features are far below optical resolution.
  The demo inspects a **printed, magnified** layout target, and the method is
  imager-agnostic: the same pipeline works with SEM or higher-magnification images.
- **Small synthetic defects:** candidate recall is lower for missing vias and opens
  (3–4 px features at 4 px/µm). See `data/*/stats.json`.
- **Spare rows/columns** are a configured budget (2 + 2). This OpenRAM configuration
  doesn't generate redundancy.
- **Llama reloads the model on every call** (the ExecuTorch Python runner bindings
  aren't stable yet), adding about 1–3 s per advice. A persistent C++ runner is the
  obvious next step.
- **Routing nets between blocks aren't extracted:** defects there go to REVIEW.

## 5. Setup notes

**OpenRAM 1.2.48 with NumPy 2.** `compiler/base/vector.py` calls `float()` on a 1-element
array, which NumPy 2 rejects. Patch the installed copy, replacing

```python
        else:
            self.x = float(x)
            self.y = float(y)
```
with
```python
        else:
            import numpy as _np
            self.x = float(_np.asarray(x).reshape(-1)[0])
            self.y = float(_np.asarray(y).reshape(-1)[0])
```
Our generated macro uses `check_lvsdrc = False` (no Magic/Netgen needed), the bundled
`scn4m_subm` technology, 16 words × 16 bits. Config: `openram_out/sram_16x16_scn4m.py`.

**Camera resolution.** The target window is about 253 × 325 µm, and the station works at
4 px/µm (1012 × 1299 px). Frame the printed target so it fills most of a 1080p image;
at 720p small defects get lost.
