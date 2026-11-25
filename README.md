<div align="center">

# Golden-Layout Inspector

**Design-aware, offline defect inspection for SRAM dies on Arm devices, with on-device Llama 3.2 via ExecuTorch**

[![Challenge](https://img.shields.io/badge/Bharat%20AI--SoC%20Student%20Challenge-PS5-1B64BE)](#)
[![Platform](https://img.shields.io/badge/platform-Arm%20Android%20%7C%20Arm%20Linux-0091BD)](#deployment)
[![LLM](https://img.shields.io/badge/LLM-Llama%203.2%201B-6E44FF)](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
[![Runtime](https://img.shields.io/badge/runtime-ExecuTorch-EE4C2C)](https://pytorch.org/executorch)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)](https://www.python.org)
[![Tests](https://img.shields.io/badge/tests-88%20passing-2EA44F)](tests/)

[Report](docs/report/GLI_Final_Report.pdf) · [Demo video](#demo) · [Architecture](docs/ARCHITECTURE.md) · [Android guide](docs/ANDROID.md) · [Test images](test_images/)

</div>

---

Optical inspection simply tells you **where** a defect is. Golden-Layout Inspector also tells you **what it breaks and how to fix it**. It compares a camera image of an SRAM die with the memory's own design layout, generated with [OpenRAM](https://openram.org), traces each defect to the exact bitcells, rows, columns and nets, decides **repair or scrap** with deterministic rules, and uses **Llama 3.2 1B running on-device** to explain the likely electrical fault and the March test that confirms it.

Everything runs offline on an Arm CPU. **No real defect images are needed**: the training data is generated from the layout itself.

> Built for the **Bharat AI-SoC Student Challenge 2026**, Problem Statement 5: *Real-time industrial defect detection & quality-control assistant*.

## Contents

- [Demo](#demo)
- [How it works](#how-it-works)
- [Results](#results)
- [Quick start](#quick-start)
- [Deployment](#deployment)
- [Llama 3.2 via ExecuTorch](#llama-32-via-executorch)
- [Using the station](#using-the-station)
- [Benchmarking](#benchmarking)
- [Repository structure](#repository-structure)
- [PS5 requirements mapping](#ps5-requirements-mapping)
- [Limitations](#limitations)
- [Team](#team)
- [Licences and acknowledgements](#licences-and-acknowledgements)

## Demo

📹 **Demo video:** _link to be added_

<p align="center">
  <img src="docs/report/fig/ui_inspection_repair.png" width="85%" alt="Inspection screen: two array defects mapped to bitcells r7 c6-c7, decision REPAIR with spare columns 6 and 7">
</p>
<p align="center"><em>An inspection: two defects mapped to bitcells r7 c6–c7, whole-column impact, REPAIR using spare columns 6 and 7.</em></p>

## How it works

<p align="center">
  <img src="docs/report/fig/fig_architecture.png" width="70%" alt="System architecture">
</p>

**Offline, once per design**
1. **Golden design:** OpenRAM generates a 16 × 16-bit SRAM (SCMOS `scn4m_subm`). `gli/layout.py` extracts the regions, all 256 bitcells and 3,584 pin shapes, and renders the golden image plus a printable target with ArUco markers.
2. **Synthetic training data:** bridges, opens, particles and missing vias are drawn into the layout's layer masks, then made to look like camera photos (blur, noise, colour cast, misalignment, JPEG). Result: 23,288 labelled crops from 8,000 tiles, with zero real defect images.
3. **Classifier:** a 147k-parameter CNN that sees the captured crop, the golden crop and their difference side by side.

**On the device, two pipelines**
- **Vision (real time):** ArUco + ECC registration → shift-tolerant golden-layout difference → candidate crops → CNN (bridge / open / particle / missing via / unknown).
- **Mapping and rules:** defect → bitcells, rows, columns, nets → severity (COSMETIC / REPAIRABLE / KILLER / REVIEW) → spare-row/column allocation → **PASS / REPAIR / SCRAP / RE-INSPECT**.
- **Advisor (background):** knowledge-base retrieval → Llama 3.2 1B (ExecuTorch) picks the fault model and confirming test and writes a one-sentence explanation → validated against the KB. The rules always decide the action; Llama explains but can't change it.

**No layout?** Upload a photo of any memory die without markers and the station falls back to a **reference-free** cell-to-cell comparison of the repeating array (`gli/refree.py`).

## Results

Measured on an Arm Android smartphone (Snapdragon 7s Gen 3, Cortex-A720/A520), CPU only, offline:

| Metric | Result |
|---|---|
| Defect classifier (held-out synthetic crops) | **97.3%** accuracy, macro recall 97.1% |
| End-to-end detection, 10 simulated dies / 30 defects (phone) | precision **1.00**, recall **0.77**, **100%** of found defects correctly classified |
| Inspect → decision | **0.6 s** median (1.1 s p95) |
| Live overlay | 4.7 FPS |
| Llama 3.2 1B: model load / time to first token | 1.9 s / 1.8 s |
| Llama 3.2 1B: prefill / decode | 161 tok/s / 16.8 tok/s |
| Full advice per defect | 6.2 s median |
| Llama answers accepted by the validator | 23/23 (benchmark), 28/30 (station logs) |
| Reference-free mode | 38/40 defects found on our array; 14/15 on a different memory design |

Full methodology, per-class results and optimisation techniques: [**final report**](docs/report/GLI_Final_Report.pdf).

## Quick start

Try it on a laptop with the **virtual camera** (no phone, camera or Llama needed):

```bash
git clone https://github.com/ZenithRizz/Golden_Layout_Inspector.git
cd Golden_Layout_Inspector
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q                       # 88 tests
./scripts/run_station.sh virtual          # then open http://127.0.0.1:8000
```

The virtual camera renders the printed target with random defects; press **New virtual die** on the Live screen. Without Llama the station uses knowledge-base template advice, clearly labelled in the UI.

## Deployment

The same code runs on any 64-bit Arm device with a camera.

| | Android smartphone (demonstrated) | Raspberry Pi 5 / Arm Linux (supported) |
|---|---|---|
| Setup | [`docs/ANDROID.md`](docs/ANDROID.md) → `scripts/android/setup_termux.sh` | `scripts/setup_pi.sh` |
| Camera | Phone camera via the IP Webcam app (localhost) | USB webcam (`usb:0`) or Pi camera (`picam`) |
| Vision runtime | OpenCV DNN | ONNX Runtime (optional Hailo AI HAT+) |
| Llama runner | `llama_main` cross-compiled with the Android NDK | `llama_main` built natively |
| Start | `bash scripts/run_station.sh http://USER:PASS@127.0.0.1:8080` | `./scripts/run_station.sh usb:0` |

### Android (Termux)

```bash
# On the laptop: build the Android Llama runner and bundle everything
bash scripts/android/build_llama_android.sh
bash scripts/android/package_for_phone.sh

# On the phone, in Termux (GitHub build of Termux, not the Play Store one)
bash ~/storage/downloads/setup_termux.sh
cd ~/gli && bash scripts/run_station.sh http://USER:PASS@127.0.0.1:8080
```

Then open `http://127.0.0.1:8000` in the phone's browser. Step-by-step instructions, camera settings and troubleshooting: [`docs/ANDROID.md`](docs/ANDROID.md).

### Raspberry Pi 5

```bash
git clone https://github.com/ZenithRizz/Golden_Layout_Inspector.git
cd Golden_Layout_Inspector
./scripts/setup_pi.sh
./scripts/run_station.sh usb:0            # or: picam
```

Recommended: Raspberry Pi 5 (8 GB), active cooler, 27 W power supply, 1080p USB webcam on a stand 25–35 cm above the target.

## Llama 3.2 via ExecuTorch

The model weights are **not** in this repository (licence and size). To produce them:

1. Request access to [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) and run `hf auth login`.
2. Export to ExecuTorch (`.pte`, 8da4w quantisation, XNNPACK), either:
   - on Google Colab: [`notebooks/export_llama_colab.ipynb`](notebooks/export_llama_colab.ipynb) (recommended; the export needs more than 11 GB RAM), or
   - locally: `scripts/llama/01_setup_executorch.sh` then `scripts/llama/02_export_llama.sh`.
3. Build the runner: `scripts/android/build_llama_android.sh` (Android) or `scripts/llama/03_build_runner.sh` (Arm Linux).
4. Put `llama3_2_1b_instruct_8da4w_xnn.pte`, `tokenizer.model` and `llama_main` in `~/gli-models/llama/`, then check with `scripts/llama/04_smoke_test.sh`.

`run_station.sh` picks Llama up automatically; the Home screen shows **Llama 3.2 1B · ExecuTorch**.

## Using the station

**Printed demo target**
1. Print `designs/sram16x16_scn4m/target_A4.pdf` at **100% / actual size**, in colour, on matte paper.
2. Add defects by hand:
   - **Bridge:** a fine pen line in the metal colour joining two neighbouring lines.
   - **Open / crack:** a thin stroke of white correction pen across a line.
   - **Particle:** a pencil dot or a crumb.
   - **Missing via:** white-out over one dark via square.
3. Keep all four corner markers in view. An unmarked print is the PASS reference.

**Uploading images** (Live screen → *Upload image*)
- A photo of the **printed target** (markers visible) gets the full layout check: bitcell mapping and the repair decision.
- A photo of **any memory die** without markers gets the reference-free cell-to-cell check; findings go to engineer review.
- Images with no repeating array are rejected with an explanation.

Ready-made images with an answer key are in [`test_images/`](test_images/).

## Benchmarking

```bash
python scripts/benchmark.py --dies 40                         # vision + detection vs ground truth
python scripts/benchmark.py --dies 10 \
  --llama-runner ~/gli-models/llama/llama_main \
  --llama-pte ~/gli-models/llama/llama3_2_1b_instruct_8da4w_xnn.pte \
  --llama-tokenizer ~/gli-models/llama/tokenizer.model        # + Llama metrics
python scripts/benchmark.py --images data/real_targets/       # photos of printed targets
```

Results are written to `bench/<timestamp>/summary.md`.

<details>
<summary><b>Rebuild everything from scratch</b> (optional; all outputs are already in the repo)</summary>

```bash
# 1. Golden SRAM with OpenRAM (16 words x 16 bits, SCMOS scn4m_subm)
cd openram_out && python $(python -c "import openram,os;print(os.path.dirname(openram.__file__))")/sram_compiler.py -n sram_16x16_scn4m.py && cd ..
#    OpenRAM 1.2.48 + NumPy 2 needs the small vector.py patch in docs/ARCHITECTURE.md > Setup notes

# 2. Design package + printable target
python scripts/build_design.py --gds openram_out/sram_16x16/sram_16x16_scn4m.gds \
  --name sram16x16_scn4m --words 16 --word-size 16

# 3. Synthetic dataset (about 2 min) and classifier (about 5 min on CPU)
python scripts/make_dataset.py --tiles 8000
python scripts/train_classifier.py --epochs 8
```
</details>

## Repository structure

```
Golden_Layout_Inspector/
├── gli/                  station code
│   ├── layout.py         GDS extraction, golden render, printable target
│   ├── synth.py          synthetic defect injection + camera simulation
│   ├── camera.py         camera sources (IP Webcam, USB, Pi camera, virtual)
│   ├── register.py       ArUco + ECC registration
│   ├── vision.py         golden difference, candidates, classifier backends
│   ├── refree.py         reference-free (cell-to-cell) inspection
│   ├── mapping.py        defect → region, nets, bitcells
│   ├── decision.py       severity rules, spare allocation, die decision
│   ├── kb.py, kb.json    fault-model knowledge base
│   ├── advisor.py        Llama 3.2 advisor (ExecuTorch) + validation
│   ├── store.py          SQLite log, CSV/JSON export, repair maps
│   ├── station.py        orchestration
│   ├── server.py         FastAPI server (Linux / Pi)
│   ├── server_lite.py    dependency-free server (Android / Termux)
│   └── static/           web UI
├── scripts/              build_design, make_dataset, train_classifier, benchmark,
│                         run_station, setup_pi, llama/*, android/*
├── designs/              golden design package (design.json, golden image, masks, target PDF)
├── models/               trained classifier (ONNX) + metrics
├── data/                 synthetic dataset statistics
├── openram_out/          OpenRAM config + generated macro (GDS, LEF, Verilog, LIB)
├── notebooks/            Colab notebook for the Llama export
├── test_images/          sample images with answer key
├── tests/                unit tests
└── docs/                 architecture, Android guide, PRD, final report
```

## PS5 requirements mapping

| Requirement | Implementation |
|---|---|
| Detect defects, cracks, anomalies | `gli/vision.py`: golden-layout difference + CNN (bridge, open/crack, particle, missing via) + `UNKNOWN` anomalies |
| Classify type and severity | CNN class; `gli/decision.py` severity (COSMETIC / REPAIRABLE / KILLER / REVIEW) and die decision |
| Alert with actionable advice | `gli/advisor.py`: Llama 3.2 1B via ExecuTorch (fault model, confirming test, action, explanation); UI banner + audio alert |
| Log every inspection | `gli/store.py`: SQLite + images, CSV/JSON export, repair maps |
| Real time, on the edge, no cloud | Arm CPU of an Android phone or a Raspberry Pi 5 / any arm64 Linux board; no network use |
| Llama + ExecuTorch | `scripts/llama/*`, `scripts/android/*`, `LlamaExecuTorchBackend` in `gli/advisor.py` |

## Limitations

- **Optical proof of concept.** A phone camera can't resolve real sub-micron features, so the demo inspects a magnified print of the layout. Real dies need a microscope or SEM and a renderer tuned to that imager; the method stays the same.
- **Accuracy figures are on simulated dies.** Real-photo accuracy hasn't been measured on a labelled set. The detector is tuned for zero false alarms, so the smallest opens and missing vias can be missed.
- **Spares are a configured budget** (2 rows + 2 columns); this OpenRAM configuration doesn't generate redundancy.
- **Product vision:** manufacturers upload their own GDS/OASIS layout as the golden reference; this proof of concept uses an OpenRAM SRAM.

## Licences and acknowledgements

- [OpenRAM](https://openram.org): BSD 3-Clause, used as a tool to generate the golden layout.
- [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct): Llama 3.2 Community License. **Built with Llama.** Weights are not redistributed here.
- [ExecuTorch](https://pytorch.org/executorch): BSD.
- OpenCV, ONNX Runtime, FastAPI: permissive licences.
- No AGPL code and no non-commercial datasets are used.
