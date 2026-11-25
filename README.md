# Golden-Layout Inspector

**Design-aware, offline defect inspection for SRAM dies on Arm devices (Android phone or Raspberry Pi), with on-device Llama 3.2 (ExecuTorch).**
PS5: Real-time industrial defect detection & quality-control assistant.

> **Bharat AI-SoC Student Challenge 2026: PS5, real-time industrial defect detection & QC assistant.**
> Demo device: **Nothing Phone (3a) Pro** (Snapdragon 7s Gen 3, Arm Cortex-A720/A520), Android + Termux, fully offline.
> Final report: [`docs/report/GLI_Final_Report.pdf`](docs/report/GLI_Final_Report.pdf) · Demo video: _link_ · Test images with answer key: [`test_images/`](test_images/)


Existing inspection says *where* a defect is. This station says *what it breaks and how to fix it*:
1. It compares the camera image against the memory's own layout, generated with
   [OpenRAM](https://openram.org).
2. It classifies each defect.
3. It maps the defect to the exact bitcell, row, column and net.
4. It decides repair (using spare rows/columns) or scrap with fixed rules.
5. An on-device Llama 3.2 1B explains the likely electrical fault and the March test that confirms it.

No real defect images are needed: training data is generated from the layout.

- Architecture & code walkthrough: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- Product spec: [`docs/prd.md`](docs/prd.md)

## Requirements mapping (PS5)

| Requirement | Where |
|---|---|
| Detect defects, cracks, anomalies | `gli/vision.py`: golden-layout difference + CNN (bridge, open/crack, particle, missing via) + `UNKNOWN` anomalies |
| Classify type and severity | CNN class; `gli/decision.py` severity (COSMETIC / REPAIRABLE / KILLER / REVIEW) and die decision |
| Alert with actionable advice | `gli/advisor.py`: Llama 3.2 1B via ExecuTorch (fault model, confirming test, action, explanation); UI banner + audio alert |
| Log every inspection | `gli/store.py`: SQLite + images, CSV/JSON export, repair maps |
| Real time, on the edge, no cloud | Arm CPU of an Android phone or a Raspberry Pi 5 / any arm64 Linux board; no network use; `scripts/benchmark.py` measures it |
| Llama + ExecuTorch | `scripts/llama/*.sh` export and runner; `LlamaExecuTorchBackend` |

## Repository layout

```
gli/                 station code (layout, synth, vision, register, mapping, decision, kb, advisor, store, server, UI)
scripts/             build_design, make_dataset, train_classifier, benchmark, setup_pi, run_station, llama/*, compile_hef (optional)
designs/<name>/      golden design package (design.json, golden image, masks, printable target)
models/              trained classifier (ONNX) + metrics
data/<name>/         synthetic training crops + stats
openram_out/         OpenRAM config + generated macro (GDS, LEF, Verilog, LIB)
tests/               unit tests (decision rules, allocation, KB coverage, advisor validation, mapping)
docs/                architecture, PRD, strategy
```

## Quick start (laptop, no hardware)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q                                     # 88 tests
./scripts/run_station.sh virtual                        # open http://127.0.0.1:8000
```

The **virtual camera** renders the printed target with random defects. Use *New virtual die* on the Live screen.

### Rebuild everything from scratch (optional; outputs are already in the repo)

```bash
# 1. Golden SRAM with OpenRAM (16 words x 16 bits, SCMOS scn4m_subm)
cd openram_out && python $(python -c "import openram,os;print(os.path.dirname(openram.__file__))")/sram_compiler.py -n sram_16x16_scn4m.py && cd ..
#    (OpenRAM 1.2.48 + NumPy 2 needs the small vector.py patch in docs/ARCHITECTURE.md > Setup notes)
# 2. Design package + printable target
python scripts/build_design.py --gds openram_out/sram_16x16/sram_16x16_scn4m.gds --name sram16x16_scn4m --words 16 --word-size 16
# 3. Synthetic dataset (about 2 min) and classifier (about 5 min CPU, faster on Colab)
python scripts/make_dataset.py --tiles 8000
python scripts/train_classifier.py --epochs 8
```

## Llama 3.2 via ExecuTorch

1. **Access:** request it at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct, then run `hf auth login`.
2. **Laptop (x86 Ubuntu):** `scripts/llama/01_setup_executorch.sh`, then `scripts/llama/02_export_llama.sh`.
   This produces `llama3_2_1b_instruct_8da4w_xnn.pte` and `tokenizer.model`.
3. **Pi 5:** `scripts/llama/01_setup_executorch.sh`, then `scripts/llama/03_build_runner.sh` (builds `llama_main`).
   Copy the `.pte` and tokenizer to `~/gli-models/llama/`.
4. Run `scripts/llama/04_smoke_test.sh`. Then `run_station.sh` uses Llama automatically
   (the status card shows "Llama 3.2 1B · ExecuTorch").

If Llama is unavailable, the station still works with knowledge-base template advice, clearly labelled in the UI.

## Android phone deployment

See [`docs/ANDROID.md`](docs/ANDROID.md): Termux + the IP Webcam app + an Android build of ExecuTorch's `llama_main`. Everything runs on the phone's Arm CPU.

## Raspberry Pi 5 deployment

```bash
git clone <this repo> && cd gli
./scripts/setup_pi.sh
./scripts/run_station.sh usb:0          # or: picam
```

**Hardware:**
- Raspberry Pi 5 (8 GB)
- active cooler
- 27 W power supply
- USB webcam (1080p recommended) on a stand, about 25–35 cm above the target, even lighting

## Physical demo target

1. Print `designs/sram16x16_scn4m/target_A4.pdf` at **100% / actual size**, in colour, on matte paper.
2. Make defects by hand, so the test set is not synthetic:
   - **Bridge:** a fine pen line in the metal colour (blue = m1, purple = m2) joining two neighbouring lines.
   - **Open / crack:** a thin stroke of white correction pen across a line.
   - **Particle:** a pencil dot, a crumb or dust.
   - **Missing via:** white-out over one dark via square.
3. Keep all four corner markers visible. Use an unmarked print as the PASS reference.

## Benchmark & evaluation

```bash
python scripts/benchmark.py --dies 40                                        # vision + detection vs ground truth
python scripts/benchmark.py --dies 20 --llama-runner ... --llama-pte ... --llama-tokenizer ...   # + Llama metrics
python scripts/benchmark.py --images data/real_targets/                     # photos of printed targets
```

Results go to `bench/<stamp>/summary.md`, ready to paste into the report.

## Uploading images

On the Live screen, **Upload image** accepts:
- a photo of the **printed target** (four corner markers visible): the full layout check, with bitcell mapping and the repair decision;
- a photo of **any memory die**, without markers: the **reference-free** cell-to-cell check of the repeating array (`gli/refree.py`). It finds and classifies anomalies, but without a layout every finding goes to engineer review.

Images with no repeating array are rejected with an explanation.

## Licences and credits

- **OpenRAM:** BSD 3-Clause, used as a tool.
- **Llama 3.2:** Llama 3.2 Community Licence. *Built with Llama.*
- **ExecuTorch:** BSD.
- **OpenCV, ONNX Runtime, FastAPI:** permissive licences.
- No Ultralytics/AGPL code and no non-commercial datasets.
