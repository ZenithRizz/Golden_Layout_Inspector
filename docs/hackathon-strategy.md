# Hackathon Strategy: Arm × Llama/ExecuTorch Edge AI Hackathon — PS5 (Manufacturing & Industry 4.0)

## Hackathon Profile

**Organisers / sponsors (inferred):** Arm (hardware, KleidiAI), Meta (Llama, ExecuTorch), Raspberry Pi / Hailo (AI HAT+ / AI HAT+ 2). The event's public page was not found; details come from the problem statements the team supplied.

**Tracks offered:** Agriculture & Food Security; Environment & Climate; Cybersecurity & Digital Safety; Education (Android only); Manufacturing & Industry 4.0 (PS5 — real-time industrial defect detection & QC assistant).

**Chosen track:** PS5 — Manufacturing & Industry 4.0.

**PS5 functional requirements:**
- Detect surface defects, cracks and anomalies
- Classify defect type and severity
- Alert operators with actionable recommendations
- Log inspection results
- Run in real time at the edge, no cloud

**Mandatory technology:** A Llama-family model (e.g., Llama 3.2 1B/3B-Instruct, or fine-tuned/distilled variant) for the advisory/reasoning component, deployed on-device with ExecuTorch as the reference runtime.

**Allowed hardware:** Raspberry Pi 5 (CPU-only); Pi 5 + AI HAT+ / AI HAT+ 2; Arm Linux devices; Arm Android devices with camera.

**Hailo rule:** If Hailo is used, vision runs via HailoRT on the NPU and Llama runs via ExecuTorch on the Pi's Arm CPU. The two pipelines must be clearly separated and documented.

**Optional tech:** OpenCV, Hailo SDK, MVTec AD / NEU / KolektorSDD, YOLO / EfficientDet, anomaly-detection frameworks.

**Deliverables:** source code, application, demo video, technical report.

**Eligibility:** Indian nationals; 1–3 members; student teams from one college, nominated by the college, with a faculty mentor and a team leader.

**Originality rules:** Entry must be the entrants' original work, solely owned by them, and must not infringe anyone's IP.

**Team:** 3 members — AI/ML, software engineering, VLSI. Currently working on OpenRAM (BSD 3-Clause SRAM compiler), which will be used strictly as a dependency/tool.

**Assumptions (unconfirmed — correct if wrong):**
1. Judging is roughly balanced across innovation, technical execution/edge optimisation, real-world impact, demo, and report.
2. Build window is several weeks, not a 24–48 h sprint.
3. The team can obtain a Raspberry Pi 5 (8 GB) and an AI HAT+ or AI HAT+ 2, plus a USB camera / USB microscope.
4. An x86-64 Ubuntu machine is available for the Hailo Dataflow Compiler (the compiler does not run on the Pi).

## Winning Project Profile

What scores well at this event, and why:

1. **Llama must reason, not chat.** Arm already publishes "run Llama on a Pi 5 with ExecuTorch" and "Android Llama chat app" learning paths, so judges have seen the basic demo many times. The winning pattern is: specialised vision model → structured detection output → Llama turns it into a grounded, domain-specific decision the operator could not easily make alone.
2. **Edge engineering is the differentiator for these sponsors.** Show 4-bit quantisation, XNNPACK/KleidiAI kernels, tokens/s, time-to-first-token, peak RSS, end-to-end latency, and a clean NPU (vision) vs CPU (Llama) split. Ablations (1B vs 3B, quantisation scheme, prompt length) make the report stand out.
3. **Visibly offline and real-time demo.** Pull the network cable on camera; a live camera feed must drive the pipeline.
4. **India relevance.** India's semiconductor packaging and test (OSAT/ATMP) industry went live in 2026 (Micron Sanand inaugurated 28 Feb 2026; Kaynes Semicon OSAT inaugurated 31 Mar 2026; Tata Assam and CG Power–Renesas sites ramping). A project aimed at this new industry hits a nationally visible priority.
5. **Domain credibility.** A team with real VLSI/memory-design experience can answer judge questions that a generic CV team cannot.
6. **Licence hygiene.** Avoid Ultralytics YOLO (AGPL-3.0; trained weights also fall under AGPL by Ultralytics' terms). Prefer Apache-2.0/MIT detectors (YOLOX, EfficientDet, MobileNet-SSD, anomalib models) or own-trained models. Credit "Built with Llama". OpenRAM (BSD-3) is fine as a credited dependency.

## Problem Rankings

Scores 1–10: Impact, Innovation, Technical complexity, Feasibility, Market need, Judge appeal, Startup potential. Overall = mean.

| Rank | # | Problem | Track | Overall | OpenRAM fit |
|---|---|---|---|---|---|
| 1 | M6 | Anomaly detection without defect samples + operator copilot | Mfg | 8.1 | Low |
| 2 | M1 | PCB assembly inspection for small Indian assemblers | Mfg | 8.0 | Low |
| 3 | M4 | Fabric defects in power-loom clusters | Mfg | 7.7 | None |
| 4 | C6 | Live scam-call coach for elderly users | Cyber | 7.6 | None |
| 5 | A3 | Pest trap counting and spray timing | Agri | 7.4 | None |
| 6 | C1 | Hinglish/regional SMS & WhatsApp scam detection | Cyber | 7.3 | None |
| 7 | M2 | Wafer-map defect patterns with root-cause reasoning | Mfg | 7.3 | Medium |
| 8 | M5 | Tablet/capsule inspection under Schedule M | Mfg | 7.1 | None |
| 9 | A1 | Offline multilingual crop-disease advisory | Agri | 7.1 | None |
| 10 | A2 | Extension-worker triage with outbreak logging | Agri | 7.1 | None |
| 11 | **M9** | **Memory-die defect inspection trained on OpenRAM-generated data** | **Mfg** | **7.0** | **High** |
| 12 | A6 | Pesticide label misreading and overuse | Agri | 7.0 | None |
| 13 | E3 | Material sorting for informal waste pickers | Env | 7.0 | None |
| 14 | C4 | OT/Modbus anomaly monitoring for small factories | Cyber | 6.9 | None |
| 15 | Ed4 | Learning pathways for chip-design talent | Edu | 6.7 | Medium |
| 16 | C2 | Home/SME network intrusion detection | Cyber | 6.7 | None |
| 17 | C3 | IoT device behaviour fingerprinting | Cyber | 6.7 | None |
| 18 | A5 | Post-harvest storage spoilage | Agri | 6.6 | None |
| 19 | E2 | E-waste identification and safe channels | Env | 6.4 | None |
| 20 | A7 | Fertiliser advice from soil health card | Agri | 6.4 | None |
| 21 | Ed3 | Skill gaps to jobs for ITI students | Edu | 6.4 | None |
| 22 | M3 | Steel surface defects (NEU) | Mfg | 6.3 | None |
| 23 | M7 | Packaging seal and label integrity | Mfg | 6.3 | None |
| 24 | C5 | QR and phishing-URL fraud | Cyber | 6.3 | None |
| 25 | Ed2 | Offline learning companion in local languages | Edu | 6.3 | None |
| 26 | Ed1 | Adaptive exam-prep planners | Edu | 5.6 | None |
| 27 | E1 | Household waste segregation coach | Env | 5.3 | None |

**Team decision:** M9 selected, by explicit team choice, because it uses the team's unique OpenRAM/VLSI capability. Its raw score is mid-table; the selected solution (below) is designed to close the gap on demo quality and PS5 fit, and borrows M6's "learn from good samples" idea.

## Market Research

### M9 — Memory-die / memory-array defect inspection with synthetic layout-derived training data

**Incumbent inspection & review equipment**
- **KLA, Applied Materials (SEMVision line), Onto Innovation, Camtek.** Applied's SEMVision H20 eBeam review system uses deep-learning image models to separate true defect types and is adopted by leading logic and memory makers. Strengths: physics-level sensitivity, fab integration. Weaknesses: multi-crore capital cost, cloud/fab-server analytics, closed models, no operator-facing natural-language reasoning on the tool, out of reach for small packaging/test lines and labs.

**AI defect-classification software startups**
- **SixSense (Singapore).** AI automated defect classification (ADC) and lot disposition for semiconductor lines; raised an $8.5M Series A led by Peak XV's Surge; deployed at GlobalFoundries and JCET; partnered with Tong Hsing (packaging/test) in Feb 2026; now building agentic root-cause analysis. Strengths: real fab data, no-code model training. Weaknesses: server/cloud software sold to large fabs; needs the customer's real defect images to train; not an edge device; no memory-architecture-aware reasoning.
- **Averroes.ai, Landing AI, Robovision, Cognex/Halcon-based in-house teams.** General or semi-specialised visual-inspection platforms; same weaknesses — data-hungry, server-based, generic.
- **Chipway (China)** — AI ultrasonic inspection hardware; different modality (packages/IGBTs), not memory-array optical inspection.

**Yield / RCA copilots**
- **PDF Solutions Exensio + Lavorro (FabAssist.ai / ToolAssist.ai)** — generative-AI virtual assistant for root cause on fab data. **SemiAI (Korea)** — agentic yield and RCA platform. These are enterprise, cloud/on-prem server systems tied to MES and yield databases; none runs on a ₹15–25k edge device next to the inspection station, and none links an image defect to a memory-specific fault model and repair decision.

**Research / open-source**
- **Siemens EDA (SPIE 2024 / ASMC 2024):** pairs SEM images with aligned design-layout clips as two input channels for a YOLOv8 detector, and uses a conditional GAN to translate (deliberately modified) layouts into synthetic defective SEM images — direct evidence that layout-driven synthetic data works.
- **IWAPS 2024 (Zhao et al.):** generates a reference SEM image from layout, then compares it with the captured image to find defects (die-to-database style).
- **Diffusion-model synthetic SEM generation (arXiv 2407.10348):** realistic noise/roughness in a limited-data regime.
- **Synthetic line-pattern SEM defects (breaks, bridges)** trained YOLOv8/SSD/EfficientNet that transferred to real SEM images.
- **REFICS (WACV 2022):** ~800k synthetic SEM images of IC layouts plus a generator tool, for hardware assurance — a public benchmark for layout-to-image realism.
- **SRAM defect/fault literature (e.g., LIRMM 8T SRAM resistive-open/short injection; cell-aware test for memories):** maps physical defects to functional fault models and March-test detection — the knowledge Llama would be grounded on.
- No public repo found that combines an open SRAM compiler (OpenRAM) → synthetic defect dataset → edge detector → LLM fault-model/repair advisor.

**Market gap (precise statement):**
Everything that exists either (a) needs the customer's own scarce, proprietary defect images to train, (b) runs on expensive tools or servers, or (c) stops at "defect class X at location Y". Nothing gives a packaging/test-line or lab operator an **offline, low-cost station that knows the design** (via the layout) and **reasons about what the defect means electrically for a memory** — which bitcells/rows/columns are affected, which functional fault it causes, which March test confirms it, and whether spare rows/columns can repair it. The gap exists because incumbents sell to fabs that already have data and servers, and the memory-design knowledge needed to bridge "image → fault model → repair" usually sits with a different team (design/DFT) than the inspection team. An open SRAM compiler makes it possible to own the design data and generate unlimited labelled training data.

**Caveats found in research:**
- India's new plants are mainly assembly/test (ATMP/OSAT), not wafer fabs; Tata Dholera's fab is targeted for first silicon around 2026–27. The near-term Indian user is therefore packaging/test lines, failure-analysis labs, university/fab-lab clean rooms, and the coming fab — not high-volume front-end wafer inspection.
- Real SEM is not possible on a Pi; the station must use optical/microscope imaging. The demo must be honest that resolution is optical-microscope-class and that the method (layout-reference + LLM reasoning) is what scales to higher-resolution imagers.

## Candidate Solutions

All five are PS5-compliant (detect → classify type & severity → recommend → log → real-time edge). Hardware baseline: Pi 5 8 GB + AI HAT+/AI HAT+ 2 + USB microscope camera, unless noted.

### S1 — Supervised synthetic-data detector + Llama report writer (baseline)
- **Core idea:** Render OpenRAM SRAM layouts, inject defects, train an Apache-licensed detector; Llama writes a plain-language inspection report.
- **Users:** Inspection operators.
- **UVP:** Zero real defect data needed.
- **AI angle:** Detector + LLM summarisation.
- **Architecture:** OpenRAM GDS → renderer + defect injector → detector (Hailo) → JSON → Llama 3.2 1B (ExecuTorch, CPU) → report + SQLite log.
- **Business:** Licensed inspection software.
- **Complexity / feasibility:** Medium / high. ~2–3 weeks.
- **Risks:** Llama role looks thin ("report writer"); sim-to-real gap.
- **Judge appeal / demo / novelty:** Moderate / moderate / moderate.

### S2 — "Golden-Layout Inspector": layout-aware defect inspection with a memory fault-and-repair copilot ★
- **Core idea:** Use the OpenRAM layout as the golden reference ("die-to-database" on the edge). The vision stage finds and classifies physical defects and **registers them to layout coordinates**, so each defect is mapped to the exact bitcell, wordline, bitline and layer. Llama, grounded on a compact memory fault knowledge base, turns that into: affected cells/rows/columns → likely functional fault (stuck-at, transition, coupling, bitline open, wordline short) → confirming March-test recommendation → repair decision using spare rows/columns (repair / scrap / re-inspect) → severity → operator action. All logged.
- **Users:** Packaging/test-line operators, failure-analysis engineers, university and fab-lab clean rooms; later, Indian OSAT/ATMP and the Dholera fab.
- **UVP:** The only edge inspector that knows the memory's design and reasons about electrical consequence and repairability, trained without any proprietary defect images.
- **AI integration:**
  1. Synthetic data engine (own code) on top of OpenRAM GDS: layer rendering, realistic optical blur/noise/illumination, defect injection (bridge/short, open/break, particle, missing contact/via, line-width deviation), automatic labels.
  2. Vision on Hailo NPU via HailoRT: small Apache-2.0 detector/segmenter (e.g., YOLOX-Nano or an EfficientDet/UNet variant) + a reference-difference channel (captured image vs rendered golden tile), following the published layout+image two-channel approach.
  3. Geometry stage on CPU (OpenCV): registration of captured image to layout, defect-to-net/bitcell mapping using the layout's cell grid.
  4. Llama 3.2 1B-Instruct (optionally 3B) via ExecuTorch on CPU, 4-bit, XNNPACK/KleidiAI; structured JSON in, constrained JSON + short operator text out; small retrieval over a team-written fault/test/repair knowledge base; optional LoRA fine-tune on team-generated (defect → fault → action) pairs.
- **Architecture:** Camera → OpenCV preprocess → [NPU] defect model → [CPU] registration + bitcell mapping → [CPU] Llama advisor → operator UI (local web UI on the Pi) + SQLite inspection log + per-die repair map export.
- **Business model:** Low-cost inspection station + per-line software licence; design-aware synthetic-data service for memory/IP vendors; extension to logic standard-cell arrays.
- **Scalability / roadmap:** Swap optical camera for higher-res imagers; add real images via few-shot adaptation; add wafer-map aggregation; support other compilers/PDKs; integrate with ATE results.
- **Complexity / feasibility:** High / medium-high. ~4–5 weeks for 3 people (fits assumed window; can fall back to S1 scope if time is cut).
- **Risks:** Sim-to-real gap (mitigate: domain randomisation + small real/printed-photo set + reference-difference channel); registration accuracy; Hailo DFC needs an x86 host; Llama hallucination (mitigate: constrained JSON, KB grounding, rule check on repairs).
- **Judge appeal / demo / novelty / patentability:** Very high / high (live microscope, defect lights up the exact bitcell, Llama explains the fault and repair, network unplugged) / high / plausible claim on layout-registered defect-to-fault-model reasoning on edge.

### S3 — Good-samples-only anomaly detector + operator copilot (M6 hybrid)
- **Core idea:** Train an unsupervised anomaly model (PatchCore/PaDiM-style) only on defect-free OpenRAM renders; heatmap → Llama explains and recommends.
- **UVP:** Catches unseen defect types.
- **Architecture:** Anomaly model (CPU or NPU) → heatmap → Llama.
- **Complexity / feasibility:** Medium / medium (anomaly models are memory-heavy on Pi; Hailo support for memory-bank methods is limited).
- **Risks:** Weak defect-type classification (PS5 asks for type + severity); Llama has less to reason over.
- **Judge appeal:** Good; novelty moderate (anomaly + LLM is becoming common).

### S4 — Android phone-microscope app for OSAT technicians
- **Core idea:** Clip-on phone microscope + Android app; detector via ExecuTorch/LiteRT; Llama 1B on phone via ExecuTorch.
- **UVP:** Cheapest form factor; very portable.
- **Complexity / feasibility:** High / medium (two on-device runtimes on Android, camera pipeline, thermal limits).
- **Risks:** Loses the NPU story; image quality of clip-on optics; heavier app engineering.
- **Judge appeal:** Moderate-high; scalable and cheap, but weaker "industrial station" feel.

### S5 — Die-image + wafer-map fusion yield copilot
- **Core idea:** Combine per-die defect detections with wafer-level spatial patterns (WM-811K style) so Llama reasons about process root cause across a wafer.
- **UVP:** Moves from single-die to process-level insight.
- **Complexity / feasibility:** High / medium-low within the window.
- **Risks:** Two datasets and two models; wafer-map part is not camera/real-time; PS5 fit weaker; Indian users (ATMP) rarely own wafer maps.
- **Judge appeal:** High technically, lower on PS5 fit and demo.

## Decision Matrix

Weights mirror the assumed rubric (innovation, technical/edge execution, Llama+ExecuTorch use, PS5 fit, demo). Scores 1–10; weighted score = Σ(weight × score)/100.

| Criterion | Weight | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|---|
| Innovation | 10 | 6 | 9 | 8 | 7 | 7 |
| Originality | 8 | 6 | 10 | 8 | 7 | 6 |
| Time required (fit to window) | 8 | 8 | 6 | 6 | 4 | 4 |
| Complexity manageability | 5 | 8 | 5 | 6 | 4 | 4 |
| Technical depth | 12 | 6 | 9 | 8 | 7 | 8 |
| Llama + ExecuTorch centrality | 12 | 6 | 9 | 8 | 5 | 7 |
| PS5 brief fit | 10 | 8 | 8 | 7 | 7 | 6 |
| Demo wow factor | 12 | 6 | 9 | 7 | 6 | 7 |
| Scalability | 4 | 7 | 7 | 6 | 9 | 7 |
| Business potential | 4 | 6 | 7 | 6 | 8 | 6 |
| Risk (higher = lower risk) | 6 | 8 | 6 | 6 | 4 | 4 |
| Cost | 2 | 9 | 8 | 8 | 8 | 9 |
| Judge appeal | 5 | 6 | 9 | 7 | 6 | 7 |
| Startup viability | 2 | 6 | 7 | 6 | 8 | 5 |
| **Weighted total** | **100** | **6.68** | **8.14** | **7.15** | **6.18** | **6.33** |

**Winner: S2 — Golden-Layout Inspector.**

**Why S2 wins:** It is the only option where Llama does reasoning no rule table or classifier would do on its own (defect → affected cells → fault model → test → repair), which directly answers "meaningful use of Llama". It has the strongest edge-engineering story (NPU vision + CPU Llama, measured separately), the most original data story (owned synthetic data from an open SRAM compiler, no licence-restricted datasets), and a demo moment judges remember (defect lights up the exact bitcell on the layout, Llama states the fault and repair, network unplugged). It is also the option only this team can credibly build.

**Why the runners-up lose:**
- **S3 (7.15):** Strong idea, but PS5 asks for defect *type* and severity, which anomaly-only methods do weakly, and memory-bank anomaly models are awkward on Hailo/Pi. Keep its idea as a **secondary "unknown defect" flag inside S2**.
- **S1 (6.68):** Safest and fastest, but Llama is decorative and the project resembles many student defect-detection entries. It is S2's **fallback scope** if time runs short.
- **S5 (6.33):** Technically rich but splits effort across two datasets, the wafer-map part is not real-time camera inspection, and India's live plants are mainly ATMP without wafer maps.
- **S4 (6.18):** Cheapest form factor, but drops the NPU story, adds Android camera + dual-runtime risk, and clip-on optics hurt detection quality. Good **post-hackathon product variant**.

## Selected Project

**Name (working):** Golden-Layout Inspector (alt: "SiliconSight", "RAMScope").

**Problem statement:**
Memory dies and arrays are inspected with expensive tools that need large proprietary defect datasets, and even then they only report "defect type at location". The operator still has to work out what the defect means for the memory — which cells fail, how to confirm it, and whether spare rows/columns can repair it. India's new packaging/test lines, failure-analysis labs and university fab labs have neither the data nor the budget for this, and need an affordable offline station that turns a defect image into a repair decision.

**Solution summary:**
An offline Raspberry Pi 5 + Hailo inspection station that uses the memory's own layout (generated with OpenRAM, used as a credited BSD-3 dependency) as a golden reference. A detector trained entirely on the team's own synthetic layout-derived dataset finds and classifies defects on a live microscope feed (Hailo NPU via HailoRT); OpenCV registers each defect to the exact bitcell/wordline/bitline; Llama 3.2 1B running through ExecuTorch on the Arm CPU reasons over the mapped defect with a memory fault/test/repair knowledge base and tells the operator the likely fault, the March test to confirm it, the severity, and whether to repair, scrap or re-inspect. Every inspection is logged with a repair map.

**Elevator pitch:**
"Chip inspection tools tell you *where* a defect is. Ours tells you *what it breaks and how to fix it* — offline, on a ₹25k Raspberry Pi. We generate unlimited labelled training data from the memory's own layout, so a new line needs zero defect images to start, and an on-device Llama acts as the memory test engineer standing next to every operator."

**One-line product vision:**
Design-aware, offline inspection that turns every defect image into a repair decision.

**Success metrics (for the demo and report):**
- Detection: mAP@0.5 ≥ 0.80 on synthetic test set; report separately on a small real/printed-image set (sim-to-real gap shown honestly).
- Localisation: ≥ 90 % of detected defects mapped to the correct bitcell/line.
- Advice quality: ≥ 85 % agreement between Llama's fault-model + repair decision and a team-written ground-truth rubric over ≥ 100 cases; 0 invalid JSON outputs (constrained decoding).
- Real-time: vision ≥ 15 FPS on Hailo; end-to-end defect-to-advice ≤ 3 s; Llama decode tokens/s, TTFT and peak RSS reported for 1B (and 3B if tested) at 4-bit.
- Offline: full demo with network disconnected.
- Logging: 100 % of inspections persisted with image, defects, mapping, advice, timestamp.

**Future startup vision:**
Start with failure-analysis labs, university/fab-lab clean rooms and Indian ATMP/OSAT lines as low-cost, design-aware inspection stations. Grow into a "design-aware inspection" platform: synthetic defect-data generation as a service for memory/IP vendors, support for logic standard cells and other PDKs, higher-resolution imaging back-ends, and integration with ATE test results so physical defects and electrical failures are reconciled automatically. Positioned as India-built inspection intelligence for the domestic semiconductor programme, complementary to (not competing with) high-end fab tools.

## Build Constraints

- **Track:** PS5 — Manufacturing & Industry 4.0 (real-time industrial defect detection & QC assistant).
- **Time limit:** Project must be complete by **23 Nov 2026**, with work starting 12 Nov (11 working days). The lean scope and dated gates are in `prd.md`, Section 4. Fallback is S1 scope (detector + Llama advice, without bitcell mapping).
- **Team:** 3 members (AI/ML, SDE, VLSI); team leader + faculty mentor required; same college; college nomination.
- **Mandatory:** Llama-family model for advisory/reasoning, deployed on-device via ExecuTorch (reference runtime) on Arm CPU.
- **Hardware:** Raspberry Pi 5 (8 GB recommended) + AI HAT+ or AI HAT+ 2 (Hailo-10H, 40 TOPS INT4, $130); USB microscope/camera; active cooling. x86-64 Ubuntu host required for Hailo Dataflow Compiler.
- **Pipeline separation (must be documented):** Vision → HailoRT on Hailo NPU. Llama → ExecuTorch on Pi Arm CPU. Report each pipeline's latency/memory separately.
- **Dependencies & licences:**
  - OpenRAM — BSD 3-Clause, used only as a tool/dependency; keep notice; credit in README/report.
  - Llama 3.2 — Llama Community Licence; "Built with Llama" attribution.
  - ExecuTorch, OpenCV, HailoRT — permissive/vendor licences; credit.
  - Detector: Apache-2.0/MIT models only (e.g., YOLOX, EfficientDet, MobileNet-SSD, own UNet). **Do not ship Ultralytics YOLO (AGPL-3.0).**
  - Layout rendering: prefer a permissively licensed GDS library (e.g., gdstk) inside shipped code; GPL tools (e.g., KLayout) only as external dev tools, not linked into the submission.
  - Avoid non-commercial datasets (e.g., MVTec AD CC BY-NC-SA) in the core product; if used, only for benchmarking and labelled as such.
- **Originality:** All glue code, synthetic-data engine, defect injector, mapping logic, knowledge base, prompts and UI written by the team; no forked end-to-end repos.
- **Submission requirements:** Source code repo; runnable application (Pi image or install script); demonstration video (live camera, defect → bitcell highlight → Llama advice → log, with network unplugged); technical report (architecture, dual-pipeline diagram, dataset generation, training, quantisation, benchmarks, ablations, limitations incl. sim-to-real gap).
