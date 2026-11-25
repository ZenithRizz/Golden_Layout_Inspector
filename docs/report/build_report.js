// Builds the final project report (.docx) from report_data.json + figures.
//   node docs/report/build_report.js  ->  docs/report/GLI_Final_Report.docx
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, HeadingLevel, LevelFormat, BorderStyle, PageBreak, Footer, PageNumber, TableOfContents, ExternalHyperlink,
} = require("docx");

const HERE = __dirname;
const D = JSON.parse(fs.readFileSync(path.join(HERE, "report_data.json"), "utf8"));
const FIG = (f) => path.join(HERE, "fig", f);
const TEXT_W = 9026; // A4 width 11906 - 2 x 1440 margins (DXA)
const INK = "1D2330", MUTED = "5B6475", ACCENT = "1B64BE", HEAD_FILL = "E8EEF7";

// ---------- helpers ----------
function runs(text, base = {}) {
  // **bold** and `code` inline markup
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), font: "Consolas", size: 19, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text), spacing: { after: 120, line: 276 }, ...opts });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const B = (text, level = 0) => new Paragraph({ numbering: { reference: "bul", level }, children: runs(text), spacing: { after: 60, line: 264 } });
const N = (text) => new Paragraph({ numbering: { reference: "num", level: 0 }, children: runs(text), spacing: { after: 60, line: 264 } });
const CODE = (text) => new Paragraph({
  children: [new TextRun({ text, font: "Consolas", size: 17, color: INK })],
  shading: { type: ShadingType.CLEAR, fill: "F3F5F8", color: "auto" }, spacing: { after: 0, line: 240 },
  indent: { left: 120 },
});
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
let figNo = 0, tabNo = 0;
function FIGURE(file, widthPx, caption) {
  const { w, h } = pngSize(FIG(file));
  const width = Math.min(widthPx, 600), height = Math.round(width * h / w);
  figNo += 1;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(FIG(file)), transformation: { width, height },
        altText: { title: `Figure ${figNo}`, description: caption, name: file } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: `Figure ${figNo}. `, bold: true, size: 18, color: MUTED }), new TextRun({ text: caption, italics: true, size: 18, color: MUTED })] }),
  ];
}
const cellBorder = { style: BorderStyle.SINGLE, size: 4, color: "C9D1DD" };
function TABLE(header, rows, widths, caption) {
  const total = widths.reduce((a, b) => a + b, 0);
  const scaled = widths.map((w) => Math.round(w * TEXT_W / total));
  scaled[scaled.length - 1] += TEXT_W - scaled.reduce((a, b) => a + b, 0);
  const mk = (cells, head) => new TableRow({
    tableHeader: head, cantSplit: true,
    children: cells.map((c, i) => new TableCell({
      width: { size: scaled[i], type: WidthType.DXA },
      borders: { top: cellBorder, bottom: cellBorder, left: cellBorder, right: cellBorder },
      shading: head ? { type: ShadingType.CLEAR, fill: HEAD_FILL, color: "auto" } : undefined,
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      children: String(c).split("\n").map((line) => new Paragraph({ spacing: { after: 0, line: 252 },
        children: runs(line, { size: 18, bold: head ? true : undefined }) })),
    })),
  });
  tabNo += 1;
  const out = [];
  if (caption) out.push(new Paragraph({ spacing: { before: 120, after: 60 }, keepNext: true,
    children: [new TextRun({ text: `Table ${tabNo}. `, bold: true, size: 18, color: MUTED }), new TextRun({ text: caption, italics: true, size: 18, color: MUTED })] }));
  out.push(new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: scaled, rows: [mk(header, true), ...rows.map((r) => mk(r, false))] }));
  out.push(new Paragraph({ spacing: { after: 160 }, children: [] }));
  return out;
}
const v = (x, unit = "", digits = null) => {
  if (x === null || x === undefined) return "pending";
  if (typeof x === "number" && digits !== null) return x.toFixed(digits) + unit;
  return String(x) + unit;
};
const pb = D.phone_bench, team = D.team, dev = D.device;

// ---------- content ----------
const body = [];

// Cover
body.push(
  new Paragraph({ spacing: { before: 1800, after: 120 }, children: [new TextRun({ text: "Bharat AI-SoC Student Challenge 2026", size: 24, color: MUTED })] }),
  new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: "PS5: Real-time industrial defect detection & quality-control assistant", size: 24, color: MUTED })] }),
  new Paragraph({ spacing: { before: 360, after: 120 }, children: [new TextRun({ text: "Golden-Layout Inspector", size: 56, bold: true, color: INK })] }),
  new Paragraph({ spacing: { after: 480 }, children: [new TextRun({ text: "Design-aware, offline SRAM die inspection with on-device Llama 3.2 (ExecuTorch) on Arm devices: Android smartphones and Arm Linux boards such as the Raspberry Pi 5", size: 28, color: ACCENT })] }),
  new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: ACCENT, space: 4 } }, spacing: { after: 240 }, children: [] }),
  P(`**Final Project Report**`),
  ...(team.name ? [P(`**Team:** ${team.name}`)] : []),
  P(`**Team leader:** ${team.leader || "[team leader]"}`),
  P(`**Team members:** ${team.members || "[member 1], [member 2]"}`),
  P(`**College:** ${team.college || "[college]"}      **Faculty mentor:** ${team.mentor || "[mentor]"}`),
  P(`**Repository:** ${team.repo || "[GitHub link]"}`),
  P(`**Demo video:** ${team.video || "[video link]"}`),
  P(`**Hardware:** ${dev.name}, ${dev.soc}, Arm CPU only (no cloud, no network). The same code runs on Arm Linux boards (Raspberry Pi 5).`),
  P(`**Date:** 25 November 2026`),
  new Paragraph({ children: [new PageBreak()] }),
);


// 1 Summary
body.push(H1("1. Executive summary"),
  P("Existing inspection tools report **where** a defect is. Golden-Layout Inspector also reports **what it breaks and how to fix it**. It compares each camera image of an SRAM die against the die's own design layout (the golden reference), generated with the open-source OpenRAM memory compiler. Every defect is traced to the exact bitcell rows, columns and nets it touches. Deterministic rules then decide repair (with spare rows/columns) or scrap, and **Llama 3.2 1B Instruct, running on-device through ExecuTorch**, explains the likely electrical fault and the test that confirms it."),
  P("Because the reference is the design itself, the training data is generated from the layout too: **no real defect images are needed**. Everything runs offline on the Arm CPU of a standard Android smartphone, and the same code base runs on Arm Linux single-board computers such as the Raspberry Pi 5 (Section 3.1)."),
  ...TABLE(["Result", "Value"], [
    ["Golden design", "OpenRAM 16 x 16-bit SRAM (SCMOS scn4m_subm): 256 bitcells, 3,584 pin shapes, 7 region types"],
    ["Training data", "8,000 synthetic layout tiles, 23,288 labelled crops, 0 real defect images"],
    ["Defect classifier", "147k-parameter CNN, 97.3% accuracy on held-out synthetic crops (macro recall 97.1%)"],
    ["End-to-end detection (30 simulated dies, 90 defects)", "precision 1.00, recall 0.63, correct class on 98.2% of found defects"],
    ["Inspect to decision", `${v(pb.inspect_to_decision_ms_p50, " ms (phone, p50)")}; 194 ms p50 on a 2-vCPU x86 dev machine`],
    ["On-device LLM", `Llama 3.2 1B Instruct, 8da4w (4-bit weights), XNNPACK; TTFT ${v(pb.llama_ttft_ms_p50, " ms")}, decode ${v(pb.llama_decode_tok_s_p50, " tok/s")} on the phone`],
    ["Advice validity", `${D.phone_log_sample.advice_from_llama}/${D.phone_log_sample.advice_total} answers produced by Llama and accepted by the validator in the phone runs of ${D.phone_log_sample.date}`],
    ["Reference-free mode (any memory die, no layout)", "95% of defects found on our array without markers; 14/15 on a different memory design; non-array images rejected"],
    ["Tests", "88 unit tests (decision rules, spare allocation, KB coverage, advisor validation, layout mapping)"],
  ], [3, 7], "Key results."),
);

// 2 Problem
body.push(H1("2. Problem, need and objectives"));
body.push(H2("2.1 Why such a device is needed"),
  P("**India is building semiconductor manufacturing capacity quickly.** The Semicon India programme was approved on 15 December 2021 with an outlay of Rs 76,000 crore (over USD 10 billion), offering 30% capital support for assembly, test and packaging (ATMP/OSAT) units and targeting at least 15 of them [1]. By February 2026, 10 projects had been approved (2 fabs and 8 packaging and assembly units, about Rs 1.6 lakh crore of investment in six states) [2]. Micron's USD 2.75 billion assembly-and-test plant at Sanand, Gujarat expects to test and assemble tens of millions of memory chips in 2026, rising to hundreds of millions in 2027 [3]. IESA and Counterpoint estimate India's semiconductor market at USD 45 billion in 2023, growing about 13% a year to over USD 100 billion by 2030 [4]."),
  P("**Inspection is expensive and concentrated.** Metrology and inspection is a market of over USD 18 billion in 2025, about 14% of all wafer-fab equipment spending, led by a few vendors (KLA, Applied Materials, Lasertec) [5]. These systems report a defect's class and location; they don't say what it breaks electrically."),
  P("**Memories dominate chips and depend on repair.** Industry roadmaps (SIA/ITRS) have long put embedded memory at half or more of SoC die area [6], and large embedded SRAMs need redundant rows/columns: without repair, yield is poor [7]. Whether a defect is repairable depends on exactly which cells and nets it hits, which requires the design layout."),
  P("**Manual inspection is unreliable.** Sandia National Laboratories' review of visual inspection found that error rates of 20% to 30% are common, and accuracy in its own study ranged from about 10% to 90% depending on the defect type [8][9]. New plants and university fab labs also lack the years of labelled defect images that conventional machine-learning inspection needs."),
  P("**The gap.** An affordable, offline station that (a) needs no real defect dataset, (b) knows the design, so it can name the failing bitcells, rows, columns and nets, (c) decides repair or scrap with auditable rules, and (d) explains the fault and the confirming test to the operator in plain language, on commodity Arm hardware."),
);
body.push(H2("2.2 Objectives"),
  P("PS5 asks for an on-device system that detects defects, classifies type and severity, alerts operators with actionable advice, logs every inspection and runs in real time at the edge. Table 2 maps each requirement to our implementation."),
  ...TABLE(["PS5 requirement", "Implementation"], [
    ["Detect surface defects, cracks and anomalies", "Golden-layout difference + CNN: bridge, open (crack/break), particle, missing via; strong unexplained differences become UNKNOWN anomalies"],
    ["Classify defect type and severity", "CNN class + rules: COSMETIC / REPAIRABLE / KILLER / REVIEW per defect; PASS / REPAIR / SCRAP / RE-INSPECT per die"],
    ["Alert operators with actionable recommendations", "Decision banner + audio alert; Llama advice: fault model, confirming March test, action, plain-language explanation"],
    ["Log inspection results", "SQLite (WAL) with images, mapping, advice, operator accept/override; CSV/JSON export; repair map"],
    ["Real time at the edge, no cloud", "Arm CPU of an Android phone (also Raspberry Pi 5 / Arm Linux); no network use; live overlay plus sub-second decision"],
    ["Llama + ExecuTorch", "Llama 3.2 1B Instruct exported to .pte (XNNPACK, 8da4w) and run by ExecuTorch's llama_main on the phone"],
  ], [3, 7], "PS5 requirements and how they are met."),
);

// 3 Architecture
body.push(H1("3. System architecture"),
  P("The system has an offline preparation stage (run once per design) and an on-device station with two separate pipelines (Figure 1). Both pipelines run on the phone's Arm CPU, and they're timed separately."),
  B("**Vision pipeline (real time):** frame, then registration (ArUco markers + ECC sub-pixel alignment), golden-layout difference, candidate crops, CNN classifier (OpenCV DNN)."),
  B("**Advisor pipeline (background worker):** knowledge-base retrieval, Llama 3.2 1B via ExecuTorch, validation and composition of the advice. It never blocks the live view."),
  ...FIGURE("fig_architecture.png", 560, "System architecture. Dashed arrows are artefacts produced offline and loaded by the station."),
  ...TABLE(["Component", "Hardware / runtime"], [
    ["Target device", `${dev.name} [17]: ${dev.soc}; ${dev.cpu}${dev.ram ? "; " + dev.ram + " RAM" : ""}`],
    ["Software on device", dev.os],
    ["Vision inference", "OpenCV DNN on the Arm CPU (ONNX Runtime on Linux/Pi; HailoRT backend implemented for an optional AI HAT+)"],
    ["LLM inference", "ExecuTorch llama_main, cross-compiled with the Android NDK (arm64-v8a, android-23), XNNPACK delegate"],
    ["Camera", "Phone rear camera streamed to the station on localhost by the IP Webcam app (password protected), or image upload"],
    ["Operator UI", "Local web UI (dependency-free Python server) opened in the phone's browser, or on a laptop over the phone's hotspot"],
  ], [2.4, 7.6], "Hardware and runtimes."),
);
body.push(H2("3.1 Deployment targets: Android and Arm Linux"),
  P("The station is plain Python with NumPy and OpenCV, and the Llama runner is ExecuTorch's C++ `llama_main`, so the same repository runs on any 64-bit Arm device with a camera. The phone is our demonstrated platform. The Raspberry Pi 5 path is scripted and uses the same code; we had no board to benchmark it on, so no Pi numbers are reported."),
  ...TABLE(["", "Android smartphone (demonstrated)", "Raspberry Pi 5 / Arm Linux (supported)"], [
    ["Hardware", `${dev.name}, ${dev.soc}`, "Raspberry Pi 5 (Broadcom BCM2712, 4x Cortex-A76 @ 2.4 GHz) [16], or any arm64 Linux board"],
    ["OS", "Android + Termux", "Raspberry Pi OS 64-bit (or another arm64 Linux)"],
    ["Setup", "`scripts/android/setup_termux.sh`", "`scripts/setup_pi.sh`"],
    ["Camera", "Phone camera via IP Webcam (localhost)", "USB webcam (`usb:0`) or Pi camera (`picam`, Picamera2)"],
    ["Vision runtime", "OpenCV DNN", "ONNX Runtime (optional Hailo AI HAT+ NPU via HailoRT)"],
    ["Llama runner", "`llama_main` cross-compiled with the Android NDK", "`llama_main` built natively (`scripts/llama/03_build_runner.sh`)"],
    ["Web server", "Dependency-free `server_lite.py`", "FastAPI (`server.py`)"],
    ["Start", "`run_station.sh http://USER:PASS@127.0.0.1:8080`", "`run_station.sh usb:0` or `run_station.sh picam`"],
  ], [1.6, 4.2, 4.2], "Supported Arm deployment targets. The .pte model, classifier, design package and UI are identical."),
);

// 4 Methodology
body.push(H1("4. Methodology"));
body.push(H2("4.1 Golden design from OpenRAM"),
  P("OpenRAM [10] (BSD-3, used as a tool) generated a 16-word x 16-bit single-port SRAM in the SCMOS scn4m_subm technology. Our extractor (`gli/layout.py`) walks the GDS instance hierarchy and builds a design package:"),
  B("**Regions** from OpenRAM instance names: bitcell array, dummy/replica cells, row decoder, precharge, sense amplifiers, write drivers, control logic."),
  B("**Bitcells:** all 256 `cell_1rw` instances, each with its row and column. Cell rectangles come from the cell's boundary layer (6.8 x 10.4 um), because the bounding boxes of mirrored rows alternate."),
  B("**Nets:** each bitcell's pin shapes (bl, br, wl, vdd, gnd, Q, Q_bar), transformed to absolute coordinates and named per instance (`bl_6`, `wl_7`, `q_7_6`), giving 3,584 pin shapes."),
  B("**Golden image:** every layer rasterised at 4 px/um (work scale), plus a 9 px/um printable target with four ArUco markers. Figure 2 shows the inspected window."),
  ...FIGURE("fig_golden.png", 430, "Golden render of the OpenRAM SRAM with extracted regions and the 16 x 16 bitcell grid (red)."),
);
body.push(H2("4.2 Synthetic defect data (no real defect images)"),
  P("Defects are injected into the **layer masks** of random 256 x 256 tiles, so they're physically consistent: a bridge adds metal between two lines of the same layer, an open cuts a line, and a missing via deletes a cut. They can also be painted like the physical demo marks (marker ink, correction fluid, dust). Defects hidden under higher layers have no optical signature, so they're rejected (contrast filter) to keep labels honest. A camera model adds residual misalignment, blur, colour cast, illumination gradient, noise and JPEG compression."),
  P("Each tile is then processed by the **station's own candidate detector**, and every candidate crop is labelled by overlap with the ground truth (`NONE` for noise). Training and runtime therefore see the same distribution."),
  ...FIGURE("fig_synth.png", 560, "Golden tiles (top) and the same tiles with an injected defect after camera simulation (bottom)."),
  ...TABLE(["Class", "Crops", "Candidate recall of the detector"], [
    ["NONE (noise / misregistration)", "12,587", "n/a"], ["BRIDGE", "3,151", "81.7%"], ["OPEN", "2,452", "68.6%"],
    ["PARTICLE", "3,979", "98.4%"], ["MISSING_VIA", "1,119", "43.2%"], ["Total (8,000 tiles)", "23,288", ""],
  ], [4, 2, 4], "Synthetic dataset. Candidate recall = share of injected defects that produce a candidate blob."),
);
body.push(H2("4.3 Vision pipeline"),
  B("**Registration:** ArUco [14] corners give a homography to layout pixels. ECC [15] (`cv2.findTransformECC`, affine) then aligns the rectified frame to the golden image to sub-pixel accuracy. The refinement is cached while the camera is still, and registration quality becomes the mapping confidence."),
  B("**Golden difference:** per colour channel, the distance of the capture outside the golden image's local [min, max] in a 3 x 3 window. This equals the minimum absolute difference over +-1 px shifts, but costs one dilate and one erode, so it tolerates residual misalignment."),
  B("**Candidates:** blobs above threshold, with a mean-contrast floor (26) that removed every resampling artefact in our tests."),
  B("**Classifier:** 8 convolution layers + global pooling, 146,613 parameters. Its 64 x 64 input has 3 channels: captured gray, golden gray and |difference|, so the network sees what should be there next to what is there. Trained 8 epochs (AdamW, one-cycle LR, label smoothing, class weights). Normalisation is inside the graph, so it accepts raw 0-255 crops."),
  B("**Reference-free mode** (uploads without markers, any memory die): the array's repeat period comes from the image's autocorrelation. Each cell is compared with its neighbours one period away (min over neighbours keeps the defect and discards its shifted copies). Only truly repeating areas are kept (local normalised cross-correlation + texture). The same CNN then runs, with the neighbouring cell as the reference."),
);
body.push(H2("4.4 Mapping and deterministic decision rules"),
  P("Each detection box is converted to layout micrometres and mapped to the most consequential overlapping region, the layers present, the nets whose pin shapes it touches, and the bitcell rows/columns. A **fault footprint** turns the physical defect into failing cells: an open on a bitline fails the whole column, a bridge touching a wordline fails the whole row, and otherwise only the covered cells fail."),
  P("Severity rules (first match wins):"),
  N("unknown anomaly or low confidence: **REVIEW**;"),
  N("periphery (decoder, sense amp, write driver, precharge, control) or a vdd-gnd short: **KILLER**;"),
  N("dummy cells / outside: **COSMETIC**;"),
  N("bitcell array: **REPAIRABLE**."),
  P("Spares (2 rows + 2 columns, a configured budget) are allocated by must-repair analysis followed by a greedy cover. The die decision is RE-INSPECT if any defect needs review, SCRAP if any is a killer or the budget is exceeded, REPAIR if anything is repairable, and PASS otherwise. These rules always decide the action. Llama explains them but can't change them, so an LLM error can never pass a bad die."),
);
body.push(H2("4.5 Llama advisor on ExecuTorch"),
  P("Llama 3.2 1B Instruct [12] was exported with ExecuTorch's [11] `export_llm`:"),
  B("8-bit dynamic activations with 4-bit grouped weights (8da4w, group 32);"),
  B("4-bit embeddings;"),
  B("KV cache with the custom SDPA operator;"),
  B("XNNPACK delegate, maximum sequence length 1024."),
  P("The resulting 0.86 GB `.pte` is run by ExecuTorch's C++ `llama_main`, cross-compiled for Android with the NDK."),
  P("For each defect, the advisor sends a compact JSON record: class, region, nets, exact affected location, the rule's action, and the candidate fault models and tests from the top KB entries. Llama picks the fault model and confirming test (March tests [13]) and writes one sentence on what fails and why. The station validates the answer and composes the final advice. Table 5 shows how the design converged; small models need narrow, checkable tasks."),
  ...TABLE(["Version", "Change", "Outcome (laptop test, 3 cases)"], [
    ["v1", "Free JSON incl. action; 518-token prompt with a worked example", "2/3 valid, but one told the operator to scrap a repairable die (copied 'row 7' from the example); 13-15 s"],
    ["v2", "Action and location given as facts; 332 tokens", "0/3: echoed the input, abbreviated names ('SA'), merged test names"],
    ["v3", "JSON prefilled ('{\"fault_model\": \"'), one-sentence cause only, test + action appended by the station, forgiving option matching", "2/3 Llama; 1 runaway generation hit the timeout"],
    ["v4 (final)", "Decode cap (seq_len = prompt + 90 tokens), salvage output on timeout, example moved to a block absent from the design", `${D.phone_log_sample.advice_from_llama}/${D.phone_log_sample.advice_total} accepted from Llama in the phone runs`],
  ], [1.3, 4.5, 4.2], "Advisor design iterations."),
  P("Validation rejects answers whose fault model or test isn't in the KB options, whose explanation mentions row/column numbers not in the record, or which say there is no defect. Rejected answers are retried once if time allows; otherwise a deterministic KB template is used and labelled as such in the UI."),
);
body.push(H2("4.6 Operator experience and logging"),
  P("The local web UI has four screens:"),
  B("**Home:** pipeline health, showing which backend each pipeline runs on and that the station is offline."),
  B("**Live:** camera overlay with registration status and live detections; the Inspect button or Space key."),
  B("**Inspection:**"),
  B("frozen frame with numbered defects;", 1),
  B("layout zoom with the affected bitcells highlighted and labelled (e.g. r7 c6);", 1),
  B("the advisor card;", 1),
  B("accept, or override with a mandatory reason;", 1),
  B("the repair map.", 1),
  B("**Log:** export as CSV/JSON."),
  P("Severity is always shown as text + icon + colour. SCRAP and RE-INSPECT trigger an audio alert. Every inspection is written atomically to SQLite (WAL) before the UI confirms it; overrides are appended, never edited."),
  ...FIGURE("ui_inspection_repair.png", 600, "Inspection of a printed-target photo: two array defects mapped to bitcells r7 c6-c7, whole-column impact, REPAIR with spare columns 6 and 7. (Captured on the development laptop, where the advisor shows template advice; on the phone the same card shows Llama's answer.)"),
);

// 5 Results
body.push(H1("5. Results"));
body.push(H2("5.1 Classifier"),
  ...FIGURE("fig_confusion.png", 400, "Confusion matrix on held-out synthetic crops. The main confusion is bridge vs particle (dark ink bridges resemble dust)."),
);
body.push(H2("5.2 End-to-end detection"),
  P("The benchmark harness (`scripts/benchmark.py`) inspects simulated dies: printed-target renders with random defects, viewed with random perspective and camera degradation. It scores detections against the ground truth."),
  ...TABLE(["Defect class", "Found / planted", "Correct class (of found)"], [
    ["PARTICLE", "18 / 18", "18 / 18"], ["BRIDGE", "22 / 30", "21 / 22"], ["OPEN", "11 / 27", "11 / 11"], ["MISSING_VIA", "6 / 15", "6 / 6"],
    ["All (30 dies)", "57 / 90 (recall 0.63)", "56 / 57 (98.2%); precision 1.00, 0 false alarms"],
  ], [3, 3.5, 3.5], "End-to-end detection on 30 simulated dies (90 planted defects)."),
  P("The misses are the smallest synthetic opens and missing vias (3-4 px features at 4 px/um), which the contrast floor deliberately trades away for zero false alarms. Hand-made defects on the printed target are larger and were detected in our physical tests."),
  P("**Reference-free mode** (no markers, no layout):"),
  B("our SRAM array photographed at random scales: 38/40 defects found, 33 correctly classified, 8 false alarms in 20 images;"),
  B("a different memory design next to non-repeating periphery: 14/15 found, all correctly classified, 0-2 false alarms per image (at the array edge);"),
  B("images with no repeating array: rejected with an explanation."),
);
body.push(H2("5.3 On-device performance"),
  ...TABLE(["Metric", `Phone (${dev.name})`, "Laptop x86 (Ryzen 7, WSL2)", "Dev container (2 vCPU x86)"], [
    ["Live vision loop", v(pb.live_fps, " FPS"), "n/a", "13.8 FPS (72.6 ms/frame)"],
    ["Registration (p50)", v(pb.register_ms_p50, " ms"), "n/a", "140 ms (with ECC refresh)"],
    ["Golden difference (p50)", v(pb.diff_ms_p50, " ms"), "n/a", "36 ms"],
    ["CNN on candidates (p50)", v(pb.classify_ms_p50, " ms"), "n/a", "2.8 ms"],
    ["Inspect to decision (p50)", v(pb.inspect_to_decision_ms_p50, " ms"), "n/a", "194 ms"],
    ["Llama model load", v(pb.llama_model_load_ms_p50, " ms"), "2.1-5.4 s", "n/a"],
    ["Llama prefill", v(pb.llama_prefill_tok_s_p50, " tok/s"), "51-80 tok/s", "n/a"],
    ["Llama decode", v(pb.llama_decode_tok_s_p50, " tok/s"), "14-25 tok/s", "n/a"],
    ["Time to first token (~300-token prompt)", v(pb.llama_ttft_ms_p50, " ms"), "3.8-6.0 s", "n/a"],
    ["Full advice per defect (p50)", v(pb.advice_latency_ms_p50, " ms"), "10-14 s", "n/a"],
  ], [3.2, 2.4, 2.2, 2.2], "Performance by platform. Laptop LLM figures are ranges from the advisor test runs."),
  P("The decision (detection, mapping, severity, repair allocation) is shown immediately. Llama's explanation arrives a few seconds later from the background worker, so the live view and the operator's decision are never blocked by the LLM."),
);

// 6 Hardware utilisation and optimisation
body.push(H1("6. Hardware utilisation and optimisation techniques"));
body.push(H2("6.1 Hardware utilisation"),
  B("**Arm CPU only.** Both pipelines run on the Snapdragon's Arm cores. Llama uses 4 threads (`--cpu_threads=4`) via XNNPACK, whose Arm micro-kernels (KleidiAI) use the cores' dot-product instructions for int4/int8 maths. OpenCV handles the vision pipeline with its own thread pool."),
  B("**Memory.** The 0.86 GB `.pte` is memory-mapped by ExecuTorch. The station keeps one golden image (about 1 MP) and per-layer masks in RAM; the CNN is 0.6 MB."),
  B("**Portability.** Nothing in either pipeline is phone-specific: on a Raspberry Pi 5 the same code runs on the Cortex-A76 cores, with XNNPACK using the same Arm int8 dot-product kernels (Section 3.1)."),
  B("**No NPU/GPU.** We had no Raspberry Pi AI HAT. The HailoRT classifier backend and the Hailo compile script are implemented so the vision model can move to an NPU without code changes, while Llama stays on the CPU through ExecuTorch."),
);
body.push(H2("6.2 Optimisation techniques"),
  ...TABLE(["Technique", "Where", "Effect"], [
    ["8da4w quantisation (4-bit grouped weights, group 32; 8-bit dynamic activations)", "Llama export", "1B model in 0.86 GB; int8/int4 kernels on Arm"],
    ["4-bit embedding quantisation", "Llama export", "Shrinks the large 128k-token embedding table"],
    ["KV cache + custom SDPA operator", "Llama export", "Decoding reuses past keys/values instead of recomputing"],
    ["XNNPACK delegate (KleidiAI micro-kernels on Arm)", "ExecuTorch runtime", "Optimised int4/int8 GEMM on Cortex-A720/A520"],
    ["Prompt compaction (518 to ~300 tokens)", "Advisor", "About 40% less prefill; TTFT about 6.6 s to about 4 s on the laptop"],
    ["JSON prefill + one-sentence answer", "Advisor", "Output 80-98 to 27-29 tokens (about 70% less decode time)"],
    ["Decode cap: seq_len = prompt + 90 tokens", "Advisor", "Bounds runaway generation; worst case drops from more than 25 s to a few seconds"],
    ["Rules decide, LLM explains; background worker", "Station", "Decision in well under a second; LLM latency hidden from the operator"],
    ["Analytic shift-tolerant difference (dilate/erode) instead of a heavy detector", "Vision", "36 ms per frame; no large detection network needed"],
    ["Tiny CNN only on candidate crops", "Vision", "2.8 ms for all candidates in a frame"],
    ["ECC refinement cached while the pose is static", "Vision", "Sub-pixel alignment without paying for ECC every frame"],
    ["Normalisation inside the CNN graph; OpenCV DNN runtime", "Vision", "Raw uint8 input; no ONNX Runtime dependency on Android"],
    ["Dependency-free web server, SQLite WAL", "Station", "Runs in Termux with only Python, NumPy and OpenCV"],
  ], [4.2, 1.6, 4.2], "Optimisation techniques."),
  P("**Engineering constraints met along the way:**"),
  B("The Llama export peaks above the 11 GB available to WSL on our laptop, so it ran on Google Colab. The `.pte` is portable, and only inference must be on-device."),
  B("The Play Store build of Termux couldn't install OpenCV or run native binaries, so the standard Termux build is used."),
  B("ExecuTorch 1.0's CMake option checks required the official `llm` preset."),
);

// 7 Limitations, 8 Future
body.push(H1("7. Limitations"),
  B("**Optical demo.** Real sub-micron features can't be resolved by a phone camera, so the demo inspects a magnified print of the layout. The method (die-to-database comparison, mapping, rules, LLM reasoning) is imager-independent, but a fabricated die needs a microscope or SEM and a renderer tuned to that imager."),
  B("**Synthetic training data.** Real-photo accuracy must be reported separately; small synthetic opens and missing vias have lower recall."),
  B("**Spare budget.** The 2 + 2 spares are a configured budget; this OpenRAM configuration doesn't generate redundancy."),
  B("**Llama load time.** `llama_main` reloads the model for every call (a persistent runner is the next step); a 1B model needs the narrow, validated task described above."),
  B("**Unextracted nets.** Routing between blocks and nets in the periphery aren't extracted, so those defects go to engineer review."),
);
body.push(H1("8. Future work: from proof of concept to product"),
  P("In this proof of concept the golden reference is an OpenRAM SRAM. In production, a manufacturer uploads their own GDS/OASIS layout, and the station builds the golden image, the bitcell and net map, and the training data from it automatically, again with no defect images."),
  B("Per-process layer-map files and region/net extraction from layout labels (beyond OpenRAM naming)."),
  B("Imager-specific rendering for microscopes and SEMs; alignment on die features instead of printed markers."),
  B("A persistent ExecuTorch runner (or the Python runner bindings once stable) and an NPU path for vision."),
  B("Reconciliation with electrical test results (March test fail maps) and redundancy settings read from the memory compiler."),
);

// 9 Reproducibility
body.push(H1("9. Reproducibility"),
  P("All code, the generated design package, the trained model, test images with an answer key, and step-by-step guides (`README.md`, `docs/ARCHITECTURE.md`, `docs/ANDROID.md`) are in the repository. The main commands are:"),
  CODE("python scripts/build_design.py --gds openram_out/sram_16x16/sram_16x16_scn4m.gds --name sram16x16_scn4m"),
  CODE("python scripts/make_dataset.py --tiles 8000 && python scripts/train_classifier.py --epochs 8"),
  CODE("notebooks/export_llama_colab.ipynb            # Llama 3.2 1B -> .pte (8da4w, XNNPACK)"),
  CODE("bash scripts/android/build_llama_android.sh  # llama_main for arm64 Android"),
  CODE("bash scripts/run_station.sh http://USER:PASS@127.0.0.1:8080   # on the phone (Termux)"),
  CODE("./scripts/setup_pi.sh && ./scripts/run_station.sh usb:0         # on a Raspberry Pi 5"),
  CODE("python scripts/benchmark.py --dies 10 --llama-runner ... --llama-pte ... --llama-tokenizer ..."),
  new Paragraph({ spacing: { after: 160 }, children: [] }),
  P("**Licences and attribution:**"),
  B("OpenRAM: BSD-3-Clause, used as a tool."),
  B("Llama 3.2: Llama 3.2 Community License. Built with Llama."),
  B("ExecuTorch: BSD."),
  B("OpenCV, ONNX Runtime, FastAPI: permissive licences."),
  B("No AGPL code and no non-commercial datasets are used. All project code was written by the team, with AI assistance, and reviewed by the team."),
);

const REFS = [
  ["Prime Minister's Office, \"Cabinet approves programme for development of semiconductors and display manufacturing ecosystem in India\", 15 Dec 2021.", "https://www.pmindia.gov.in/en/news_updates/cabinet-approves-programme-for-development-of-semiconductors-and-display-manufacturing-e"],
  ["Devdiscourse, \"India approves Rs 1.6 lakh crore semiconductor push under Semicon India\" (Lok Sabha written reply), 4 Feb 2026.", "https://www.devdiscourse.com/article/law-order/3793443-india-approves-rs-16-lakh-crore-semiconductor-push-under-semicon-india"],
  ["Computer Weekly, \"Micron opens $2.75bn chip assembly plant in India\", 2026.", "https://www.computerweekly.com/news/366639653/Micron-opens-275bn-chip-assembly-plant-in-India"],
  ["News On AIR, \"India's semiconductor market projected to surpass $100 billion by 2030\" (IESA and Counterpoint Research), 16 Oct 2024.", "https://newsonair.gov.in/indias-semiconductor-market-projected-to-surpass-100-billion-by-2030"],
  ["Yole Group, \"+15% in 2025: metrology & inspection accelerate semiconductor innovation\", press release, 2025.", "https://www.yolegroup.com/press-release/15-in-2025-metrology-inspection-accelerate-semiconductor-innovation/"],
  ["C. Brown, \"SoC embedded memories: overview\", EE Times, 2003 (republished by Design & Reuse).", "https://www.design-reuse.com/articles/5417/soc-embedded-memories-overview.html"],
  ["S. Favre, \"Are you wasting your SRAM memory redundancy?\", Siemens EDA (Mentor) Calibre blog, 20 Mar 2017.", "https://blogs.sw.siemens.com/calibre/2017/03/20/are-you-wasting-your-sram-memory-redundancy/"],
  ["J. E. See, \"Visual Inspection: A Review of the Literature\", Sandia National Laboratories, SAND2012-8590, Oct 2012.", "https://www.sandia.gov/research/publications/details/visual-inspection-a-review-of-the-literature-2012-10-01/"],
  ["J. E. See, \"Visual Inspection Reliability Study\", Sandia National Laboratories, SAND2014-2785C, Apr 2014.", "https://www.osti.gov/servlets/purl/1142011"],
  ["M. R. Guthaus et al., \"OpenRAM: An Open-Source Memory Compiler\", Proc. IEEE/ACM ICCAD, 2016.", "https://openram.org"],
  ["PyTorch, ExecuTorch documentation (LLM export and runtime).", "https://pytorch.org/executorch"],
  ["Meta, Llama 3.2 1B Instruct model card.", "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct"],
  ["A. J. van de Goor, Testing Semiconductor Memories: Theory and Practice, Wiley, 1991.", null],
  ["S. Garrido-Jurado et al., \"Automatic generation and detection of highly reliable fiducial markers under occlusion\", Pattern Recognition 47(6), 2014.", null],
  ["G. D. Evangelidis and E. Z. Psarakis, \"Parametric image alignment using enhanced correlation coefficient maximization\", IEEE TPAMI 30(10), 2008.", null],
  ["Raspberry Pi Ltd, Raspberry Pi 5 product page.", "https://www.raspberrypi.com/products/raspberry-pi-5/"],
  ["GSMArena, Nothing Phone (3a) Pro specifications.", "https://www.gsmarena.com/nothing_phone_(3a)_pro-13649.php"]
];
body.push(new Paragraph({ children: [new PageBreak()] }), H1("References"),
  ...REFS.map(([t, u], i) => new Paragraph({ spacing: { after: 90, line: 264 }, indent: { left: 540, hanging: 540 },
    children: [new TextRun({ text: `[${i + 1}]\t${t}` }), ...(u ? [new TextRun({ text: " " }), new ExternalHyperlink({ link: u, children: [new TextRun({ text: u, style: "Hyperlink", color: ACCENT, size: 18 })] })] : [])] })),
);

// ---------- document ----------
const doc = new Document({
  creator: team.leader || "Golden-Layout Inspector team",
  title: "Golden-Layout Inspector: Final Project Report",
  styles: {
    default: { document: { run: { font: "Arial", size: 21, color: INK } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: "Arial", size: 30, bold: true, color: INK }, paragraph: { spacing: { before: 320, after: 140 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: "Arial", size: 24, bold: true, color: ACCENT }, paragraph: { spacing: { before: 220, after: 100 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bul", levels: [
      { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
      { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1000, hanging: 270 } } } },
    ] },
    { reference: "num", levels: [
      { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } },
    ] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1300, bottom: 1300, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "Golden-Layout Inspector  |  page ", size: 16, color: MUTED }), new TextRun({ children: [PageNumber.CURRENT], size: 16, color: MUTED })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then((buf) => {
  const out = path.join(HERE, "GLI_Final_Report.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out);
});
