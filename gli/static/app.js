// Golden-Layout Inspector operator UI (no build step, no external requests).
const $ = (id) => document.getElementById(id);
const api = async (path, opts = {}) => {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw Object.assign(new Error(body.error || body.detail || r.statusText), { body });
  return body;
};
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function toast(msg, bad = false) {
  let t = document.getElementById("toast");
  if (!t) { t = document.createElement("div"); t.id = "toast"; document.body.appendChild(t); }
  t.textContent = msg; t.className = "toast show" + (bad ? " bad" : "");
  clearTimeout(t._h); t._h = setTimeout(() => (t.className = "toast"), 3500);
}
const regionName = (r) => (r === "UNMAPPED" ? "no layout" : r.replace("_", " ").toLowerCase());
const SEV_ICON = { KILLER: "✖", REVIEW: "?", REPAIRABLE: "↻", COSMETIC: "·" };
const DIE_ICON = { PASS: "✔", REPAIR: "↻", SCRAP: "✖", RE_INSPECT: "?" };

let view = "home", status = null, cur = null, sel = 0, pollTimer = null;

function show(v) {
  view = v;
  document.querySelectorAll(".view").forEach((e) => e.classList.toggle("on", e.id === "v-" + v));
  document.querySelectorAll("nav .nav").forEach((b) => b.classList.toggle("on", b.dataset.view === v));
  $("feed").src = v === "live" ? "/stream.mjpg?" + Date.now() : "";
  if (v === "log") loadLog();
}
document.querySelectorAll("nav .nav").forEach((b) => (b.onclick = () => show(b.dataset.view)));
$("startBtn").onclick = () => show("live");
$("homeLog").onclick = () => show("log");

// ---------------------------------------------------------------- status
function card(title, value, sub, state) {
  return `<div class="card ${state}"><h4>${esc(title)}</h4><div class="v">${esc(value)}</div><div class="s">${esc(sub)}</div></div>`;
}
async function refreshStatus() {
  try { status = await api("/api/status"); } catch (e) { return; }
  const s = status, L = s.live || {};
  const cam = s.camera.ok ? card("Camera", s.camera.source, L.fps ? `${L.fps} FPS` : "running", "ok")
                          : card("Camera", "Unavailable", s.camera.error, "bad");
  const VB = { "hailo-npu": "Hailo NPU (HailoRT)", "cpu-onnx": "CPU (ONNX Runtime)", "cpu-opencv-dnn": "CPU (OpenCV DNN)", "diff-only": "Diff only (no model)" };
  const vis = card("Vision pipeline", VB[s.vision.backend] || s.vision.backend,
                   "golden-layout diff + CNN classifier", s.vision.backend === "diff-only" ? "warn" : "ok");
  const adv = card("Advisor pipeline", s.advisor.backend === "template" ? "Template advice only" : "Llama 3.2 1B · ExecuTorch",
                   s.advisor.backend === "template" ? "Llama runner/model not found" : `Arm CPU · queue ${s.advisor.queue}`,
                   s.advisor.backend === "template" ? "warn" : "ok");
  const sto = card("Storage", `${(s.storage.free_mb / 1000).toFixed(1)} GB free`, s.storage.free_mb < 500 ? "Low: images not saved" : "inspection log OK",
                   s.storage.free_mb < 500 ? "bad" : "ok");
  const reg = card("Registration", L.locked ? "Locked" : "Not locked", L.locked ? `${L.markers}/4 markers · ECC ${L.ecc}` : (L.reason || ""), L.locked ? "ok" : "warn");
  $("cards").innerHTML = cam + reg + vis + adv + sto;
  const d = s.design;
  $("designInfo").innerHTML = `<b>${esc(d.name)}</b> · ${d.rows}×${d.cols} bitcells · ${esc(d.technology)} · spare rows ${d.spare_rows}, spare columns ${d.spare_cols}${d.spares_virtual ? " (configured budget)" : ""}`;
  $("notes").innerHTML = (s.vision.notes || []).map((n) => `<div class="note">${esc(n)}</div>`).join("");
  $("startBtn").disabled = !s.camera.ok;
  // live side panel
  $("virtualBox").classList.toggle("hidden", s.camera.source !== "virtual");
  $("srcInfo").textContent = `Source: ${s.camera.source}`;
  const t = L.timings || {};
  $("liveStats").innerHTML = [["FPS", L.fps ?? "–"], ["Focus", L.focus ?? "–"], ["Register", ms(t.register_ms)], ["Diff", ms(t.diff_ms)],
    ["Classify", ms(t.classify_ms)], ["Vision total", ms(t.vision_total_ms)]].map(([k, v]) => `<span>${k}</span><b>${v}</b>`).join("");
  const counts = {};
  (L.dets || []).forEach((d) => (counts[d.cls] = (counts[d.cls] || 0) + 1));
  $("liveCounts").innerHTML = Object.keys(counts).length ? Object.entries(counts).map(([k, v]) => `<span>${k} × ${v}</span>`).join("") : "No defects in view";
  const ban = $("liveBanner");
  const msg = !s.camera.ok ? s.camera.error : !L.locked ? L.reason : "";
  ban.textContent = msg || "";
  ban.classList.toggle("hidden", !msg);
  $("inspectBtn").disabled = !s.camera.ok || !L.locked;
}
const ms = (v) => (v == null ? "–" : `${v.toFixed(0)} ms`);
setInterval(refreshStatus, 1000);
refreshStatus();

// ---------------------------------------------------------------- inspect
async function doInspect() {
  $("inspectMsg").textContent = "Inspecting…";
  try {
    const rec = await api("/api/inspect", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ die_id: $("dieId").value.trim() || null }) });
    $("inspectMsg").textContent = "";
    openInspection(rec);
  } catch (e) { $("inspectMsg").textContent = e.message; }
}
$("inspectBtn").onclick = doInspect;
$("upload").onchange = async (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  const fd = new FormData(); fd.append("file", f);
  $("inspectMsg").textContent = "Inspecting uploaded image… (large photos take a few seconds)";
  try { openInspection(await api("/api/upload", { method: "POST", body: fd })); $("inspectMsg").textContent = ""; }
  catch (e) { $("inspectMsg").textContent = e.message; }
  ev.target.value = "";
};
$("newDie").onclick = async () => {
  await api("/api/virtual/new_die", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ defects: +$("nDef").value }) });
  $("inspectMsg").textContent = "";
};
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.code === "Space" && view === "live") { e.preventDefault(); if (!$("inspectBtn").disabled) doInspect(); }
  if (view === "detail" && cur) {
    if (e.key === "ArrowRight") selectDef(Math.min(sel + 1, cur.defects.length - 1));
    if (e.key === "ArrowLeft") selectDef(Math.max(sel - 1, 0));
  }
});

function beep(kind) {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const tones = kind === "SCRAP" ? [880, 660, 880] : [700, 700];
    tones.forEach((f, i) => {
      const o = ctx.createOscillator(), g = ctx.createGain();
      o.frequency.value = f; o.connect(g); g.connect(ctx.destination);
      g.gain.setValueAtTime(0.15, ctx.currentTime + i * 0.22);
      g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + i * 0.22 + 0.2);
      o.start(ctx.currentTime + i * 0.22); o.stop(ctx.currentTime + i * 0.22 + 0.21);
    });
  } catch (e) {}
}

function openInspection(rec) {
  cur = rec; cur.advice = cur.advice || {}; sel = 0;
  $("navDetail").disabled = false;
  show("detail");
  const dd = rec.decision.die_decision;
  const b = $("dieBanner");
  b.className = `die-banner d-${dd} flash`;
  const al = rec.decision.allocation;
  const spares = dd === "REPAIR" ? ` · spares used: rows ${al.spare_rows_used.length}/${al.spare_rows}, columns ${al.spare_cols_used.length}/${al.spare_cols}` : "";
  const rf = rec.mode === "reference_free" ? ` · no layout: cell-to-cell check (repeat ${rec.reference_free.px || "–"}×${rec.reference_free.py || "–"} px)` : "";
  b.innerHTML = `<span>${DIE_ICON[dd]} ${esc(rec.die_id)}: ${dd.replace("_", "-")}</span><span class="why">${esc(rec.decision.die_reason)}${spares}${rf}</span><span class="grow"></span><span id="decided" class="decided"></span>`;
  const last = (rec.actions || []).slice(-1)[0];
  setDecided(last ? (last.action.startsWith("override") ? `Overridden → ${last.final}` : "Accepted ✓") : "");
  if (dd === "SCRAP" || dd === "RE_INSPECT") beep(dd);
  $("frozen").src = `/api/inspection/${rec.id}/image/overlay?${Date.now()}`;
  renderDefList();
  renderSummary();
  if (rec.defects.length) selectDef(0); else {
    $("zoom").removeAttribute("src"); $("mapInfo").innerHTML = ""; $("zoomTitle").textContent = "Layout view";
    $("advCard").innerHTML = `<div class="muted">No defects found. Advisor not needed.</div>`; $("advSrc").textContent = "";
  }
  if (pollTimer) clearInterval(pollTimer);
  const t0 = Date.now();
  pollTimer = setInterval(async () => {
    const r = await api(`/api/inspection/${rec.id}`).catch(() => null);
    if (!r || !cur || cur.id !== rec.id) return;
    cur.advice = r.advice || {};
    renderDefList(); renderAdvice(); renderSummaryTable();
    const done = Object.keys(cur.advice).length >= (rec.advice_expected || 0);
    if (done || Date.now() - t0 > 120000) clearInterval(pollTimer);
  }, 1000);
}

function renderDefList() {
  const sv = cur.decision.defects;
  $("defList").innerHTML = cur.defects.map((d, i) => {
    const s = sv[i].severity, a = cur.advice[d.id];
    return `<div class="def ${i === sel ? "on" : ""}" data-i="${i}"><span class="num">${i + 1}</span>
      <b>${esc(d.cls)}</b><span class="muted small">${(d.conf * 100).toFixed(0)}%</span><span class="grow"></span>
      <span class="small muted">${esc(regionName(cur.mapping[i].region))}</span>
      <span class="sev s-${s}">${SEV_ICON[s]} ${s}</span>${a ? "" : '<span class="small muted">…</span>'}</div>`;
  }).join("");
  document.querySelectorAll(".def").forEach((e) => (e.onclick = () => selectDef(+e.dataset.i)));
}

function selectDef(i) {
  sel = i; renderDefList();
  const d = cur.defects[i], mp = cur.mapping[i], sv = cur.decision.defects[i];
  $("zoomTitle").textContent = `Defect ${i + 1}: ${d.cls} · ${regionName(mp.region)}`;
  $("zoom").src = `/api/inspection/${cur.id}/defect/${d.id}/zoom.png`;
  const rows = sv.full_rows.length ? `${sv.full_rows.join(", ")} (whole row)` : mp.affected_rows.join(", ") || "–";
  const cols = sv.full_cols.length ? `${sv.full_cols.join(", ")} (whole column)` : mp.affected_cols.join(", ") || "–";
  $("mapInfo").innerHTML = [["Severity", `<span class="sev s-${sv.severity}">${SEV_ICON[sv.severity]} ${sv.severity}</span> ${esc(sv.reason)}`],
    ["Layers", esc(mp.layers.join(", ") || "–")], ["Nets", esc(mp.nets.slice(0, 10).join(", ") || "–")],
    ["Rows", esc(rows)], ["Columns", esc(cols)], ["Layout (µm)", mp.layout_bbox_um ? esc(mp.layout_bbox_um.map((v) => v.toFixed(1)).join(", ")) : "– (no layout)"],
    ["Confidence", `detection ${(d.conf * 100).toFixed(0)}% · ` + (mp.region === "UNMAPPED" ? "no layout to map" : `mapping ${(mp.mapping_confidence * 100).toFixed(0)}%`)]]
    .map(([k, v]) => `<b>${k}</b><span>${v}</span>`).join("");
  renderAdvice();
}

function renderAdvice() {
  if (!cur || !cur.defects.length) return;
  const d = cur.defects[sel], r = cur.advice[d.id];
  if (!r) {
    $("advSrc").textContent = "";
    $("advCard").innerHTML = `<div class="muted">Analysing fault… (rule-based decision already applied)</div><div class="skel"></div><div class="skel"></div><div class="skel" style="width:60%"></div>`;
    return;
  }
  if (!r.advice) { $("advCard").innerHTML = `<div class="conflict">Advisor error: ${esc((r.problems || []).join("; "))}</div>`; return; }
  const a = r.advice, st = r.stats || {};
  $("advSrc").className = "tag " + (r.source === "llama" ? "llama" : "template");
  $("advSrc").textContent = r.source === "llama" ? "Llama 3.2 1B · on-device" : "Template advice";
  const perf = r.source === "llama" ? `TTFT ${st.ttft_ms ?? "–"} ms · decode ${st.decode_tok_s ?? "–"} tok/s · total ${(r.latency_ms / 1000).toFixed(1)} s`
                                    : `${r.latency_ms} ms${(r.problems || []).length ? " · Llama fallback: " + esc(r.problems[0]) : ""}`;
  $("advCard").innerHTML = `
    <div><div class="adv-k">Likely fault</div><div class="adv-v">${esc(a.fault_model)}</div></div>
    <div><div class="adv-k">Affected</div><div class="adv-v">${esc(a.affected)}</div></div>
    <div><div class="adv-k">Confirm with</div><div class="adv-v">${esc(a.confirm_test)}</div></div>
    <div><div class="adv-k">Action</div><div class="adv-v">${esc(r.final_action.replaceAll("_", " "))}</div></div>
    <div class="adv-expl">${esc(a.explanation)}</div>
    ${r.conflict ? `<div class="conflict">Advisor suggested ${esc(a.recommended_action)}; rule-based decision ${esc(r.final_action)} applied.</div>` : ""}
    <div class="muted small">Confidence ${esc(a.confidence)} · KB ${esc((r.kb_ids || []).join(", "))} · ${perf}</div>`;
}

function renderSummary() {
  $("repairMap").src = `/api/inspection/${cur.id}/repairmap.png?${Date.now()}`;
  $("rmJson").href = `/api/inspection/${cur.id}/repairmap.json`;
  $("rmPng").href = `/api/inspection/${cur.id}/repairmap.png`;
  $("saved").textContent = `Saved ✓ · inspection ${cur.id} · decision in ${cur.timings.inspect_to_decision_ms.toFixed(0)} ms` + (cur.warning ? ` · ${cur.warning}` : "");
  renderSummaryTable();
}
function renderSummaryTable() {
  const sv = cur.decision.defects;
  $("defTable").innerHTML = `<tr><th>#</th><th>Class</th><th>Region</th><th>Severity</th><th>Fault</th><th>Test</th></tr>` +
    cur.defects.map((d, i) => {
      const a = (cur.advice[d.id] || {}).advice || {};
      return `<tr><td>${i + 1}</td><td>${esc(d.cls)}</td><td>${esc(cur.mapping[i].region)}</td><td><span class="sev s-${sv[i].severity}">${sv[i].severity}</span></td><td>${esc(a.fault_model || "…")}</td><td>${esc(a.confirm_test || "…")}</td></tr>`;
    }).join("");
}

$("prevDef").onclick = () => cur && selectDef(Math.max(sel - 1, 0));
$("nextDef").onclick = () => cur && selectDef(Math.min(sel + 1, cur.defects.length - 1));
$("reinspectBtn").onclick = () => show("live");
$("nextDie").onclick = () => { $("dieId").value = ""; show("live"); };
function setDecided(text) {
  const el = $("decided");
  if (el) el.textContent = text;
  const done = !!text;
  $("acceptBtn").disabled = done;
  $("acceptBtn").textContent = done && text.startsWith("Accepted") ? "Accepted ✓" : "Accept decision";
}
$("acceptBtn").onclick = async () => {
  if (!cur) return;
  const btn = $("acceptBtn");
  btn.disabled = true; btn.textContent = "Saving…";
  try {
    const r = await api(`/api/inspection/${cur.id}/action`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action: "accept" }) });
    setDecided("Accepted ✓");
    $("saved").textContent = `Decision ${r.final} accepted and logged ✓`;
    toast(`${cur.die_id}: ${r.final} accepted and logged`);
  } catch (e) {
    btn.disabled = false; btn.textContent = "Accept decision";
    toast("Could not save the decision: " + e.message, true);
  }
};
$("overrideBtn").onclick = () => { $("ovrReason").value = ""; $("ovrNote").value = ""; $("ovr").showModal(); };
$("ovrOk").onclick = async (e) => {
  if (!$("ovrReason").value) { e.preventDefault(); $("ovrReason").focus(); toast("Choose a reason for the override", true); return; }
  try {
    const r = await api(`/api/inspection/${cur.id}/action`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "override:" + $("ovrDecision").value, reason: $("ovrReason").value, note: $("ovrNote").value }) });
    setDecided(`Overridden → ${r.final}`);
    $("saved").textContent = `Overridden to ${r.final} and logged ✓`;
    toast(`${cur.die_id}: overridden to ${r.final} (${$("ovrReason").value}) and logged`);
  } catch (err) { toast("Could not save the override: " + err.message, true); }
};

// ---------------------------------------------------------------- log
async function loadLog() {
  const rows = await api("/api/log");
  $("logTable").innerHTML = `<tr><th>Time</th><th>Die</th><th>Decision</th><th>Final</th><th>Defects</th><th>Advice (Llama/total)</th></tr>` +
    (rows.length ? rows.map((r) => `<tr data-id="${r.id}"><td>${new Date(r.ts * 1000).toLocaleString()}</td><td>${esc(r.die_id)}</td>
      <td><span class="sev s-${{ PASS: "COSMETIC", REPAIR: "REPAIRABLE", SCRAP: "KILLER", RE_INSPECT: "REVIEW" }[r.decision]}">${r.decision}</span></td>
      <td>${esc(r.final || "–")}${r.operator_action && r.operator_action.startsWith("override") ? " (override)" : ""}</td><td>${r.n_defects}</td><td>${r.llama_advice}/${r.advice}</td></tr>`).join("")
      : `<tr><td colspan="6" class="muted">No inspections yet: start one from Home.</td></tr>`);
  document.querySelectorAll("#logTable tr[data-id]").forEach((tr) => (tr.onclick = async () => openInspection(await api(`/api/inspection/${tr.dataset.id}`))));
}
$("refreshLog").onclick = loadLog;
