"""Inspection station: ties camera, registration, vision, mapping, decision,
advisor and log together.  Used by the web server and the benchmark script.

Two pipelines, measured separately:
  * Vision pipeline  - camera -> registration -> golden diff -> CNN classifier
                       (ONNX Runtime on the Pi CPU, or HailoRT on the NPU if fitted)
  * Advisor pipeline - Llama 3.2 1B via ExecuTorch on the Pi Arm CPU, in a
                       background worker so it never blocks the live feed.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict

import cv2
import numpy as np

from .advisor import Advisor, AdvisorWorker, LlamaExecuTorchBackend
from .camera import open_source, VirtualCamera
from .decision import decide
from .layout import Design
from .mapping import Mapper
from .refree import inspect_reference_free, neighbour_diff
from .register import Registrar
from .store import Store, repair_map_png
from .vision import VisionConfig, detect, load_classifier

CLASS_COLORS = {"BRIDGE": (0, 140, 255), "OPEN": (255, 0, 200), "PARTICLE": (0, 0, 220),
                "MISSING_VIA": (200, 160, 0), "UNKNOWN": (0, 200, 200)}
SEV_ORDER = {"KILLER": 0, "REVIEW": 1, "REPAIRABLE": 2, "COSMETIC": 3}


class Station:
    def __init__(self, design_dir, source="virtual", onnx=None, hef=None, prefer="hailo",
                 llama=None, runs="runs", focus_min=15.0):
        self.design_dir = design_dir
        self.d = Design.load(design_dir)
        self.golden = cv2.imread(os.path.join(design_dir, "golden_work.png"))
        self.masks = dict(np.load(os.path.join(design_dir, "masks_work.npz")))
        self.reg = Registrar(self.d, self.golden)
        self.mapper = Mapper(self.d, self.masks)
        self.cfg = VisionConfig()
        self.classifier, self.notes = load_classifier(onnx, hef, prefer)
        backend = LlamaExecuTorchBackend(**llama) if llama else None
        self.advisor = Advisor(backend)
        if llama and self.advisor.mode == "template":
            self.notes.append("Advisor offline: Llama runner/model not found, template advice only")
        self.store = Store(runs)
        self.worker = AdvisorWorker(self.advisor, self._on_advice)
        self.worker.start()
        self.focus_min = focus_min
        self.source_spec = source
        self.src = None
        self.src_error = ""
        self.latest = None
        self.latest_ts = 0
        self.live = {"locked": False, "reason": "Starting camera...", "dets": [], "fps": 0.0, "timings": {}}
        self.live_jpeg = None
        self.lock = threading.Lock()
        self.pending = {}            # inspection id -> {defect_id: result}
        self.fps_hist = deque(maxlen=30)
        self.die_counter = 1
        self._stop = False
        self.set_source(source)
        threading.Thread(target=self._grab_loop, daemon=True).start()
        threading.Thread(target=self._proc_loop, daemon=True).start()

    # ------------------------------------------------------------------ camera
    def set_source(self, spec):
        try:
            if self.src is not None:
                self.src.close()
        except Exception:
            pass
        try:
            self.src = open_source(spec, self.design_dir)
            self.source_spec = getattr(self.src, "display", spec)
            self.src_error = ""
        except Exception as e:
            self.src = None
            self.source_spec = _redact(spec)
            self.src_error = f"Camera unavailable ({_redact(str(e))}). Use 'Upload image' or the virtual camera."

    def _grab_loop(self):
        fails = 0
        while not self._stop:
            src = self.src
            if src is None:
                time.sleep(0.2)
                continue
            try:
                f = src.read()
            except Exception:
                f = None
            if f is None:
                fails += 1
                if fails > 30:
                    self.src_error = "Camera disconnected. Reconnect it or upload an image."
                time.sleep(0.05)
                continue
            fails = 0
            self.src_error = ""
            with self.lock:
                self.latest, self.latest_ts = f, time.time()

    def _proc_loop(self):
        last = 0
        while not self._stop:
            with self.lock:
                f, ts = self.latest, self.latest_ts
            if f is None or ts == last:
                time.sleep(0.01)
                continue
            last = ts
            try:
                res, overlay = self.process(f)
            except Exception as e:   # keep the live loop alive
                res, overlay = {"locked": False, "reason": f"Processing error: {e}", "dets": [], "timings": {}}, f
            self.fps_hist.append(time.time())
            if len(self.fps_hist) > 1:
                res["fps"] = round((len(self.fps_hist) - 1) / (self.fps_hist[-1] - self.fps_hist[0] + 1e-6), 1)
            ok, jpg = cv2.imencode(".jpg", _fit(overlay, 1100), [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self.lock:
                self.live = res
                self.live_jpeg = jpg.tobytes() if ok else None

    # ------------------------------------------------------------------ vision
    def focus(self, frame):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (640, int(640 * g.shape[0] / g.shape[1])))
        return float(cv2.Laplacian(g, cv2.CV_64F).var())

    def process(self, frame, force_refine=False):
        t = {}
        t0 = time.perf_counter()
        if force_refine:
            self.reg._cache = None
        r = self.reg.register(frame)
        t["register_ms"] = (time.perf_counter() - t0) * 1000
        res = {"locked": r.locked, "reason": r.reason, "markers": r.n_markers, "reproj_px": round(r.reproj_px, 2)
               if np.isfinite(r.reproj_px) else None, "ecc": round(r.ecc, 3), "reg_quality": r.quality,
               "focus": round(self.focus(frame), 1), "dets": [], "timings": t}
        overlay = frame.copy()
        if not r.locked:
            _banner(overlay, r.reason or "Not registered", (0, 140, 255))
            return res, overlay
        t1 = time.perf_counter()
        rect = self.reg.rectify(frame, r.H)
        valid = self.reg.valid_mask(frame.shape, r.H)
        t["rectify_ms"] = (time.perf_counter() - t1) * 1000
        dets, dmap = detect(rect, self.golden, self.classifier, self.cfg, valid, t)
        t["vision_total_ms"] = (time.perf_counter() - t0) * 1000
        res["dets"] = [asdict(d) for d in dets]
        res["_H"], res["_rect"], res["_valid"] = r.H, rect, valid
        for d in dets:
            x, y, w, h = d.box
            pts = Registrar.to_frame(r.H, [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]).astype(np.int32)
            col = CLASS_COLORS.get(d.cls, (0, 255, 0))
            c = pts.mean(0)
            rad = int(max(14, np.abs(pts - c).max() + 8))
            cv2.circle(overlay, tuple(c.astype(int)), rad, col, 2, cv2.LINE_AA)
            cv2.putText(overlay, f"{d.cls} {d.conf:.2f}", (int(c[0]) + rad, int(c[1]) - rad // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        _banner(overlay, f"Registered  |  {len(dets)} defect(s)  |  {self.classifier.name}", (40, 160, 40))
        return res, overlay

    # ------------------------------------------------------------------ inspect
    def inspect(self, frame=None, die_id=None, allow_reference_free=False):
        if frame is None:
            with self.lock:
                frame = None if self.latest is None else self.latest.copy()
        if frame is None:
            return {"error": self.src_error or "No camera frame available"}
        fval = self.focus(frame)
        if fval < self.focus_min:
            return {"error": f"Adjust focus (sharpness {fval:.0f} < {self.focus_min:.0f})"}
        t0 = time.perf_counter()
        res, overlay = self.process(frame, force_refine=True)
        if not res["locked"]:
            if allow_reference_free and res.get("markers", 0) == 0:
                return self._inspect_reference_free(frame, die_id, t0)
            return {"error": res["reason"] or "Align die: fiducials not found"}
        dets = res["dets"]
        t1 = time.perf_counter()
        maps = []
        for d in dets:
            x, y, w, h = d["box"]
            W, H = self.d.work_size
            partial = x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2
            maps.append(self.mapper.map(d["box"], res["reg_quality"], partial))
        dec = decide(dets, maps, self.d)
        t2 = time.perf_counter()
        iid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        die = die_id or f"DIE-{self.die_counter:04d}"
        self.die_counter += 1
        timings = dict(res["timings"])
        timings.update({"map_decide_ms": (t2 - t1) * 1000, "inspect_to_decision_ms": (t2 - t0) * 1000})
        rec = {"id": iid, "ts": time.time(), "design": self.d.name, "die_id": die,
               "defects": dets, "mapping": maps, "decision": dec, "timings": timings,
               "registration": {k: res[k] for k in ("markers", "reproj_px", "ecc", "reg_quality", "focus")},
               "vision_backend": self.classifier.name, "advisor_backend": self.advisor.mode,
               "source": self.source_spec}
        if isinstance(self.src, VirtualCamera):
            rec["ground_truth"] = self.src.gts
        _banner(overlay, f"{die}: {dec['die_decision']}  ({dec['die_reason']})",
                {"PASS": (40, 160, 40), "REPAIR": (0, 160, 230), "SCRAP": (0, 0, 220), "RE_INSPECT": (0, 140, 255)}
                [dec["die_decision"]])
        for i, d in enumerate(dets):
            x, y, w, h = d["box"]
            p = Registrar.to_frame(res["_H"], [[x + w / 2, y + h / 2]])[0].astype(int)
            cv2.putText(overlay, str(i + 1), (p[0] - 6, p[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(overlay, str(i + 1), (p[0] - 6, p[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        paths, imgs_ok = self.store.save_inspection(rec, frame, overlay, res["_rect"])
        if not imgs_ok:
            rec["warning"] = "Low disk space: images not saved, metadata only"
        self.pending[iid] = {}
        items = []
        for d, mp, sv in zip(dets, maps, dec["defects"]):
            items.append((d, mp, sv))
        items.sort(key=lambda it: SEV_ORDER.get(it[2]["severity"], 9))
        return self._finish(rec, items, dec)

    def _finish(self, rec, items, dec):
        items.sort(key=lambda it: SEV_ORDER.get(it[2]["severity"], 9))
        if items and (dec["die_decision"] != "PASS" or any(sv["severity"] != "COSMETIC" for sv in dec["defects"])):
            self.worker.submit(rec["id"], items)
            rec["advice_expected"] = min(len(items), self.worker.max_queue)
        else:
            rec["advice_expected"] = 0
        return rec

    def _inspect_reference_free(self, frame, die_id, t0):
        """Uploaded photo of a die we have no layout for: cell-to-cell comparison (refree.py)."""
        r = inspect_reference_free(frame, self.classifier, self.cfg)
        if "error" in r:
            return {"error": r["error"]}
        dets = [asdict(d) for d in r["dets"]]
        maps = [{"layout_bbox_um": None, "region": "UNMAPPED", "layers": [], "nets": [], "affected_rows": [],
                 "affected_cols": [], "affected_cells": [], "mapping_confidence": 0.0, "partial": False}
                for _ in dets]
        dec = decide(dets, maps, self.d)
        if not dets:
            dec["die_reason"] = "No anomalies found in the repeating array"
        t2 = time.perf_counter()
        iid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        die = die_id or f"UPLOAD-{self.die_counter:04d}"
        self.die_counter += 1
        timings = {k: v for k, v in r["timings"].items()}
        timings["inspect_to_decision_ms"] = (t2 - t0) * 1000
        rec = {"id": iid, "ts": time.time(), "design": "unknown die (no layout)", "die_id": die,
               "mode": "reference_free", "reference_free": {**r["period"], "array_fraction": r["array_fraction"],
                                                            "scale": r["scale"]},
               "defects": dets, "mapping": maps, "decision": dec, "timings": timings,
               "vision_backend": self.classifier.name, "advisor_backend": self.advisor.mode, "source": "upload"}
        img = r["image"]
        overlay = img.copy()
        dim = r["mask"] == 0
        overlay[dim] = (overlay[dim] * 0.55).astype(np.uint8)       # show which area was inspected
        for i, d in enumerate(dets):
            x, y, w, h = d["box"]
            col = CLASS_COLORS.get(d["cls"], (0, 255, 0))
            cv2.rectangle(overlay, (x - 6, y - 6), (x + w + 6, y + h + 6), col, 2)
            cv2.putText(overlay, f"{i + 1} {d['cls']}", (x, max(14, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(overlay, f"{i + 1} {d['cls']}", (x, max(14, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1)
        _banner(overlay, f"{die}: {dec['die_decision']}  (no layout - cell-to-cell, repeat {r['period']['px']}x{r['period']['py']} px)",
                {"PASS": (40, 160, 40), "REPAIR": (0, 160, 230), "SCRAP": (0, 0, 220), "RE_INSPECT": (0, 140, 255)}
                [dec["die_decision"]])
        paths, imgs_ok = self.store.save_inspection(rec, frame, overlay, img)
        if not imgs_ok:
            rec["warning"] = "Low disk space: images not saved, metadata only"
        self.pending[iid] = {}
        return self._finish(rec, list(zip(dets, maps, dec["defects"])), dec)

    def _on_advice(self, iid, did, res):
        self.store.save_advice(iid, did, res)
        self.pending.setdefault(iid, {})[did] = res

    # ------------------------------------------------------------------ views
    def get(self, iid):
        return self.store.get(iid)

    def action(self, iid, action, reason="", note=""):
        rec = self.store.get(iid)
        if rec is None:
            return {"error": "unknown inspection"}
        orig = rec["decision"]["die_decision"]
        if action == "accept":
            final = orig
        elif action.startswith("override:"):
            if not reason:
                return {"error": "An override needs a reason"}
            final = action.split(":", 1)[1]
        else:
            final = orig
        self.store.save_action(iid, action, reason, note, orig, final)
        return {"ok": True, "final": final}

    def defect_zoom(self, iid, did, scale=5):
        rec = self.store.get(iid)
        if rec is None:
            return None
        idx = next((i for i, d in enumerate(rec["defects"]) if d["id"] == did), None)
        if idx is None:
            return None
        d, mp = rec["defects"][idx], rec["mapping"][idx]
        sv = rec["decision"]["defects"][idx]
        if rec.get("mode") == "reference_free":
            return self._zoom_reference_free(rec, d, scale)
        x, y, w, h = d["box"]
        cx, cy = x + w / 2, y + h / 2
        half = max(48, int(max(w, h) * 1.5))
        x0, y0 = int(cx - half), int(cy - half)
        side = 2 * half
        rect = cv2.imread(rec["paths"]["rect"]) if rec["paths"]["rect"] else None

        def crop(img):
            p = cv2.copyMakeBorder(img, side, side, side, side, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            c = p[y0 + side:y0 + 2 * side, x0 + side:x0 + 2 * side]
            return cv2.resize(c, (side * scale // 2 * 2, side * scale // 2 * 2), interpolation=cv2.INTER_NEAREST)

        gold = crop(self.golden)
        s = gold.shape[0] / side
        cells = {tuple(c) for c in mp["affected_cells"]}
        fail = {tuple(c) for c in sv["failing_cells"]}
        for b in self.d.bitcells:
            bx0, by1 = self.d.um_to_work(b["rect"][0], b["rect"][1])
            bx1, by0 = self.d.um_to_work(b["rect"][2], b["rect"][3])
            p0 = (int((bx0 - x0) * s), int((by0 - y0) * s))
            p1 = (int((bx1 - x0) * s), int((by1 - y0) * s))
            if p1[0] < 0 or p1[1] < 0 or p0[0] > gold.shape[1] or p0[1] > gold.shape[0]:
                continue
            key = (b["r"], b["c"])
            if key in cells:
                ov = gold.copy()
                cv2.rectangle(ov, p0, p1, (0, 220, 255), -1)
                gold = cv2.addWeighted(ov, 0.35, gold, 0.65, 0)
                cv2.rectangle(gold, p0, p1, (0, 160, 255), 3)
                cv2.putText(gold, f"r{b['r']} c{b['c']}", (p0[0] + 4, p0[1] + 18), 0, 0.55, (0, 0, 0), 2)
            elif key in fail:
                cv2.rectangle(gold, p0, p1, (60, 60, 230), 1)
        bx, by = int((x - x0) * s), int((y - y0) * s)
        cv2.rectangle(gold, (bx, by), (bx + int(w * s), by + int(h * s)), (0, 0, 255), 2)
        panels = [gold]
        if rect is not None:
            cap = crop(rect)
            cv2.rectangle(cap, (bx, by), (bx + int(w * s), by + int(h * s)), (0, 0, 255), 2)
            panels = [cap, gold]
        out = np.hstack([cv2.copyMakeBorder(p, 28, 0, 0, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for p in panels])
        labels = ["Captured (aligned to layout)", "Golden layout + affected bitcells"] if len(panels) == 2 else ["Golden layout"]
        for i, lab in enumerate(labels):
            cv2.putText(out, lab, (i * (panels[0].shape[1] + 6) + 6, 20), 0, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
        return out

    def _zoom_reference_free(self, rec, d, scale=4):
        img = cv2.imread(rec["paths"]["rect"]) if rec["paths"].get("rect") else None
        if img is None:
            return None
        rf = rec["reference_free"]
        _, ref, _ = neighbour_diff(img, rf.get("px"), rf.get("py"), blur=0)
        x, y, w, h = d["box"]
        half = max(40, int(max(w, h) * 1.6))
        cx, cy = x + w // 2, y + h // 2

        def crop(im):
            p = cv2.copyMakeBorder(im, half, half, half, half, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            c = p[cy:cy + 2 * half, cx:cx + 2 * half]
            return cv2.resize(c, (2 * half * scale // 2, 2 * half * scale // 2), interpolation=cv2.INTER_NEAREST)

        a, b = crop(img), crop(ref)
        s = a.shape[0] / (2 * half)
        for im in (a, b):
            bx, by = int((x - cx + half) * s), int((y - cy + half) * s)
            cv2.rectangle(im, (bx, by), (bx + int(w * s), by + int(h * s)), (0, 0, 255), 2)
        out = np.hstack([cv2.copyMakeBorder(p, 28, 0, 0, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for p in (a, b)])
        for i, lab in enumerate(["Uploaded image", "Reference: neighbouring cell (one repeat away)"]):
            cv2.putText(out, lab, (i * (a.shape[1] + 6) + 6, 20), 0, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
        return out

    def repair_map(self, iid):
        rec = self.store.get(iid)
        if rec is None:
            return None
        if rec.get("mode") == "reference_free":
            img = np.full((150, 520, 3), 255, np.uint8)
            for i, t in enumerate(["No layout for this die:", "a bitcell repair map needs the design file.",
                                   "Findings are listed for engineer review."]):
                cv2.putText(img, t, (16, 40 + 32 * i), 0, 0.6, (60, 60, 60), 1, cv2.LINE_AA)
            return img
        return repair_map_png(rec, self.d.rows, self.d.cols)

    def status(self):
        with self.lock:
            live = {k: v for k, v in self.live.items() if not k.startswith("_")}
        return {
            "design": {"name": self.d.name, "rows": self.d.rows, "cols": self.d.cols, "technology": self.d.technology,
                       "spare_rows": self.d.spare_rows, "spare_cols": self.d.spare_cols,
                       "spares_virtual": self.d.spares_virtual},
            "camera": {"source": self.source_spec, "ok": self.src is not None and not self.src_error,
                       "error": self.src_error},
            "vision": {"backend": self.classifier.name, "notes": self.notes},
            "advisor": {"backend": self.advisor.mode, "queue": self.worker.q.qsize()},
            "storage": {"free_mb": round(self.store.free_mb())},
            "live": live,
        }


def _redact(text):
    """Hide user:password@ in camera URLs."""
    import re
    return re.sub(r"//[^/@\s]+@", "//***@", text)


def _fit(img, max_w):
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    return cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)


def _banner(img, text, color):
    h = max(30, img.shape[0] // 28)
    cv2.rectangle(img, (0, 0), (img.shape[1], h), color, -1)
    cv2.putText(img, text, (10, int(h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, h / 45, (255, 255, 255), 2, cv2.LINE_AA)
