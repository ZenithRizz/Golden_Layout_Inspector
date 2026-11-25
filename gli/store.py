"""Inspection log (F10): SQLite (WAL) + image files, CSV/JSON export, repair map."""
from __future__ import annotations

import csv
import io
import json
import os
import shutil
import sqlite3
import threading
import time

import cv2
import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections(
  id TEXT PRIMARY KEY, ts REAL, design TEXT, die_id TEXT, decision TEXT, reason TEXT,
  n_defects INTEGER, record TEXT, frame_path TEXT, overlay_path TEXT, rect_path TEXT);
CREATE TABLE IF NOT EXISTS advice(
  inspection_id TEXT, defect_id TEXT, ts REAL, source TEXT, latency_ms REAL, conflict INTEGER,
  result TEXT, PRIMARY KEY(inspection_id, defect_id));
CREATE TABLE IF NOT EXISTS actions(
  inspection_id TEXT, ts REAL, action TEXT, reason TEXT, note TEXT, original TEXT, final TEXT);
"""


class Store:
    def __init__(self, root="runs"):
        self.root = root
        self.img_dir = os.path.join(root, "images")
        os.makedirs(self.img_dir, exist_ok=True)
        self.db_path = os.path.join(root, "inspections.db")
        self.lock = threading.Lock()
        self.db = sqlite3.connect(self.db_path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.db.commit()

    def free_mb(self):
        return shutil.disk_usage(self.root).free / 1e6

    def save_inspection(self, rec, frame, overlay, rect):
        iid = rec["id"]
        paths = {}
        keep_images = self.free_mb() > 500
        for kind, img in (("frame", frame), ("overlay", overlay), ("rect", rect)):
            if img is None or not keep_images:
                paths[kind] = None
                continue
            p = os.path.join(self.img_dir, f"{iid}_{kind}.jpg")
            cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            paths[kind] = p
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO inspections VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                            (iid, rec["ts"], rec["design"], rec["die_id"], rec["decision"]["die_decision"],
                             rec["decision"]["die_reason"], len(rec["defects"]), json.dumps(rec),
                             paths["frame"], paths["overlay"], paths["rect"]))
        return paths, keep_images

    def save_advice(self, iid, did, res):
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO advice VALUES (?,?,?,?,?,?,?)",
                            (iid, did, time.time(), res.get("source"), res.get("latency_ms"),
                             int(bool(res.get("conflict"))), json.dumps(res)))

    def save_action(self, iid, action, reason, note, original, final):
        with self.lock, self.db:
            self.db.execute("INSERT INTO actions VALUES (?,?,?,?,?,?,?)",
                            (iid, time.time(), action, reason, note, original, final))

    def get(self, iid):
        cur = self.db.execute("SELECT record, frame_path, overlay_path, rect_path FROM inspections WHERE id=?", (iid,))
        row = cur.fetchone()
        if not row:
            return None
        rec = json.loads(row[0])
        rec["paths"] = {"frame": row[1], "overlay": row[2], "rect": row[3]}
        rec["advice"] = {d: json.loads(r) for d, r in
                         self.db.execute("SELECT defect_id, result FROM advice WHERE inspection_id=?", (iid,))}
        acts = list(self.db.execute("SELECT ts, action, reason, note, original, final FROM actions WHERE inspection_id=? ORDER BY ts", (iid,)))
        rec["actions"] = [dict(zip(["ts", "action", "reason", "note", "original", "final"], a)) for a in acts]
        return rec

    def list(self, limit=200):
        q = """SELECT i.id, i.ts, i.design, i.die_id, i.decision, i.n_defects,
                      (SELECT COUNT(*) FROM advice a WHERE a.inspection_id=i.id AND a.source='llama'),
                      (SELECT COUNT(*) FROM advice a WHERE a.inspection_id=i.id),
                      (SELECT final FROM actions x WHERE x.inspection_id=i.id ORDER BY ts DESC LIMIT 1),
                      (SELECT action FROM actions x WHERE x.inspection_id=i.id ORDER BY ts DESC LIMIT 1)
               FROM inspections i ORDER BY i.ts DESC LIMIT ?"""
        keys = ["id", "ts", "design", "die_id", "decision", "n_defects", "llama_advice", "advice", "final", "operator_action"]
        return [dict(zip(keys, r)) for r in self.db.execute(q, (limit,))]

    def export(self, fmt="csv", ids=None):
        rows = []
        for item in self.list(100000):
            if ids and item["id"] not in ids:
                continue
            rec = self.get(item["id"])
            for d, sv, mp in zip(rec["defects"], rec["decision"]["defects"], rec["mapping"]):
                adv = rec["advice"].get(d["id"], {}).get("advice") or {}
                rows.append({
                    "inspection_id": rec["id"], "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(rec["ts"])),
                    "die_id": rec["die_id"], "die_decision": rec["decision"]["die_decision"],
                    "final_decision": item["final"] or rec["decision"]["die_decision"],
                    "defect_id": d["id"], "class": d["cls"], "confidence": d["conf"], "region": mp["region"],
                    "nets": " ".join(mp["nets"]), "rows": " ".join(map(str, mp["affected_rows"])),
                    "cols": " ".join(map(str, mp["affected_cols"])), "severity": sv["severity"],
                    "fault_model": adv.get("fault_model", ""), "confirm_test": adv.get("confirm_test", ""),
                    "advice_source": rec["advice"].get(d["id"], {}).get("source", ""),
                    "explanation": adv.get("explanation", ""),
                })
            if not rec["defects"]:
                rows.append({"inspection_id": rec["id"], "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(rec["ts"])),
                             "die_id": rec["die_id"], "die_decision": rec["decision"]["die_decision"],
                             "final_decision": item["final"] or rec["decision"]["die_decision"]})
        if fmt == "json":
            return json.dumps(rows, indent=1)
        buf = io.StringIO()
        keys = ["inspection_id", "time", "die_id", "die_decision", "final_decision", "defect_id", "class", "confidence",
                "region", "nets", "rows", "cols", "severity", "fault_model", "confirm_test", "advice_source", "explanation"]
        w = csv.DictWriter(buf, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        return buf.getvalue()


def repair_map_png(rec, rows, cols, cell=26):
    """Array grid: failing cells red, allocated spare rows/cols highlighted."""
    alloc = rec["decision"]["allocation"]
    fail = {tuple(c) for d in rec["decision"]["defects"] for c in d["failing_cells"]}
    pad_l, pad_t = 46, 30
    W, H = pad_l + cols * cell + 20, pad_t + rows * cell + 60
    img = np.full((H, W, 3), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            y = pad_t + (rows - 1 - r) * cell
            x = pad_l + c * cell
            col = (235, 235, 235)
            if r in alloc["spare_rows_used"] or c in alloc["spare_cols_used"]:
                col = (210, 235, 200)
            if (r, c) in fail:
                col = (60, 60, 220) if not (r in alloc["spare_rows_used"] or c in alloc["spare_cols_used"]) else (40, 140, 40)
            cv2.rectangle(img, (x + 1, y + 1), (x + cell - 2, y + cell - 2), col, -1)
        cv2.putText(img, str(r), (8, pad_t + (rows - 1 - r) * cell + cell - 8), 0, 0.4, (90, 90, 90), 1)
    for c in range(cols):
        cv2.putText(img, str(c), (pad_l + c * cell + 6, pad_t - 10), 0, 0.4, (90, 90, 90), 1)
    def _used(v, n):
        return f"{len(v)}/{n}" + (f" ({', '.join(map(str, v))})" if v else "")
    txt = (f"{rec['decision']['die_decision']}   spare rows used {_used(alloc['spare_rows_used'], alloc['spare_rows'])}"
           f"   spare cols used {_used(alloc['spare_cols_used'], alloc['spare_cols'])}")
    cv2.putText(img, txt, (pad_l, H - 30), 0, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(img, "red = failing, green = repaired by spare, row 0 at bottom", (pad_l, H - 12), 0, 0.4,
                (90, 90, 90), 1, cv2.LINE_AA)
    return img
