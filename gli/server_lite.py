"""Dependency-free web server (Python standard library only) for the operator UI.

Same API and UI as gli/server.py (FastAPI), for devices where FastAPI's compiled
dependencies don't install, e.g. Android/Termux.  run_station.sh picks it automatically.

Run:  python -m gli.server_lite --design designs/sram16x16_scn4m --source http://127.0.0.1:8080
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import time
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from .station import Station

HERE = os.path.dirname(__file__)
STATIC = os.path.join(HERE, "static")
ST: Station | None = None


def _public(rec):
    rec = dict(rec)
    rec.pop("paths", None)
    return rec


class Handler(BaseHTTPRequestHandler):
    server_version = "GLI/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):   # quiet
        pass

    # ---------------------------------------------------------------- helpers
    def _send(self, code, body: bytes, ctype="application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, default=str).encode())

    def _err(self, code, msg):
        self._json({"error": msg, "detail": msg}, code)

    def _png(self, img):
        if img is None:
            return self._err(404, "not found")
        ok, b = cv2.imencode(".png", img)
        self._send(200, b.tobytes(), "image/png")

    def _file(self, path, ctype=None):
        if not path or not os.path.isfile(path):
            return self._err(404, "not found")
        with open(path, "rb") as f:
            data = f.read()
        self._send(200, data, ctype or mimetypes.guess_type(path)[0] or "application/octet-stream")

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 25 * 1024 * 1024:
            raise ValueError("too large")
        return self.rfile.read(n) if n else b""

    def _jbody(self):
        b = self._body()
        try:
            return json.loads(b or b"{}")
        except json.JSONDecodeError:
            return {}

    # ---------------------------------------------------------------- GET
    def do_GET(self):
        u = urlparse(self.path)
        p, q = u.path, parse_qs(u.query)
        try:
            if p == "/":
                return self._file(os.path.join(STATIC, "index.html"), "text/html; charset=utf-8")
            if p.startswith("/static/"):
                f = os.path.normpath(os.path.join(STATIC, p[len("/static/"):]))
                if not f.startswith(STATIC):
                    return self._err(403, "forbidden")
                return self._file(f)
            if p == "/api/status":
                return self._json(ST.status())
            if p == "/stream.mjpg":
                return self._stream()
            if p == "/api/log":
                return self._json(ST.store.list(int(q.get("limit", ["200"])[0])))
            if p == "/api/export":
                fmt = q.get("fmt", ["csv"])[0]
                body = ST.store.export(fmt).encode()
                stamp = time.strftime("%Y%m%d-%H%M%S")
                return self._send(200, body, "application/json" if fmt == "json" else "text/csv",
                                  {"Content-Disposition": f'attachment; filename="inspections-{stamp}.{fmt}"'})
            m = re.fullmatch(r"/api/inspection/([^/]+)", p)
            if m:
                rec = ST.get(m.group(1))
                return self._json(_public(rec)) if rec else self._err(404, "unknown inspection")
            m = re.fullmatch(r"/api/inspection/([^/]+)/image/(frame|overlay|rect)", p)
            if m:
                rec = ST.get(m.group(1))
                return self._file(rec["paths"].get(m.group(2)) if rec else None, "image/jpeg")
            m = re.fullmatch(r"/api/inspection/([^/]+)/defect/([^/]+)/zoom.png", p)
            if m:
                return self._png(ST.defect_zoom(m.group(1), m.group(2)))
            m = re.fullmatch(r"/api/inspection/([^/]+)/repairmap.png", p)
            if m:
                return self._png(ST.repair_map(m.group(1)))
            m = re.fullmatch(r"/api/inspection/([^/]+)/repairmap.json", p)
            if m:
                rec = ST.get(m.group(1))
                if not rec:
                    return self._err(404, "unknown inspection")
                fail = sorted({tuple(c) for d in rec["decision"]["defects"] for c in d["failing_cells"]})
                return self._json({"die_id": rec["die_id"], "design": rec["design"],
                                   "decision": rec["decision"]["die_decision"],
                                   "allocation": rec["decision"]["allocation"], "failing_cells": fail})
            return self._err(404, "not found")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        last = None
        try:
            while True:
                jpg = ST.live_jpeg
                if jpg is not None and jpg is not last:
                    last = jpg
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                     str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    self.wfile.flush()
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    # ---------------------------------------------------------------- POST
    def do_POST(self):
        p = urlparse(self.path).path
        try:
            if p == "/api/inspect":
                rec = ST.inspect(die_id=(self._jbody().get("die_id") or None))
                return self._json(rec, 409) if "error" in rec else self._json(_public(rec))
            if p == "/api/upload":
                img = self._upload_image()
                if img is None:
                    return self._err(415, "Not a JPG/PNG image")
                rec = ST.inspect(frame=img, allow_reference_free=True)
                return self._json(rec, 409) if "error" in rec else self._json(_public(rec))
            if p == "/api/source":
                ST.set_source(self._jbody().get("source", "virtual"))
                return self._json(ST.status()["camera"])
            if p == "/api/virtual/new_die":
                from .camera import VirtualCamera
                if not isinstance(ST.src, VirtualCamera):
                    return self._err(400, "Only available with the virtual camera")
                n = int(self._jbody().get("defects", 3))
                return self._json({"ground_truth": ST.src.new_die(max(0, min(n, 8)))})
            m = re.fullmatch(r"/api/inspection/([^/]+)/action", p)
            if m:
                b = self._jbody()
                r = ST.action(m.group(1), b.get("action", ""), b.get("reason", ""), b.get("note", ""))
                return self._json(r, 400) if "error" in r else self._json(r)
            return self._err(404, "not found")
        except ValueError as e:
            return self._err(413, str(e))
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _upload_image(self):
        ctype = self.headers.get("Content-Type", "")
        body = self._body()
        if "multipart/form-data" not in ctype:
            return None
        msg = BytesParser(policy=email_policy).parsebytes(
            b"Content-Type: " + ctype.encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + body)
        for part in msg.iter_parts():
            data = part.get_payload(decode=True)
            if data:
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    return img
        return None


def main():
    global ST
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="designs/sram16x16_scn4m")
    ap.add_argument("--source", default="virtual", help="virtual | http://127.0.0.1:8080 | usb:0 | picam | image path")
    ap.add_argument("--onnx", default="models/defect_cls.onnx")
    ap.add_argument("--hef", default=None)
    ap.add_argument("--llama-runner", default=os.environ.get("GLI_LLAMA_RUNNER"))
    ap.add_argument("--llama-pte", default=os.environ.get("GLI_LLAMA_PTE"))
    ap.add_argument("--llama-tokenizer", default=os.environ.get("GLI_LLAMA_TOKENIZER"))
    ap.add_argument("--llama-threads", type=int, default=int(os.environ.get("GLI_LLAMA_THREADS", "4")))
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    llama = None
    if a.llama_runner and a.llama_pte and a.llama_tokenizer:
        llama = {"runner": a.llama_runner, "pte": a.llama_pte, "tokenizer": a.llama_tokenizer,
                 "threads": a.llama_threads}
    ST = Station(a.design, a.source, onnx=a.onnx if os.path.exists(a.onnx) else None, hef=a.hef,
                 llama=llama, runs=a.runs)
    print(f"Golden-Layout Inspector (lite server) on http://{a.host}:{a.port}  "
          f"vision={ST.classifier.name}  advisor={ST.advisor.mode}", flush=True)
    for n in ST.notes:
        print("note:", n, flush=True)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
