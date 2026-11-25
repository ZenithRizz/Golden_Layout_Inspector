"""Local web server for the operator UI (F9).  Binds to localhost by default.

Run:  python -m gli.server --design designs/sram16x16_scn4m --source virtual
"""
from __future__ import annotations

import argparse
import asyncio
import os
import time

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .station import Station

HERE = os.path.dirname(__file__)
app = FastAPI(title="Golden-Layout Inspector")
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")
ST: Station | None = None


def _png(img):
    ok, b = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(500, "encode failed")
    return Response(b.tobytes(), media_type="image/png")


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/api/status")
def status():
    return ST.status()


@app.get("/stream.mjpg")
async def stream():
    async def gen():
        last = None
        while True:
            jpg = ST.live_jpeg
            if jpg is not None and jpg is not last:
                last = jpg
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            await asyncio.sleep(0.03)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


class InspectReq(BaseModel):
    die_id: str | None = None


@app.post("/api/inspect")
def inspect(req: InspectReq):
    rec = ST.inspect(die_id=req.die_id or None)
    if "error" in rec:
        return JSONResponse(rec, status_code=409)
    return _public(rec)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(413, "Image larger than 20 MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(415, "Not a JPG/PNG image")
    rec = ST.inspect(frame=img, allow_reference_free=True)
    if "error" in rec:
        return JSONResponse(rec, status_code=409)
    return _public(rec)


@app.get("/api/inspection/{iid}")
def get_inspection(iid: str):
    rec = ST.get(iid)
    if rec is None:
        raise HTTPException(404, "unknown inspection")
    return _public(rec)


@app.get("/api/inspection/{iid}/image/{kind}")
def image(iid: str, kind: str):
    rec = ST.get(iid)
    if rec is None or kind not in ("frame", "overlay", "rect") or not rec["paths"].get(kind):
        raise HTTPException(404)
    return FileResponse(rec["paths"][kind], media_type="image/jpeg")


@app.get("/api/inspection/{iid}/defect/{did}/zoom.png")
def zoom(iid: str, did: str):
    img = ST.defect_zoom(iid, did)
    if img is None:
        raise HTTPException(404)
    return _png(img)


@app.get("/api/inspection/{iid}/repairmap.png")
def repairmap(iid: str):
    img = ST.repair_map(iid)
    if img is None:
        raise HTTPException(404)
    return _png(img)


@app.get("/api/inspection/{iid}/repairmap.json")
def repairmap_json(iid: str):
    rec = ST.get(iid)
    if rec is None:
        raise HTTPException(404)
    fail = sorted({tuple(c) for d in rec["decision"]["defects"] for c in d["failing_cells"]})
    return {"die_id": rec["die_id"], "design": rec["design"], "decision": rec["decision"]["die_decision"],
            "allocation": rec["decision"]["allocation"], "failing_cells": fail}


class ActionReq(BaseModel):
    action: str
    reason: str = ""
    note: str = ""


@app.post("/api/inspection/{iid}/action")
def action(iid: str, req: ActionReq):
    r = ST.action(iid, req.action, req.reason, req.note)
    if "error" in r:
        return JSONResponse(r, status_code=400)
    return r


@app.get("/api/log")
def log(limit: int = 200):
    return ST.store.list(limit)


@app.get("/api/export")
def export(fmt: str = "csv"):
    body = ST.store.export(fmt)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    media = "application/json" if fmt == "json" else "text/csv"
    return Response(body, media_type=media,
                    headers={"Content-Disposition": f'attachment; filename="inspections-{stamp}.{fmt}"'})


class SourceReq(BaseModel):
    source: str


@app.post("/api/source")
def set_source(req: SourceReq):
    ST.set_source(req.source)
    return ST.status()["camera"]


class DieReq(BaseModel):
    defects: int = 3


@app.post("/api/virtual/new_die")
def new_die(req: DieReq):
    from .camera import VirtualCamera
    if not isinstance(ST.src, VirtualCamera):
        raise HTTPException(400, "Only available with the virtual camera")
    gts = ST.src.new_die(max(0, min(req.defects, 8)))
    return {"ground_truth": gts}


def _public(rec):
    rec = dict(rec)
    rec.pop("paths", None)
    return rec


def main():
    global ST
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="designs/sram16x16_scn4m")
    ap.add_argument("--source", default="virtual", help="virtual | usb:0 | picam | path/to/image.jpg")
    ap.add_argument("--onnx", default="models/defect_cls.onnx")
    ap.add_argument("--hef", default=None, help="Hailo .hef (only if an AI HAT is fitted)")
    ap.add_argument("--llama-runner", default=os.environ.get("GLI_LLAMA_RUNNER"))
    ap.add_argument("--llama-pte", default=os.environ.get("GLI_LLAMA_PTE"))
    ap.add_argument("--llama-tokenizer", default=os.environ.get("GLI_LLAMA_TOKENIZER"))
    ap.add_argument("--llama-threads", type=int, default=4)
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
    print(f"Golden-Layout Inspector on http://{a.host}:{a.port}  vision={ST.classifier.name}  advisor={ST.advisor.mode}")
    for n in ST.notes:
        print("note:", n)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
