#!/usr/bin/env python3
"""Train the defect classifier on synthetic crops and export ONNX (F2).

Outputs (models/):
  defect_cls.pt            PyTorch weights
  defect_cls.onnx          dynamic batch, for ONNX Runtime on the Pi CPU
  defect_cls_b1.onnx       static batch 1, input for the Hailo Dataflow Compiler
  calib_nhwc.npy           512 calibration crops (NHWC, 0-255) for Hailo optimisation
  metrics.json             accuracy, per-class precision/recall, confusion matrix
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gli.model import DefectNet  # noqa: E402
from gli.vision import CLASSES  # noqa: E402


def augment(x):
    # x: B,3,H,W float.  Flips/90-degree rotations keep defect semantics.
    if np.random.rand() < 0.5:
        x = torch.flip(x, [3])
    if np.random.rand() < 0.5:
        x = torch.flip(x, [2])
    k = np.random.randint(4)
    if k:
        x = torch.rot90(x, k, [2, 3])
    g = torch.empty(x.size(0), 1, 1, 1).uniform_(0.85, 1.15)
    x = x.clone()
    x[:, :1] = (x[:, :1] * g).clamp(0, 255)   # brightness jitter on the captured channel only
    return x


def evaluate(model, X, Y, bs=512):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            preds.append(model(X[i:i + bs]).argmax(1))
    p = torch.cat(preds).numpy()
    y = Y.numpy()
    n = len(CLASSES)
    cm = np.zeros((n, n), int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    per = {}
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]
        per[c] = {"precision": round(tp / max(cm[:, i].sum(), 1), 3),
                  "recall": round(tp / max(cm[i].sum(), 1), 3), "support": int(cm[i].sum())}
    return float((p == y).mean()), per, cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sram16x16_scn4m/crops.npz")
    ap.add_argument("--out", default="models")
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--max-none", type=int, default=6000, help="cap NONE crops to balance classes / speed up")
    a = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)

    d = np.load(a.data)
    Xn, Yn = d["X"], d["y"]
    none_idx = np.where(Yn == 0)[0]
    if len(none_idx) > a.max_none:
        drop = np.random.choice(none_idx, len(none_idx) - a.max_none, replace=False)
        keep = np.setdiff1d(np.arange(len(Yn)), drop)
        Xn, Yn = Xn[keep], Yn[keep]
    X = torch.from_numpy(Xn).permute(0, 3, 1, 2).float()
    Y = torch.from_numpy(Yn).long()
    idx = torch.randperm(len(Y))
    n_val = int(0.15 * len(Y))
    vi, ti = idx[:n_val], idx[n_val:]
    Xtr, Ytr, Xva, Yva = X[ti], Y[ti], X[vi], Y[vi]
    counts = torch.bincount(Ytr, minlength=len(CLASSES)).float()
    w = (counts.sum() / (len(CLASSES) * counts.clamp(min=1))).clamp(max=6.0)

    model = DefectNet()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * int(np.ceil(len(Ytr) / a.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
    lossf = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    print(f"train {len(Ytr)}  val {len(Yva)}  class weights {w.numpy().round(2)}  "
          f"params {sum(p.numel() for p in model.parameters())}")
    best, best_state = -1, None
    for ep in range(a.epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(Ytr))
        tot = 0.0
        for i in range(0, len(perm), a.bs):
            b = perm[i:i + a.bs]
            xb = augment(Xtr[b])
            loss = lossf(model(xb), Ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        acc, per, _ = evaluate(model, Xva, Yva)
        macro = np.mean([v["recall"] for v in per.values()])
        print(f"epoch {ep + 1:2d} loss {tot / len(Ytr):.3f} val acc {acc:.3f} macro-recall {macro:.3f} ({time.time() - t0:.0f}s)")
        if macro > best:
            best, best_state = macro, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    acc, per, cm = evaluate(model, Xva, Yva)
    torch.save(model.state_dict(), os.path.join(a.out, "defect_cls.pt"))

    model.eval()
    dummy = torch.zeros(1, 3, 64, 64)
    torch.onnx.export(model, dummy, os.path.join(a.out, "defect_cls.onnx"), input_names=["input"],
                      output_names=["logits"], dynamic_axes={"input": {0: "n"}, "logits": {0: "n"}},
                      opset_version=13, dynamo=False)
    torch.onnx.export(model, dummy, os.path.join(a.out, "defect_cls_b1.onnx"), input_names=["input"],
                      output_names=["logits"], opset_version=13, dynamo=False)
    sel = np.random.choice(len(Xn), min(512, len(Xn)), replace=False)
    np.save(os.path.join(a.out, "calib_nhwc.npy"), Xn[sel].astype(np.float32))
    metrics = {"val_accuracy": round(acc, 4), "macro_recall": round(float(best), 4), "per_class": per,
               "confusion_matrix": {"labels": CLASSES, "matrix": cm.tolist()},
               "train_size": int(len(Ytr)), "val_size": int(len(Yva)), "epochs": a.epochs}
    json.dump(metrics, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
