#!/usr/bin/env python3
"""OPTIONAL - only if a Raspberry Pi AI HAT+ / AI HAT+ 2 is fitted.

Compiles models/defect_cls_b1.onnx to a Hailo .hef with the Hailo Dataflow
Compiler (DFC).  Run on an x86-64 Ubuntu machine with the DFC installed
(Hailo Developer Zone).  The DFC version must match the HailoRT version on the Pi.

  python scripts/compile_hef.py --arch hailo8l      # AI HAT+ 13 TOPS
  python scripts/compile_hef.py --arch hailo8       # AI HAT+ 26 TOPS
  python scripts/compile_hef.py --arch hailo10h     # AI HAT+ 2

Then copy models/defect_cls.hef to the Pi; run_station.sh picks it up and the
vision pipeline runs on the NPU via HailoRT (Llama stays on the CPU via ExecuTorch).
"""
import argparse

import numpy as np
from hailo_sdk_client import ClientRunner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="models/defect_cls_b1.onnx")
    ap.add_argument("--calib", default="models/calib_nhwc.npy")
    ap.add_argument("--arch", default="hailo8l")
    ap.add_argument("--out", default="models/defect_cls.hef")
    a = ap.parse_args()
    runner = ClientRunner(hw_arch=a.arch)
    runner.translate_onnx_model(a.onnx, "defect_cls", start_node_names=["input"], end_node_names=["logits"],
                                net_input_shapes={"input": [1, 3, 64, 64]})
    calib = np.load(a.calib).astype(np.float32)          # NHWC, 0-255 (normalisation is inside the model)
    runner.optimize(calib)
    hef = runner.compile()
    with open(a.out, "wb") as f:
        f.write(hef)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
