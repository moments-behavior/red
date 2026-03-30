#!/usr/bin/env python3
"""
Convert already-traced JARVIS TorchScript .pt files to TensorRT .engine files.

Usage (from jarvis conda env):
    conda run -n jarvis python3 compile_jarvis_trt.py

Steps:
    1. Load the CPU-traced .pt files
    2. Export to ONNX on CUDA
    3. Call trtexec to compile to .engine (FP16)

Outputs (in the same directory as the .pt files):
    centerDetect.engine
    keypointDetect.engine
"""

import os
import subprocess
import sys

import torch

JARVIS_ROOT = "/home/user/src/JARVIS-HybridNet"
PROJECT_NAME = "mouseJan30"
PT_DIR = os.path.join(JARVIS_ROOT, "projects", PROJECT_NAME, "trt-models", "predict2D")
TRTEXEC = "/home/user/nvidia/TensorRT/targets/x86_64-linux-gnu/bin/trtexec"

CENTER_PT   = os.path.join(PT_DIR, "centerDetect.pt")
KP_PT       = os.path.join(PT_DIR, "keypointDetect.pt")
CENTER_ONNX = os.path.join(PT_DIR, "centerDetect.onnx")
KP_ONNX     = os.path.join(PT_DIR, "keypointDetect.onnx")
CENTER_ENG  = os.path.join(PT_DIR, "centerDetect.engine")
KP_ENG      = os.path.join(PT_DIR, "keypointDetect.engine")

CENTER_SIZE = 320
KP_SIZE     = 704   # mouseJan30 KEYPOINTDETECT.BOUNDING_BOX_SIZE


def export_onnx(pt_path, onnx_path, input_shape, label):
    import onnx
    from onnx import helper

    print(f"\n[INFO] Exporting {label} to ONNX …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(pt_path, map_location=device)
    model.eval()

    example = torch.randn(*input_shape, device=device)
    tmp_onnx = onnx_path + ".tmp"
    with torch.no_grad():
        torch.onnx.export(
            model,
            example,
            tmp_onnx,
            input_names=["input"],
            opset_version=17,
            do_constant_folding=True,
        )

    # The JARVIS model returns a tuple; keep only the first output (heatmap)
    proto = onnx.load(tmp_onnx)
    if len(proto.graph.output) > 1:
        first_out_name = proto.graph.output[0].name
        # Remove all outputs except the first
        while len(proto.graph.output) > 1:
            del proto.graph.output[-1]
        # Rename first output to "output" for consistency
        proto.graph.output[0].name = "output"
        # Rename the corresponding node's output
        for node in proto.graph.node:
            for i, o in enumerate(node.output):
                if o == first_out_name:
                    node.output[i] = "output"
    import os
    os.remove(tmp_onnx)
    onnx.save(proto, onnx_path)
    print(f"[INFO] Saved ONNX ({len(proto.graph.output)} output(s)) → {onnx_path}")


def compile_trt(onnx_path, engine_path, label):
    print(f"\n[INFO] Compiling {label} with trtexec (FP16) …")
    cmd = [
        TRTEXEC,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        "--noDataTransfers",
    ]
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"[ERROR] trtexec failed for {label}")
        sys.exit(1)
    print(f"[INFO] Saved engine → {engine_path}")


if __name__ == "__main__":
    os.makedirs(PT_DIR, exist_ok=True)

    # Check .pt files exist
    for p in [CENTER_PT, KP_PT]:
        if not os.path.exists(p):
            print(f"[ERROR] Missing: {p}")
            print("  Run tools/trace_jarvis_torchscript.py first (in jarvis env)")
            sys.exit(1)

    # Check trtexec
    if not os.path.isfile(TRTEXEC):
        print(f"[ERROR] trtexec not found at: {TRTEXEC}")
        sys.exit(1)

    export_onnx(CENTER_PT, CENTER_ONNX, (1, 3, CENTER_SIZE, CENTER_SIZE), "CenterDetect")
    export_onnx(KP_PT,     KP_ONNX,     (1, 3, KP_SIZE,     KP_SIZE),     "KeypointDetect")

    compile_trt(CENTER_ONNX, CENTER_ENG, "CenterDetect")
    compile_trt(KP_ONNX,     KP_ENG,    "KeypointDetect")

    print(f"\n[DONE] TensorRT engines written to: {PT_DIR}")
    print(f"  {CENTER_ENG}")
    print(f"  {KP_ENG}")
    print("\nIn RED → Load JARVIS Models, select the .engine files for GPU inference.")
    print("Note: engines are GPU-architecture-specific (compiled for RTX 3090 / sm_86).")
