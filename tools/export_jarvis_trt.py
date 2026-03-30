#!/usr/bin/env python3
"""
Export JARVIS CenterDetect + KeypointDetect models to TensorRT engines.

Usage:
    python3 export_jarvis_trt.py --project mouseHybrid24 \
        --output /path/to/engines/

Outputs:
    centerDetect.engine
    keypointDetect.engine
"""

import argparse
import os
import sys

import torch
import torch_tensorrt

# Add JARVIS-HybridNet to path if needed
JARVIS_ROOT = os.environ.get("JARVIS_ROOT",
                             os.path.join(os.path.dirname(__file__), "..", "..",
                                          "JARVIS-HybridNet"))
if os.path.isdir(JARVIS_ROOT):
    sys.path.insert(0, JARVIS_ROOT)


def load_jarvis_models(project_name: str):
    """Return (center_model, kp_model) as eval-mode torch.nn.Modules."""
    try:
        from jarvis.prediction.jarvis2D import JARVISPredictor2D  # type: ignore
        predictor = JARVISPredictor2D(project_name)
        center_model = predictor.centerDetect_net.eval()
        kp_model     = predictor.keypointDetect_net.eval()
        return center_model, kp_model
    except ImportError:
        print("[ERROR] Cannot import JARVIS. Set JARVIS_ROOT or add it to PYTHONPATH.")
        sys.exit(1)


def compile_to_trt(model, example_input, output_path: str, label: str):
    """Compile a PyTorch model to a TensorRT engine and save it."""
    print(f"[INFO] Compiling {label} with torch_tensorrt …")
    trt_model = torch_tensorrt.compile(
        model,
        inputs=[torch_tensorrt.Input(example_input.shape, dtype=torch.float32)],
        enabled_precisions={torch.float16},
        truncate_long_and_double=True,
    )

    # Serialize the TRT engine via torch_tensorrt
    torch_tensorrt.save(trt_model, output_path, inputs=[example_input])
    print(f"[INFO] Saved {label} engine → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export JARVIS models to TensorRT")
    parser.add_argument("--project",  required=True,
                        help="JARVIS project name (e.g. mouseHybrid24)")
    parser.add_argument("--output",   required=True,
                        help="Output directory for .engine files")
    parser.add_argument("--device",   default="cuda:0",
                        help="CUDA device (default: cuda:0)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device)

    center_model, kp_model = load_jarvis_models(args.project)
    center_model.to(device)
    kp_model.to(device)

    # Example inputs matching model specs
    center_example = torch.randn(1, 3, 320, 320, device=device)
    kp_example     = torch.randn(1, 3, 832, 832, device=device)

    center_out = os.path.join(args.output, "centerDetect.engine")
    kp_out     = os.path.join(args.output, "keypointDetect.engine")

    compile_to_trt(center_model, center_example, center_out, "CenterDetect")
    compile_to_trt(kp_model,     kp_example,     kp_out,    "KeypointDetect")

    print("\n[DONE] Engine files written to:", args.output)
    print("  centerDetect.engine")
    print("  keypointDetect.engine")
    print("\nNote: TensorRT engines are GPU-architecture-specific.")
    print("Re-export on every machine where RED is used.")


if __name__ == "__main__":
    main()
