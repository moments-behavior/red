#!/usr/bin/env python3
"""
convert_pth_to_trt.py — Convert JARVIS-HybridNet .pth checkpoints to TensorRT engines

Two-stage pipeline:
  1. .pth → ONNX (opset 17, dynamic batch size)
  2. ONNX → TensorRT .engine (FP16, optimized for RTX 3090)

Usage:
  # Full pipeline: .pth → ONNX → TensorRT
  python convert_pth_to_trt.py --jarvis_project /path/to/jarvis/project --output_dir ./engines

  # ONNX only (skip TensorRT):
  python convert_pth_to_trt.py --jarvis_project /path/to/project --output_dir ./onnx --onnx_only

  # Convert existing ONNX files to TensorRT:
  python convert_pth_to_trt.py --onnx_dir ./onnx --output_dir ./engines --trt_only

  # Use trtexec CLI instead of Python API:
  python convert_pth_to_trt.py --jarvis_project /path/to/project --output_dir ./engines --use_trtexec

Requirements:
  - PyTorch (for .pth loading and ONNX export)
  - JARVIS-HybridNet (pip install jarvis-hybridnet, or available in PYTHONPATH)
  - For TensorRT conversion: either tensorrt Python package or trtexec CLI

Expected model I/O shapes:
  CenterDetect:
    Input:  "image"        — [batch, 3, 320, 320]   float32 (ImageNet-normalized)
    Output: "heatmap_low"  — [batch, 1, 80, 80]     float32 (stride-4 center heatmap)
    Output: "heatmap_high" — [batch, 1, 160, 160]   float32 (stride-2 center heatmap, used for argmax)

  KeypointDetect:
    Input:  "image"        — [batch, 3, 704, 704]   float32 (ImageNet-normalized)
    Output: "heatmap_low"  — [batch, N, 176, 176]   float32 (stride-4 keypoint heatmaps)
    Output: "heatmap_high" — [batch, N, 352, 352]   float32 (stride-2 keypoint heatmaps, used for argmax)

  Where N = number of joints (e.g., 24 for a typical skeleton).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Stage 1: .pth → ONNX
# ---------------------------------------------------------------------------

def find_latest_checkpoint(module_dir: Path) -> Path | None:
    """Find the latest 'final' .pth checkpoint in a JARVIS module directory.

    JARVIS organizes checkpoints as:
      <module>/Run_XXXX/<model>_final_weights.pth
    We pick the highest-numbered Run_ directory.
    """
    if not module_dir.is_dir():
        return None
    runs = sorted([d for d in module_dir.iterdir()
                   if d.is_dir() and d.name.startswith("Run_")])
    if not runs:
        return None
    for pth in sorted(runs[-1].glob("*final*.pth"), reverse=True):
        return pth
    # Fallback: any .pth in the latest run
    for pth in sorted(runs[-1].glob("*.pth"), reverse=True):
        return pth
    return None


def load_jarvis_config(project_dir: Path) -> dict:
    """Load JARVIS project config.yaml for model architecture info."""
    config_path = project_dir / "config.yaml"
    if not config_path.exists():
        print(f"Warning: config.yaml not found in {project_dir}")
        return {}
    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    except ImportError:
        print("Warning: PyYAML not installed, cannot parse config.yaml")
        return {}


def export_onnx(model, dummy_input, output_path: str, input_names: list,
                output_names: list, opset: int = 17):
    """Export a PyTorch model to ONNX with dynamic batch size."""
    import torch
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            input_names[0]: {0: "batch_size"},
            **{name: {0: "batch_size"} for name in output_names},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"  Exported ONNX: {output_path}")


def convert_pth_to_onnx(project_dir: Path, output_dir: Path) -> dict:
    """Convert JARVIS .pth checkpoints to ONNX format.

    Returns dict with paths and metadata for model_info.json.
    """
    import torch

    # Try to import JARVIS model classes
    try:
        from jarvis.efficienttrack.efficienttrack import EfficientTrack
    except ImportError:
        print("Error: JARVIS-HybridNet not found. Install with:")
        print("  pip install jarvis-hybridnet")
        print("  # or: conda install -c conda-forge jarvis-hybridnet")
        sys.exit(1)

    config = load_jarvis_config(project_dir)
    models_dir = project_dir / "models"

    # Resolve checkpoint paths
    cd_dir = models_dir / "CenterDetect"
    kd_dir = models_dir / "KeypointDetect"
    cd_pth = find_latest_checkpoint(cd_dir)
    kd_pth = find_latest_checkpoint(kd_dir)

    if not cd_pth:
        print(f"Error: No CenterDetect checkpoint found in {cd_dir}")
        sys.exit(1)
    if not kd_pth:
        print(f"Error: No KeypointDetect checkpoint found in {kd_dir}")
        sys.exit(1)

    print(f"CenterDetect checkpoint:  {cd_pth}")
    print(f"KeypointDetect checkpoint: {kd_pth}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract model parameters from config
    center_input_size = 320
    keypoint_input_size = 704
    num_joints = 24
    model_size = "medium"

    if config:
        cd_cfg = config.get("CenterDetect", {})
        kd_cfg = config.get("KeypointDetect", {})
        center_input_size = cd_cfg.get("input_size", center_input_size)
        keypoint_input_size = kd_cfg.get("input_size", keypoint_input_size)
        num_joints = kd_cfg.get("num_joints", config.get("num_joints", num_joints))
        model_size = cd_cfg.get("model_size", model_size)

    # Load and export CenterDetect
    print(f"\nExporting CenterDetect ({center_input_size}x{center_input_size})...")
    cd_model = EfficientTrack("CenterDetect", str(project_dir))
    cd_model.load_weights(str(cd_pth))
    cd_model.model.eval()

    dummy_cd = torch.randn(1, 3, center_input_size, center_input_size)
    cd_onnx = str(output_dir / "center_detect.onnx")
    export_onnx(cd_model.model, dummy_cd, cd_onnx,
                input_names=["image"],
                output_names=["heatmap_low", "heatmap_high"])

    # Load and export KeypointDetect
    print(f"\nExporting KeypointDetect ({keypoint_input_size}x{keypoint_input_size}, "
          f"{num_joints} joints)...")
    kd_model = EfficientTrack("KeypointDetect", str(project_dir))
    kd_model.load_weights(str(kd_pth))
    kd_model.model.eval()

    dummy_kd = torch.randn(1, 3, keypoint_input_size, keypoint_input_size)
    kd_onnx = str(output_dir / "keypoint_detect.onnx")
    export_onnx(kd_model.model, dummy_kd, kd_onnx,
                input_names=["image"],
                output_names=["heatmap_low", "heatmap_high"])

    # Write model_info.json (consumed by RED's parse_jarvis_model_info)
    project_name = config.get("project_name", project_dir.name)
    info = {
        "project_name": project_name,
        "center_detect": {
            "input_size": center_input_size,
            "model_size": model_size,
        },
        "keypoint_detect": {
            "input_size": keypoint_input_size,
            "num_joints": num_joints,
        },
        "tensorrt_info": {
            "precision": "fp16",
            "note": "ImageNet normalization applied in preprocessing (not baked into model)",
        },
    }
    info_path = output_dir / "model_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Wrote {info_path}")

    return {
        "center_onnx": cd_onnx,
        "keypoint_onnx": kd_onnx,
        "center_input_size": center_input_size,
        "keypoint_input_size": keypoint_input_size,
        "num_joints": num_joints,
    }


# ---------------------------------------------------------------------------
# Stage 2: ONNX → TensorRT .engine
# ---------------------------------------------------------------------------

def convert_onnx_to_trt_python(onnx_path: str, engine_path: str,
                                fp16: bool = True,
                                max_batch: int = 1,
                                workspace_gb: float = 4.0):
    """Convert ONNX to TensorRT engine using the tensorrt Python API."""
    try:
        import tensorrt as trt
    except ImportError:
        print("Error: tensorrt Python package not found.")
        print("Install with: pip install tensorrt")
        print("Or use --use_trtexec to use the trtexec CLI instead.")
        sys.exit(1)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    print(f"  Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Parse error: {parser.get_error(i)}")
            sys.exit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                  int(workspace_gb * (1 << 30)))

    if fp16 and builder.platform_has_fast_fp16:
        print("  Enabling FP16 precision")
        config.set_flag(trt.BuilderFlag.FP16)
    elif fp16:
        print("  Warning: FP16 not supported on this GPU, using FP32")

    # Set optimization profile for dynamic batch
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_shape = input_tensor.shape  # e.g., [-1, 3, 320, 320]
    min_shape = (1,) + tuple(input_shape[1:])
    opt_shape = (1,) + tuple(input_shape[1:])
    max_shape = (max_batch,) + tuple(input_shape[1:])
    profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    print(f"  Building engine (this may take a few minutes)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("  Error: engine build failed")
        sys.exit(1)

    with open(engine_path, "wb") as f:
        f.write(serialized)
    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"  Wrote engine: {engine_path} ({size_mb:.1f} MB)")


def convert_onnx_to_trt_trtexec(onnx_path: str, engine_path: str,
                                 fp16: bool = True):
    """Convert ONNX to TensorRT engine using the trtexec CLI."""
    trtexec = shutil.which("trtexec")
    if not trtexec:
        # Common install locations on Windows
        for candidate in [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT\bin\trtexec.exe",
            r"C:\TensorRT\bin\trtexec.exe",
        ]:
            if os.path.exists(candidate):
                trtexec = candidate
                break

    if not trtexec:
        print("Error: trtexec not found in PATH.")
        print("Install TensorRT and add its bin/ directory to PATH,")
        print("or use the Python API (omit --use_trtexec).")
        sys.exit(1)

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--workspace=4096",
    ]
    if fp16:
        cmd.append("--fp16")

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  trtexec failed (exit {result.returncode}):")
        print(result.stderr[-500:] if result.stderr else result.stdout[-500:])
        sys.exit(1)

    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"  Wrote engine: {engine_path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert JARVIS .pth checkpoints to TensorRT engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--jarvis_project", type=str,
                        help="Path to JARVIS project directory (contains config.yaml, models/)")
    parser.add_argument("--onnx_dir", type=str,
                        help="Path to directory with existing ONNX files (for --trt_only)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for .onnx and/or .engine files")
    parser.add_argument("--onnx_only", action="store_true",
                        help="Only export to ONNX, skip TensorRT conversion")
    parser.add_argument("--trt_only", action="store_true",
                        help="Only convert ONNX→TensorRT (requires --onnx_dir)")
    parser.add_argument("--use_trtexec", action="store_true",
                        help="Use trtexec CLI instead of tensorrt Python API")
    parser.add_argument("--fp32", action="store_true",
                        help="Use FP32 precision instead of FP16")
    parser.add_argument("--max_batch", type=int, default=1,
                        help="Max batch size for TensorRT engine (default: 1)")
    parser.add_argument("--workspace_gb", type=float, default=4.0,
                        help="TensorRT workspace size in GB (default: 4.0)")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fp16 = not args.fp32

    # Validate arguments
    if args.trt_only and not args.onnx_dir:
        parser.error("--trt_only requires --onnx_dir")
    if not args.trt_only and not args.jarvis_project:
        parser.error("--jarvis_project is required (unless using --trt_only)")

    # Stage 1: .pth → ONNX
    if args.trt_only:
        onnx_dir = Path(args.onnx_dir)
        cd_onnx = str(onnx_dir / "center_detect.onnx")
        kd_onnx = str(onnx_dir / "keypoint_detect.onnx")
        if not os.path.exists(cd_onnx) or not os.path.exists(kd_onnx):
            print(f"Error: ONNX files not found in {onnx_dir}")
            sys.exit(1)
        # Copy model_info.json if present
        info_src = onnx_dir / "model_info.json"
        info_dst = output_dir / "model_info.json"
        if info_src.exists() and not info_dst.exists():
            shutil.copy2(info_src, info_dst)
    else:
        project_dir = Path(args.jarvis_project)
        if not project_dir.is_dir():
            print(f"Error: Project directory not found: {project_dir}")
            sys.exit(1)
        result = convert_pth_to_onnx(project_dir, output_dir)
        cd_onnx = result["center_onnx"]
        kd_onnx = result["keypoint_onnx"]

    if args.onnx_only:
        print("\nDone (ONNX only). To build TensorRT engines, run again with --trt_only.")
        return

    # Stage 2: ONNX → TensorRT
    print(f"\n--- Converting to TensorRT ({'FP16' if fp16 else 'FP32'}) ---")

    convert_fn = convert_onnx_to_trt_trtexec if args.use_trtexec else convert_onnx_to_trt_python

    cd_engine = str(output_dir / "center_detect.engine")
    print(f"\nBuilding CenterDetect engine...")
    convert_fn(cd_onnx, cd_engine, fp16=fp16,
               **({"max_batch": args.max_batch, "workspace_gb": args.workspace_gb}
                  if not args.use_trtexec else {}))

    kd_engine = str(output_dir / "keypoint_detect.engine")
    print(f"\nBuilding KeypointDetect engine...")
    convert_fn(kd_onnx, kd_engine, fp16=fp16,
               **({"max_batch": args.max_batch, "workspace_gb": args.workspace_gb}
                  if not args.use_trtexec else {}))

    print(f"\nDone. Engine files written to {output_dir}/")
    print(f"  center_detect.engine")
    print(f"  keypoint_detect.engine")
    print(f"\nTo use in RED, place these files (along with model_info.json) in your")
    print(f"project's jarvis_models/<model_name>/ directory.")


if __name__ == "__main__":
    main()
