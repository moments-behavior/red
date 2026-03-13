#!/usr/bin/env python3
"""Convert JARVIS .pth checkpoints to CoreML .mlpackage format.

Reconstructs EfficientTrack PyTorch models from the JARVIS source code,
loads trained .pth weights, traces with torch.jit.trace, and converts
via coremltools' unified PyTorch converter.

Preprocessing baked into the CoreML model:
  - Input: BGR color layout (matches CVPixelBuffer from Metal pipeline)
  - Scale: 1/255.0 (normalize to [0,1])
  - No ImageNet normalization (JARVIS models expect [0,1] range)

Usage:
    conda run -n jarvis python scripts/pth_to_coreml.py \
        --jarvis_project /path/to/JARVIS/project \
        --output_dir /path/to/output

Requirements:
    pip install coremltools torch  (in jarvis conda env)
"""

import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn


class SimpleConfig:
    """Minimal config object that EfficientTrackBackbone expects."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def find_latest_pth(module_dir):
    """Find the latest *_final.pth checkpoint in a JARVIS module directory.

    Looks for: <module_dir>/Run_*/EfficientTrack-*_final.pth
    Returns the path from the latest Run_* directory, or None.
    """
    if not os.path.isdir(module_dir):
        return None
    runs = sorted(glob.glob(os.path.join(module_dir, "Run_*")))
    if not runs:
        return None
    # Search latest run first, then older runs
    for run_dir in reversed(runs):
        finals = glob.glob(os.path.join(run_dir, "EfficientTrack-*_final.pth"))
        if finals:
            return finals[0]
        # Fall back to any .pth file
        pths = glob.glob(os.path.join(run_dir, "*.pth"))
        if pths:
            return sorted(pths)[-1]
    return None


def read_jarvis_config(jarvis_project):
    """Read model configuration from JARVIS project config.yaml.

    Returns dict with keys: num_joints, center_input_size,
    keypoint_input_size, model_size, project_name.
    """
    config = {
        "num_joints": 24,
        "center_input_size": 320,
        "keypoint_input_size": 512,
        "model_size": "medium",
        "project_name": "",
    }

    # Search for config.yaml in multiple locations:
    # 1. jarvis_project/config.yaml (direct)
    # 2. jarvis_project/../config.yaml (models subdir → parent)
    # 3. jarvis_project/../../config.yaml (models/Run_xxx → grandparent)
    candidates = [
        os.path.join(jarvis_project, "config.yaml"),
        os.path.join(jarvis_project, "..", "config.yaml"),
        os.path.join(jarvis_project, "..", "..", "config.yaml"),
    ]
    cfg_path = None
    for c in candidates:
        if os.path.exists(c):
            cfg_path = os.path.realpath(c)
            break
    if cfg_path is None:
        print(f"  WARNING: config.yaml not found. Searched:")
        for c in candidates:
            print(f"    {c}")
        print(f"  Using defaults: {config}")
        return config
    print(f"  Found config: {cfg_path}")

    try:
        import yaml
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        config["project_name"] = cfg.get("PROJECT_NAME", "")

        cd_cfg = cfg.get("CENTERDETECT", {})
        config["center_input_size"] = cd_cfg.get("IMAGE_SIZE", 320)

        kd_cfg = cfg.get("KEYPOINTDETECT", {})
        config["keypoint_input_size"] = kd_cfg.get("BOUNDING_BOX_SIZE", 512)
        config["num_joints"] = kd_cfg.get("NUM_JOINTS", 24)

        # Model size from either section
        config["model_size"] = cd_cfg.get("MODEL_SIZE", kd_cfg.get("MODEL_SIZE", "medium"))

        config["_has_config"] = True
        print(f"  Config: {config['num_joints']} joints, "
              f"center={config['center_input_size']}, "
              f"keypoint={config['keypoint_input_size']}, "
              f"size={config['model_size']}")
    except Exception as e:
        print(f"  WARNING: failed to parse config.yaml: {e}")
        print(f"  Using defaults")

    return config


def infer_output_channels_from_weights(weights_path):
    """Infer the number of output channels from a checkpoint's final_conv1 weight."""
    if not os.path.exists(weights_path):
        return None
    sd = torch.load(weights_path, map_location="cpu")
    if "final_conv1.weight" in sd:
        return sd["final_conv1.weight"].shape[0]
    return None


class NormalizedModel(nn.Module):
    """Wraps a model with ImageNet normalization as the first step.

    JARVIS training applies: (pixel/255 - mean) / std
    CoreML ImageType gives us: pixel/255 (i.e., [0,1] float)
    This wrapper applies: (x - mean) / std on the [0,1] input.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        # ImageNet mean/std (RGB order — CoreML handles BGR→RGB via color_layout)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)


def build_model(mode, weights_path, config):
    """Build EfficientTrackBackbone and load weights.

    Args:
        mode: 'center' or 'keypoint'
        weights_path: path to .pth state_dict file
        config: dict from read_jarvis_config

    Returns:
        (model in eval mode, input_size)
    """
    from jarvis.efficienttrack.model import EfficientTrackBackbone

    num_joints = config["num_joints"]
    cd_input = config["center_input_size"]
    kd_input = config["keypoint_input_size"]
    model_size = config["model_size"]

    if mode == "center":
        output_channels = 1
        input_size = cd_input
    else:
        # Check actual output channels from checkpoint
        actual = infer_output_channels_from_weights(weights_path)
        if actual is not None and actual != num_joints:
            print(f"  NOTE: checkpoint has {actual} output channels, "
                  f"config says {num_joints}. Using checkpoint value.")
            output_channels = actual
            num_joints = actual
        else:
            output_channels = num_joints
        input_size = kd_input

    cfg = SimpleConfig(
        MODEL_SIZE=model_size,
        IMAGE_SIZE=cd_input,
        BOUNDING_BOX_SIZE=kd_input,
        NUM_JOINTS=num_joints,
    )

    model = EfficientTrackBackbone(cfg, model_size=model_size,
                                    output_channels=output_channels)

    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded weights: {weights_path}")
    else:
        print(f"  WARNING: weights not found: {weights_path}")

    model.eval()
    return model, input_size


def convert_to_coreml(model, input_size, output_path, model_name):
    """Convert PyTorch model to CoreML .mlpackage.

    Args:
        model: PyTorch model in eval mode
        input_size: int, spatial dimension
        output_path: str, path to write .mlpackage
        model_name: str, human-readable name

    Returns:
        (size_mb, convert_time_s)
    """
    import coremltools as ct

    print(f"  Converting {model_name} to CoreML...")

    # Trace the model directly (no normalization wrapper).
    # The EfficientTrack backbone handles [0,1] input via internal batch norm.
    # Testing confirmed: predictions are BETTER without ImageNet normalization
    # baked in, matching the original convert_onnx_to_coreml.py behavior.
    t0 = time.time()
    dummy_input = torch.randn(1, 3, input_size, input_size)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_input)
    print(f"  Traced in {time.time() - t0:.2f}s")

    # Convert with TensorType input (accepts MLMultiArray for manual preprocessing).
    # The C++ inference code handles BGRA→RGB conversion and ImageNet normalization
    # before feeding the tensor to CoreML.
    t0 = time.time()
    tensor_input = ct.TensorType(
        name="image",
        shape=(1, 3, input_size, input_size),
    )

    coreml_model = ct.convert(
        traced,
        inputs=[tensor_input],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16,
    )
    input_mode = "TensorType (normalized float tensor)"

    convert_time = time.time() - t0
    print(f"  Converted in {convert_time:.2f}s ({input_mode})")

    # Save
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    coreml_model.save(output_path)

    total_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(output_path) for f in fns
    )
    size_mb = total_size / (1024 * 1024)
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")

    return size_mb, convert_time


def main():
    parser = argparse.ArgumentParser(
        description="Convert JARVIS .pth checkpoints to CoreML .mlpackage")
    parser.add_argument("--jarvis_project", required=True,
                        help="JARVIS training project directory "
                             "(contains models/CenterDetect and models/KeypointDetect)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for .mlpackage files and model_info.json")
    parser.add_argument("--jarvis_root", default=os.path.expanduser("~/src/JARVIS-HybridNet"),
                        help="Path to JARVIS-HybridNet source (default: ~/src/JARVIS-HybridNet)")
    parser.add_argument("--keypoint_input_size", type=int, default=0,
                        help="Override keypoint detect input size (0 = use config.yaml value)")
    args = parser.parse_args()

    # Add JARVIS to path
    if os.path.isdir(args.jarvis_root):
        sys.path.insert(0, args.jarvis_root)
        print(f"JARVIS source: {args.jarvis_root}")
    else:
        print(f"ERROR: JARVIS source not found at {args.jarvis_root}")
        print(f"Install JARVIS-HybridNet or specify --jarvis_root")
        sys.exit(1)

    # Verify imports
    try:
        from jarvis.efficienttrack.model import EfficientTrackBackbone
    except ImportError as e:
        print(f"ERROR: Cannot import JARVIS: {e}")
        print(f"Make sure JARVIS-HybridNet is at {args.jarvis_root}")
        sys.exit(1)

    import coremltools as ct
    print(f"coremltools: {ct.__version__}, torch: {torch.__version__}")

    # Resolve paths
    models_dir = os.path.join(args.jarvis_project, "models")
    if not os.path.isdir(models_dir):
        # Maybe jarvis_project already points to the models/ dir
        if (os.path.isdir(os.path.join(args.jarvis_project, "CenterDetect")) and
            os.path.isdir(os.path.join(args.jarvis_project, "KeypointDetect"))):
            models_dir = args.jarvis_project
        else:
            print(f"ERROR: Cannot find models directory.")
            print(f"  Tried: {models_dir}")
            print(f"  Expected CenterDetect/ and KeypointDetect/ subdirectories.")
            sys.exit(1)

    # Find checkpoints
    cd_pth = find_latest_pth(os.path.join(models_dir, "CenterDetect"))
    kd_pth = find_latest_pth(os.path.join(models_dir, "KeypointDetect"))

    if not cd_pth:
        print(f"ERROR: No CenterDetect .pth checkpoint found in {models_dir}/CenterDetect/")
        sys.exit(1)
    if not kd_pth:
        print(f"ERROR: No KeypointDetect .pth checkpoint found in {models_dir}/KeypointDetect/")
        sys.exit(1)

    print(f"CenterDetect checkpoint:  {cd_pth}")
    print(f"KeypointDetect checkpoint: {kd_pth}")

    # Read config
    config = read_jarvis_config(args.jarvis_project)

    # Override keypoint input size if specified
    if args.keypoint_input_size > 0:
        print(f"  Overriding keypoint_input_size: {config['keypoint_input_size']} -> {args.keypoint_input_size}")
        config["keypoint_input_size"] = args.keypoint_input_size

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Convert CenterDetect
    print(f"\n--- CenterDetect ---")
    cd_model, cd_size = build_model("center", cd_pth, config)
    cd_path = os.path.join(args.output_dir, "center_detect.mlpackage")
    cd_mb, cd_time = convert_to_coreml(cd_model, cd_size, cd_path, "CenterDetect")

    # Convert KeypointDetect
    print(f"\n--- KeypointDetect ---")
    kd_model, kd_size = build_model("keypoint", kd_pth, config)
    kd_path = os.path.join(args.output_dir, "keypoint_detect.mlpackage")
    kd_mb, kd_time = convert_to_coreml(kd_model, kd_size, kd_path, "KeypointDetect")

    # Infer actual num_joints from the keypoint model
    actual_joints = infer_output_channels_from_weights(kd_pth)
    num_joints = actual_joints if actual_joints else config["num_joints"]

    # Write/update model_info.json
    meta_path = os.path.join(args.output_dir, "model_info.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
    else:
        metadata = {
            "center_detect": {},
            "keypoint_detect": {},
        }

    metadata["center_detect"]["mlpackage_file"] = "center_detect.mlpackage"
    metadata["center_detect"]["mlpackage_size_mb"] = round(cd_mb, 1)
    # Preserve existing input_size if no config was found (it may have been
    # set correctly from an earlier conversion or manual configuration)
    if "input_size" not in metadata["center_detect"] or config.get("_has_config"):
        metadata["center_detect"]["input_size"] = config["center_input_size"]
    metadata["keypoint_detect"]["mlpackage_file"] = "keypoint_detect.mlpackage"
    metadata["keypoint_detect"]["mlpackage_size_mb"] = round(kd_mb, 1)
    if "input_size" not in metadata["keypoint_detect"] or config.get("_has_config"):
        metadata["keypoint_detect"]["input_size"] = config["keypoint_input_size"]
    metadata["keypoint_detect"]["num_joints"] = num_joints
    metadata["model_size"] = config["model_size"]
    if config["project_name"]:
        metadata["project_name"] = config["project_name"]
    metadata["coreml_info"] = {
        "format": "mlprogram (.mlpackage)",
        "precision": "float16",
        "minimum_deployment_target": "macOS13",
        "input_color_layout": "BGR",
        "input_scale": 1.0 / 255.0,
        "input_bias": [0.0, 0.0, 0.0],
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nUpdated: {meta_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"CONVERSION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  CenterDetect:   {cd_mb:.1f} MB ({cd_time:.1f}s)")
    print(f"  KeypointDetect: {kd_mb:.1f} MB ({kd_time:.1f}s)")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
