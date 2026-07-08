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

import torch


class SimpleConfig:
    """Minimal config object that EfficientTrackBackbone / V2VNet expect."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


import torch.nn as nn


class ManualInstanceNorm3d(nn.Module):
    """CoreML/ORT-safe InstanceNorm3d (explicit mean/var; safe on zero-variance
    channels). Mirrors scripts/export_jarvis_onnx.py."""
    def __init__(self, num_features, eps=1e-5, affine=False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        self.num_features = num_features
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        var = x.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x = x * self.weight.view(1, -1, 1, 1, 1) + self.bias.view(1, -1, 1, 1, 1)
        return x


def replace_instance_norm_3d(model):
    for name, module in model.named_children():
        if isinstance(module, nn.InstanceNorm3d):
            r = ManualInstanceNorm3d(module.num_features, eps=module.eps,
                                     affine=module.affine)
            if module.affine and module.weight is not None:
                r.weight.data = module.weight.data.clone()
                r.bias.data = module.bias.data.clone()
            setattr(model, name, r)
        else:
            replace_instance_norm_3d(module)


def find_latest_hybridnet_pth(models_dir):
    """Find the latest HybridNet-*_final.pth under <models_dir>/HybridNet/Run_*/."""
    hn_dir = os.path.join(models_dir, "HybridNet")
    if not os.path.isdir(hn_dir):
        return None
    runs = sorted(glob.glob(os.path.join(hn_dir, "Run_*")))
    for run_dir in reversed(runs):
        finals = glob.glob(os.path.join(run_dir, "HybridNet-*_final.pth"))
        if finals:
            return finals[0]
        pths = glob.glob(os.path.join(run_dir, "*.pth"))
        if pths:
            return sorted(pths)[-1]
    return None


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
        "num_cameras": 16,
        "roi_cube_size": 200,
        "grid_spacing": 2,
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

        hn_cfg = cfg.get("HYBRIDNET", {})
        config["num_cameras"] = hn_cfg.get("NUM_CAMERAS", 16)
        config["roi_cube_size"] = hn_cfg.get("ROI_CUBE_SIZE", 200)
        config["grid_spacing"] = hn_cfg.get("GRID_SPACING", 2)

        config["_has_config"] = True
        print(f"  Config: {config['num_joints']} joints, "
              f"center={config['center_input_size']}, "
              f"keypoint={config['keypoint_input_size']}, "
              f"size={config['model_size']}")
    except Exception as e:
        print(f"  WARNING: failed to parse config.yaml: {e}")
        print(f"  Using defaults")

    return config


def extract_hn_efftrack_state(hn_pth):
    """Extract the effTrack.* sub-state_dict from a HybridNet checkpoint.

    CRITICAL: the 2D keypoint detector used by the volumetric 3D pipeline is the
    effTrack baked INTO the HybridNet checkpoint (jointly trained with V2VNet),
    NOT the standalone KeypointDetect checkpoint. V2VNet only produces accurate,
    confident output when fed heatmaps from ITS OWN effTrack; feeding it heatmaps
    from the standalone KeypointDetect model degrades accuracy from ~2mm to ~20mm
    and confidence from ~0.9 to ~0.28 (verified on the vertical_cyl dataset).
    """
    state = torch.load(hn_pth, map_location="cpu", weights_only=True)
    eff = {k[len("effTrack."):]: v for k, v in state.items()
           if k.startswith("effTrack.")}
    return eff if eff else None


def infer_output_channels_from_weights(weights):
    """Infer the number of output channels from a checkpoint's final_conv1 weight.

    `weights` may be a path to a .pth file or an already-loaded state_dict.
    """
    if isinstance(weights, dict):
        sd = weights
    elif os.path.exists(weights):
        sd = torch.load(weights, map_location="cpu")
    else:
        return None
    if "final_conv1.weight" in sd:
        return sd["final_conv1.weight"].shape[0]
    return None


def build_model(mode, weights_path, config):
    """Build EfficientTrackBackbone and load weights.

    Args:
        mode: 'center' or 'keypoint'
        weights_path: path to .pth state_dict file, OR an in-memory state_dict
            (used to source the keypoint detector from a HybridNet checkpoint's
            internal effTrack).
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

    state_dict = None
    if isinstance(weights_path, dict):
        state_dict = weights_path
    elif os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu")
    if state_dict is not None:
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys or result.unexpected_keys:
            print(f"  WARNING: {len(result.missing_keys)} missing, "
                  f"{len(result.unexpected_keys)} unexpected keys")
            for k in result.missing_keys[:5]:
                print(f"    missing: {k}")
            for k in result.unexpected_keys[:5]:
                print(f"    unexpected: {k}")
        src = "<in-memory state_dict>" if isinstance(weights_path, dict) else weights_path
        print(f"  Loaded weights: {src}")
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
    try:
        with torch.no_grad():
            traced = torch.jit.trace(model, dummy_input)
    except Exception as e:
        print(f"  ERROR: torch.jit.trace failed: {e}")
        sys.exit(1)
    print(f"  Traced in {time.time() - t0:.2f}s")

    # Convert with TensorType input (accepts MLMultiArray for manual preprocessing).
    # The C++ inference code handles BGRA→RGB conversion and ImageNet normalization
    # before feeding the tensor to CoreML.
    t0 = time.time()
    tensor_input = ct.TensorType(
        name="image",
        shape=(1, 3, input_size, input_size),
    )

    try:
        coreml_model = ct.convert(
            traced,
            inputs=[tensor_input],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS13,
            compute_precision=ct.precision.FLOAT16,
        )
    except Exception as e:
        print(f"  ERROR: coremltools conversion failed: {e}")
        sys.exit(1)
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


def convert_v2vnet_to_coreml(hn_pth_path, config, output_path):
    """Convert the V2VNet (3D CNN inside HybridNet) to CoreML .mlpackage.

    V2VNet is the only learned network in the 3D 'hybrid3d' stage; the
    reprojection (voxel grid) is deterministic geometry done in host C++.
    Input grid = ROI_CUBE_SIZE/GRID_SPACING (net downsamples to half internally).

    Returns (size_mb, convert_time_s, grid_in, grid_out) or None on failure.
    """
    import coremltools as ct
    from jarvis.hybridnet.v2vnet import V2VNet

    nj = config["num_joints"]
    grid_in = int(config["roi_cube_size"] / config["grid_spacing"])   # e.g. 100
    grid_out = grid_in // 2                                            # e.g. 50

    print(f"  Building V2VNet({nj},{nj}), grid_in={grid_in} -> grid_out={grid_out}")
    net = V2VNet(nj, nj)
    state = torch.load(hn_pth_path, map_location="cpu", weights_only=True)
    v2v = {k[len("v2vNet."):]: v for k, v in state.items()
           if k.startswith("v2vNet.")}
    if not v2v:
        print(f"  ERROR: no v2vNet.* weights in {hn_pth_path}")
        return None
    result = net.load_state_dict(v2v, strict=False)
    if result.missing_keys or result.unexpected_keys:
        print(f"  WARNING: {len(result.missing_keys)} missing, "
              f"{len(result.unexpected_keys)} unexpected V2VNet keys")
    replace_instance_norm_3d(net)
    net.eval()

    t0 = time.time()
    dummy = torch.randn(1, nj, grid_in, grid_in, grid_in)
    try:
        with torch.no_grad():
            traced = torch.jit.trace(net, dummy)
    except Exception as e:
        print(f"  ERROR: V2VNet trace failed: {e}")
        return None
    print(f"  Traced in {time.time() - t0:.2f}s")

    t0 = time.time()
    try:
        ml = ct.convert(
            traced,
            inputs=[ct.TensorType(name="vox_in", shape=(1, nj, grid_in, grid_in, grid_in))],
            outputs=[ct.TensorType(name="vox_out")],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS13,
            compute_precision=ct.precision.FLOAT16,
        )
    except Exception as e:
        print(f"  ERROR: V2VNet coreml conversion failed: {e}")
        return None
    convert_time = time.time() - t0
    print(f"  Converted in {convert_time:.2f}s")

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    ml.save(output_path)
    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fns in os.walk(output_path) for f in fns)
    size_mb = total / (1024 * 1024)
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")
    return size_mb, convert_time, grid_in, grid_out


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
    parser.add_argument("--center_input_size", type=int, default=0,
                        help="Override center detect input size (0 = use config.yaml value)")
    parser.add_argument("--num_joints", type=int, default=0,
                        help="Override number of joints (0 = use config.yaml value)")
    parser.add_argument("--world_scale", type=float, default=1.0,
                        help="Scale applied to the hybridnet roi_cube_size and grid_spacing "
                             "written to model_info.json, converting the model's TRAINING world "
                             "units into the INFERENCE calibration's units. The host reprojection "
                             "builds the voxel grid in the calibration's units, so these MUST match. "
                             "E.g. a fly model trained in 0.1mm units used with a millimetre DLT "
                             "calibration -> --world_scale 0.1 (48/1 -> 4.8/0.1). grid_in/grid_out "
                             "are dimensionless tensor sizes and are NOT scaled. Default 1.0.")
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

    # Resolve paths — support three layouts:
    #   1. JARVIS project: <dir>/models/CenterDetect/Run_*/*.pth
    #   2. models subdir:  <dir>/CenterDetect/Run_*/*.pth
    #   3. flat layout:    <dir>/center_detect.pth + keypoint_detect.pth
    cd_pth = None
    kd_pth = None

    # Try flat layout first (RED project jarvis_models/<name>/)
    flat_cd = os.path.join(args.jarvis_project, "center_detect.pth")
    flat_kd = os.path.join(args.jarvis_project, "keypoint_detect.pth")
    if os.path.exists(flat_cd) and os.path.exists(flat_kd):
        cd_pth = flat_cd
        kd_pth = flat_kd
        print(f"  Using flat layout: {args.jarvis_project}")
    else:
        # Try JARVIS project layout
        models_dir = os.path.join(args.jarvis_project, "models")
        if not os.path.isdir(models_dir):
            if (os.path.isdir(os.path.join(args.jarvis_project, "CenterDetect")) and
                os.path.isdir(os.path.join(args.jarvis_project, "KeypointDetect"))):
                models_dir = args.jarvis_project
            else:
                print(f"ERROR: Cannot find models directory.")
                print(f"  Tried: {models_dir}")
                print(f"  Also tried flat layout: {flat_cd}")
                print(f"  Expected CenterDetect/ and KeypointDetect/ subdirectories,")
                print(f"  or center_detect.pth + keypoint_detect.pth files.")
                sys.exit(1)

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

    # Override config values if specified on command line
    if args.center_input_size > 0:
        print(f"  Overriding center_input_size: {config['center_input_size']} -> {args.center_input_size}")
        config["center_input_size"] = args.center_input_size
    if args.keypoint_input_size > 0:
        print(f"  Overriding keypoint_input_size: {config['keypoint_input_size']} -> {args.keypoint_input_size}")
        config["keypoint_input_size"] = args.keypoint_input_size
    if args.num_joints > 0:
        print(f"  Overriding num_joints: {config['num_joints']} -> {args.num_joints}")
        config["num_joints"] = args.num_joints

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Convert CenterDetect
    print(f"\n--- CenterDetect ---")
    cd_model, cd_size = build_model("center", cd_pth, config)
    cd_path = os.path.join(args.output_dir, "center_detect.mlpackage")
    cd_mb, cd_time = convert_to_coreml(cd_model, cd_size, cd_path, "CenterDetect")

    # Resolve the HybridNet checkpoint first: it decides which keypoint weights
    # the keypoint_detect.mlpackage must carry.
    hn_models_dir = models_dir if "models_dir" in dir() else None
    hn_pth = find_latest_hybridnet_pth(hn_models_dir) if hn_models_dir else None

    # Convert KeypointDetect.
    # For a 3D (HybridNet) project, the keypoint detector MUST be the effTrack
    # baked into the HybridNet checkpoint — V2VNet was jointly trained with it and
    # only produces accurate/confident output when fed that effTrack's heatmaps.
    # The standalone KeypointDetect checkpoint yields ~20mm/conf~0.28 instead of
    # ~2mm/conf~0.9. Fall back to the standalone checkpoint only for 2D-only
    # projects (no HybridNet checkpoint).
    print(f"\n--- KeypointDetect ---")
    kp_weights = kd_pth
    if hn_pth:
        hn_eff = extract_hn_efftrack_state(hn_pth)
        if hn_eff:
            kp_weights = hn_eff
            print(f"  3D project: sourcing keypoint detector from HybridNet "
                  f"checkpoint's internal effTrack: {hn_pth}")
        else:
            print(f"  WARNING: no effTrack.* weights in {hn_pth}; "
                  f"falling back to standalone KeypointDetect")
    kd_model, kd_size = build_model("keypoint", kp_weights, config)
    kd_path = os.path.join(args.output_dir, "keypoint_detect.mlpackage")
    kd_mb, kd_time = convert_to_coreml(kd_model, kd_size, kd_path, "KeypointDetect")

    # Convert HybridNet V2VNet (3D CNN) if a HybridNet checkpoint is present.
    # Optional: absence just yields a 2D-only model. Failure does not abort.
    v2v_result = None
    if hn_pth:
        print(f"\n--- HybridNet V2VNet (3D) ---")
        print(f"HybridNet checkpoint: {hn_pth}")
        v2v_path = os.path.join(args.output_dir, "v2vnet.mlpackage")
        try:
            v2v_result = convert_v2vnet_to_coreml(hn_pth, config, v2v_path)
        except Exception as e:
            print(f"  ERROR: V2VNet conversion raised: {e}")
            v2v_result = None
    else:
        print(f"\n--- No HybridNet checkpoint found; 2D-only conversion ---")

    # Infer actual num_joints from the keypoint model actually converted
    actual_joints = infer_output_channels_from_weights(kp_weights)
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
        "note": "TensorType input — C++ applies ImageNet normalization manually",
    }

    if v2v_result is not None:
        v2v_mb, v2v_time, grid_in, grid_out = v2v_result
        # roi_cube_size / grid_spacing define the voxel grid's PHYSICAL size. The host
        # reprojection builds the grid in the inference calibration's units, so scale the
        # training-unit values by --world_scale to match. grid_in/grid_out are the V2VNet
        # tensor dimensions (dimensionless) and are deliberately left unscaled.
        roi_scaled = config["roi_cube_size"] * args.world_scale
        spacing_scaled = config["grid_spacing"] * args.world_scale
        if args.world_scale != 1.0:
            print(f"  world_scale={args.world_scale}: roi_cube_size "
                  f"{config['roi_cube_size']} -> {roi_scaled}, grid_spacing "
                  f"{config['grid_spacing']} -> {spacing_scaled} "
                  f"(grid_in={grid_in}, grid_out={grid_out} unchanged)")
        metadata["hybridnet"] = {
            "mlpackage_file": "v2vnet.mlpackage",
            "mlpackage_size_mb": round(v2v_mb, 1),
            "num_cameras": config["num_cameras"],
            "roi_cube_size": roi_scaled,     # physical, in inference-calibration units
            "grid_spacing": spacing_scaled,  # physical, in inference-calibration units
            "grid_in": grid_in,      # V2VNet input voxel grid side (reprojection output)
            "grid_out": grid_out,    # V2VNet output side = soft-argmax grid
            "num_joints": num_joints,
            "world_scale": args.world_scale,  # training-units -> calibration-units factor
            "note": "3D CNN only; reprojection + soft-argmax done in host C++. "
                    "roi_cube_size/grid_spacing are physical (inference-calibration units).",
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
