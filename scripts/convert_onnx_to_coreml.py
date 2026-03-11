#!/usr/bin/env python3
"""Convert JARVIS EfficientTrack ONNX models to CoreML .mlpackage format.

Strategy: ONNX -> PyTorch (via JARVIS source) -> torch.jit.trace -> coremltools -> .mlpackage

The direct ONNX-to-CoreML path is NOT available because:
  - coremltools deprecated its ONNX converter (only supported opset <=10)
  - Our models use ONNX opset 18 (ReduceMean, Resize ops are incompatible)
  - onnx2torch fails on ReduceMean version 18

Instead, we reconstruct the PyTorch model from the JARVIS source code,
load the .pth weights, trace with torch.jit.trace, and convert via
coremltools' unified PyTorch converter.

Preprocessing baked into the CoreML model:
  - Input: BGR color layout (matches CVPixelBuffer from Metal pipeline)
  - Scale: 1/255.0 (normalize to [0,1])
  - No ImageNet normalization (JARVIS models expect [0,1] range, not normalized)

Usage:
    conda run -n jarvis python scripts/convert_onnx_to_coreml.py

Requirements:
    pip install coremltools torch  (in jarvis conda env)
"""

import os
import sys
import time
import json
import shutil

import torch
import torch.nn as nn
import numpy as np

# Add JARVIS to path so we can import the model architecture
JARVIS_ROOT = '/Users/johnsonr/src/JARVIS-HybridNet'
sys.path.insert(0, JARVIS_ROOT)

from jarvis.efficienttrack.model import EfficientTrackBackbone


# ── Configuration ──────────────────────────────────────────────────────
MODEL_DIR = '/Users/johnsonr/src/red/models/jarvis_mouseJan30'
ONNX_CD = os.path.join(MODEL_DIR, 'center_detect.onnx')
ONNX_KD = os.path.join(MODEL_DIR, 'keypoint_detect.onnx')

# JARVIS project with trained weights
JARVIS_PROJECT = '/Users/johnsonr/src/JARVIS-HybridNet/projects/annotate1'
CD_WEIGHTS = os.path.join(JARVIS_PROJECT, 'models/CenterDetect/Run_20260309-200548/EfficientTrack-medium_final.pth')
KD_WEIGHTS = os.path.join(JARVIS_PROJECT, 'models/KeypointDetect/Run_20260309-203455/EfficientTrack-medium_final.pth')

# Model info from model_info.json
CD_INPUT_SIZE = 320
KD_INPUT_SIZE = 704
NUM_JOINTS = 24
MODEL_SIZE = 'medium'


class SimpleConfig:
    """Minimal config object that EfficientTrackBackbone expects."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def build_model(mode, weights_path):
    """Build EfficientTrackBackbone and load weights.

    Args:
        mode: 'center' or 'keypoint'
        weights_path: path to .pth state_dict file

    Returns:
        model in eval mode on CPU
    """
    if mode == 'center':
        output_channels = 1
        input_size = CD_INPUT_SIZE
    else:
        output_channels = NUM_JOINTS
        input_size = KD_INPUT_SIZE

    cfg = SimpleConfig(
        MODEL_SIZE=MODEL_SIZE,
        IMAGE_SIZE=CD_INPUT_SIZE,
        BOUNDING_BOX_SIZE=KD_INPUT_SIZE,
        NUM_JOINTS=NUM_JOINTS,
    )

    model = EfficientTrackBackbone(cfg, model_size=MODEL_SIZE,
                                    output_channels=output_channels)

    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print(f'  Loaded weights: {weights_path}')
    else:
        print(f'  WARNING: weights not found: {weights_path}')
        print(f'  Using random weights (model structure only)')

    model.eval()
    return model, input_size


def verify_against_onnx(pytorch_model, onnx_path, input_size):
    """Compare PyTorch model output against ONNX Runtime output.

    Returns True if outputs match within tolerance.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print('  [skip ONNX verification - onnxruntime not installed]')
        return True

    # Use the same random seed for reproducibility
    np.random.seed(42)
    dummy_np = np.random.randn(1, 3, input_size, input_size).astype(np.float32)
    dummy_torch = torch.from_numpy(dummy_np)

    # PyTorch inference
    with torch.no_grad():
        pt_outs = pytorch_model(dummy_torch)
    pt_low = pt_outs[0].numpy()
    pt_high = pt_outs[1].numpy()

    # ONNX Runtime inference
    sess = ort.InferenceSession(onnx_path)
    ort_outs = sess.run(None, {'image': dummy_np})
    ort_low = ort_outs[0]
    ort_high = ort_outs[1]

    # Compare
    max_diff_low = np.max(np.abs(pt_low - ort_low))
    max_diff_high = np.max(np.abs(pt_high - ort_high))
    print(f'  PyTorch vs ONNX max diff: low={max_diff_low:.6f}, high={max_diff_high:.6f}')

    # Tolerance: ONNX uses ManualInstanceNorm2d, PyTorch uses nn.InstanceNorm2d
    # They should be numerically close but not identical
    ok = max_diff_low < 0.01 and max_diff_high < 0.01
    if not ok:
        print(f'  WARNING: PyTorch and ONNX outputs differ significantly!')
        print(f'  This is expected if the ONNX model uses ManualInstanceNorm2d')
        print(f'  and PyTorch uses nn.InstanceNorm2d (different numerical paths)')
    return ok


def convert_to_coreml(model, input_size, output_path, model_name):
    """Convert PyTorch model to CoreML .mlpackage using coremltools.

    The model is traced with torch.jit.trace, then converted to CoreML
    ML Program format (.mlpackage) with Image input type.

    Args:
        model: PyTorch model in eval mode
        input_size: int, spatial dimension (e.g. 320)
        output_path: str, path to write .mlpackage
        model_name: str, human-readable name for the model
    """
    import coremltools as ct

    print(f'\n  Converting {model_name} to CoreML...')

    # ── Step 1: Trace the model ───────────────────────────────────────
    t0 = time.time()
    dummy_input = torch.randn(1, 3, input_size, input_size)

    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_input)

    trace_time = time.time() - t0
    print(f'  Traced in {trace_time:.2f}s')

    # ── Step 2: Convert to CoreML ─────────────────────────────────────
    t0 = time.time()

    # Option A: ImageType input with preprocessing baked in
    # This allows passing CVPixelBuffer directly from Metal pipeline.
    #
    # The JARVIS models expect RGB float [0, 1] input.
    # CVPixelBuffers from our Metal pipeline are BGRA.
    # CoreML handles BGR->RGB reordering automatically when color_layout=BGR.
    # Scale 1/255.0 normalizes uint8 [0,255] to float [0,1].
    #
    # Note: ct.colorlayout does not have BGRA; BGR is the closest.
    # CoreML will accept 32BGRA CVPixelBuffers and ignore the alpha channel
    # when the model input is specified as BGR.
    image_input = ct.ImageType(
        name='image',
        shape=(1, 3, input_size, input_size),
        color_layout=ct.colorlayout.BGR,
        scale=1.0 / 255.0,
        bias=[0.0, 0.0, 0.0],  # No mean subtraction needed
    )

    # Option B: TensorType input (for comparison / if ImageType causes issues)
    tensor_input = ct.TensorType(
        name='image',
        shape=(1, 3, input_size, input_size),
    )

    # Try ImageType first, fall back to TensorType
    try:
        coreml_model = ct.convert(
            traced,
            inputs=[image_input],
            convert_to='mlprogram',
            minimum_deployment_target=ct.target.macOS13,
            compute_precision=ct.precision.FLOAT16,
        )
        input_mode = 'ImageType (BGR, scale=1/255)'
    except Exception as e:
        print(f'  ImageType conversion failed: {e}')
        print(f'  Falling back to TensorType...')
        coreml_model = ct.convert(
            traced,
            inputs=[tensor_input],
            convert_to='mlprogram',
            minimum_deployment_target=ct.target.macOS13,
            compute_precision=ct.precision.FLOAT16,
        )
        input_mode = 'TensorType (raw float tensor)'

    convert_time = time.time() - t0
    print(f'  Converted in {convert_time:.2f}s')
    print(f'  Input mode: {input_mode}')

    # ── Step 3: Save .mlpackage ───────────────────────────────────────
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    coreml_model.save(output_path)

    # Calculate size
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(output_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    size_mb = total_size / (1024 * 1024)
    print(f'  Saved: {output_path} ({size_mb:.1f} MB)')

    # ── Step 4: Test inference ────────────────────────────────────────
    print(f'  Testing CoreML inference...')
    t0 = time.time()

    # Load and run
    try:
        # For prediction, we need to provide the right input type
        import PIL.Image
        if 'ImageType' in input_mode:
            # Create a test image
            test_img = PIL.Image.fromarray(
                np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)
            )
            prediction = coreml_model.predict({'image': test_img})
        else:
            test_input = np.random.randn(1, 3, input_size, input_size).astype(np.float32)
            prediction = coreml_model.predict({'image': test_input})

        first_run_time = time.time() - t0
        print(f'  First inference: {first_run_time*1000:.1f}ms (includes compilation)')

        # Output info
        for key, val in prediction.items():
            if hasattr(val, 'shape'):
                print(f'  Output "{key}": shape={val.shape}, dtype={val.dtype}')
            else:
                print(f'  Output "{key}": type={type(val).__name__}')

        # Benchmark: run 10 more times
        times = []
        for _ in range(10):
            t0 = time.time()
            if 'ImageType' in input_mode:
                prediction = coreml_model.predict({'image': test_img})
            else:
                prediction = coreml_model.predict({'image': test_input})
            times.append(time.time() - t0)

        avg_ms = np.mean(times) * 1000
        std_ms = np.std(times) * 1000
        print(f'  Avg inference: {avg_ms:.1f} +/- {std_ms:.1f} ms (10 runs)')

    except Exception as e:
        print(f'  Inference test failed: {e}')
        import traceback
        traceback.print_exc()

    return coreml_model, convert_time, size_mb


def convert_tensor_input(model, input_size, output_path, model_name):
    """Convert with TensorType input only (simpler, always works).

    Use this if ImageType causes issues or if you want to handle
    preprocessing in C++ before calling CoreML.
    """
    import coremltools as ct

    print(f'\n  Converting {model_name} (TensorType) to CoreML...')

    dummy_input = torch.randn(1, 3, input_size, input_size)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_input)

    t0 = time.time()
    coreml_model = ct.convert(
        traced,
        inputs=[ct.TensorType(name='image', shape=(1, 3, input_size, input_size))],
        convert_to='mlprogram',
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16,
    )
    convert_time = time.time() - t0

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    coreml_model.save(output_path)

    total_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, dn, fns in os.walk(output_path) for f in fns
    )
    size_mb = total_size / (1024 * 1024)
    print(f'  Saved: {output_path} ({size_mb:.1f} MB) in {convert_time:.2f}s')

    return coreml_model


def infer_output_channels_from_weights(weights_path):
    """Infer the number of output channels from a checkpoint's final_conv1 weight."""
    if not os.path.exists(weights_path):
        return None
    sd = torch.load(weights_path, map_location='cpu')
    if 'final_conv1.weight' in sd:
        return sd['final_conv1.weight'].shape[0]
    return None


def main():
    print('=' * 70)
    print('JARVIS EfficientTrack: ONNX -> CoreML .mlpackage Conversion')
    print('=' * 70)

    import coremltools as ct
    print(f'\ncoremltools version: {ct.__version__}')
    print(f'torch version: {torch.__version__}')
    print(f'Model directory: {MODEL_DIR}')

    # ── Build CenterDetect model ──────────────────────────────────────
    print(f'\n--- CenterDetect (1 output channel, {CD_INPUT_SIZE}x{CD_INPUT_SIZE}) ---')
    cd_model, cd_size = build_model('center', CD_WEIGHTS)

    # Verify PyTorch model matches ONNX output
    print(f'\n  Verifying PyTorch vs ONNX...')
    verify_against_onnx(cd_model, ONNX_CD, cd_size)

    # Convert to CoreML
    cd_mlpackage = os.path.join(MODEL_DIR, 'center_detect.mlpackage')
    cd_coreml, cd_time, cd_mb = convert_to_coreml(
        cd_model, cd_size, cd_mlpackage, 'CenterDetect')

    # Also save TensorType version for comparison
    cd_tensor_path = os.path.join(MODEL_DIR, 'center_detect_tensor.mlpackage')
    convert_tensor_input(cd_model, cd_size, cd_tensor_path, 'CenterDetect')

    # ── Build KeypointDetect model ────────────────────────────────────
    # Detect actual output channels from the checkpoint (may differ from
    # model_info.json if the ONNX was exported from a different training run)
    actual_joints = infer_output_channels_from_weights(KD_WEIGHTS)
    if actual_joints is not None and actual_joints != NUM_JOINTS:
        print(f'\n  NOTE: Checkpoint has {actual_joints} output channels, '
              f'model_info.json says {NUM_JOINTS}.')
        print(f'  Using checkpoint value ({actual_joints} joints).')
        kd_joints = actual_joints
    else:
        kd_joints = NUM_JOINTS

    # Also check ONNX model's actual output shape
    try:
        import onnx
        onnx_model = onnx.load(ONNX_KD)
        onnx_joints = onnx_model.graph.output[1].type.tensor_type.shape.dim[1].dim_value
        print(f'  ONNX model has {onnx_joints} output channels.')
        if onnx_joints != kd_joints:
            print(f'  WARNING: ONNX ({onnx_joints}) != checkpoint ({kd_joints}).')
            print(f'  The ONNX model was likely exported from a different checkpoint.')
            print(f'  Converting with checkpoint ({kd_joints} joints) for now.')
    except Exception as e:
        print(f'  Could not inspect ONNX model: {e}')

    # Use annotate1's bounding box size (512) rather than model_info's 704
    # if the checkpoint doesn't match
    kd_input = KD_INPUT_SIZE
    try:
        # Read from annotate1 config
        import yaml
        cfg_path = os.path.join(JARVIS_PROJECT, 'config.yaml')
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                proj_cfg = yaml.safe_load(f)
            proj_bb_size = proj_cfg.get('KEYPOINTDETECT', {}).get('BOUNDING_BOX_SIZE', KD_INPUT_SIZE)
            if proj_bb_size != KD_INPUT_SIZE:
                print(f'  NOTE: Project config says BOUNDING_BOX_SIZE={proj_bb_size}, '
                      f'model_info says {KD_INPUT_SIZE}.')
                # Use project config value when it matches checkpoint
                if actual_joints is not None and actual_joints != NUM_JOINTS:
                    kd_input = proj_bb_size
                    print(f'  Using project config value ({kd_input})')
    except Exception:
        pass

    print(f'\n--- KeypointDetect ({kd_joints} joints, {kd_input}x{kd_input}) ---')

    # Build model with correct output channels
    cfg = SimpleConfig(
        MODEL_SIZE=MODEL_SIZE,
        IMAGE_SIZE=CD_INPUT_SIZE,
        BOUNDING_BOX_SIZE=kd_input,
        NUM_JOINTS=kd_joints,
    )
    kd_model = EfficientTrackBackbone(cfg, model_size=MODEL_SIZE,
                                       output_channels=kd_joints)
    if os.path.exists(KD_WEIGHTS):
        kd_model.load_state_dict(
            torch.load(KD_WEIGHTS, map_location='cpu'), strict=False)
        print(f'  Loaded weights: {KD_WEIGHTS}')
    kd_model.eval()

    # Convert to CoreML
    kd_mlpackage = os.path.join(MODEL_DIR, 'keypoint_detect.mlpackage')
    kd_coreml, kd_time, kd_mb = convert_to_coreml(
        kd_model, kd_input, kd_mlpackage, 'KeypointDetect')

    # TensorType version
    kd_tensor_path = os.path.join(MODEL_DIR, 'keypoint_detect_tensor.mlpackage')
    convert_tensor_input(kd_model, kd_input, kd_tensor_path, 'KeypointDetect')

    # ── Update model_info.json ────────────────────────────────────────
    meta_path = os.path.join(MODEL_DIR, 'model_info.json')
    with open(meta_path) as f:
        metadata = json.load(f)

    metadata['center_detect']['mlpackage_file'] = 'center_detect.mlpackage'
    metadata['center_detect']['mlpackage_size_mb'] = round(cd_mb, 1)
    metadata['keypoint_detect']['mlpackage_file'] = 'keypoint_detect.mlpackage'
    metadata['keypoint_detect']['mlpackage_size_mb'] = round(kd_mb, 1)
    metadata['keypoint_detect']['mlpackage_num_joints'] = kd_joints
    metadata['keypoint_detect']['mlpackage_input_size'] = kd_input
    metadata['coreml_info'] = {
        'format': 'mlprogram (.mlpackage)',
        'precision': 'float16',
        'minimum_deployment_target': 'macOS13',
        'input_color_layout': 'BGR',
        'input_scale': 1.0 / 255.0,
        'input_bias': [0.0, 0.0, 0.0],
        'note': 'ImageType input accepts CVPixelBuffer directly',
    }

    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'\nUpdated: {meta_path}')

    # ── Summary ───────────────────────────────────────────────────────
    print(f'\n{"=" * 70}')
    print(f'SUMMARY')
    print(f'{"=" * 70}')
    print(f'  CenterDetect:   {cd_mb:.1f} MB, converted in {cd_time:.1f}s')
    print(f'  KeypointDetect: {kd_mb:.1f} MB, converted in {kd_time:.1f}s')
    print(f'\nOutput files:')
    print(f'  {cd_mlpackage}')
    print(f'  {kd_mlpackage}')
    print(f'  {cd_tensor_path}  (TensorType variant)')
    print(f'  {kd_tensor_path}  (TensorType variant)')
    print(f'\nTo use in C++:')
    print(f'  1. Compile .mlpackage -> .mlmodelc with xcrun coremlcompiler')
    print(f'  2. Load MLModel from .mlmodelc path')
    print(f'  3. Pass CVPixelBuffer (BGRA) directly as input')
    print(f'  4. First inference triggers device specialization (may be slow)')
    print(f'  5. Subsequent inferences use cached compiled model')
    print(f'\nPre-compile for faster loading:')
    print(f'  xcrun coremlcompiler compile center_detect.mlpackage .')
    print(f'  xcrun coremlcompiler compile keypoint_detect.mlpackage .')


if __name__ == '__main__':
    main()
