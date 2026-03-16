#!/usr/bin/env python3
"""Convert SuperPoint PyTorch model to CoreML .mlpackage format.

Exports raw semi (1,65,H/8,W/8) + desc (1,256,H/8,W/8) heads with NO
post-processing. The C++ inference code handles softmax, NMS, top-K,
descriptor interpolation, and L2-normalization.

Usage:
    pip install torch torchvision coremltools
    python scripts/convert_superpoint_coreml.py --output_dir models/superpoint

Requirements:
    torch, torchvision (for pretrained SuperPoint), coremltools >= 7.0
"""

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn


class SuperPointRawHeads(nn.Module):
    """Wrapper that returns raw semi + desc heads from torchvision SuperPoint."""

    def __init__(self):
        super().__init__()
        from torchvision.models import feature_extraction
        import torchvision.models as models

        # Load pretrained SuperPoint from torchvision
        self.model = models.detection.superpoint.SuperPoint()
        # Load the default pretrained weights
        from torchvision.models.detection.superpoint import SuperPoint_Weights
        state_dict = SuperPoint_Weights.DEFAULT.get_state_dict(progress=True)
        self.model.load_state_dict(state_dict)

    def forward(self, x):
        # x: [1, 1, H, W] grayscale, [0,1] range
        # Run through the backbone + heads, return raw outputs
        # SuperPoint backbone: shared encoder → two heads
        out = self.model._forward(x)
        # _forward returns dict with 'semi' and 'desc'
        return out["semi"], out["desc"]


class SuperPointFromScratch(nn.Module):
    """Minimal SuperPoint architecture matching torchvision weights.
    Falls back to this if torchvision's _forward is not available."""

    def __init__(self, weights_path=None):
        super().__init__()
        # Shared encoder
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256
        self.conv1a = nn.Conv2d(1, c1, 3, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, 3, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, 3, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, 3, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, 3, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, 3, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, 3, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, 3, padding=1)

        # Detector head
        self.convPa = nn.Conv2d(c4, c5, 3, padding=1)
        self.convPb = nn.Conv2d(c5, 65, 1)

        # Descriptor head
        self.convDa = nn.Conv2d(c4, c5, 3, padding=1)
        self.convDb = nn.Conv2d(c5, 256, 1)

        if weights_path and os.path.exists(weights_path):
            sd = torch.load(weights_path, map_location="cpu")
            self.load_state_dict(sd)

    def forward(self, x):
        # Shared encoder
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Detector head: [1, 65, H/8, W/8]
        semi = self.convPb(self.relu(self.convPa(x)))

        # Descriptor head: [1, 256, H/8, W/8]
        desc = self.convDb(self.relu(self.convDa(x)))

        return semi, desc


def main():
    parser = argparse.ArgumentParser(
        description="Convert SuperPoint to CoreML .mlpackage")
    parser.add_argument("--output_dir", default="models/superpoint",
                        help="Output directory for .mlpackage")
    parser.add_argument("--input_height", type=int, default=480,
                        help="Input image height (must be divisible by 8)")
    parser.add_argument("--input_width", type=int, default=640,
                        help="Input image width (must be divisible by 8)")
    args = parser.parse_args()

    assert args.input_height % 8 == 0, "Height must be divisible by 8"
    assert args.input_width % 8 == 0, "Width must be divisible by 8"

    import coremltools as ct
    print(f"coremltools: {ct.__version__}, torch: {torch.__version__}")

    # Load SuperPoint with pretrained weights.
    # Try torchvision first (has SuperPoint since v0.17), then fall back to
    # MagicLeap pretrained weights with our from-scratch architecture.
    print("Loading SuperPoint model...")
    model = None

    # Method 1: torchvision (if available)
    try:
        model = SuperPointRawHeads()
        dummy = torch.randn(1, 1, args.input_height, args.input_width)
        with torch.no_grad():
            semi, desc = model(dummy)
        print(f"  torchvision SuperPoint loaded")
        print(f"  semi: {semi.shape}, desc: {desc.shape}")
    except Exception as e:
        print(f"  torchvision not available ({e})")
        model = None

    # Method 2: MagicLeap pretrained weights (official release)
    if model is None:
        print("  Loading MagicLeap pretrained weights...")
        SUPERPOINT_URL = "https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth"
        model = SuperPointFromScratch()
        try:
            sd = torch.hub.load_state_dict_from_url(SUPERPOINT_URL, map_location="cpu")
            result = model.load_state_dict(sd, strict=False)
            if result.missing_keys:
                print(f"  WARNING: {len(result.missing_keys)} missing keys")
            print(f"  Loaded pretrained weights ({len(sd)} parameters)")
        except Exception as e:
            print(f"  ERROR: Failed to download weights: {e}")
            print(f"  Model will have random weights — results will be meaningless")
            sys.exit(1)

    model.eval()

    # Trace
    print(f"Tracing with input shape [1, 1, {args.input_height}, {args.input_width}]...")
    dummy = torch.randn(1, 1, args.input_height, args.input_width)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)

    # Convert to CoreML
    print("Converting to CoreML...")
    t0 = time.time()

    coreml_model = ct.convert(
        traced,
        inputs=[ct.TensorType(
            name="image",
            shape=(1, 1, args.input_height, args.input_width),
        )],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16,
    )

    convert_time = time.time() - t0
    print(f"  Converted in {convert_time:.1f}s")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "superpoint.mlpackage")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    coreml_model.save(output_path)

    total_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(output_path) for f in fns
    )
    size_mb = total_size / (1024 * 1024)
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")

    # Write model info
    import json
    info = {
        "model": "SuperPoint",
        "input_channels": 1,
        "input_height": args.input_height,
        "input_width": args.input_width,
        "output_semi_shape": [1, 65, args.input_height // 8, args.input_width // 8],
        "output_desc_shape": [1, 256, args.input_height // 8, args.input_width // 8],
        "precision": "float16",
        "note": "Raw heads — C++ handles softmax, NMS, top-K, descriptor interp, L2-norm",
    }
    info_path = os.path.join(args.output_dir, "model_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Info: {info_path}")

    print(f"\nDone! Model at {output_path}")


if __name__ == "__main__":
    main()
