# JARVIS + CoTracker Integration in RED

## Overview

RED supports two new AI-assisted labeling buttons in the **Labeling Tool** panel:

| Button | What it does |
|--------|-------------|
| **Load JARVIS Models** | Loads CenterDetect + KeypointDetect engines |
| **Run JARVIS — This Frame** | Runs 2D JARVIS inference on all cameras for the current frame |
| **Load CoTracker Model** | Loads the TorchScript CoTracker3 model |
| **Run CoTracker — Fill Gaps** | Propagates labeled keypoints across buffered frames |

---

## One-time model export

### 1. Export JARVIS → TensorRT engines

```bash
cd /path/to/red
python3 tools/export_jarvis_trt.py \
    --project mouseHybrid24 \
    --output /data/models/jarvis_engines/
```

Outputs:
- `centerDetect.engine`   — CenterDetect TRT engine (input 1×3×320×320)
- `keypointDetect.engine` — KeypointDetect TRT engine (input 1×3×832×832)

**Alternatively**, if `torch_tensorrt` is unavailable, provide raw TorchScript `.pt`
files from `JARVIS-HybridNet/projects/<project>/models/`. RED will automatically
fall back to LibTorch inference when it sees `.pt` extensions.

### 2. Export CoTracker3 → TorchScript

```bash
python3 tools/export_cotracker_torchscript.py \
    --output /data/models/cotracker3_offline.pt
```

This downloads CoTracker3 via `torch.hub` and traces it to a portable `.pt` file.

---

## Using the buttons in RED

1. Open a project with a 24-keypoint skeleton loaded.
2. Load a video and navigate to a frame where the animal is visible.
3. Click **Load JARVIS Models** → select `centerDetect.engine` and `keypointDetect.engine`
   (or `.pt` files). A status message confirms success.
4. Click **Run JARVIS — This Frame** to run inference. Keypoints appear on all camera
   views. If ≥ 2 cameras detected the animal, triangulation is run automatically.
5. Click **Load CoTracker Model** → select `cotracker3_offline.pt`.
6. Click **Run CoTracker — Fill Gaps**. RED collects all frames currently in the
   decoder buffer, uses any labeled keypoints as anchor queries, and propagates them
   through all buffered frames. Frames where visibility > 0.5 are stored.

---

## Tensor format conventions

| Stage | Format | Notes |
|-------|--------|-------|
| Frame buffer | `unsigned char*` RGBA, uint8, row-major | No Y-flip at input |
| JARVIS input | `(1, 3, H, W)` float32, BGR channel order | Normalized with ImageNet mean/std in BGR order |
| CenterDetect output | `(1, 1, 320, 320)` float32 | Raw logit heatmap; sigmoid gives confidence |
| KeypointDetect output | `(1, 24, 416, 416)` float32 | Raw logit per-keypoint heatmap |
| CoTracker video | `(1, T, 3, H, W)` float32, RGB | Values in [0, 255]; downsampled 4× for speed |
| CoTracker queries | `(1, N, 3)` float32 | `[frame_idx, x_px, y_px]` in downsampled space |
| CoTracker output tracks | `(1, T, N, 2)` float32 | `(x, y)` in downsampled space → upscaled by RED |
| CoTracker output vis | `(1, T, N)` float32 | Raw logit; sigmoid > 0.5 → visible |

---

## TensorRT engine portability

TRT engines are compiled for a specific GPU architecture.
**You must re-run `export_jarvis_trt.py` on every machine where RED is used.**

LibTorch `.pt` files are fully portable across machines with the same CUDA version.

---

## Known limitations

- **JARVIS** runs one frame at a time (no batching across frames).
  For dense annotation, run it frame-by-frame or use CoTracker to propagate.
- **CoTracker** is limited to frames currently held in the decoder circular buffer
  (`scene->size_of_buffer` frames). Increasing `seek_interval` enlarges the buffer.
- **CoTracker** runs the full sequence of buffered frames on each button press.
  For very long buffers this can take several seconds.
- **CenterDetect** confidence threshold defaults to 0.3. Adjust via
  `g_jarvis_confidence_threshold` if detections are missed or spurious.
- Keypoints predicted by JARVIS have `confidence < 1.0`, same convention as YOLO
  auto-detections. Manual labels always have `confidence = 1.0`.
