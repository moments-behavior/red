# JARVIS + CoTracker Integration in RED

## Status Summary (2026-03-30)

- **JARVIS**: Working. 2D keypoint detection runs per-frame on all cameras. You can edit the predicted keypoints and press Triangulate to get 3D labels. This is the primary labeling workflow right now.
- **CoTracker**: NOT WORKING. See [CoTracker Issues](#cotracker-issues) below. Should be disabled for now.

---

## Overview

RED supports AI-assisted labeling buttons in the **Labeling Tool** panel:

| Button | What it does | Status |
|--------|-------------|--------|
| **Load JARVIS Models** | Loads CenterDetect + KeypointDetect engines | Working |
| **Run JARVIS — This Frame** | Runs 2D JARVIS inference on all cameras for the current frame | Working |
| **Load CoTracker Model** | Loads the TorchScript CoTracker3 model | Loads OK |
| **Run CoTracker — Fill Gaps** | Propagates labeled keypoints across buffered frames | **Not working** |

---

## Current Workflow (JARVIS only)

1. Open a project with a 24-keypoint skeleton loaded.
2. Load a video and navigate to a frame where the animal is visible.
3. Click **Load JARVIS Models** → select `centerDetect.engine` and `keypointDetect.engine`
   (or `.pt` files). A status message confirms success.
4. Click **Run JARVIS — This Frame** to run inference. Keypoints appear on all camera views.
5. **Edit** the predicted keypoints manually as needed (JARVIS is a 2D network, so predictions may need correction).
6. Press **Triangulate** to compute 3D positions from the 2D labels across cameras. This also helps with labelling new unseen frames by providing consistent 3D → 2D reprojections.

---

## CoTracker Issues

CoTracker3 is integrated in code but **does not work in practice**. The intended use case — label one frame with JARVIS, then propagate keypoints to neighboring frames with CoTracker — fails. After pressing "Run CoTracker — Fill Gaps", keypoints remain only on the originally labeled frame and do not propagate to other frames.

### Known problems

1. **No propagation across frames.** The core issue: labelling 1 frame and pressing CoTracker does not fill in any additional frames. The output stays stuck on the single labeled frame.

2. **Fixed trace dimensions.** The TorchScript model was traced with fixed `T=200` frames and `N=24` queries (`cotracker_infer.cpp:7-8`). If the actual frame count or query count doesn't match these exact dimensions, the model receives zero-padded dummy data. It's unclear how CoTracker3 handles this padding — it may effectively ignore the real queries.

3. **Aggressive downsampling.** Frames are downsampled to 25% (`CT_SCALE = 0.25`) before being fed to CoTracker (`cotracker_infer.h:34`). For small animals or fine keypoints, this may destroy the spatial detail needed for accurate tracking.

4. **Query coordinate space.** Query points are scaled by `CT_SCALE` before being passed to the model (`cotracker_infer.cpp:102-103`), and results are scaled back. Any mismatch between the coordinate conventions (e.g., Y-flip between ImPlot space and pixel space) could cause the model to receive nonsensical anchor positions, though the code does attempt to handle this (`red.cpp` around line 2865).

5. **Visibility threshold.** Tracked points are only stored when `vis > 0.5` (`red.cpp` around line 2930). If CoTracker's visibility scores are systematically low (e.g., due to the padding or downsampling), all propagated points may be silently discarded.

6. **TorchScript tracing limitations.** CoTracker3 uses attention and correlation mechanisms internally. TorchScript tracing can silently bake in tensor shapes or control flow from the trace inputs, which may cause incorrect behavior at inference time with different-shaped inputs.

### Recommendation

Disable the CoTracker UI for now. It adds complexity without providing value in its current state. Focus on the JARVIS + manual edit + triangulate workflow, which is working.

---

## Next Steps

1. **Disable CoTracker in the GUI** — hide or grey out the CoTracker buttons so users don't waste time on it.
2. **Upload a JARVIS dataset** — export labeled data from RED and upload it.
3. **Merge with cluster dataset** — combine the uploaded dataset with existing data on the cluster for retraining or evaluation.

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

### 2. Export CoTracker3 → TorchScript (currently broken — see above)

```bash
python3 tools/export_cotracker_torchscript.py \
    --output /data/models/cotracker3_offline.pt
```

This downloads CoTracker3 via `torch.hub` and traces it to a portable `.pt` file.

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

- **JARVIS** is a 2D network — it runs one frame at a time on each camera independently (no batching across frames). Predictions may need manual editing before triangulation.
- **CoTracker** is not working — see [CoTracker Issues](#cotracker-issues) above.
- **CenterDetect** confidence threshold defaults to 0.3. Adjust via
  `g_jarvis_confidence_threshold` if detections are missed or spurious.
- Keypoints predicted by JARVIS have `confidence < 1.0`, same convention as YOLO
  auto-detections. Manual labels always have `confidence = 1.0`.
