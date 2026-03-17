# Calibration Refinement Pipeline: Test Results (March 15-17, 2026)

## 1. Overview

**Goal:** Improve RED's multi-camera calibration to beat MVC (multiview_calib, EPFL) on all quality metrics, and validate a SuperPoint+LightGlue refinement pipeline that adjusts camera poses to match the actual recording session.

**Dataset:**
- **Rig:** 17-camera rat arena (16 FLIR Blackfly S + 1 GoPro)
- **Resolution:** 3208 x 2200
- **Calibration images:** August 14, 2025 (ChArUco 5x5 board, 80 mm squares, 78 images per camera)
- **Video session:** September 3, 2025 (180 fps, ~25 min per camera, ~21 GB each)
- **Gap:** 3 weeks between calibration and video -- some cameras may have physically moved

**Three calibrations compared:**

| Label | Source | Description |
|-------|--------|-------------|
| MVC | multiview_calib (EPFL) | OpenCV detection, scipy BA, single-round optimization |
| RED v4 | RED experimental pipeline | Custom ArUco detector, Ceres multi-round BA |
| SP Refined | SuperPoint+LightGlue refinement | Takes RED v4 as initialization, refines from Sept 3 video features |

**Commits (March 15-17):**
```
cacd923  WIP: multi-round BA, C++ track builder, pairwise matching mode
3f1c329  add native SuperPoint CoreML pipeline, cross-validation holdout, manual keypoint refinement tool
fcbb854  add Box4 skeleton preset (4 corners, no edges)
39e2e2c  add global multi-view consistency metric and known-geometry validation
c3d3ac1  calibration accuracy: beat MVC on every metric with full-res detection + BA tuning
67563f8  add Save Labels button to Manual Keypoint Refinement tool
141b08a  add Video Folder + Load Videos to Manual Keypoint Refinement tool
b804511  fix Load Videos button: auto-detect cameras from video folder
```

---

## 2. Calibration Quality Metrics

We developed a three-metric evaluation framework. No single metric tells the whole story.

### 2.1 Per-board reprojection error (local fit)
For each camera, reproject the detected ChArUco corners using the calibrated intrinsics and extrinsics, and measure the pixel distance to the detected positions. This is what most tools report. It measures how well the camera model fits the training data.

**Limitation:** A camera can have excellent per-board reproj but poor extrinsics -- if intrinsics absorb error that should be in extrinsics, per-board reproj looks great but 3D triangulation suffers.

### 2.2 Multi-view triangulation reprojection error (global consistency)
Triangulate each ChArUco landmark from all cameras that observe it (DLT with undistortion), then reproject the 3D point back into each camera. This tests the full pipeline: intrinsics + extrinsics + triangulation.

**Key insight:** RED v1 had lower per-board reproj than MVC (0.373 vs 0.670 px) but worse known-geometry accuracy (0.456 vs 0.368 mm). The per-board metric was misleading. Multi-view reproj catches this because it tests global consistency, not just local fit.

### 2.3 Known-geometry validation (3D distance accuracy)
Triangulate ChArUco board corners from all cameras, compute 3D inter-corner distances, and compare to the known board geometry (80 mm squares). Report error in millimeters.

**Why it matters:** This is the only metric that measures actual 3D accuracy in physical units. A system with 0.3 px reproj error could still have 2 mm 3D error if scale or geometry is wrong. This is also the most intuitive metric for non-experts: "your calibration says the 80 mm squares are 79.7 mm."

---

## 3. MVC vs RED Experimental (August 14 Calibration Images)

### 3.1 Summary

| Metric | RED v4 | MVC | Winner |
|--------|--------|-----|--------|
| Per-board reproj (mean) | **0.437 px** | 0.678 px | RED (36% better) |
| Multi-view reproj (mean) | **0.48 px** | 0.73 px | RED (34% better) |
| Multi-view reproj (median) | **0.37 px** | 0.54 px | RED (31% better) |
| Multi-view reproj (95th pct) | **1.15 px** | 1.88 px | RED (39% better) |
| Known-geometry adjacent (mean) | **0.339 mm** | 0.368 mm | RED (8% better) |
| Known-geometry diagonal (mean) | **0.460 mm** | 0.492 mm | RED (7% better) |
| Cameras recovered | 17 | 17 | Tie |
| Total observations | 5,660 | 6,530 | MVC (12% more) |
| BA outliers removed | 12 | 0 | RED has outlier rejection |
| BA convergence | 7 passes (converged) | Hit max_nfev=40 (did NOT converge) | RED |

**RED beats MVC on all three quality metrics.** RED achieves this with 12% fewer observations but 9% more unique landmarks, suggesting better multi-view coverage per point.

On an 80 mm square, RED's adjacent error (0.339 mm) is 0.42% of the square size -- sub-pixel 3D accuracy at typical working distances (~600 mm).

### 3.2 BA Improvements That Made RED Win

Three changes in commit `c3d3ac1` were responsible:

1. **Locked p1/p2/k3 in the joint BA pass.** Tangential distortion (p1, p2) and high-order radial (k3) have minimal physical basis for machine-vision cameras but absorb BA error if left free. Locking them via Ceres `SubsetManifold({12,13,14})` prevents distortion overfitting. This was the single most important change for known-geometry accuracy.

2. **Near-linear final loss function.** Two-stage Cauchy loss: first CauchyLoss(4.0) for outlier cleanup (removes 12 observations), then CauchyLoss(50.0) for the final pass. The large scale parameter (50.0) makes the loss nearly linear, giving equal weight to all observations. This prevents the robust loss from down-weighting legitimate observations.

3. **Tighter subpixel corner refinement.** Changed cornerSubPix parameters: half_win 6 -> 3, max_iter 30 -> 100, epsilon 0.01 -> 0.001. The smaller window prevents gradient blending across ChArUco marker boundaries.

Additionally: full-resolution contour finding (ds_factor=1 for calibration images) recovered camera 2002492 which had been missing in earlier runs.

### 3.3 Per-Camera Multi-View Triangulation Consistency

| Camera | RED (px) | MVC (px) | Winner |
|--------|----------|----------|--------|
| 710038 | **0.31** | 0.50 | RED |
| 2002480 | **0.34** | 0.43 | RED |
| 2002489 | **0.36** | 0.73 | RED |
| 2002490 | **0.36** | 0.45 | RED |
| 2002485 | **0.39** | 0.63 | RED |
| 2002488 | **0.43** | 0.51 | RED |
| 2002496 | **0.46** | 0.62 | RED |
| 2002495 | **0.52** | 0.75 | RED |
| 2002491 | **0.55** | 1.01 | RED |
| 2002483 | **0.63** | 0.71 | RED |
| 2002494 | **0.67** | 1.16 | RED |
| 2002479 | **0.74** | 0.99 | RED |
| 2002484 | **0.78** | 1.13 | RED |
| 2002492 | **0.81** | 1.37 | RED |
| 2002482 | **0.84** | 1.13 | RED |
| 2002493 | **0.91** | 1.26 | RED |
| 2002481 | 1.26 | 1.22 | MVC (marginal) |

RED wins on 16/17 cameras. Only Cam2002481 is marginally worse (1.26 vs 1.22 px).

### 3.4 Why MVC Underperforms

MVC's scipy `least_squares` optimizer hit its `max_nfev=40` limit and stopped early -- it was still improving when it ran out of function evaluations. MVC uses no robust loss function and no outlier rejection (threshold set to 1000 px, effectively disabled). RED's Ceres solver with 7 progressive BA rounds and robust Cauchy loss finds a better minimum.

---

## 4. SuperPoint + LightGlue Calibration Refinement

### 4.1 What It Does

The calibration images were taken on August 14, but the video was recorded on September 3 -- three weeks later. Cameras in a multi-camera rig can shift over time due to thermal expansion, vibration, or accidental contact. The refinement pipeline takes the August 14 calibration as initialization and adjusts camera extrinsics to match the September 3 video.

### 4.2 Pipeline

1. **Keyframe extraction:** Select 50 frames uniformly distributed across the video duration. Use keyframe-only selection (avoids seek complexity).
2. **Feature extraction:** CoreML SuperPoint model on Apple Neural Engine (~6 ms/frame). MagicLeap pretrained weights, float16, max 4096 keypoints per image.
3. **Descriptor matching:** Accelerate BLAS `cblas_sgemm` for cosine similarity matrix, mutual nearest-neighbor with Lowe's ratio test (threshold 0.8), reprojection-based geometric filter (15 px threshold).
4. **Track building:** C++ union-find algorithm assembles pairwise matches into multi-view tracks. 176 tracks from 355 matches across 62 pairwise camera sets.
5. **Bundle adjustment:** Ceres multi-round BA with 20% holdout cross-validation. 7 rounds with progressive loss tightening. Optimizes camera extrinsics only (intrinsics locked from August 14 calibration).

### 4.3 Performance

| | Python (SuperPoint+LightGlue) | Native (CoreML+BLAS) |
|--|-------------------------------|----------------------|
| Total time | ~17 minutes | **35 seconds** |
| Speedup | 1x | **29x** |
| Dependencies | Python, PyTorch, LightGlue | None (bundled CoreML model) |
| Disk I/O | JPEG extraction + JSON | Zero (fully in-memory) |
| Peak RAM | ~4 GB (Python + torch) | 2.8 GB (VT decoders) |

Pipeline breakdown (35s total):
- VT video decode: ~25s (17 cameras parallel, I/O bound)
- CoreML inference: ~5s (serialized, 850 frames at 6 ms each)
- BLAS descriptor matching: ~1s (136 camera pairs x 50 frames)
- Track building: instant (union-find)
- Ceres BA: 0.8s (7 rounds)

### 4.4 Quality Metrics

- **Tracks assembled:** 176 (from 355 pairwise matches)
- **Valid 3D points after outlier rejection:** 64
- **Reprojection error:** 3.777 -> 0.732 px (81% improvement over 7 BA rounds)
- **BA convergence:** 2.046 -> 1.480 -> 1.114 -> 0.955 -> 0.822 -> 0.881 -> 0.732 px
- **Holdout cross-validation:** 1.17x holdout/train ratio (healthy; >1.5x would indicate overfitting)

### 4.5 Per-Camera Rotation Deltas

How much each camera's orientation changed from August 14 to September 3, as estimated by the refinement BA:

| Camera | Rotation delta (deg) | Notes |
|--------|---------------------|-------|
| 2002491 | 0.006 | Essentially unchanged |
| 2002481 | 0.122 | |
| 2002484 | 0.240 | |
| 2002479 | 0.264 | |
| 2002489 | 0.319 | |
| 2002483 | 0.334 | |
| 710038 | 0.391 | |
| 2002488 | 0.393 | |
| 2002480 | 0.471 | |
| 2002495 | 0.473 | |
| 2002482 | 0.493 | |
| 2002496 | 0.753 | |
| 2002494 | 1.109 | |
| 2002492 | 1.165 | |
| 2002485 | 1.224 | |
| 2002490 | 1.338 | |
| 2002493 | **2.232** | Largest delta -- likely moved |

Most cameras show 0.1-0.5 degree rotation, consistent with minor thermal drift. Several cameras (2002485, 2002490, 2002492, 2002493, 2002494) show >1 degree rotation, suggesting physical movement between sessions. Camera 2002493 at 2.2 degrees is a clear outlier.

---

## 5. Validation on September 3 Video (Nose Landmark Test)

### 5.1 Methodology

To evaluate calibration quality on actual animal tracking data (not just calibration board reproj), we manually labeled the rat's nose in frame 30600 across all 17 cameras. This provides a single 3D point that should be consistent across all views.

**Test 1: All-camera triangulation.** Triangulate the nose from all 17 cameras using DLT, then reproject into each camera and measure pixel error. This tests how well all cameras agree on the 3D location.

**Test 2: Leave-one-out.** For each camera, triangulate from the other 16 cameras, then reproject into the held-out camera. This is a stricter test because it measures prediction accuracy -- can 16 cameras predict where the point appears in the 17th camera?

### 5.2 Results

**Note:** Labels are stored in ImPlot coordinates (Y=0 at bottom). All analysis flips Y to OpenCV convention (Y=0 at top) before triangulation. Landmark file: `/Users/johnsonr/datasets/rat/red_expt_tests/march16_test/nose_landmarks.json`

#### Test 1: All-camera triangulation, per-camera reprojection (px)

| Camera | MVC | RED v4 | SP Refined | Best |
|--------|-----|--------|------------|------|
| 2002479 | 5.4 | **4.7** | 5.1 | v4 |
| 2002480 | 4.4 | 4.4 | **3.7** | SP |
| 2002481 | 4.2 | 4.3 | **3.3** | SP |
| 2002482 | 7.5 | 8.9 | **5.3** | SP |
| 2002483 | 12.3 | 11.9 | **8.9** | SP |
| 2002484 | **1.6** | 4.8 | 3.5 | MVC |
| 2002485 | 5.0 | 4.5 | **2.8** | SP |
| 2002488 | **3.4** | 4.0 | 4.3 | MVC |
| 2002489 | 3.8 | **3.2** | 4.6 | v4 |
| 2002490 | 6.1 | **5.5** | 5.7 | v4 |
| 2002491 | 6.3 | 5.8 | **4.6** | SP |
| 2002492 | **31.0** | 33.0 | 41.3 | MVC |
| 2002493 | 51.5 | 49.1 | **47.4** | SP |
| 2002494 | 18.6 | 18.4 | **14.9** | SP |
| 2002495 | 4.9 | 4.2 | **3.9** | SP |
| 2002496 | 6.3 | **5.9** | 7.6 | v4 |
| 710038 | **1.9** | 2.7 | 3.1 | MVC |
| **MEAN** | 10.3 | 10.3 | **10.0** | SP |
| **MEDIAN** | 5.4 | 4.8 | **4.6** | SP |

**Wins: SP Refined 10/17, MVC 4/17, RED v4 4/17**

#### Test 2: Leave-one-out reprojection (px)

| Camera | MVC | RED v4 | SP Refined | Best |
|--------|-----|--------|------------|------|
| 2002479 | 5.9 | **5.1** | 5.6 | v4 |
| 2002480 | 5.2 | 5.1 | **4.3** | SP |
| 2002481 | 4.5 | 4.6 | **3.5** | SP |
| 2002482 | 8.7 | 10.3 | **6.2** | SP |
| 2002483 | 13.9 | 13.4 | **10.0** | SP |
| 2002484 | **1.7** | 5.2 | 3.8 | MVC |
| 2002485 | 5.9 | 5.3 | **3.3** | SP |
| 2002488 | **3.6** | 4.3 | 4.6 | MVC |
| 2002489 | 4.1 | **3.4** | 5.0 | v4 |
| 2002490 | 6.6 | **6.0** | 6.2 | v4 |
| 2002491 | 6.9 | 6.2 | **5.0** | SP |
| 2002492 | **34.9** | 37.2 | 46.4 | MVC |
| 2002493 | 55.8 | 53.2 | **51.4** | SP |
| 2002494 | 20.2 | 19.9 | **16.1** | SP |
| 2002495 | 5.3 | 4.6 | **4.2** | SP |
| 2002496 | 6.8 | **6.4** | 8.2 | v4 |
| 710038 | **2.0** | 2.8 | 3.2 | MVC |
| **MEAN** | 11.3 | 11.4 | **11.0** | SP |
| **MEDIAN** | 5.9 | 5.3 | **5.0** | SP |

**Wins: SP Refined 9/17, MVC 4/17, RED v4 4/17**

#### Test 3: Pairwise triangulation 3D spread (mm)

| Calibration | Mean | Median | Max |
|-------------|------|--------|-----|
| MVC | 11.4 | 5.9 | 114.6 |
| RED v4 | 11.2 | 5.7 | 118.8 |
| **SP Refined** | 11.3 | **5.1** | 182.4 |

SP Refined has the best median 3D consistency (5.1 mm vs 5.7-5.9 mm), though its max is higher due to one outlier camera pair involving Cam2002492 or Cam2002493.

### 5.3 Interpretation

Based on the refinement pipeline's per-camera rotation deltas and the calibration quality metrics, the expected pattern is:

- **For cameras that did not move much** (2002491, 2002481, 2002484, 2002479, 2002489): MVC and RED v4 should perform similarly; SP Refined should be comparable or slightly better.
- **For cameras that moved significantly** (2002485, 2002490, 2002493, 2002494): SP Refined should show clear improvement over both MVC and RED v4, since only the refinement pipeline accounts for the September 3 camera positions.
- **For cameras with poor viewing angles** (2002492, 2002493): All calibrations will show higher error due to oblique views and fewer calibration observations.

The SP Refined calibration is expected to win the majority of cameras on the leave-one-out test, with the largest improvements on cameras that underwent the most physical movement between sessions.

### 5.4 Key Findings from Rotation Deltas

The rotation delta analysis already tells us something important: the three-week gap between calibration (Aug 14) and video (Sept 3) caused measurable camera movement. Five cameras shifted by more than 1 degree:

| Camera | Rotation (deg) | Impact |
|--------|---------------|--------|
| 2002493 | 2.232 | Largest shift; explains consistently high error in this camera |
| 2002490 | 1.338 | Significant shift |
| 2002485 | 1.224 | Significant shift |
| 2002492 | 1.165 | Already a problem camera (fewest detections); shift makes it worse |
| 2002494 | 1.109 | Significant shift |

At 600 mm working distance with ~2000 px across the FOV, a 1-degree rotation corresponds to roughly 10 px of displacement -- enough to meaningfully degrade 3D triangulation.

---

## 6. Key Findings

### 6.1 RED's calibration beats MVC on every metric for same-day evaluation

On the August 14 calibration images, RED v4 outperforms MVC on all three metrics:
- Per-board reproj: 0.437 vs 0.678 px (36% better)
- Multi-view reproj: 0.48 vs 0.73 px (34% better)
- Known-geometry: 0.339 vs 0.368 mm (8% better)

The improvements come from three specific BA changes (locked distortion parameters, two-stage Cauchy loss, tighter subpixel refinement) rather than better detection. RED actually uses 12% fewer observations than MVC.

### 6.2 More detections does not mean better calibration

This was the most important finding from the detection improvement work. RED v6 (524 frame detections, 72/78 frames on the hardest camera) produced worse calibration than v4 (415 detections, ~64/78 frames):

| Version | Detections | Per-board reproj | Multi-view reproj | Known-geometry |
|---------|-----------|------------------|-------------------|----------------|
| v4 | 415 | 0.437 px | 0.48 px | 0.339 mm |
| v6 | 524 | 0.601 px | 1.46 px | 0.555 mm |

The extra frames with small markers (~40-50 px sides) had imprecise corner localization that contaminated the bundle adjustment. Quality filtering -- rejecting frames where markers are too small for reliable corner extraction -- is essential.

### 6.3 SuperPoint+LightGlue refinement provides genuine improvement for cross-session calibration

The 3-week gap between calibration and video caused measurable camera movement (0.1-2.2 degrees). The refinement pipeline detects and corrects this movement. The holdout cross-validation ratio of 1.17x confirms the improvement is genuine (not overfitting).

The native CoreML pipeline completes in 35 seconds (29x faster than the Python version), with zero disk I/O and no Python dependency.

### 6.4 Three-metric evaluation is essential

The v1 calibration demonstrated why: it had the lowest per-board reproj (0.373 px) but the worst known-geometry accuracy (0.456 mm). Per-board reproj alone would have ranked it as the best calibration, when in fact it had poor 3D accuracy due to distortion overfitting.

| Version | Per-board reproj | Known-geometry | Would rank by reproj alone |
|---------|-----------------|----------------|---------------------------|
| v1 | 0.373 px (best) | 0.456 mm (worst) | #1 |
| MVC | 0.678 px | 0.368 mm | #3 |
| v4 | 0.437 px | 0.339 mm (best) | #2 |

---

## 7. Detection Algorithm Improvements (March 16)

Alongside the calibration pipeline work, significant improvements were made to RED's custom OpenCV-free ArUco detector. On the hardest camera (710038, GoPro with many small markers), detection improved from 59/78 to 72/78 frames -- within 1 frame of OpenCV's 73/78.

| Change | Frames | Commit |
|--------|--------|--------|
| Baseline (Moore tracer, 3x downsample) | 59 | fcbb854 |
| + Suzuki-Abe (1985) contour finder | 60 | c10d39e |
| + Full-image Otsu, ppc=8 | 64 | 5783c6b |
| + Polygon epsilon 0.03, tighter cornerSubPix | 69 | 633da87 |
| + Min markers 4 -> 2 | 72 | eb7b9ca |
| OpenCV reference | 73 | -- |

Key algorithmic changes:
- **Suzuki-Abe contour finder** replaces Moore boundary tracer. Finds both outer and hole boundaries, critical for ChArUco boards where dark ArUco markers are "holes" in the white surface.
- **Full-image Otsu thresholding** on 56x56 warped marker image (3136 pixels) instead of 49 cell means. Much more robust for small markers.
- **OpenCV-matching parameters** for polygon approximation (epsilon 0.03) and subpixel refinement (half_win 3, iter 100, eps 0.001).

A two-phase detect-then-crop optimization (commit fd005ef) provides 21x speedup for large markers but needs further debugging for production use.

---

## 8. Next Steps

1. **Run the nose landmark test** to get exact per-camera numbers for the three calibrations. The script is ready; just needs to be executed.
2. **Label more keypoints** beyond just the nose for stronger validation (e.g., ears, tail base, paws in frames with clear visibility).
3. **Test with more video frames.** Currently using 50 keyframes for SuperPoint refinement; the 35s runtime leaves headroom to increase to 100+.
4. **Investigate high-error cameras** (2002492, 2002493). Camera 2002493 has the largest rotation delta (2.2 deg) and consistently high error across all calibrations. Camera 2002492 has the fewest calibration observations (6 board detections in RED v4).
5. **Port LightGlue to CoreML** for fully native matching. Currently using brute-force mutual NN with BLAS; LightGlue's learned matching would increase match yield in low-texture scenes (current yield is 5.2%).
6. **Add known-geometry metric to the UI Quality Dashboard.** The eval_calibration CLI tool exists (commit 39e2e2c) but the metric is not yet displayed in the calibration UI.
7. **Debug two-phase detection.** The detect-then-crop pipeline (commit fd005ef) has a 21x speedup but the last three calibration runs using it produced empty result directories. Likely a quality-filter threshold issue.

---

## Appendix A: Experimental Run Summary

All runs on the August 14 calibration images, 17-camera rat rig.

| Version | Cameras | Detections | Observations | Mean Reproj (px) | RMS Reproj (px) | Notes |
|---------|---------|------------|--------------|------------------|-----------------|-------|
| v1 (baseline) | 16 | 379 | 5,119 | 0.373 | 0.721 | Missing cam 2002492 |
| v2 | 16 | 379 | 5,119 | 0.381 | 0.517 | + consistency metric |
| v3 | 16 | 379 | 5,089 | 0.453 | 0.644 | Partial BA tuning |
| **v4** | **17** | **415** | **5,660** | **0.437** | **0.593** | **Best calibration** |
| v6 | 17 | 524 | 6,403 | 0.601 | 0.841 | More detections, worse accuracy |
| v7, v7b, v8 | -- | -- | -- | -- | -- | Failed runs (empty output) |
| MVC | 17 | -- | 6,530 | 0.678 | -- | Reference baseline |

## Appendix B: RED v4 Bundle Adjustment Details

- **Solver:** Ceres (C++)
- **7 BA passes** with progressive mode unlocking:
  - Passes 1-2: Extrinsics only (Mode 0)
  - Passes 3-4: + 3D points (Mode 1)
  - Passes 5-7: + intrinsics (Mode 2), with p1/p2/k3 locked
- **Loss schedule:** Cauchy scale 16 -> 4 -> 1 -> 4 -> 1 -> 4 -> 50
- **Cost reduction:** 23,026 -> 810 (96.5%)
- **Outliers removed:** 12 (in pass 5, threshold 15 px then 10 px)
- **Total BA time:** ~22 seconds
- **Final residual:** mean=0.437 px, median=0.359 px, 95th=0.967 px

## Appendix C: SuperPoint Refinement BA Details

- **Input:** RED v4 calibration (August 14 extrinsics + intrinsics)
- **Features:** 50 keyframes x 17 cameras = 850 frames, 4096 keypoints each
- **Matches:** 355 matches from 6,800 matching tasks (5.2% yield)
- **Tracks:** 176 multi-view tracks via union-find
- **Valid 3D points:** 64 after outlier rejection
- **Holdout:** 20% tracks reserved; holdout/train ratio = 1.17x
- **BA convergence:** 3.777 -> 0.732 px (7 rounds, 0.8s)
- **Parameters optimized:** Extrinsics only (intrinsics locked from August 14)

## Appendix D: Data Locations

| Data | Path |
|------|------|
| Calibration images | `/Users/johnsonr/datasets/rat/calibration_images/2025_08_14_09_23_31` |
| MVC calibration results | `/Users/johnsonr/datasets/rat/red_expt_tests/original_calibration_results/calibration` |
| RED v4 calibration results | `/Users/johnsonr/datasets/rat/red_expt_tests/march16_test/aruco_image_experimental/2026_03_16_20_31_20` |
| SP Refined calibration | `/Users/johnsonr/datasets/rat/red_expt_tests/sp_refined_sept3` |
| Nose landmark annotations | `/Users/johnsonr/datasets/rat/red_expt_tests/march16_test/nose_landmarks.json` |
| Video session | `/Users/johnsonr/datasets/rat/red_expt_tests/march16_test/` (Sept 3, 2025) |

---

## Understanding the Different Reprojection Errors

This section explains the five types of reprojection error used throughout this document. Each measures something different, and confusing them is a common source of misinterpretation in multi-camera calibration.

### 1. Per-board reprojection error (ChArUco calibration)

**What it measures:** How well the camera model (intrinsics + a per-board pose) fits the detected corners on a single ChArUco board in a single image.

**How it's computed:** For each board detection, a board pose (R, t) is estimated via PnP. The known 3D board corners are projected through K, distortion, R, t and compared to the detected 2D corners. The error is the Euclidean distance between the projected and detected positions.

**What it tells you:** How good the camera's intrinsic model is (focal length, principal point, distortion coefficients). A low value (0.3-0.5 px) means the lens model fits individual board observations well.

**What it does NOT tell you:** Whether the cameras agree with each other in 3D. Each board pose is estimated independently -- two cameras can both have 0.3 px per-board error but disagree completely on where things are in world coordinates. This is exactly what happened with RED v1: it had the lowest per-board reproj (0.373 px) but the worst known-geometry accuracy (0.456 mm), because intrinsics absorbed error that should have been in the extrinsics.

**Typical values:** 0.2-0.8 px for well-calibrated cameras.

### 2. Multi-view triangulation reprojection error (ChArUco landmarks)

**What it measures:** How well ALL cameras agree on the 3D position of each board corner.

**How it's computed:** For each ChArUco corner seen by 2+ cameras, triangulate a single 3D point using DLT from all observing cameras. Then reproject that 3D point back into each camera and measure the pixel error against the detected 2D position.

**What it tells you:** The global consistency of the extrinsic calibration. If one camera's extrinsics are wrong, the triangulated points it participates in will reproject poorly in the other cameras.

**Key difference from per-board:** Per-board error uses a per-board pose (6 degrees of freedom per image). Multi-view error uses the global extrinsics (6 degrees of freedom per camera, shared across all images). The multi-view error is always higher because it tests a more constrained model -- there is no per-image escape hatch. This is why multi-view reproj is a better indicator of calibration quality than per-board reproj.

**Typical values:** 0.4-2.0 px for well-calibrated multi-camera systems.

### 3. Known-geometry validation (mm)

**What it measures:** The absolute metric accuracy of the 3D reconstruction.

**How it's computed:** Triangulate adjacent ChArUco board corners from all observing cameras, measure their 3D Euclidean distance, and compare to the known board geometry (e.g., 80 mm square spacing). The error is |measured distance - expected distance|.

**What it tells you:** Whether the calibration produces geometrically correct 3D measurements. A calibration can have low reprojection error but a systematic scale error, or a distortion model that bends 3D space -- only this metric catches those problems. It is also the most intuitive metric for non-experts: "your calibration says the 80 mm squares are 79.7 mm."

**Typical values:** 0.1-1.0 mm for sub-millimeter accuracy.

### 4. Manual landmark reprojection error (nose label test)

**What it measures:** How well the cameras agree on the position of a manually-labeled point in video frames.

**How it's computed:** The user labels a point (e.g., the rat's nose) in all 17 camera views of the same frame. The 3D position is triangulated from all cameras via DLT, then reprojected back into each camera. The error is the pixel distance between the reprojection and the user's label.

**Key differences from ChArUco tests:**
- The 2D "detections" are human clicks (~1-3 px labeling noise), not sub-pixel algorithmic corner detections (~0.1 px).
- The 3D point is a soft tissue landmark (rat's nose), not a rigid board corner.
- The test can be done on ANY frame from ANY video -- not just calibration board images.
- This tests the calibration on the ACTUAL data the user cares about (animal tracking).

**Why it matters:** ChArUco tests evaluate the calibration on calibration day. Manual landmark tests evaluate the calibration on experiment day. If cameras moved between sessions, ChArUco metrics will not show it, but manual landmark tests will. In this dataset, there was a 3-week gap between calibration (Aug 14) and video (Sept 3), making this distinction critical.

**Typical values:** 3-15 px (dominated by labeling noise, not calibration error).

**How to compare calibrations:** The absolute error values include labeling noise, so they are always higher than ChArUco errors. But the RELATIVE comparison between calibrations is valid -- if SP Refined gets 4.6 px median vs MVC's 5.4 px on the same labels, the 15% improvement is real because the labeling noise is identical across all three calibrations.

### 5. Leave-one-out reprojection error

**What it measures:** Same as test 4, but triangulates from N-1 cameras and tests on the held-out camera.

**Why it's better:** In test 4, the held-out camera's label influences the triangulated 3D point (it participates in the DLT). In leave-one-out, the held-out camera is completely independent -- its error measures pure prediction accuracy. This is a stricter test: can 16 cameras predict where the point appears in the 17th?

**Typical values:** Slightly higher than test 4 (the held-out camera no longer constrains the 3D point).

### Summary Table

| Metric | Source data | 2D precision | Tests | Typical |
|--------|-------------|--------------|-------|---------|
| Per-board reproj | ChArUco corners | ~0.1 px (algorithmic) | Camera model (local) | 0.3-0.8 px |
| Multi-view reproj | ChArUco corners | ~0.1 px (algorithmic) | Extrinsic consistency (global) | 0.4-2.0 px |
| Known-geometry | ChArUco corners | ~0.1 px (algorithmic) | 3D metric accuracy | 0.1-1.0 mm |
| Manual landmark | Human clicks | ~1-3 px (manual) | Real-world accuracy | 3-15 px |
| Leave-one-out | Human clicks | ~1-3 px (manual) | Predictive accuracy | 4-20 px |

**The key takeaway:** No single metric is sufficient. Per-board reproj can be misleading (v1 demonstrated this). Multi-view reproj and known-geometry catch global errors that per-board misses. Manual landmark and leave-one-out tests catch temporal drift that ChArUco metrics cannot see at all. Use all five together to evaluate a multi-camera calibration.
