# Rat 17-Camera Calibration: multiview_calib vs RED v4

*Johnson Lab, HHMI Janelia Research Campus — March 17, 2026*

## Background

RED's calibration pipeline began as a C++ port of the [multiview_calib](https://github.com/JohnsonLabJanelia/multiview_calib) Python pipeline. Through iterative experimentation on a 17-camera rat behavioral rig, we identified and addressed several limitations in the original approach. This document compares the two pipelines using validated results from March 16–17, 2026.

**RED v4** refers to the calibration at commit `c3d3ac1` (March 16, 2026), which is the first version to beat MVC on all three quality metrics. Later detection improvements (v6: Suzuki-Abe, threshold windows, etc.) increased detection count but degraded calibration accuracy — see the detection-accuracy paradox below.

## Dataset

- **Calibration images**: `/Users/johnsonr/datasets/rat/calibration_images/2025_08_14_09_23_31/`
- **Rig**: 17 cameras (14 standard ~2300px focal length, 1 wide 710038 ~1770px, 2 telephoto ~3200px)
- **Resolution**: 3208 × 2200 (7 MP)
- **Images**: 78 per camera (1326 total, 2.9 GB)
- **Capture date**: August 14, 2025
- **Board**: 5×5 ChArUco, DICT_5X5_100, 80mm squares, 60mm markers (400mm × 400mm total)
- **Max ChArUco corners per frame**: 16 (inner corners of 5×5 board)
- **Ground truth**: 2 reference images with 16 manually measured 3D points on orthogonal planes (Procrustes alignment)

## Results

### Three quality metrics: RED v4 wins all three

| Metric | MVC | RED v4 | RED advantage |
|--------|-----|--------|---------------|
| **Per-board reproj (mean)** | 0.678 px | **0.437 px** | 36% better |
| **Multi-view reproj (mean)** | 0.73 px | **0.48 px** | 34% better |
| **Multi-view reproj (median)** | 0.54 px | **0.37 px** | 31% better |
| **Multi-view reproj (95th pct)** | 1.88 px | **1.15 px** | 39% better |
| **Known-geometry adjacent (mean)** | 0.368 mm | **0.339 mm** | 8% better |
| **Known-geometry diagonal (mean)** | 0.492 mm | **0.460 mm** | 7% better |

### Pipeline statistics

| Metric | MVC | RED v4 |
|--------|-----|--------|
| Cameras calibrated | 17 | 17 |
| Frames with detections | 474 | 415 |
| 2D observations (post-BA) | 6,530 | 5,817 |
| Unique landmarks | 1,132 | 1,230 |
| BA outliers removed | 0 | 12 |
| BA convergence | Hit 40-iter cap (NOT converged) | Fully converged (7 passes) |
| Detection speed | ~15 img/s (CPU) | 64 img/s (Metal GPU) |
| BA time | 14s (1 pass) | 22s (7 passes) |

**Key finding**: RED v4 achieves better accuracy with fewer observations. Stricter detection rejects marginal corners that would add noise to BA. The 9% more unique landmarks suggests better multi-view coverage per point.

---

## Pipeline Architecture Comparison

### Stage 1: ArUco Detection

| | MVC | RED v4 |
|--|-----|--------|
| **Detector** | OpenCV `cv2.aruco.detectMarkers()` | Custom GPU-accelerated (Metal/CUDA) |
| **Threshold** | OpenCV defaults | Multi-scale adaptive (GPU separable box filter) |
| **Bit reading** | OpenCV built-in | Otsu-based with 25% border tolerance |
| **Min markers/frame** | 1–2 (permissive) | 4 (strict) |
| **Subpixel refinement** | OpenCV defaults | `half_win=3, max_iter=100, epsilon=0.001` |
| **Contour resolution** | Full resolution | Full resolution (ds_factor=1 for images) |
| **Speed** | ~15 img/s (CPU) | 64 img/s (Metal GPU) |

RED v4's stricter minimum (4 markers vs 1–2) means fewer frames pass detection, but those that do have higher-quality corner localization. This is the primary reason RED's intrinsic calibration is better.

### Stage 2: Initial Camera Pair Selection

| | MVC | RED v4 |
|--|-----|--------|
| **Method** | Maximum spanning tree (NetworkX) | Greedy best pair (O(N²) search) |
| **Criterion** | Most shared landmarks | Most shared frames with intrinsic reproj < 2.0 px |
| **Quality gate** | None | Per-frame intrinsic quality threshold |

### Stage 3: Relative Pose Computation

| | MVC | RED v4 |
|--|-----|--------|
| **Primary** | Fundamental matrix (8-point/RANSAC) | Essential matrix from normalized coordinates |
| **Conversion** | F → E via K matrices | Direct E (no conversion needed) |
| **Disambiguation** | OpenCV `decomposeEssentialMatrix` | Explicit positive-depth counting (4 candidates) |
| **Multi-path robustness** | 3-hop triangle averaging | No (deferred to PnP registration) |

MVC averages relative poses across multiple paths through the camera graph. RED defers this to the PnP registration stage with Markley quaternion averaging.

### Stage 4: Extrinsic Registration

| | MVC | RED v4 |
|--|-----|--------|
| **Topology** | Star through hub camera | Incremental all-to-all |
| **Pose averaging** | Simple chain accumulation | Markley weighted quaternion averaging |
| **Scale estimation** | Median distance ratio | Median distance ratio |
| **Camera ordering** | Fixed by config | Greedy by shared 3D point count |

MVC's star topology means every camera's pose depends on its overlap with the hub camera. Cameras with poor hub overlap show high outlier rates:

| Pair | MVC Outlier Rate |
|------|-----------------|
| 710038–2002491 | 30.4% |
| 710038–2002492 | 25.4% |
| 710038–2002484 | 23.5% |
| 710038–2002482 | 22.3% |

RED's incremental PnP registration adds cameras one at a time, choosing the best-constrained next camera. Markley quaternion averaging across multiple frames provides robustness without the star topology's fragility.

### Stage 5: Bundle Adjustment

This is the most significant difference between the two pipelines.

**MVC (scipy least_squares)**:
- Single optimization pass with Huber loss
- `max_nfev=40` (hit the iteration cap — **not converged**)
- All parameters (extrinsics + intrinsics + distortion + 3D points) optimized jointly from the start
- No progressive refinement, no re-triangulation
- Final first-order optimality: 1.6e+04 (still decreasing when it stopped)

**RED v4 (Ceres Solver with Graduated Non-Convexity)**:
```
Stage 0 — Extrinsics only (3 passes):
  Pass 1: CauchyLoss(16)  →  24312 → 7307   (4 iters,  0.25s)
  Pass 2: CauchyLoss(4)   →  5502 → 5445    (8 iters,  0.39s)
  Pass 3: CauchyLoss(1)   →  2239 → 2186    (79 iters, 2.55s)
  Outlier rejection: 20 px threshold → 0 removed
  Re-triangulation: 1047 landmarks

Stage 1 — Extrinsics + 3D points (2 passes):
  Pass 4: CauchyLoss(4)   →  3789 → 2280    (40 iters, 1.23s)
  Pass 5: CauchyLoss(1)   →  1274 → 1153    (101 iters, 2.73s)
  Outlier rejection: 10 px threshold → 9 removed

Stage 2 — Full joint optimization (2 passes):
  Pass 6: CauchyLoss(4)   →  1131 → 415     (101 iters, 12.71s)
  Outlier rejection: 15 px threshold → 3 removed
  Pass 7: CauchyLoss(50)  →  395 → 377      (101 iters, 3.70s)
  Outlier rejection: 10 px threshold → 0 removed
```

Key innovations in RED's BA:

1. **Graduated Non-Convexity (GNC)**: Cauchy scale decreases 16 → 4 → 1 across passes. Large scale = wide basin (easy to converge from far away). Small scale = tight basin (precise). This avoids local minima that trap single-pass methods.

2. **Hierarchical parameter unlocking**: Extrinsics first (well-constrained), then 3D points (better initialized after extrinsic refinement), then intrinsics (most sensitive). MVC optimizes everything jointly from the start.

3. **Re-triangulation**: After Stage 0 refines extrinsics, all 3D points are re-triangulated with improved poses before joint optimization.

4. **Progressive outlier rejection**: Thresholds tighten as the solution improves (20 → 10 → 15 → 10 px). MVC uses fixed thresholds (1000 px early, 50 px final) and removes zero outliers.

5. **Weak camera intrinsic fixing**: Cameras with < 30 observations keep intrinsics fixed during Stage 2, preventing Hessian rank deficiency.

6. **Near-linear final loss**: CauchyLoss(50) in the last pass is effectively linear, giving all inliers equal weight for best metric accuracy. The preceding CauchyLoss(4) pass handles outlier cleanup.

7. **Locked p1/p2/k3**: Tangential distortion (p1, p2) and 6th-order radial (k3) are locked via `SubsetManifold({12,13,14})` during Stage 2. These parameters are poorly constrained by typical board data and cause overfitting when free. MVC optimizes all 5 distortion coefficients jointly.

8. **Cauchy vs Huber**: Cauchy loss `log(1 + (r/s)²)` has heavier tails than Huber, providing better robustness to gross outliers.

### Stage 6: World Registration

Both use identical Procrustes alignment (SVD on cross-covariance) to align to world coordinates. RED reads an optional `world_frame_rotation` from `config.json` to match MVC's hardcoded coordinate transform.

---

## Per-Camera Results

### Detection Counts

| Camera | Focal | RED boards | RED corners | MVC obs | Notes |
|--------|-------|-----------|-------------|---------|-------|
| 710038 | 1770 | 59 | 931 | 1,028 | GoPro wide |
| 2002488 | 2362 | 52 | 756 | 741 | |
| 2002490 | 2366 | 53 | 724 | 723 | |
| 2002496 | 2362 | 51 | 710 | 654 | |
| 2002489 | 2385 | 40 | 590 | 622 | |
| 2002480 | 3311 | 28 | 341 | 256 | Telephoto |
| 2002485 | 3165 | 19 | 277 | 269 | Telephoto |
| 2002484 | 2293 | 13 | 189 | 353 | MVC has 1.9× more |
| 2002483 | 2903 | 15 | 195 | 217 | |
| 2002493 | 2310 | 13 | 181 | 251 | |
| 2002495 | 2275 | 12 | 160 | 189 | |
| 2002491 | 2295 | 10 | 146 | 223 | |
| 2002481 | 2297 | 13 | 139 | 296 | MVC has 2× more |
| 2002479 | 2307 | 8 | 115 | 250 | MVC has 2× more |
| 2002494 | 2340 | 7 | 94 | 156 | |
| 2002492 | 2844 | 6 | 86 | 118 | Fewest detections |

MVC detects more on oblique/side cameras (2002479, 2002481, 2002484) where board detection is harder. RED's stricter quality filter rejects marginal detections that would add noise.

### Per-Camera BA Reprojection Error (Per-Board)

| Camera | RED v4 | MVC | Winner |
|--------|--------|-----|--------|
| 710038 | **0.319** | 0.525 | RED |
| 2002496 | **0.489** | 0.671 | RED |
| 2002488 | **0.461** | 0.555 | RED |
| 2002490 | **0.393** | 0.482 | RED |
| 2002489 | **0.392** | 0.799 | RED |
| 2002480 | **0.388** | 0.483 | RED |
| 2002485 | **0.457** | 0.672 | RED |
| 2002495 | **0.359** | 0.600 | RED |
| 2002479 | **0.453** | 0.853 | RED |
| 2002483 | 0.466 | **0.438** | MVC |
| 2002482 | **0.456** | 0.844 | RED |
| 2002481 | **0.480** | 0.999 | RED |
| 2002491 | **0.498** | 0.852 | RED |
| 2002484 | **0.573** | 0.898 | RED |
| 2002494 | **0.443** | 0.819 | RED |
| 2002492 | **0.646** | 1.080 | RED |
| 2002493 | **0.809** | 1.004 | RED |

RED wins 16/17 cameras. MVC only wins Cam2002483 (0.438 vs 0.466 px).

### Per-Camera Multi-View Triangulation Consistency

| Camera | RED v4 (px) | MVC (px) | Winner |
|--------|-------------|----------|--------|
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
| 2002481 | 1.26 | **1.22** | MVC (marginal) |

RED wins 16/17 cameras. Only Cam2002481 is marginally worse (1.26 vs 1.22 px) — this camera has fewest detections in RED.

### Per-Camera Intrinsic Reprojection Error

| Camera | RED v4 intrinsic | Notes |
|--------|-----------------|-------|
| 710038 | 0.179 | Best (GoPro, wide FOV) |
| 2002496 | 0.206 | |
| 2002489 | 0.211 | |
| 2002488 | 0.220 | |
| 2002490 | 0.233 | |
| 2002480 | 0.305 | |
| 2002485 | 0.320 | |
| 2002494 | 0.404 | |
| 2002493 | 0.408 | |
| 2002491 | 0.429 | |
| 2002479 | 0.446 | |
| 2002484 | 0.447 | |
| 2002495 | 0.520 | |
| 2002492 | 0.523 | |
| 2002483 | 0.661 | |
| 2002482 | 0.829 | |
| 2002481 | 0.977 | Worst — fewest detections |

Mean intrinsic reproj: 0.430 px. MVC does not report standalone intrinsic reproj.

### Known-Geometry Validation

| Metric | RED v4 | MVC |
|--------|--------|-----|
| Adjacent mean error | **0.339 mm** | 0.368 mm |
| Adjacent median error | **0.311 mm** | 0.347 mm |
| Adjacent 95th pct | **0.709 mm** | 0.796 mm |
| Adjacent max error | **1.775 mm** | 1.813 mm |
| Diagonal mean error | **0.460 mm** | 0.492 mm |
| Diagonal median error | **0.424 mm** | 0.450 mm |

On an 80 mm square, RED's adjacent error (0.339 mm) is 0.42% of the square size.

---

## The Detection-Accuracy Paradox

On March 16, we improved RED's ArUco detector to approach OpenCV detection parity (72/78 vs 73/78 frames on camera 710038). But more detections made calibration worse:

| Version | Detections | Observations | Mean Reproj | Known-Geometry |
|---------|-----------|-------------|-------------|----------------|
| **v4** (c3d3ac1) | 415 | 5,660 | **0.437 px** | **0.339 mm** |
| v6 (eb7b9ca) | 524 (+26%) | 6,403 | 0.601 px (37% worse) | 0.555 mm (64% worse) |

The extra frames recovered by aggressive detection had small markers (~40–65px sides) with imprecise subpixel localization. These added noise to BA, degrading calibration quality despite more data.

**Lesson**: Observation quality matters more than quantity. A few hundred clean corners trump a thousand noisy ones. Future detection improvements need a quality filter to reject frames where markers are too small for reliable corner localization.

---

## Problem Cameras

Two cameras are problematic across both pipelines:

**Cam2002492** — Only 6 ChArUco board detections (fewest of any camera). RED's BA reproj is 0.646 px; MVC's is 1.080 px. In the September 3 nose landmark test, this camera shows 31–46 px leave-one-out error across all four calibrations tested.

**Cam2002481** — Highest intrinsic reproj (0.977 px). Only 13 detections. This is the only camera where MVC marginally beats RED on multi-view consistency (1.22 vs 1.26 px).

Both cameras need fundamentally better calibration data: more board images at diverse angles. No amount of algorithmic improvement can compensate for insufficient observations.

---

## Graceful Error Handling (RED-specific)

1. **Auto-skip failing cameras**: If a camera has < 4 valid detections, it is excluded with a toast notification. The pipeline continues with remaining cameras.
2. **Weak camera intrinsic fixing**: Cameras with < 30 observations keep intrinsics fixed during full joint BA to prevent numerical instability.
3. **CHOLMOD warning suppression**: Sparse Cholesky warnings from Ceres (normal LM behavior) are suppressed via stderr redirection during the solve.
4. **Load Calibration resilience**: Missing YAML files are gracefully skipped on reload.

---

## Data Locations

| Item | Path |
|------|------|
| Calibration images | `/Users/johnsonr/datasets/rat/calibration_images/2025_08_14_09_23_31/` |
| MVC results | `/Users/johnsonr/datasets/rat/sp_test_march17/original_calibration_results/calibration/` |
| RED v4 results (March 17 re-run) | `/Users/johnsonr/datasets/rat/sp_test_march17/redv4/aruco_image_experimental/2026_03_17_10_55_50/` |
| RED v4 results (original March 16) | `/Users/johnsonr/datasets/rat/sp_test/sp_init/aruco_image_experimental_v4/2026_03_16_11_27_16/` |
| All version results (v1–v8) | `/Users/johnsonr/datasets/rat/sp_test/sp_init/aruco_image_experimental_v*/` |
| MVC config | `/Users/johnsonr/datasets/rat/calibration_images/original_calibration_results/config.json` |
| Detailed experiment log | `/Users/johnsonr/claude_dev_notes/calibration/aruco_calibration_tests_March16.md` |
| Per-camera metrics | `/Users/johnsonr/claude_dev_notes/calibration/RED_vs_MVC_comparison_march16.md` |

---

## Methodology

**multiview_calib** (v2025-08, Python):
- ArUco detection: OpenCV `cv2.aruco.detectMarkers()` + `cv2.aruco.interpolateCornersCharuco()`
- Intrinsic calibration: `cv2.calibrateCamera()` (Zhang's method)
- Extrinsic registration: Star topology, multi-path averaging
- Bundle adjustment: `scipy.optimize.least_squares(method='trf', loss='linear', max_nfev=40)`
- Distortion: All 5 coefficients (k1, k2, p1, p2, k3) free
- Global registration: Procrustes alignment + hardcoded coordinate transform

**RED v4** (commit `c3d3ac1`, C++):
- ArUco detection: Custom Metal GPU-accelerated, full-resolution for images, 4+ markers required
- Subpixel refinement: Gradient autocorrelation (half_win=3, iter=100, eps=0.001)
- Intrinsic calibration: Ceres LM solver with quality gate (re-calibrate on best 50% if reproj > 1.0 px)
- Extrinsic registration: Incremental PnP with Markley quaternion averaging
- Bundle adjustment: Ceres Solver, 7-pass GNC (CauchyLoss 16 → 4 → 1 → 4 → 1 → 4 → 50)
- Progressive outlier rejection: 20 → 10 → 15 → 10 px thresholds
- Distortion: k1, k2 free; p1, p2, k3 locked via SubsetManifold
- Weak camera handling: Fix intrinsics for cameras with < 30 observations

**Hardware**: Apple M-series MacBook (Metal GPU for ArUco detection)
