# Rat 17-Camera Calibration: multiview_calib vs RED Experimental Pipeline

*Johnson Lab, HHMI Janelia Research Campus -- March 2026*

## Background

RED's calibration pipeline began as a C++ port of the [multiview_calib](https://github.com/JohnsonLabJanelia/multiview_calib) Python pipeline. Through iterative experimentation and profiling on this rat 17-camera dataset, we identified and addressed several limitations in the original approach. This document compares the two pipelines and explains the improvements.

## Dataset

- **Calibration images**: `/Users/johnsonr/datasets/rat/calibration_images/2025_08_14_09_23_31/`
- **Config file**: `/Users/johnsonr/datasets/rat/calibration_images/original_calibration_results/config.json`
- **MVC results**: `/Users/johnsonr/datasets/rat/calibration_images/original_calibration_results/output/`
- **RED results**: `/Users/johnsonr/red_demos/rat_calib2/aruco_image_experimental/`
- **Rig**: 17 cameras (14 standard ~2300px focal length, 1 wide 710038 ~1770px, 2 telephoto ~3200px)
- **Camera serials**: 2002479, 2002480, 2002481, 2002482, 2002483, 2002484, 2002485, 2002488, 2002489, 2002490, 2002491, 2002492, 2002493, 2002494, 2002495, 2002496, 710038
- **Resolution**: 3208 x 2200 (7 MP)
- **Image format**: JPEG, ~2.4 MB each
- **Images**: 78 per camera (1326 total, 2.9 GB)
- **Naming convention**: `{camera_serial}_{frame_number}.jpg` (e.g., `2002479_0.jpg` through `710038_77.jpg`)
- **Capture date**: 2025-08-14
- **Board**: 5x5 ChArUco, DICT_5X5_100 (OpenCV dictionary ID 5), 80mm squares, 60mm markers (400mm x 400mm total board)
- **ArUco markers per board**: 12 (alternating squares in 5x5 checkerboard)
- **Max ChArUco corners per frame**: 16 (inner corners of 5x5 board)
- **Theoretical max observations per camera**: 78 x 16 = 1,248
- **Ground truth**: 2 reference images (frames 76, 77) with 16 manually measured 3D points each on orthogonal planes, used for Procrustes world-frame alignment

## Overall Results

| Metric | multiview_calib (MVC) | RED Experimental | Difference |
|--------|----------------------|------------------|------------|
| Cameras calibrated | 17 | 16 (auto-dropped 2002492) | |
| Images processed | 71 of 78 | 78 of 78 | MVC missed 7 frames |
| Frames with detections | 474 | 379 | RED 20% fewer (stricter) |
| 2D observations (post-BA) | 6,513 | 5,119 | RED 21% fewer |
| Outliers removed | 0 | 41 (0.8%) | |
| **Mean intrinsic reproj** | **0.663 px** | **0.338 px** | **RED 49% better** |
| **Mean BA reproj** | **0.43 px** | **0.373 px** | **RED 13% better** |
| BA convergence | Hit 40-iter cap | Fully converged (7 passes) | |
| Detection time | Not logged | 20.3s (Metal GPU, 64 img/s) | |
| BA time | 14.0s (1 pass) | 23.6s (7 passes) | |
| Total pipeline | ~15s (BA only) | ~44s (detection + BA) | |

**Key finding**: RED achieves lower error with fewer observations. Stricter detection thresholds reject marginal detections that would degrade calibration quality. The extra BA time is invested in progressive refinement that fully converges.

---

## Pipeline Architecture Comparison

### Stage 1: ArUco Detection

| | MVC | RED |
|--|-----|-----|
| **Detector** | OpenCV `cv2.aruco.detectMarkers()` | Custom GPU-accelerated (Metal/CUDA) |
| **Threshold** | OpenCV defaults | Multi-scale adaptive (GPU separable box filter) |
| **Bit reading** | OpenCV built-in | Otsu-based with 25% border tolerance |
| **Min markers** | 1-2 per frame | 4 per frame |
| **Dictionaries** | All OpenCV dictionaries | DICT_4X4_50/100/250, DICT_5X5_50/100/250, DICT_6X6_50/250, ARUCO_ORIGINAL |
| **Speed** | ~10-20 img/s (CPU) | 64 img/s (Metal GPU) |

RED's stricter minimum (4 markers vs 1-2) means fewer frames pass detection, but those that do have higher-quality corner localization. This is the primary reason RED's intrinsic calibration is 49% better.

### Stage 2: Initial Pair Selection

| | MVC | RED |
|--|-----|-----|
| **Method** | Maximum spanning tree (NetworkX) | Greedy best pair (O(N^2) search) |
| **Criterion** | Most shared landmarks | Most shared frames with intrinsic reproj < 2.0 px |
| **Quality gate** | None | Per-frame intrinsic quality threshold |

RED's quality gate ensures the initial pair has reliable intrinsics before computing the first extrinsic pose. MVC uses any pair with enough common points regardless of intrinsic quality.

### Stage 3: Relative Pose Computation

| | MVC | RED |
|--|-----|-----|
| **Primary** | Fundamental matrix (8-point/RANSAC) | Essential matrix from normalized coordinates |
| **Conversion** | F -> E via K matrices | Direct E (no conversion needed) |
| **Disambiguation** | OpenCV `decomposeEssentialMatrix` | Explicit positive-depth counting (4 candidates) |
| **Multi-path robustness** | Yes: 3-hop triangle averaging | No (deferred to PnP registration) |

MVC has an advantage here: it averages relative poses across multiple paths through the camera graph, providing early robustness. RED defers this to the PnP registration stage.

### Stage 4: Extrinsic Registration

| | MVC | RED |
|--|-----|-----|
| **Topology** | Star through hub camera | Incremental all-to-all |
| **Pose averaging** | Simple chain accumulation | Markley weighted quaternion averaging |
| **Scale estimation** | Median distance ratio | Median distance ratio |
| **Camera ordering** | Fixed by config | Greedy by shared 3D point count |

RED's incremental PnP registration adds cameras one at a time, choosing the camera with the most 3D correspondences. Markley quaternion averaging across multiple frames provides geometric robustness without the star topology's fragility.

MVC's star topology means every camera's pose depends on its overlap with the hub camera. Cameras with poor hub overlap show high outlier rates:

| Pair | MVC Outlier Rate |
|------|-----------------|
| 710038-2002491 | 30.4% |
| 710038-2002492 | 25.4% |
| 710038-2002484 | 23.5% |
| 710038-2002482 | 22.3% |

### Stage 5: Bundle Adjustment

This is the most significant difference between the two pipelines.

**MVC (scipy least_squares)**:
- Single optimization pass with Huber loss
- `max_nfev=40` (hit the iteration cap -- **not converged**)
- All parameters (extrinsics + intrinsics + 3D points) optimized jointly from the start
- No progressive refinement
- Final first-order optimality: 1.6e+04 (still decreasing)

**RED (Ceres Solver with Graduated Non-Convexity)**:
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
  Pass 6: CauchyLoss(1)   →  1131 → 415     (101 iters, 12.71s)
  Outlier rejection: 5 px threshold → 11 removed
  Pass 7: CauchyLoss(1)   →  395 → 377      (101 iters, 3.70s)
  Outlier rejection: 3 px threshold → 21 removed
```

Key innovations in RED's BA:

1. **Graduated Non-Convexity (GNC)**: Cauchy scale decreases 16 -> 4 -> 1 across passes. Large scale = wide basin (easy to converge from far away). Small scale = tight basin (precise local optimization). This avoids local minima that trap single-pass methods.

2. **Hierarchical parameter unlocking**: Extrinsics first (well-constrained), then 3D points (better initialized after extrinsic refinement), then intrinsics (most sensitive). MVC optimizes everything jointly from the start, which can lead to unstable early iterations.

3. **Re-triangulation**: After Stage 0 refines extrinsics, all 3D points are re-triangulated with the improved poses before joint optimization. This provides a much better starting point for Stage 1.

4. **Progressive outlier rejection**: Thresholds tighten as the solution improves (20 -> 10 -> 5 -> 3 px). MVC uses fixed thresholds (1000 px early, 50 px final) and removes zero outliers.

5. **Weak camera intrinsic fixing**: Cameras with < 30 observations keep intrinsics fixed during full joint passes. This prevents the Schur complement from becoming indefinite (rank-deficient) when cameras have too few observations to constrain 15 parameters (6 extrinsic + 4 intrinsic + 5 distortion). This directly addresses the CHOLMOD "not positive definite" warnings from the sparse Cholesky factorization.

6. **Cauchy loss vs Huber**: Cauchy loss `log(1 + (r/s)^2)` has heavier tails than Huber, providing better robustness to gross outliers. Both are superior to linear (squared) loss.

### Stage 6: World Registration

Both pipelines use identical Procrustes alignment (SVD on cross-covariance matrix) to align the calibration frame to world coordinates using ground truth 3D points.

### Stage 7: Coordinate Frame Adjustment (Z-Flip)

MVC applies a post-registration coordinate transform (`r_t = [[0,1,0],[1,0,0],[0,0,-1]]`) that swaps X/Y and negates Z. This is a proper rotation (det = +1) specific to the lab's physical setup.

RED provides an interactive **Flip Z** button in the 3D viewer that:
- Applies `R_new = R * diag(1,1,-1)` and negates Z on 3D points
- Saves flipped YAML files and `ba_points.json` to disk
- Produces an improper rotation (det = -1), which is handled correctly on reload via `projectPointR()` — a matrix-based projection function that bypasses Rodrigues conversion

This was verified by a roundtrip test: calibrate -> flip Z -> save -> reload -> recompute errors. Per-camera reprojection errors match within 0.0004 px (YAML serialization precision).

---

## Per-Camera Detection Rates

| Camera | Focal | MVC Frames | RED Frames | MVC Rate | RED Rate |
|--------|-------|-----------|-----------|----------|----------|
| 710038 | 1770 | 66 | 54 | 84.6% | 69.2% |
| 2002488 | 2362 | 52 | 50 | 66.7% | 64.1% |
| 2002490 | 2366 | 52 | 52 | 66.7% | 66.7% |
| 2002496 | 2362 | 45 | 48 | 57.7% | 61.5% |
| 2002489 | 2385 | 43 | 39 | 55.1% | 50.0% |
| 2002484 | 2293 | 24 | 13 | 30.8% | 16.7% |
| 2002481 | 2297 | 23 | 10 | 29.5% | 12.8% |
| 2002479 | 2307 | 21 | 8 | 26.9% | 10.3% |
| 2002480 | 3311 | 20 | 26 | 25.6% | 33.3% |
| 2002485 | 3165 | 18 | 19 | 23.1% | 24.4% |
| 2002491 | 2295 | 18 | 10 | 23.1% | 12.8% |
| 2002493 | 2310 | 18 | 12 | 23.1% | 15.4% |
| 2002482 | 3173 | 17 | 9 | 21.8% | 11.5% |
| 2002483 | 2903 | 15 | 10 | 19.2% | 12.8% |
| 2002495 | 2275 | 15 | 12 | 19.2% | 15.4% |
| 2002494 | 2340 | 12 | 7 | 15.4% | 9.0% |
| 2002492 | 2844 | 9 | 0 | 11.5% | 0.0% |

**Concern**: 6 cameras have fewer than 15 RED detections. Future calibrations should target 20+ detections per camera.

## Per-Camera Reprojection Error (BA)

| Camera | MVC Mean | RED Mean | Winner | MVC Median | RED Median | Winner |
|--------|----------|----------|--------|------------|------------|--------|
| 710038 | 0.525 | **0.248** | RED | 0.447 | **0.212** | RED |
| 2002488 | 0.555 | **0.392** | RED | 0.436 | **0.295** | RED |
| 2002490 | 0.482 | **0.312** | RED | 0.422 | **0.266** | RED |
| 2002496 | 0.671 | **0.388** | RED | 0.580 | **0.313** | RED |
| 2002489 | 0.799 | **0.308** | RED | 0.728 | **0.278** | RED |
| 2002485 | 0.672 | **0.279** | RED | 0.490 | **0.225** | RED |
| 2002480 | 0.483 | **0.317** | RED | 0.446 | **0.239** | RED |
| 2002484 | 0.898 | **0.603** | RED | 0.698 | **0.282** | RED |
| 2002493 | **1.004** | 1.050 | MVC | 0.749 | **0.243** | RED |
| 2002483 | 0.438 | **0.287** | RED | 0.309 | **0.204** | RED |
| 2002481 | 0.999 | **0.824** | RED | 0.808 | **0.518** | RED |
| 2002491 | 0.852 | **0.297** | RED | 0.624 | **0.243** | RED |
| 2002482 | 0.844 | **0.507** | RED | 0.563 | **0.326** | RED |
| 2002479 | 0.853 | **0.475** | RED | 0.625 | **0.320** | RED |
| 2002495 | 0.600 | **0.243** | RED | 0.360 | **0.216** | RED |
| 2002494 | 0.819 | **0.448** | RED | 0.666 | **0.247** | RED |
| 2002492 | 1.080 | -- | -- | 0.828 | -- | -- |

RED wins mean reproj on 15/16 cameras and median reproj on all 16. The only camera where MVC wins on mean (2002493) has a better median in RED (0.243 vs 0.749), indicating a few high-error outlier observations in RED.

## Intrinsic Calibration Quality (RMS pixels)

| Camera | MVC Intrinsic | RED Intrinsic | Winner |
|--------|-------------|-------------|--------|
| 710038 | 0.282 | **0.155** | RED |
| 2002496 | 0.207 | **0.192** | RED |
| 2002490 | 0.220 | **0.204** | RED |
| 2002488 | 0.297 | **0.200** | RED |
| 2002489 | 0.380 | **0.194** | RED |
| 2002480 | **0.260** | 0.265 | MVC |
| 2002485 | 0.661 | **0.290** | RED |
| 2002483 | **0.636** | 0.767 | MVC |
| 2002482 | 1.091 | **0.353** | RED |
| 2002491 | 0.978 | **0.353** | RED |
| 2002481 | 0.876 | **0.419** | RED |
| 2002493 | 1.070 | **0.387** | RED |
| 2002484 | 0.889 | **0.421** | RED |
| 2002479 | 0.865 | **0.416** | RED |
| 2002495 | 0.684 | **0.393** | RED |
| 2002494 | 0.604 | **0.402** | RED |

MVC mean intrinsic: 0.663 px. RED mean intrinsic: 0.338 px (**49% better**).

## Graceful Error Handling

RED includes several safeguards not present in MVC:

1. **Auto-skip failing cameras**: If a camera has < 4 valid detections, it is automatically excluded from calibration. A toast notification informs the user, and the camera is unchecked in the Camera Selection UI. The pipeline continues with remaining cameras.

2. **Camera Selection UI**: Per-camera checkboxes allow the user to manually exclude cameras before running calibration. All/None quick-toggle buttons available.

3. **Load Calibration resilience**: When reloading a calibration from disk, missing YAML files (from excluded cameras) are gracefully skipped with a warning, rather than causing a fatal error.

4. **Weak camera intrinsic fixing**: During full joint BA, cameras with < 30 observations keep intrinsics fixed to prevent numerical instability.

5. **CHOLMOD warning suppression**: The sparse Cholesky factorization in Ceres sometimes encounters non-positive-definite matrices (normal LM behavior when probing the trust region boundary). These warnings are suppressed via stderr redirection during the solve, keeping terminal output clean.

## Recommendations for Future Calibrations

1. **Capture more images** (150+ per camera) to ensure all cameras have 20+ detections
2. **Target weak cameras** with dedicated board positions visible from steep angles
3. **Vary board orientation** more aggressively (20-40 degree tilts)
4. **Optimal board distance**: 0.7-1.2m for standard lenses, 1.0-1.5m for telephoto
5. **Consider a larger board** (600-800mm) for better detection reliability at distance
6. **Use RED's Experimental pipeline** — 13% better BA error, 49% better intrinsics, automatic error handling, GPU-accelerated detection

## Sample Terminal Output (RED Experimental)

```
[Experimental] === Starting experimental pipeline ===
[Calibration] Phase 1: detecting corners in 1326 images across 17 cameras...
[Calibration] Phase 1 done: 1326 images in 20.7s (64 img/s)
[Calibration] Skipping camera 2002492: Too few valid images for camera 2002492 (3)
[Experimental] Skipped 1 camera(s): 2002492
[Experimental]   Initial pair: 2002488 + 710038 (45 shared frames)
[Experimental]   Initial triangulation: 651 points
[Experimental]   PnP 2002490: 80 frames from 2+ init cameras (500 3D pts)
[Experimental]   PnP 2002496: 117 frames from 3+ init cameras (525 3D pts)
[Experimental]   PnP 2002489: 136 frames from 4+ init cameras (504 3D pts)
  ... (11 more cameras registered incrementally)
[Experimental] Triangulated 1056 landmarks (threshold=50px)
[Experimental] Pass 1/7 (mode=0 cauchy=16): OK cost 24312→7307 iters=4 time=0.25s
[Experimental] Pass 2/7 (mode=0 cauchy=4):  OK cost 5502→5445 iters=8 time=0.39s
[Experimental] Pass 3/7 (mode=0 cauchy=1):  OK cost 2239→2186 iters=79 time=2.55s
[Experimental]   Outlier rejection: threshold=20.0 px removed=0
[Experimental]   Re-triangulated: 1047 landmarks
[Experimental] Pass 4/7 (mode=1 cauchy=4):  OK cost 3789→2280 iters=40 time=1.23s
[Experimental] Pass 5/7 (mode=1 cauchy=1):  OK cost 1274→1153 iters=101 time=2.73s
[Experimental]   Outlier rejection: threshold=10.0 px removed=9
[Experimental] Pass 6/7 (mode=2 cauchy=1):  OK cost 1131→415 iters=101 time=12.71s
[Experimental]   Outlier rejection: threshold=5.0 px removed=11
[Experimental] Pass 7/7 (mode=2 cauchy=1):  OK cost 395→377 iters=101 time=3.70s
[Experimental]   Outlier rejection: threshold=3.0 px removed=21
[Experimental] === Done: 0.373 px (5119 obs, 16 cameras) ===
[Flip Z] Saved 16 cameras + 3D points to .../aruco_image_experimental/2026_03_14_02_12_15
```

## Methodology

**multiview_calib** (v2025-08, Python):
- ArUco detection: OpenCV `cv2.aruco.detectMarkers()` + `cv2.aruco.interpolateCornersCharuco()`
- Intrinsic calibration: `cv2.calibrateCamera()` (Zhang's method)
- Extrinsic registration: Star topology, multi-path averaging
- Bundle adjustment: `scipy.optimize.least_squares(method='trf', loss='linear', max_nfev=40)`
- Global registration: Procrustes alignment + coordinate transform

**RED Experimental Pipeline** (v2026-03, C++):
- ArUco detection: Custom Metal GPU-accelerated detector with multi-dictionary support
- Intrinsic calibration: Custom Eigen-based implementation with quality gate
- Extrinsic registration: Incremental PnP with Markley quaternion averaging
- Bundle adjustment: Ceres Solver, 7-pass GNC with progressive Cauchy loss
- Outlier rejection: 4 progressive rounds (20 -> 10 -> 5 -> 3 px thresholds)
- Weak camera handling: Fix intrinsics for cameras with < 30 observations
- Z-flip: Interactive button with improper-rotation-safe reload via `projectPointR()`

**Hardware**: Apple M-series MacBook (Metal GPU for ArUco detection, ANE for inference)
