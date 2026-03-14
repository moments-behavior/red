# Rat 17-Camera Calibration: multiview_calib vs RED Experimental Pipeline

*Johnson Lab, HHMI Janelia Research Campus -- March 2026*

## Dataset

- **Rig**: 17 cameras (16 standard ~2300px focal length, 1 wide 710038 ~1770px, 2 telephoto ~3200px)
- **Resolution**: 3208 x 2200 (7 MP)
- **Images**: 78 per camera (1326 total)
- **Board**: 5x5 ChArUco, DICT_5X5_100, 80mm squares, 60mm markers (400mm total)
- **Max corners per frame**: 16 (inner corners of 5x5 board)
- **Theoretical max observations per camera**: 78 x 16 = 1248

## Overall Results

| Metric | multiview_calib (MVC) | RED Experimental | Difference |
|--------|----------------------|------------------|------------|
| Cameras calibrated | 17 | 16 (dropped 2002492) | |
| Images processed | 71 of 78 | 78 of 78 | MVC missed 7 frames |
| Frames with detections | 474 | 379 | RED 20% fewer |
| 2D observations (pre-BA) | 6,616 | 5,303 | RED 20% fewer |
| 2D observations (post-BA) | 6,513 | 5,119 | RED 21% fewer |
| Outliers removed | 0 | 41 (0.8%) | |
| **Mean intrinsic reproj** | **0.663 px** | **0.338 px** | **RED 49% better** |
| **Mean BA reproj** | **0.43 px** | **0.373 px** | **RED 13% better** |
| BA convergence | Hit 40-iter cap | Fully converged (7 passes) | |
| Total time | ~5 min (CPU) | ~44s (Metal GPU + Ceres) | RED 7x faster |

**Key finding**: RED achieves lower error with fewer observations. Stricter detection thresholds reject marginal detections that would degrade calibration quality.

## Per-Camera Detection Rates

Detection rates vary dramatically across cameras, reflecting viewing geometry and board visibility:

| Camera | Focal | MVC Frames | RED Frames | MVC Rate | RED Rate | Notes |
|--------|-------|-----------|-----------|----------|----------|-------|
| 710038 | 1770 | 66 | 54 | 84.6% | 69.2% | Wide lens, best coverage |
| 2002488 | 2362 | 52 | 50 | 66.7% | 64.1% | |
| 2002490 | 2366 | 52 | 52 | 66.7% | 66.7% | Identical |
| 2002496 | 2362 | 45 | 48 | 57.7% | 61.5% | RED found more |
| 2002489 | 2385 | 43 | 39 | 55.1% | 50.0% | |
| 2002484 | 2293 | 24 | 13 | 30.8% | 16.7% | Low overlap zone |
| 2002481 | 2297 | 23 | 10 | 29.5% | 12.8% | Low overlap zone |
| 2002479 | 2307 | 21 | 8 | 26.9% | 10.3% | Dangerously low |
| 2002480 | 3311 | 20 | 26 | 25.6% | 33.3% | Telephoto, RED better |
| 2002485 | 3165 | 18 | 19 | 23.1% | 24.4% | Telephoto |
| 2002491 | 2295 | 18 | 10 | 23.1% | 12.8% | Dangerously low |
| 2002493 | 2310 | 18 | 12 | 23.1% | 15.4% | |
| 2002482 | 3173 | 17 | 9 | 21.8% | 11.5% | Telephoto, low |
| 2002483 | 2903 | 15 | 10 | 19.2% | 12.8% | |
| 2002495 | 2275 | 15 | 12 | 19.2% | 15.4% | |
| 2002494 | 2340 | 12 | 7 | 15.4% | 9.0% | Dangerously low |
| 2002492 | 2844 | 9 | 0 | 11.5% | 0.0% | RED dropped entirely |

**Concern**: 6 cameras have fewer than 15 RED detections. Cameras 2002479 (8), 2002494 (7), and 2002482 (9) are at the reliability threshold. With 16 corners per frame, these cameras contribute only 112-144 observations each.

## Intrinsic Calibration Quality

Per-camera intrinsic reprojection error (RMS pixels), calibrated independently before bundle adjustment:

| Camera | MVC Intrinsic | RED Intrinsic | Winner | MVC BA | RED BA | Winner |
|--------|-------------|-------------|--------|--------|--------|--------|
| 710038 | 0.282 | **0.155** | RED | 0.525 | **0.248** | RED |
| 2002496 | 0.207 | **0.192** | RED | 0.671 | **0.388** | RED |
| 2002490 | 0.220 | **0.204** | RED | 0.482 | **0.312** | RED |
| 2002488 | 0.297 | **0.200** | RED | 0.555 | **0.392** | RED |
| 2002489 | 0.380 | **0.194** | RED | 0.799 | **0.308** | RED |
| 2002480 | **0.260** | 0.265 | MVC | 0.483 | **0.317** | RED |
| 2002485 | 0.661 | **0.290** | RED | 0.672 | **0.279** | RED |
| 2002483 | **0.636** | 0.767 | MVC | 0.438 | **0.287** | RED |
| 2002482 | 1.091 | **0.353** | RED | 0.844 | **0.507** | RED |
| 2002491 | 0.978 | **0.353** | RED | 0.852 | **0.297** | RED |
| 2002481 | 0.876 | **0.419** | RED | 0.999 | **0.824** | RED |
| 2002493 | 1.070 | **0.387** | RED | **1.004** | 1.050 | MVC |
| 2002484 | 0.889 | **0.421** | RED | 0.898 | **0.603** | RED |
| 2002479 | 0.865 | **0.416** | RED | 0.853 | **0.475** | RED |
| 2002495 | 0.684 | **0.393** | RED | 0.600 | **0.243** | RED |
| 2002494 | 0.604 | **0.402** | RED | 0.819 | **0.448** | RED |
| 2002492 | 1.269 | -- | -- | 1.080 | -- | -- |

- RED wins intrinsic on 14/16 cameras
- RED wins BA reproj on 15/16 cameras
- MVC has 7 cameras with intrinsic RMS > 0.8 px; RED has zero
- The only camera where MVC beats RED on BA (2002493) has outlier observations pulling RED's mean up (RED median 0.243 vs MVC median 0.749)

## Focal Length Agreement

Post-BA focal lengths agree within 1-2%, confirming both pipelines see the same optics:

| Camera | MVC fx | RED fx | Delta | Delta % |
|--------|--------|--------|-------|---------|
| 710038 | 1770.9 | 1761.1 | -9.8 | -0.55% |
| 2002488 | 2375.3 | 2370.7 | -4.5 | -0.19% |
| 2002490 | 2363.5 | 2352.8 | -10.7 | -0.45% |
| 2002496 | 2335.6 | 2309.2 | -26.4 | -1.13% |
| 2002480 | 3323.2 | 3271.0 | -52.3 | -1.57% |
| 2002485 | 3156.2 | 3130.5 | -25.7 | -0.81% |
| 2002489 | 2394.0 | 2384.6 | -9.5 | -0.40% |

Telephoto cameras (2002480, 2002485, 2002482) show larger absolute differences (~25-52 px) but percentage differences remain small. This is expected: with fewer detections and longer focal length, small biases in corner localization produce larger focal length shifts.

## Why Fewer Detections but Better Error

### 1. Detection Selectivity

RED's custom ArUco detector (`aruco_detect.h`) applies stricter thresholds than OpenCV's default:
- Requires minimum 4 markers detected before accepting a frame
- Multi-scale adaptive threshold (GPU-accelerated Metal on macOS)
- Otsu-based marker bit thresholding (better black/white separation)
- Spatial consistency check rejects markers with inconsistent scale ratios

OpenCV's ArUco detector (used by MVC) is more permissive, accepting frames with as few as 1-2 markers. These marginal detections have lower corner precision, degrading intrinsic calibration.

**Evidence**: MVC's mean intrinsic RMS is 0.663 px vs RED's 0.338 px. The stricter detector produces 49% better intrinsic calibration despite using fewer frames.

### 2. Different Frames Detected

The two detectors find **different subsets of the 78 images**, with surprisingly low frame overlap. This is because:
- Different adaptive threshold window sizes and constants
- Different contour finding parameters
- Different subpixel corner refinement algorithms (OpenCV vs RED's custom gradient-based method)
- RED's 3x downsampling for contour detection (optimized for speed, may miss very small markers)

### 3. Bundle Adjustment Convergence

**MVC**: Uses `scipy.optimize.least_squares` with linear loss and `max_nfev=40`. The optimization hit the iteration cap with first-order optimality still at 1.6e+04 -- **not converged**. No outlier rejection. All 6,513 observations used, including potentially corrupted ones.

**RED**: Uses Ceres Solver with progressive refinement:
1. Passes 1-3: Cauchy loss (scale 16 -> 4 -> 1), extrinsics only
2. Passes 4-5: Cauchy loss (scale 4 -> 1), extrinsics + intrinsics
3. Passes 6-7: Cauchy loss (scale 1), all parameters + 3D points
4. Three rounds of outlier rejection (thresholds 20 -> 10 -> 5 -> 3 px)
5. Total: ~434 iterations across 7 passes, fully converged

The progressive strategy prevents the optimizer from getting stuck in local minima. Separating extrinsics from intrinsics in early passes ensures a good pose estimate before refining lens parameters.

### 4. MVC Star Topology Issue

MVC computes all relative poses through camera 710038 as a hub (star topology). This means every camera-camera relationship is mediated by 710038's observations. Cameras with poor overlap with 710038 show high outlier rates in the initial extrinsic computation:

| Pair | Outlier Rate |
|------|-------------|
| 710038-2002491 | 30.4% |
| 710038-2002492 | 25.4% |
| 710038-2002484 | 23.5% |
| 710038-2002482 | 22.3% |

These corrupted initial extrinsics feed into bundle adjustment, which cannot fully recover due to the 40-iteration cap and linear loss function.

RED uses incremental PnP registration from the best initial pair (2002488 + 710038), adding cameras one at a time based on the number of shared 3D points. This avoids the star topology's fragility.

## Recommendations for Future Calibrations

### 1. Capture More Images (150+ per camera)

Current: 78 images, 6 cameras below 15 RED detections.
Target: 150+ images with dedicated positions for weak cameras.
Expected impact: All cameras above 25 detections, more robust intrinsic calibration.

### 2. Target Weak Cameras with Dedicated Board Positions

Cameras 2002479, 2002491, 2002494, and 2002482 have detection rates below 15%. These cameras likely view the calibration volume from extreme angles or distances. Recommendation:
- Place the board in positions specifically visible to these cameras
- Verify board visibility in the camera viewports before capture
- Aim for the board to fill 30-60% of the frame for optimal corner precision

### 3. Vary Board Orientation More Aggressively

Strong perspective tilt constrains distortion parameters. Current images may have the board mostly face-on to the nearest cameras. Recommendation:
- Tilt the board 20-40 degrees in multiple directions
- Include images at different distances (0.5m, 1.0m, 1.5m)
- This is especially important for telephoto cameras (2002480, 2002485, 2002482)

### 4. Optimal Board Distance

For 3208x2200 resolution at fx ~2350:
- 400mm board at 1.0m distance subtends ~940px (29% of frame width)
- 400mm board at 0.7m distance subtends ~1340px (42% of frame width)
- **Sweet spot: 0.7-1.2m** for good corner precision without losing coverage
- For telephoto cameras (fx ~3200): move board to 1.0-1.5m

### 5. Consider a Larger Board for This Rig

The 400mm board is relatively small for the arena volume. A 600mm or 800mm board would:
- Be visible from more camera angles simultaneously
- Produce larger markers in the image (better detection reliability)
- Provide more corners per detection (if using a 7x7 or 8x8 pattern)
- Trade-off: harder to manipulate physically, may not fit in tight spaces

### 6. If Continuing with MVC

- Increase `max_nfev` from 40 to at least 200
- Switch from linear loss to Huber or Cauchy (scale ~2.0)
- Add outlier rejection between BA passes
- Consider switching to Ceres solver (C++ bindings from Python available)

### 7. Use RED's Experimental Pipeline

RED achieves better results in less time with fewer manual steps:
- 13% lower BA reprojection error
- 49% lower intrinsic reprojection error
- Automatic outlier rejection
- GPU-accelerated detection (7x faster)
- Automatic handling of failing cameras
- Interactive 3D visualization of results

## Appendix: Methodology

**multiview_calib** (v2025-08, Python):
- ArUco detection: OpenCV `cv2.aruco.detectMarkers()` + `cv2.aruco.interpolateCornersCharuco()`
- Intrinsic calibration: `cv2.calibrateCamera()` (Zhang's method)
- Extrinsic registration: Star topology through camera 710038
- Bundle adjustment: `scipy.optimize.least_squares(method='trf', loss='linear', max_nfev=40)`
- Global registration: Procrustes alignment to ground truth points

**RED Experimental Pipeline** (v2026-03, C++):
- ArUco detection: Custom Metal GPU-accelerated detector (`aruco_detect.h`)
- Intrinsic calibration: Custom Eigen-based implementation with OpenCV-style model
- Extrinsic registration: Incremental PnP from best initial pair
- Bundle adjustment: Ceres Solver with progressive Cauchy loss (7 passes)
- Outlier rejection: 3 progressive rounds (20px -> 10px -> 5px -> 3px thresholds)

**Hardware**: Apple M-series MacBook (Metal GPU for ArUco detection, ANE for inference)
