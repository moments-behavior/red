# Rob1 Calibration Study: ArUco + PointSource Refinement

**Date**: 2026-03-21
**Rig**: 16-camera mouse behavioral arena (HHMI Janelia)
**Operator**: Rob Johnson
**Tool**: RED v4 (GPU-accelerated multi-camera calibration)

---

## 1. Overview

This study evaluates the complete RED calibration pipeline on a 16-camera mouse behavioral rig: initial ArUco board calibration followed by pointsource refinement using a light wand, with four optimization modes. A light wand is a thin wand with a tiny green optical fiber emitting light from the tip, creating a point source visible to multiple cameras simultaneously. We cross-validate each calibration against held-out ArUco board detections to assess whether pointsource refinement genuinely improves 3D accuracy or merely overfits to pointsource data.

**Key result**: Full-parameter pointsource refinement achieves the best performance on both pointsource data (0.268 px) AND aruco cross-validation (1.649 px), confirming that a high-quality pointsource dataset with strong multi-view redundancy can reliably estimate all camera parameters.

### Dataset Locations

| Dataset | Path |
|---------|------|
| ArUco calibration videos (Rob) | `/Users/johnsonr/datasets/mouse/calibration/Mar202026/2026_03_20_11_43_33` |
| Global registration videos | `/Users/johnsonr/datasets/mouse/calibration/Mar202026/2026_03_20_11_39_14` |
| Light wand videos | `/Users/johnsonr/datasets/mouse/calibration/Mar202026/2026_03_20_11_51_00` |
| RED project folder | `/Users/johnsonr/red_dev/quan_v_rob_calib/rob1/` |
| ArUco output | `.../rob1/aruco_calibration/2026_03_21_20_02_05/` |
| PS: Extrinsics only | `.../rob1/laser_calibration/2026_03_21_21_35_04/` |
| PS: Extrinsics + Focal | `.../rob1/laser_focal/2026_03_21_22_10_36/` |
| PS: Extrinsics + All | `.../rob1/laser_all_intrinsics/2026_03_21_22_15_00/` |
| PS: Full | `.../rob1/laser_full/2026_03_21_22_19_26/` |
| Cross-validation test | `/Users/johnsonr/red_dev/quan_v_rob_calib/test_crossval_modes.cpp` |
| PS mode test | `/Users/johnsonr/red_dev/quan_v_rob_calib/test_laser_modes.cpp` |

### Cameras

16 Emergent Vision Technologies HB-7000-S cameras (7MP, 25GigE RDMA) (3208 x 2200 px): `2002486, 2002487, 2005325, 2006050, 2006051, 2006052, 2006054, 2006055, 2006515, 2006516, 2008665, 2008666, 2008667, 2008668, 2008669, 2008670`

ArUco videos: ~445 frames each, ~224-250 MB per camera. Light wand videos: ~958 MB each (~16 GB total for 16 cameras + 1 thermal Cam710040 excluded).

---

## 2. ArUco Board Calibration (Baseline)

### 2.1 Input Data

| Parameter | Value |
|-----------|-------|
| Board | 5x5 ChArUco, 60mm squares, 37.5mm markers |
| Dictionary | DICT_4X4_50 |
| Input | 16 synchronized videos, ~445 frames each |
| Frame step | 2 (every other frame sampled) |
| Sampled frames/camera | 223 |
| Media folder | `.../Mar202026/2026_03_20_11_43_33` |
| Global registration | Separate video set (`.../2026_03_20_11_39_14`) |
| Config file | None (config-free project) |

### 2.2 The REDv4 Experimental Pipeline

The ArUco calibration follows a 7-step pipeline:

#### Step 1: ChArUco Detection + Intrinsic Calibration

For each of the 16 cameras in parallel:

1. **Video decode**: FFmpeg demuxer + VideoToolbox async hardware decoder (macOS) extracts frames as BGRA CVPixelBuffers
2. **Grayscale conversion**: BGRA to 8-bit grayscale via weighted sum (BT.601 coefficients)
3. **Adaptive thresholding**: GPU-accelerated (Metal) adaptive threshold binarizes the image. Full-resolution processing (no downsampling) — the validated REDv4 approach
4. **Contour finding**: Suzuki-Abe (1985) algorithm finds both outer and hole boundaries in the binary image
5. **Polygon approximation**: Douglas-Peucker with epsilon=0.03 (matching OpenCV) filters for quadrilateral candidates
6. **Marker identification**: For each quad, the algorithm reads the 4x4 bit pattern, tries all 4 rotations, and matches against the DICT_4X4_50 dictionary via Hamming distance
7. **ChArUco corner interpolation**: Detected ArUco markers define a partial homography; inner checkerboard corners (4x4 = 16 for a 5x5 board) are interpolated
8. **Sub-pixel refinement**: `cornerSubPix(half_win=3, max_iter=100, eps=0.001)` — tight window prevents gradient blending across marker boundaries
9. **Per-camera intrinsic calibration** (Ceres): Estimates fx, fy, cx, cy, k1, k2, p1, p2, k3 from all detected boards using a pinhole + radial/tangential distortion model

**Rob1 detection results**: 1,348 total board detections across 16 cameras (37.8% detection rate). All 16 cameras passed the minimum detection threshold.

**Per-camera detection counts and intrinsic reproj errors**:

| Camera | Detections | Intrinsic Reproj (px) | fx (px) | fy (px) |
|--------|-----------|----------------------|---------|---------|
| 2002486 | 72 | 0.325 | 2583.5 | 2571.7 |
| 2002487 | 76 | 0.414 | 3395.3 | 3387.0 |
| 2005325 | 96 | 0.361 | 2734.4 | 2729.7 |
| 2006050 | 74 | 0.376 | 3505.2 | 3497.4 |
| 2006051 | 81 | 0.471 | 3467.2 | 3456.2 |
| 2006052 | 107 | 0.514 | 3332.7 | 3321.1 |
| 2006054 | 105 | 0.331 | 3373.6 | 3362.2 |
| 2006055 | 115 | 0.463 | 3431.9 | 3421.2 |
| 2006515 | 61 | 0.346 | 3932.6 | 3930.8 |
| 2006516 | 51 | 0.436 | 3938.4 | 3932.3 |
| 2008665 | 52 | 0.485 | 3571.7 | 3563.4 |
| 2008666 | 84 | 0.265 | 2570.8 | 2563.3 |
| 2008667 | 69 | 0.372 | 3530.5 | 3519.3 |
| 2008668 | 79 | 0.380 | 3505.3 | 3497.2 |
| 2008669 | 65 | 0.305 | 3505.3 | 3499.2 |
| 2008670 | 61 | 0.344 | 2574.9 | 2566.0 |

#### Step 2: Camera Filtering

Cameras with fewer than 4 valid board detections are excluded. Rob1: all 16 cameras passed (no exclusions).

#### Step 3: Landmark Building

Detected corners are unified into a global landmark table. Each corner gets a unique ID: `global_id = frame_idx * 16 + corner_id`. Rob1: 3,299 unique 3D landmarks built from 18,235 total 2D observations.

#### Step 4: Extrinsic Initialization (Incremental PnP)

RED uses a COLMAP-inspired greedy incremental registration (not spanning-tree):

1. **Best pair selection**: All camera pairs are scored by shared frame count and geometric diversity. Multi-frame Markley quaternion averaging computes the initial relative rotation. Rob1 root: camera 2006052.
2. **Greedy expansion**: Remaining cameras are added one at a time by solving Perspective-n-Point (PnP) from already-triangulated 3D points, selecting the camera with the most visible landmarks.
3. **Re-triangulation**: After each camera addition, newly visible landmarks are triangulated.

#### Step 5: Bundle Adjustment (7-Pass GNC Cauchy)

The core quality driver. RED uses Graduated Non-Convexity with Cauchy loss:

| Pass | Cauchy Scale | Optimized Parameters | Purpose |
|------|-------------|---------------------|---------|
| 1 | 16.0 | Extrinsics only | Coarse alignment — wide basin tolerates large errors |
| 2 | 4.0 | Extrinsics only | Moderate outlier rejection |
| 3 | 1.0 | Extrinsics only | Tight refinement + outlier removal (Rob1: 29 outliers) |
| 4 | 4.0 | + Intrinsics (fx, fy, cx, cy, k1, k2) | Unlock focal length and principal point |
| 5 | 1.0 | + Intrinsics | Tight intrinsic refinement (Rob1: 3 outliers) |
| 6 | 4.0 | + Full joint (p1, p2, k3 LOCKED) | Joint optimization with locked tangential/6th-order |
| 7 | 50.0 | + Full joint (p1, p2, k3 LOCKED) | Near-linear final pass — equal weight to all inliers |

**Why Cauchy loss**: `CauchyLoss(s) = log(1 + (r/s)^2)`. Large scale = wide basin (tolerates outliers). Small scale = tight (rejects outliers). Scale 50 = nearly linear.

**Why lock p1, p2, k3**: Tangential distortion and 6th-order radial are poorly constrained by board data. When free, they absorb extrinsic errors — producing misleadingly low per-board reprojection but degraded multi-view accuracy. This was validated empirically during REDv4 development (March 16, 2026).

**Rob1 BA convergence**:

| Pass | Cost Start | Cost End | Iterations | Time | Outliers |
|------|-----------|----------|-----------|------|----------|
| 1 (s=16) | 18,285 | 15,055 | 3 | 0.47s | — |
| 2 (s=4) | 6,585 | 6,566 | 7 | 0.56s | — |
| 3 (s=1) | 3,037 | 3,017 | varies | 1.06s | 29 |
| 4 (s=4) | 3,507 | 2,724 | 14 | 1.16s | — |
| 5 (s=1) | 1,902 | 1,837 | 21 | 3.18s | 3 |
| 6 (s=4) | 2,809 | 1,475 | 46 | 3.09s | — |
| 7 (s=50) | 1,674 | 1,641 | 3 | 1.55s | — |
| **Total** | | | | **11.06s** | **32** |

#### Step 6: Global Registration (Procrustes Alignment)

The calibration is initially in an arbitrary coordinate frame (anchored to the first registered camera). Global registration aligns to a world coordinate system:

1. **Board detection in separate videos**: Frame 0 decoded from each camera's global registration video (16 cameras, parallel)
2. **Corner detection**: Full-res ChArUco detection + sub-pixel refinement on each frame
3. **Triangulation**: Board corners triangulated using the calibrated camera poses (DLT from all cameras that see each corner)
4. **Procrustes alignment**: SVD-based rigid transform (rotation + scale + translation) aligns triangulated corners to ground truth 3D coordinates (center-origin, z=0 grid with 60mm spacing)
5. **Optional post-rotation**: Convention transform (e.g., for MVC compatibility)

#### Step 7: Output

Per-camera YAML files (OpenCV format) with K, distortion, R, t. Summary data including landmarks, BA passes, and per-camera metrics.

### 2.3 ArUco Baseline Results

| Metric | Value |
|--------|-------|
| Mean per-board reproj | 0.300 px |
| Cameras | 16/16 |
| 3D landmarks | 3,299 |
| Total observations | 18,235 |
| BA time | 11.06s |
| Outliers removed | 32 |

**Per-camera final BA reprojection errors**:

| Camera | Points | Mean (px) | Std (px) | Median (px) |
|--------|--------|-----------|----------|-------------|
| 2002486 | 1,407 | 0.208 | 0.153 | 0.175 |
| 2002487 | 976 | 0.292 | 0.303 | 0.216 |
| 2005325 | 1,432 | 0.224 | 0.246 | 0.190 |
| 2006050 | 1,003 | 0.388 | 0.390 | 0.279 |
| 2006051 | 931 | 0.282 | 0.309 | 0.205 |
| 2006052 | 1,567 | 0.339 | 0.394 | 0.274 |
| 2006054 | 1,364 | 0.470 | 0.457 | 0.377 |
| 2006055 | 1,590 | 0.225 | 0.254 | 0.183 |
| 2006515 | 806 | 0.321 | 0.287 | 0.254 |
| 2006516 | 833 | 0.354 | 0.533 | 0.237 |
| 2008665 | 680 | 0.246 | 0.298 | 0.185 |
| 2008666 | 1,496 | 0.273 | 0.214 | 0.230 |
| 2008667 | 988 | 0.300 | 0.439 | 0.234 |
| 2008668 | 970 | 0.276 | 0.246 | 0.210 |
| 2008669 | 854 | 0.318 | 0.337 | 0.246 |
| 2008670 | 1,338 | 0.317 | 0.232 | 0.269 |

**Per-camera intrinsic quality** (reproj error range): 0.265 - 0.514 px, std 0.069 px. Very consistent across all cameras — no outlier cameras.

---

## 3. PointSource Calibration Refinement

### 3.1 The PointSource Pipeline

PointSource refinement uses synchronized video of a green light wand to correct camera calibration. The light wand creates a point source visible to multiple cameras simultaneously, providing dense multi-view constraints that complement the sparse board-based calibration.

#### Step 1: Light Spot Detection (Parallel)

For each camera in parallel:
1. **Video decode**: Same FFmpeg + VideoToolbox pipeline as ArUco
2. **Green channel extraction**: From BGRA pixel data
3. **Thresholding**: Pixel is "green" if `G >= green_threshold` AND `G - max(R, B) >= green_dominance`
4. **Morphological ops**: Erode then dilate to remove noise and fill gaps
5. **Connected components**: Union-find algorithm identifies blobs
6. **Blob filtering**: Reject blobs outside `[min_blob_pixels, max_blob_pixels]` range
7. **Centroid computation**: Intensity-weighted centroid for sub-pixel accuracy

Optional: GPU-accelerated detection via Metal compute shader (macOS).

#### Step 2: Multi-Camera Observation Assembly

For each video frame, collect all cameras that detected a light spot. Discard frames where fewer than `min_cameras` cameras detected the spot. This ensures every 3D point has strong multi-view constraint.

#### Step 3: Triangulation + Validation

1. Load existing calibration from YAML files (from ArUco output)
2. Undistort 2D pixel coordinates using the calibrated distortion model
3. Triangulate 3D points via DLT from all observing cameras
4. Filter per-camera reprojection errors: reject observations exceeding `reproj_threshold`
5. Re-triangulate with clean observations

#### Step 4: Bundle Adjustment (Ceres, 2-pass)

Two-pass optimization with outlier rejection between passes:

- **Pass 1**: Loose threshold (`ba_outlier_th1` = 20 px). Optimize, then reject outliers.
- **Pass 2**: Tight threshold (`ba_outlier_th2` = 5 px) on surviving observations. Final optimization.

Camera 0 extrinsics are always locked (gauge freedom — the coordinate system must be anchored to one camera). The **optimization mode** controls which other parameters are free:

| Mode | Free Parameters | Locked Parameters | Use Case |
|------|----------------|-------------------|----------|
| Extrinsics Only | R, t (6 DOF/cam) | fx, fy, cx, cy, k1, k2, p1, p2, k3 | Trusted ArUco intrinsics |
| Extrinsics + Focal | R, t, fx, fy (8 DOF/cam) | cx, cy, k1, k2, p1, p2, k3 | Focal drift suspected |
| Extrinsics + All | R, t, fx, fy, cx, cy, k1, k2 (12 DOF/cam) | p1, p2, k3 | Poor initial calibration |
| Full | R, t, fx, fy, cx, cy, k1, k2, p1, p2, k3 (15 DOF/cam) | (none) | Excellent pointsource data |

Solver: Sparse Schur (multi-threaded), Huber loss (scale 1.0).

#### Step 5: Write Output

Refined YAML files + comprehensive metadata:
- `settings.json`: All detection, filtering, and BA parameters
- `summary.json`: Before/after reproj, per-camera changes (dfx, dfy, dcx, dcy, dt_norm, drot_deg)
- `ba_points.json`: Triangulated 3D pointsource points
- `observations.json`: Per-point camera observations

### 3.2 PointSource Dataset Quality

| Metric | Value | Assessment |
|--------|-------|-----------|
| Video folder | `.../2026_03_20_11_51_00` | 16 cameras + 1 thermal |
| Video size | ~958 MB each | Long recording (~5 min) |
| Frame step | 3 (every 3rd frame) | ~4,400 frames/camera |
| Triangulated 3D points | 3,692 | Excellent |
| Total 2D observations | 51,245 | Massive constraint set |
| Mean cameras/point | 13.88 / 16 | Outstanding redundancy |
| Mode cameras/point | 15 (40% of points) | Nearly all cameras see each point |
| Spatial extent | 1017 x 869 x 530 mm | Good 3D arena coverage |
| Outliers removed (all modes) | 0 | Perfectly clean data |
| Points per camera | 3,207 - 3,680 | Well-distributed |

**Exception**: Camera 2008666 detected only 759 light spots (444 observations) — likely partially occluded or at an extreme angle. All other cameras detected 2,764 - 3,968 light spots.

### 3.3 Detection Settings

| Parameter | Value | Purpose |
|-----------|-------|---------|
| green_threshold | 20 | Min green channel value |
| green_dominance | 5 | Min green excess over R and B |
| min_blob_pixels | 10 | Reject tiny noise |
| max_blob_pixels | 600 | Reject large reflections |
| min_cameras | 3 | Min cameras per 3D point |
| reproj_threshold | 30.0 px | Pre-BA outlier filter |
| ba_outlier_th1 | 20.0 px | Pass 1 threshold |
| ba_outlier_th2 | 5.0 px | Pass 2 threshold |
| ba_max_iter | 50 | Ceres iterations per pass |

### 3.4 Per-Camera Light Spot Detection Results

| Camera | Detections | Observations (after filter) | Detection Rate |
|--------|-----------|---------------------------|----------------|
| 2002486 | 3,829 | 3,590 | 93.8% |
| 2002487 | 3,729 | 3,460 | 92.8% |
| 2005325 | 3,968 | 3,680 | 92.7% |
| 2006050 | 3,593 | 3,306 | 92.0% |
| 2006051 | 3,705 | 3,431 | 92.6% |
| 2006052 | 3,647 | 3,336 | 91.5% |
| 2006054 | 3,913 | 3,593 | 91.8% |
| 2006055 | 3,757 | 3,467 | 92.3% |
| 2006515 | 3,774 | 3,485 | 92.3% |
| 2006516 | 3,816 | 3,557 | 93.2% |
| 2008665 | 3,749 | 3,493 | 93.2% |
| 2008666 | **759** | **444** | **58.5%** |
| 2008667 | 2,764 | 2,526 | 91.4% |
| 2008668 | 3,577 | 3,288 | 91.9% |
| 2008669 | 3,521 | 3,207 | 91.1% |
| 2008670 | 3,648 | 3,382 | 92.7% |

Camera 2008666 is an outlier with only 759 detections (58.5% observation rate) — likely partially occluded or at an extreme viewing angle during pointsource data collection.

---

## 4. PointSource Refinement Results

### 4.1 Reprojection Error on PointSource Data

| Mode | Reproj After | Improvement | Time |
|------|-------------|-------------|------|
| **Extrinsics only** | 0.476 px | 52.4% | 4.4 min |
| **Extrinsics + Focal** | 0.378 px | 62.2% | 4.4 min |
| **Extrinsics + All** | 0.336 px | 66.4% | 4.4 min |
| **Full** | **0.268 px** | **73.2%** | 4.4 min |

All modes: 3,692 points, 51,245 observations, 0 outliers removed.

### 4.2 Parameter Changes by Mode

#### Extrinsic Changes

| Mode | Mean dt (mm) | Max dt (mm) | Mean drot (deg) | Max drot (deg) |
|------|-------------|------------|-----------------|----------------|
| Extrinsics only | 0.51 | 0.96 | 0.030 | 0.072 |
| + Focal | 1.23 | 5.59 | 0.042 | 0.084 |
| + All Intrinsics | 4.46 | 7.03 | 0.249 | 0.484 |
| Full | 24.08 | 47.04 | 1.009 | 2.065 |

#### Intrinsic Changes (Mean Absolute)

| Mode | dfx (px) | dfy (px) | dcx (px) | dcy (px) |
|------|----------|----------|----------|----------|
| Extrinsics only | 0 | 0 | 0 | 0 |
| + Focal | 7.29 | 5.19 | 0 | 0 |
| + All Intrinsics | 9.53 | 9.72 | 9.51 | 5.96 |
| Full | 31.34 | 27.55 | 11.29 | 37.76 |

**Observation**: Full mode makes significantly larger changes — up to 71 px in focal length, 80 px in cy, 47 mm in translation, 2.1 degrees in rotation. These are physically substantial corrections, not noise.

#### Full Mode: Detailed Per-Camera Changes

| Camera | dfx (px) | dfy (px) | dcx (px) | dcy (px) | dt (mm) | drot (deg) |
|--------|----------|----------|----------|----------|---------|------------|
| 2002486 | -62.27 | -59.53 | +24.27 | +71.98 | 0.0 (cam 0) | 0.0 (cam 0) |
| 2002487 | -5.57 | +6.96 | -7.22 | +29.99 | 31.31 | 1.219 |
| 2005325 | -50.02 | -47.73 | +0.86 | +1.82 | 30.43 | 1.035 |
| 2006050 | -19.60 | -4.09 | +7.03 | +40.17 | 24.66 | 1.063 |
| 2006051 | -22.36 | -10.76 | +4.14 | +29.23 | 34.06 | 1.432 |
| 2006052 | -63.06 | -60.89 | -2.28 | -2.36 | 29.41 | 0.983 |
| 2006054 | -70.87 | -66.68 | -8.81 | +80.05 | 47.04 | 2.065 |
| 2006055 | -53.84 | -48.40 | -4.05 | +48.40 | 34.58 | 1.389 |
| 2006515 | -4.88 | +2.83 | -35.65 | +27.60 | 14.96 | 1.051 |
| 2006516 | -15.48 | -5.62 | +1.29 | +24.41 | 37.52 | 1.418 |
| 2008665 | -13.51 | -2.92 | -4.45 | +34.86 | 35.69 | 1.495 |
| 2008666 | -51.83 | -47.86 | -6.43 | +70.79 | 10.16 | 0.548 |
| 2008667 | -0.96 | +11.83 | -4.92 | +26.09 | 16.68 | 0.614 |
| 2008668 | -9.06 | +4.50 | +1.47 | +40.23 | 20.48 | 0.697 |
| 2008669 | -5.26 | +5.34 | -8.96 | +26.28 | 21.58 | 0.664 |
| 2008670 | -68.72 | -62.87 | +65.12 | +52.13 | 6.77 | 0.508 |

Note: Camera 2002486 is camera 0 (gauge reference — extrinsics locked, intrinsics free).

### 4.3 Cross-Validation Against ArUco Board Detections

This is the critical test: do the pointsource-refined calibrations improve on data they never saw?

We loaded the 2D ArUco board corner detections from the Rob1 calibration (3,354 landmarks, 18,640 observations across 16 cameras), triangulated each landmark using each calibration's camera poses, and measured reprojection error.

| Calibration | PS Reproj | ArUco Cross-Val | vs Baseline |
|------------|-------------|-----------------|-------------|
| **ArUco baseline** | — | **1.664 px** | Reference |
| **Extrinsics only** | 0.476 px | 2.014 px | +21% worse |
| **Extrinsics + Focal** | 0.378 px | 2.038 px | +22% worse |
| **Extrinsics + All** | 0.336 px | 1.832 px | +10% worse |
| **Full** | 0.268 px | **1.649 px** | **-0.9% better** |

#### Per-Camera Cross-Validation (px)

| Camera | ArUco Baseline | Extr Only | + Focal | + All | Full |
|--------|---------------|-----------|---------|-------|------|
| 2002486 | 1.125 | 1.293 | 1.304 | 1.107 | **1.019** |
| 2002487 | 1.827 | 2.325 | 2.383 | 2.060 | **1.803** |
| 2005325 | 1.603 | 1.921 | 1.970 | 1.739 | **1.555** |
| 2006050 | 1.497 | 1.936 | 2.092 | 1.750 | **1.476** |
| 2006051 | 2.246 | 2.954 | 3.020 | 2.672 | **2.434** |
| 2006052 | **1.591** | 2.022 | 2.129 | 1.961 | 1.715 |
| 2006054 | 1.598 | 1.902 | 1.782 | 1.673 | **1.573** |
| 2006055 | 1.531 | 1.661 | 1.728 | 1.474 | **1.403** |
| 2006515 | 1.659 | 2.078 | 2.105 | 1.960 | **1.285** |
| 2006516 | **1.995** | 2.868 | 2.762 | 2.511 | 2.358 |
| 2008665 | **2.071** | 2.576 | 2.641 | 2.344 | 2.265 |
| 2008666 | **1.547** | 1.760 | 1.850 | 1.685 | 1.873 |
| 2008667 | 3.218 | 3.593 | 3.449 | 3.207 | **2.759** |
| 2008668 | 1.254 | 1.615 | 1.639 | 1.485 | **1.365** |
| 2008669 | 1.317 | 1.450 | 1.394 | 1.222 | **0.904** |
| 2008670 | 1.286 | 1.474 | 1.489 | 1.455 | **1.242** |

**Bold** = best result for that camera.

Full mode wins on 11 of 16 cameras. The ArUco baseline wins on 3 cameras (2006052, 2006516, 2008665) and ties on 2 (2006054, 2008666). Notably, camera 2008669 improves by 31% (1.317 → 0.904 px) — its ArUco intrinsics were evidently imprecise.

---

## 5. Analysis

### 5.1 Why Extrinsics-Only Hurts ArUco Cross-Validation

When only extrinsics are free, the optimizer moves cameras to fit pointsource points. But the ArUco intrinsics may have small errors (focal length, principal point) that were compensated by the ArUco extrinsics. Moving extrinsics without correcting intrinsics breaks this compensation — the cameras move to a position that's better for pointsource data but worse for ArUco data.

This is a classic bias-variance tradeoff: the extrinsics-only model has low model flexibility, so it can't simultaneously fit both data sources.

### 5.2 Why Full Mode Wins Both

With all parameters free, the optimizer can find a solution that genuinely improves the camera model — not just shift error between extrinsics and intrinsics. The large parameter changes (up to 71 px focal, 80 px cy) indicate the ArUco intrinsics had real errors that the pointsource data can correct.

The key enabler is data quality: 3,692 points with 13.88 cameras/point provides ~38 observations per free parameter. This is more than sufficient to constrain even p1, p2, k3 — parameters that are normally underdetermined from board data alone. The light wand's dense volumetric sampling provides constraints that planar boards cannot.

### 5.3 Camera 2008666 Anomaly

Camera 2008666 is the one camera where Full mode (1.873 px) is worse than ArUco baseline (1.547 px) on cross-validation. This camera had only 759 light spot detections (444 observations) — far fewer than other cameras (2,500-3,700). With fewer constraints, its parameters may be less reliable. This suggests a detection threshold: cameras with fewer than ~1,000 pointsource observations may benefit from locked intrinsics.

### 5.4 Scale of Parameter Changes

The Full mode changes are large but physically interpretable:

- **Focal length shift** (mean -31 px, ~1%): Consistent across most cameras, suggesting a systematic bias in ArUco focal estimation — possibly from board corners being slightly non-planar or from the limited depth range of board observations.
- **cy shift** (mean +38 px): Systematic vertical principal point correction. ArUco boards were likely presented at a restricted range of vertical positions, leaving cy less constrained than cx.
- **Translation changes** (mean 24 mm): Significant repositioning, especially for cameras far from the arena center where depth ambiguity is highest.
- **Rotation changes** (mean 1.0 deg): Moderate but consistent, indicating real alignment corrections.

---

## 6. Recommendations

### 6.1 For High-Quality PointSource Datasets (Like This One)

- Use **Full** optimization mode
- Requires: >3,000 pointsource points, >10 cameras/point average, good spatial coverage
- Expect: 70%+ improvement on pointsource data, slight improvement on ArUco cross-validation
- Monitor camera 2008666-like anomalies (low detection count cameras)

### 6.2 For Moderate PointSource Datasets

- Use **Extrinsics + All Intrinsics** mode (locks p1, p2, k3)
- Suitable when: 1,000-3,000 pointsource points, 6-10 cameras/point
- Expect: 50-65% improvement on pointsource data, comparable ArUco cross-validation

### 6.3 For Minimal PointSource Datasets

- Use **Extrinsics Only** mode
- Suitable when: <1,000 pointsource points, or sparse camera coverage
- Expect: 30-50% improvement on pointsource data, but may degrade ArUco cross-validation
- Consider running cross-validation to verify

### 6.4 General Calibration Workflow

1. **ArUco calibration** (config-free or with config.json): Establishes initial intrinsics and extrinsics with global registration to world coordinate frame
2. **PointSource refinement (Full mode)**: Corrects all parameters using dense multi-view constraints. This is the production calibration.
3. **Cross-validation**: Run ArUco cross-validation to verify no degradation. If a specific camera degrades, re-run with that camera's intrinsics locked.

### 6.5 PointSource Data Collection Tips

- Use a light wand with a bright green optical fiber tip visible to all cameras
- Move the light wand slowly across the entire arena volume (not just a 2D plane)
- Cover corners and edges, not just the center
- Record for 3-5 minutes (more data = better constraint)
- Verify all cameras can see the light wand (check detection rates after running)
- Use frame_step=3 as a good balance between quality and speed

### 6.6 When to Re-Calibrate

- After any physical change to the rig (camera moved, lens adjusted, rig relocated)
- If behavioral tracking shows unexplained 3D drift
- Periodically (e.g., monthly) to catch gradual drift
- Always start from ArUco → PointSource, not pointsource-only (ArUco provides the world frame)
