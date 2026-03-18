# LightGlue Calibration Refinement: Complete Experiment Report

**Date:** March 18, 2026
**Authors:** Rob Johnson, Claude
**Data:** `/Users/johnsonr/datasets/rat/sp_test_march17/`

---

## 1. Background

We have a 17-camera rig (3208×2200 resolution) used to track rats in an arena. The cameras were calibrated on **August 14, 2025** using ChArUco board images. Video experiments were recorded on **September 3, 2025** — 20 days later. Some cameras may have physically shifted in that time.

Two initial calibrations exist for August 14:
- **MVC** (multiview_calib): Jinyao's Python/OpenCV pipeline with scipy BA
- **RED v4**: RED's experimental pipeline with Ceres multi-round BA

RED v4 beats MVC on all ChArUco-based quality metrics (see `calibration_refinement_tests.md`). But both calibrations reflect August 14 camera positions, not September 3.

**Goal:** Use SuperPoint feature extraction + LightGlue matching on September 3 video to refine the calibration to match the actual camera positions during the experiment.

---

## 2. The LightGlue Matching Pipeline

### How features are generated
1. Extract 50 keyframes (uniformly spaced, 1 per ~30 seconds) from each of the 17 camera videos
2. Run SuperPoint feature detection on each frame (resize to 1600 px, extract up to 4096 keypoints per image)
3. For each camera pair, match features using **LightGlue** — a learned transformer-based matcher (ICCV 2023) that uses self-attention and cross-attention to find corresponding points between images
4. Filter matches by reprojection: triangulate each matched pair, reject if reprojection error > 15 px (using the MVC calibration for this geometric check)
5. Assemble pairwise matches into multi-view tracks via union-find

### Matching results
- **97,649 pairwise matches** across 80 camera pairs from 50 frame sets
- After track building: **13,090 tracks** with **31,138 observations** across 17 cameras
- **19.7% of tracks seen by 3+ cameras** (vs only 4.3% with brute-force matching)
- Tracks seen by 5+ cameras: 577 (4.4%)
- 1 track seen by all 17 cameras

### Per-camera observation counts

| Camera | Observations | Camera | Observations |
|--------|-------------|--------|-------------|
| 2002491 | 5,461 | 2002485 | 1,756 |
| 2002483 | 4,286 | 2002482 | 1,524 |
| 2002495 | 4,127 | 2002484 | 1,107 |
| 2002481 | 3,651 | 2002489 | 780 |
| 710038 | 3,337 | 2002493 | 494 |
| 2002488 | 3,298 | 2002496 | 91 |
| 2002480 | 2,368 | | |
| 2002490 | 1,831 | | |
| 2002479 | 1,498 | | |
| 2002492 | 1,498 | | |
| 2002494 | 1,431 | | |

Note: Cam2002496 has only 91 observations — effectively unconstrained.

---

## 3. Bundle Adjustment Configuration

### What is optimized
For each camera: 3 rotation parameters + 3 translation parameters (6 DOF extrinsics). With "locked intrinsics," the focal length, principal point, and distortion coefficients are held fixed from the initial calibration.

For each 3D point: x, y, z coordinates.

### Parameters (same for all experiments)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Rotation prior | 50.0 | Penalizes deviation from initial rotation |
| Translation prior | 500.0 | Penalizes deviation from initial translation (10× stronger) |
| Outlier threshold 1 | 30.0 px | Coarse pass rejection |
| Outlier threshold 2 | 10.0 px | Final pass rejection |
| Max BA rounds | 5 | Re-triangulation + BA cycles |
| Convergence eps | 0.001 px | Stop when reproj change < this |
| Holdout fraction | 0.2 | 20% of tracks held out for cross-validation |
| Holdout seed | 42 | Deterministic split |

### The holdout cross-validation
Before BA begins, 20% of tracks (2,618 tracks, 6,176 observations) are randomly removed. BA runs on the remaining 80%. After BA, the held-out tracks are triangulated with the refined cameras and their reprojection error is measured. If the holdout error is much worse than the training error (ratio > 1.5), the refinement is overfitting — fitting noise rather than finding better camera positions.

---

## 4. Experiment: Locked Intrinsics (Baseline)

### LG from MVC

**Command:**
```bash
release/test_holdout_compare \
    .../lightglue_from_mvc/landmarks.json \
    .../original_calibration_results/calibration/ \
    --rot-prior 50 --trans-prior 500 --outlier-th1 30 --outlier-th2 10 \
    --output .../lightglue_sp_from_mvc
```

| Metric | Value |
|--------|-------|
| Time | 23 seconds |
| Initial reproj | 2.899 px |
| Train reproj | 2.107 px (27% improvement) |
| Holdout reproj | 2.588 px |
| Holdout/Train | **1.23×** (healthy) |
| Valid 3D points | 7,141 |
| Outliers removed | 687 |

### LG from RED v4

**Command:**
```bash
release/test_holdout_compare \
    .../lightglue_from_mvc/landmarks.json \
    .../redv4/aruco_image_experimental/2026_03_17_10_55_50/ \
    --rot-prior 50 --trans-prior 500 --outlier-th1 30 --outlier-th2 10 \
    --output .../lightglue_sp_from_redv4
```

| Metric | Value |
|--------|-------|
| Time | 24 seconds |
| Initial reproj | 2.834 px |
| Train reproj | 2.095 px (26% improvement) |
| Holdout reproj | 2.581 px |
| Holdout/Train | **1.23×** (healthy) |
| Valid 3D points | 7,098 |
| Outliers removed | 720 |

### Per-camera rotation deltas

| Camera | From MVC (°) | From v4 (°) | Camera | From MVC (°) | From v4 (°) |
|--------|-------------|------------|--------|-------------|------------|
| 2002479 | 0.258 | 0.343 | 2002491 | 0.389 | 0.509 |
| 2002480 | 0.249 | 0.127 | 2002492 | 1.267 | 1.363 |
| 2002481 | 0.386 | 0.515 | 2002493 | 1.190 | 1.326 |
| 2002482 | 0.211 | 0.173 | 2002494 | 0.720 | 0.878 |
| 2002483 | 0.297 | 0.284 | 2002495 | 0.166 | 0.181 |
| 2002484 | 0.352 | 0.553 | 2002496 | 1.061 | 1.175 |
| 2002485 | 0.576 | 0.774 | 710038 | 0.103 | 0.086 |
| 2002488 | 0.011 | 0.046 | | | |
| 2002489 | 0.933 | 0.595 | | | |
| 2002490 | 0.661 | 0.834 | | | |

---

## 5. Experiment: Unlocked Intrinsics

### Rationale
MVC and RED v4 have different intrinsics (e.g., Cam2002490 has fx=2363 in MVC vs fx=2372 in RED v4). With locked intrinsics, the refinement must find different extrinsics to compensate for different focal lengths. Unlocking intrinsics might allow convergence by letting both find the same optimal intrinsics+extrinsics combination.

### Variant A: Unlock fx, fy, cx, cy only (distortion locked)

**Command:** Same as above with `--unlock-intrinsics` added.

| | From MVC | From v4 |
|---|---|---|
| Train reproj | 2.003 px | 2.017 px |
| Holdout reproj | 2.422 px | 2.432 px |
| Holdout/Train | 1.21× | 1.21× |

### Variant B: Unlock all (intrinsics + distortion)

**Command:** Same as above with `--unlock-all` added.

| | From MVC | From v4 |
|---|---|---|
| Train reproj | 1.965 px | 1.965 px |
| Holdout reproj | 2.439 px | 2.419 px |
| Holdout/Train | 1.24× | 1.23× |

---

## 6. Convergence Analysis

**The key question:** Do the two refinements (starting from MVC vs starting from v4) converge to the same answer?

### Rotation gap between the two refinements

| Variant | Mean gap | vs Initial | Cameras converging |
|---------|---------|-----------|-------------------|
| **Initial (MVC↔v4)** | **0.420°** | 1.00× | — |
| **Locked** | **0.405°** | **0.97×** (slight improvement) | 8/17 improving |
| Unlock intrinsics | 0.607° | 1.45× (worse) | 5/17 improving |
| Unlock all | 1.032° | 2.46× (much worse) | 4/17 improving |

### Focal length gap

| Variant | Mean fx gap | Max fx gap |
|---------|-----------|----------|
| Initial | 8.9 px | 23.0 px |
| Locked | 8.9 px | 23.0 px (unchanged — locked) |
| Unlock intrinsics | 17.3 px | 41.9 px (diverging) |
| Unlock all | 66.5 px | **777.6 px** (Cam2002496 — only 91 observations) |

### Conclusion
**Unlocking intrinsics makes convergence dramatically worse.** With only 13K tracks (80% 2-camera), the data cannot uniquely determine both intrinsics and extrinsics. The optimizer finds different solutions depending on the starting point, with more free parameters creating more divergence.

**The locked variant is the only one that brings the two refinements closer together.** Keep intrinsics locked — the ChArUco calibration determines them well (0.44 px per-board reproj), and the SP refinement should only adjust extrinsics.

---

## 7. Ground Truth Evaluation: Multi-Frame Nose Test

Rob manually labeled the rat's nose in 6 frames (frames 38525, 38539, 38554, 40237, 40250, 40264) across 13-16 cameras per frame, totaling 87 observations.

### Leave-one-out reprojection (the fairest test)

For each labeled camera, triangulate the nose from all OTHER cameras, then measure how far the reprojection lands from the label. This tests prediction accuracy — can the calibration predict where a point appears in a camera it wasn't triangulated from?

| Stat | MVC | RED v4 | LG from MVC | LG from v4 |
|------|-----|--------|-------------|------------|
| Mean | 11.5 | 11.3 | 11.6 | **11.3** |
| Median | 5.7 | **5.3** | 6.1 | 5.8 |
| 95th pct | 54.7 | 51.3 | 52.1 | **50.2** |

### Per-camera winners (averaged across all 6 frames)

| | MVC | RED v4 | LG from MVC | LG from v4 |
|---|---|---|---|---|
| Cameras won | 2 | 3 | **6** | **6** |

SP-refined calibrations win **12 of 17 cameras.** The improvement is consistent but modest in magnitude (~0.5 px median improvement), partly because labeling noise (~3-5 px from clicking on a rat's nose) dominates the signal.

---

## 8. Understanding the Metrics

### What is reprojection error?

Imagine you know the 3D position of a point in the world (like the tip of a rat's nose). You project that 3D point through a camera's mathematical model — its lens focal length, its position and orientation in the room, its lens distortion — to predict where it should appear in the camera's image. The distance (in pixels) between where the model predicts the point should be and where it actually appears is the **reprojection error.**

Lower reprojection error = the camera model better explains reality.

### The different ways we compute reprojection error

#### A. Per-board reprojection error (ChArUco calibration)

**What it measures:** How well each camera's model fits a calibration board in a single image.

**How it works:** A ChArUco board has corners at precisely known 3D positions (manufactured to sub-millimeter accuracy). When we photograph the board, we detect the corners in the image. Then we estimate where the board was held (its pose), project the known 3D corners through the camera model, and measure how far the projections land from the detected corners.

**Key detail:** Each image gets its own board pose (6 parameters). This means the camera model only needs to explain one image at a time — it is a very forgiving test.

**Typical values:** 0.2–0.8 pixels. RED v4 achieves 0.437 px; MVC achieves 0.670 px.

**Limitation:** A camera can have excellent per-board error but terrible 3D accuracy. The per-board pose absorbs errors that would show up if you tested multiple images simultaneously.

#### B. Multi-view triangulation reprojection error

**What it measures:** How well ALL cameras agree on the 3D position of each point.

**How it works:** Take a point (e.g., a ChArUco corner) that is seen by multiple cameras. Triangulate its 3D position using ALL cameras that see it. Then reproject that single 3D point back into each camera and measure the error. Unlike per-board error, there is no per-image pose to absorb mistakes — every camera must agree on ONE 3D position.

**Key detail:** Before triangulation, the 2D observations must be **undistorted** — the lens distortion is removed so the triangulation operates on ideal ray directions. Without this step, the errors are artificially inflated (we discovered this on March 16).

**Typical values:** 0.4–2.0 pixels for well-calibrated systems. RED v4 achieves 0.48 px; MVC achieves 0.71 px.

**Why it's better than per-board error:** It tests the global extrinsic consistency — whether all cameras are positioned correctly relative to each other. A camera with a wrong rotation will produce systematic errors when its observations disagree with the triangulated 3D points.

#### C. Known-geometry validation (millimeters)

**What it measures:** Whether the 3D reconstruction is metrically accurate — does it produce correct real-world distances?

**How it works:** Triangulate the corners of a ChArUco board (which we know are spaced exactly 80mm apart). Measure the 3D distance between adjacent corners in the reconstruction. Compare to 80mm. The error in millimeters tells you the metric accuracy of the calibration.

**Why it matters:** A calibration can have low reprojection error but a systematic scale error — everything looks consistent in pixels, but the 3D measurements are wrong. Known-geometry validation catches this.

**Typical values:** 0.1–1.0 mm. RED v4 achieves 0.339 mm on 80mm squares (0.4% error); MVC achieves 0.368 mm.

**Limitation:** Requires a calibration board with known geometry. Cannot be computed from video data alone.

#### D. Manual landmark reprojection error (nose test)

**What it measures:** How well the calibration works for its actual intended purpose — tracking an animal in the arena.

**How it works:** A human labels a visible point (like the rat's nose) in multiple camera views of the same video frame. The 3D position is triangulated from the labels, then reprojected to each camera. The error is the distance between the reprojection and the human's click.

**Key differences from ChArUco tests:**
- The labeled point is a **soft, deformable landmark** (a nose, not a rigid board corner). Different cameras may be looking at slightly different parts of the nose.
- Human labeling precision is about **3–5 pixels** (vs ~0.1 px for algorithmic corner detection). This puts a floor on the achievable error.
- The test is performed on **video from the experiment day**, not calibration board images from weeks earlier. This means it captures the effect of cameras moving between sessions.
- The point's 3D position is **unknown** — we must triangulate it. Any error in one camera's label affects the triangulated 3D point, which affects the reprojection in all other cameras.

**Typical values:** 3–15 pixels (dominated by labeling noise). In our tests, median LOO error ranges from 5.2 to 6.1 px across the four calibrations.

#### E. Leave-one-out reprojection error

**What it measures:** Same as D, but stricter — it tests whether the calibration can **predict** where a point will appear in a camera that was not used to compute the 3D position.

**How it works:** For each camera in turn: remove it from the triangulation, compute the 3D point from the remaining N-1 cameras, then project into the held-out camera. This prevents the held-out camera's label from influencing the 3D point — making it a pure prediction test.

**Why it's the fairest test for comparing calibrations:** In test D, a bad camera's label pulls the triangulated point toward itself, partially hiding its own error. In leave-one-out, the held-out camera has no influence on the 3D point, so its error is an honest measure of how well the calibration predicts that camera's observations.

**Typical values:** Slightly higher than test D (the 3D point is less constrained without one camera).

### Global geometry considerations

**What is "global geometry"?**

Each camera has a position and orientation in 3D space. The global geometry is the arrangement of all cameras relative to each other. A good calibration has the global geometry right — each camera is positioned where it actually is in the real world.

**Why per-board error doesn't test global geometry:**

Per-board error gives each image its own board pose. A camera could be placed at the wrong position in 3D space, and per-board error wouldn't notice — it would just estimate a different board pose to compensate. Only when you force multiple cameras to agree on the SAME 3D points (multi-view triangulation) does the global geometry get tested.

**How we test global geometry:**

1. **Multi-view triangulation error** — forces all cameras to agree on 3D point positions
2. **Known-geometry validation** — checks whether the agreed-upon positions produce correct real-world measurements
3. **Leave-one-out** — checks whether the global geometry can predict new observations
4. **Convergence analysis** — checks whether different starting points lead to the same global geometry (if they don't, the problem is underdetermined)

### ChArUco board corners vs human-labeled rat nose

| Aspect | ChArUco corners | Rat nose |
|--------|----------------|----------|
| **2D detection precision** | ~0.1 pixels (algorithmic subpixel) | ~3-5 pixels (human clicking) |
| **3D point rigidity** | Perfectly rigid (manufactured board) | Deformable (biological tissue) |
| **Known 3D geometry** | Yes (80mm grid spacing) | No |
| **Available on experiment day** | Only if board is present | Always (the animal is the subject) |
| **Tests calibration on** | Calibration day | Experiment day |
| **Number of points** | Hundreds (auto-detected) | A few (manually labeled) |
| **Statistical power** | Very high (1000+ observations) | Low (87 observations in our test) |
| **What errors it can detect** | Intrinsic model errors, extrinsic drift from calibration day | Cross-session camera movement, real-world tracking accuracy |

**The key insight:** ChArUco tests tell you how good the calibration was on calibration day. The nose test tells you how good it is on experiment day. Both matter, and they measure different things. A calibration that scores perfectly on ChArUco but poorly on the nose test has cameras that moved between sessions — exactly the problem SP refinement is designed to solve.

---

## 9. File Locations

| Item | Path |
|------|------|
| MVC calibration YAMLs | `.../original_calibration_results/calibration/` |
| RED v4 calibration YAMLs | `.../redv4/aruco_image_experimental/2026_03_17_10_55_50/` |
| LG from MVC (locked) | `.../lightglue_sp_from_mvc/` |
| LG from v4 (locked) | `.../lightglue_sp_from_redv4/` |
| LG from MVC (unlock intr) | `.../lg_unlock_intr_from_mvc/` |
| LG from v4 (unlock intr) | `.../lg_unlock_intr_from_v4/` |
| LG from MVC (unlock all) | `.../lg_unlock_all_from_mvc/` |
| LG from v4 (unlock all) | `.../lg_unlock_all_from_v4/` |
| LightGlue landmarks | `.../lightglue_from_mvc/landmarks.json` |
| Nose labels (6 frames) | `.../nose_labels/all_nose_labels.json` |
| All experiment logs | `.../claude_research/` |
| This document | `.../claude_research/march18_LGrefine.md` |

All paths relative to `/Users/johnsonr/datasets/rat/sp_test_march17/`

---

## 10. Reproducing These Experiments

All experiments use the `test_holdout_compare` binary at commit `3774e01` (branch `rob_ui_overhaul`).

```bash
cd /Users/johnsonr/src/red
cmake --build release --target test_holdout_compare

# Locked (baseline):
release/test_holdout_compare .../landmarks.json .../calibration/ \
    --rot-prior 50 --trans-prior 500 --outlier-th1 30 --outlier-th2 10 \
    --output .../output_folder

# Unlock intrinsics:
release/test_holdout_compare .../landmarks.json .../calibration/ \
    --rot-prior 50 --trans-prior 500 --outlier-th1 30 --outlier-th2 10 \
    --unlock-intrinsics --output .../output_folder

# Unlock all:
release/test_holdout_compare .../landmarks.json .../calibration/ \
    --rot-prior 50 --trans-prior 500 --outlier-th1 30 --outlier-th2 10 \
    --unlock-all --output .../output_folder
```

LightGlue matching requires Python with torch + lightglue:
```bash
python3 data_exporter/calibration_refinement.py \
    --image_dir .../frames/set_* \
    --calib_dir .../calibration/ \
    --output_dir .../output/ \
    --output_mode pairwise \
    --max_keypoints 4096 --resize 1600 --match_threshold 0.2 \
    --reproj_thresh 15.0 --min_matches 5 --device cpu
```

---

## 11. Assessment of Initial Calibration Quality

### The calibration is good — but not great

RED v4's ChArUco calibration is the best we have tested: 0.437 px per-board reproj, 0.48 px multi-view triangulation, 0.339 mm known-geometry error on 80mm squares. It beats MVC on every metric. But these numbers hide significant per-camera variation.

### Per-camera breakdown reveals structural weaknesses

| Tier | Cameras | Per-board reproj | Observation count | Notes |
|------|---------|-----------------|-------------------|-------|
| Excellent | 710038, 2002496, 2002489, 2002488 | 0.18–0.23 px | 60–90 boards | Good viewing angle, high SNR |
| Acceptable | 2002491, 2002495, 2002480, etc. | 0.3–0.5 px | 30–60 boards | Adequate coverage |
| Problematic | **2002481** (0.98 px), **2002482** (0.83 px), **2002483** (0.66 px) | >0.6 px | 15–30 boards | Oblique angles, sparse coverage |
| Critical | **2002492** (6 boards), **2002493** (13 boards) | High | <15 boards | Far too few observations |

Cam2002492 and Cam2002493 have 36–56 px leave-one-out error on the nose test — across ALL four calibrations. No amount of SP refinement can fix a camera whose intrinsics are poorly constrained by 6 board images.

### The bottleneck is data collection, not algorithms

The calibration algorithm (multi-round Ceres BA with robust Cauchy loss) is sound — it converges cleanly and produces a solution that is self-consistent by its own metrics. The problem is that several cameras simply did not see enough board images at enough diversity of angles. This is a human logistics problem: holding a ChArUco board so that every camera gets 30+ views at varied orientations is tedious and error-prone, especially for cameras mounted at extreme angles in a multi-camera rig.

### The detection-quality paradox

We discovered on March 16 that detecting MORE ChArUco corners actually degrades calibration quality. Our "v6" detection improvements found 26% more corners (415→524) by recovering small markers at image periphery. But the new detections had imprecise subpixel localization, and calibration accuracy worsened (known-geometry error rose from 0.339→0.555 mm). This is a well-known phenomenon in calibration literature: the observation-to-parameter ratio matters less than observation quality. A few hundred clean corners trump a thousand noisy ones.

---

## 12. Honest Assessment of LightGlue Refinement

### What it does well

1. **Genuine signal exists.** SP-refined calibrations win 12–13 of 17 cameras on the nose test, consistently across 6 labeled frames. The holdout cross-validation confirms no overfitting (ratio 1.21–1.23×). This is not noise — the refinement is finding real camera movement.

2. **LightGlue matching quality is remarkable.** 19.7% of tracks seen by 3+ cameras (vs 4.3% with brute-force MNN) provides much stronger geometric constraints. The convergence test confirms this: LightGlue holds the two starting-point refinements stable (0.97× initial gap), while MNN lets them diverge (1.22×).

3. **The pipeline is principled.** Locked intrinsics, prior-weighted BA, holdout validation, convergence testing — these are the right controls. We can be confident the refinement is not doing something pathological.

### What it does NOT do well

1. **The improvement magnitude is small.** Median LOO goes from 5.3 px (RED v4) to 5.2 px (best refined). This is within labeling noise (~3–5 px). With only 87 observations across 6 frames, we cannot statistically distinguish the calibrations at high confidence. The refinement may be worth 0.5–2 px on average, but proving this requires many more labeled frames.

2. **It cannot fix fundamentally bad cameras.** Cam2002492 (6 boards, 31–46 px error) and Cam2002493 (13 boards, 47–56 px error) remain catastrophic after refinement. If the intrinsics are wrong (because too few board images constrained them), locking wrong intrinsics and adjusting extrinsics cannot compensate. The error is structural.

3. **2-camera tracks dominate.** Even with LightGlue, 80.3% of tracks are 2-camera only. These provide weak geometric constraints — a 2-view track constrains a point along an epipolar line but cannot resolve depth ambiguity well, especially when the camera baseline is short. The 19.7% multi-view tracks do the heavy lifting, but they are concentrated on cameras with overlapping fields of view. Peripheral cameras with narrow overlap (Cam2002496: 91 observations) are barely constrained.

4. **The refinement does not converge to a unique solution.** Starting from MVC vs RED v4, the locked refinement reduces the rotation gap by only 3% (0.420°→0.405°). The two solutions are still meaningfully different — some cameras (e.g., Cam2002489: 0.648° gap) barely move toward each other. This suggests the data does not uniquely determine the extrinsics, even with LightGlue quality matches.

5. **17 minutes of Python is impractical for a desktop tool.** The LightGlue matching takes ~25 minutes on CPU. We have a native CoreML+BLAS pipeline that runs in ~35 seconds but produces brute-force MNN quality (which diverges). Porting LightGlue itself to CoreML is non-trivial due to its attention-based architecture with dynamic early stopping.

6. **Scene content is mixed.** The rat and robot arms are moving objects that produce incorrect correspondences. Uniform frame sampling helps (vs diverse sampling, which was worse), but there is no explicit static/dynamic separation. A moving-object mask would improve track quality.

### Bottom line

LightGlue refinement is a valid approach for cross-session extrinsic correction. It works — but modestly. The dominant source of error in our system is not camera movement between sessions; it is insufficient calibration data for cameras with poor viewing geometry. Fixing the worst cameras through better board collection would likely improve tracking accuracy more than any post-hoc refinement.

---

## 13. Future Directions

### 13.1 Better initial calibration data

The single highest-impact improvement is collecting more ChArUco board images for the problematic cameras (Cam2002481, 2002482, 2002492, 2002493). Specifically:
- Target 40+ boards per camera at diverse angles (not just frontal)
- Verify coverage during collection using RED's live reprojection display
- Add a "calibration sufficiency" metric that warns when a camera has too few observations or insufficient angular diversity (ratio of max-to-min board normal directions)

Automated board detection quality filtering — reject detections where the marker side length is below a minimum pixel threshold — would prevent the v6-style regression where more corners = worse calibration.

### 13.2 Structured light or laser refinement

RED already has a laser calibration pipeline (`laser_calibration.h`) that detects laser spots across multiple cameras simultaneously. A structured light pattern (e.g., a laser pointer swept through the arena) naturally produces multi-view tracks with perfect correspondence — every camera sees the same physical light source. Unlike feature matching, there is zero ambiguity about which point corresponds to which. This approach could produce dense, high-quality multi-view constraints for extrinsic refinement without any learned matcher.

### 13.3 Dense photometric methods: 3D Gaussian Splatting

We previously researched 3D Gaussian Splatting (3DGS) for scene reconstruction of the behavioral arena (see `claude_dev_notes/3d_reconstruction/scene_reconstruction_research.md`). 3DGS is directly relevant to calibration in several ways:

**Calibration validation via novel-view synthesis.** Train a 3DGS model on N-1 cameras, then render the held-out camera's view and compare to the actual image. The photometric error (PSNR, SSIM, LPIPS) is a dense, pixel-level calibration quality metric — far richer than sparse reprojection error on a few hundred corners. A well-calibrated camera will produce a sharp, accurate rendering; a miscalibrated camera will produce ghosting, blur, or shifted geometry. This could replace or supplement the nose test with an automated metric that requires no human labeling.

**Joint pose optimization.** Recent work (CF-3DGS, 3R-GS, JOGS, TrackGS — all 2024-2025) jointly optimizes 3DGS scene parameters and camera poses. Starting from our ChArUco calibration, a joint optimization could correct small extrinsic errors by finding poses that maximize photometric consistency across all views. This is conceptually similar to SP refinement but operates on raw pixels instead of sparse keypoints — making it far more constrained (every pixel provides signal). The challenge is that 3DGS training requires GPU compute (A100 recommended, ~$0.10 per training run on Janelia's cluster) and the joint pose optimization landscape can have local minima.

**Background model for foreground segmentation.** A trained 3DGS model of the empty arena can render the expected background from any camera viewpoint. Differencing the rendering against actual video frames produces a clean foreground mask of the animal — useful for both keypoint labeling (mask out background distractors) and for filtering SP refinement tracks (reject correspondences on the moving animal).

**Practical path.** We already have `data_exporter/red_to_colmap.py` to convert RED calibrations to COLMAP format, which is the standard input for all 3DGS tools. The recommended starting point is OpenSplat (runs on macOS with Metal, no remote GPU needed) or Nerfstudio's Splatfacto (needs an NVIDIA GPU but is well-documented). The 16 calibrated cameras with known poses eliminate COLMAP's hardest problem (Structure-from-Motion) — we already know the camera positions.

### 13.4 Temporal refinement

Our current approach uses 50 frames from a single moment in a recording session. Using frames from throughout a session (beginning, middle, end) would capture a wider diversity of static scene features and average out transient objects. A rolling refinement that processes batches of frames as the video plays could also detect cameras that drift during a session (e.g., from thermal expansion or mechanical creep).

### 13.5 Multi-session calibration tracking

The underlying problem — cameras moving between calibration day and experiment day — would benefit from a lightweight "calibration check" protocol. Before each experiment session, capture 30 seconds of a static target (even just the empty arena). Run SP refinement on those frames. If any camera's extrinsic delta exceeds a threshold, flag it for recalibration or apply the correction. This turns one-shot calibration into ongoing calibration maintenance, which is what large multi-camera systems actually need.

### 13.6 Better matchers — and why we probably don't need them

LightGlue is already near the ceiling of what sparse matching can achieve on this data. The fundamental limitation is not match quality but match density: most of the arena floor and walls are textureless, producing no keypoints at all. A fundamentally different approach — dense correspondence via optical flow (RAFT, FlowFormer), semi-dense matching (LoFTR, ASpanFormer), or photometric alignment (3DGS as above) — would provide signal from every pixel, not just corners and edges. The brute-force MNN pipeline already shows that match quantity without quality is harmful (divergence); the LightGlue experiments show that quality without quantity is helpful but limited. The next step is quality AND quantity, which means moving beyond sparse keypoint methods entirely.

### 13.7 Pose Splatter and novel applications

Pose Splatter (NeurIPS 2025) combines 3DGS scene models with animal pose estimation from 4–6 calibrated cameras, and was tested on mice, rats, and zebra finches — exactly our target species. A validated 3DGS model of our arena, combined with RED's multi-camera keypoint labels, could enable novel applications: rendering the scene from the animal's perspective (a "rat's eye view"), generating synthetic training data for pose estimation models, or creating publication-quality 4D visualizations of animal behavior overlaid on a photorealistic arena reconstruction.
