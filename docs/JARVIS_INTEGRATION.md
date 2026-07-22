# JARVIS ↔ red Fly 3D Training — Integration Plan

Working notes / design doc for exporting red-labeled fly data to JARVIS-HybridNet and
(eventually) launching training + managing JARVIS projects from inside red.

- red repo: `/home/rob/src/red` (branch `xp`)
- JARVIS repo: `/home/rob/src/JARVIS-HybridNet` — **use branch `dlt_fly_20260331`**
- First target project: `/home/rob/alanna/Proboscis/Proboscis.redproj`
  (telecentric, 7 cams, skeleton `proboscis4.json`, 4 nodes)

---

## Background: why the export was failing

The fly rig is **telecentric** (`"telecentric": true` in the `.redproj`), so its calibration
is **DLT**: the calibration folder holds `Cam*_dlt.csv` (11 DLT coefficients each), **not**
`<cam>.yaml`. red's built-in JARVIS exporter (`src/jarvis_export.h`) was ported from the
*standard pinhole* Python exporter and hard-codes reading `<cam>.yaml` for image dims and
`camera_matrix`/`rc_ext`/`tc_ext`. No such YAML exists for the fly rig → the error
`Cannot open calibration: .../Cam2012630.yaml`.

The correct fly logic lives in `data_exporter/red3d2jarvis.py` on red's `dltv2` branch
(see `data_exporter/FLY.md` there).

## The ×10 scaling hack (fly rig)

- JARVIS annotations contain **only per-camera 2D keypoints** (from red's `<cam>.csv`).
  3D is **re-triangulated inside JARVIS** from the 2D points + the `projectionMatrix`.
- So the "scale fly data ×10" is carried **entirely by the projection matrix**:
  `projectionMatrix[0:2, 0:3] *= 0.1`. Only the top two rows are scaled because a
  telecentric camera's DLT third row is `[0, 0, 0, 1]` (affine, `w ≡ 1`), so it's irrelevant.
- **Why:** JARVIS forces `HYBRIDNET.GRID_SPACING` to an **integer number of mm** in `[0,10]`,
  with `ROI_CUBE_SIZE % (4·spacing) == 0`. A ~3 mm fly would get ~3 voxels — useless.
  Inflating world units ×10 gives ~0.1 mm effective resolution.
- The YAML `scale: 10` field is **documentation only** — JARVIS never reads it.
- **3D keypoint values are never pre-scaled** anywhere.
- Predicted 3D from JARVIS comes out in ×10 units → divide by 10 to get mm.

Decision: ×10 is exposed as a **"Scale 10×" checkbox, checked by default** on telecentric
projects. It stays a toggle so that once JARVIS is patched to allow sub-integer mm grid
spacing (future work), we can simply uncheck it.

---

## Goal 1 — Fix the exporter (telecentric/DLT + Scale 10×)   [IMPLEMENTED — calib path verified]

> **History:** an interim change (commit `d37eabc`, 2026-07-20) had the telecentric
> path copy the raw `<cam>_dlt.csv` **verbatim** instead of writing a YAML. The
> JARVIS fork can't read a raw CSV (`cv2.FileStorage(...).getNode('projectionMatrix')`
> needs a `<cam>.yaml`), so that produced **untrainable** datasets (e.g. the
> 2026-07-22 `Feeding/jarvis_merged` with raw `_dlt.csv` + no YAML). Restored below.

Status (re-verified 2026-07-22): telecentric export/merge now writes a
JARVIS-readable `projectionMatrix` YAML via **`JarvisExport::write_projection_yaml`**
(builds the 3×4 from the 11 DLT coeffs + append `1.0`, then `[0:2,0:3]*=0.1` when
`scale_10x`). Wired into **both** `export_jarvis_dataset` overloads
(`src/jarvis_export.h`), the unified `src/export_formats.h` +
`src/gui/export_window.h`, the legacy `src/gui/jarvis_export_window.h`, **and the
Group merge** (`src/jarvis_merge.h` → `MergeConfig::scale_10x`,
`src/gui/group_export_window.h`). A **"Scale 10× (fly/telecentric)"** checkbox
(default on, shown only for telecentric) drives `scale_10x` in all three windows.
`build release` passes. The writer output was checked byte-for-byte against the
known-good Proboscis reference YAML on the real `Cam2012630` DLT data:
`data:[ 8.18532, 0.139856, -0.0797258, -33.2725, 0.162519, -8.20211, -0.206741,
414.813, 0, 0, 0, 1 ]`, `scale: 10`, and round-trips as a (3,4) matrix.
`write_dlt_calibration` (verbatim copy) is retained but **unused**, kept for Goal-3
"plan (b)" (teach the fork to read raw `_dlt.csv`). Remaining: a full GUI
export+merge run to confirm annotations + JPEG extraction end-to-end, then create-project
+ train on the re-exported Feeding merge.


Key seam: red already loads DLT calibration into `pm.camera_params` (each `CameraParams`
has a 3×4 `projection_mat` and `telecentric=true`), via `camera_load_params_from_dlt_csv`
in `src/camera.h`. The DLT loader leaves `image_width/height = 0`, so telecentric dims come
from the **video** (`ffmpeg_reader::FrameReader.width()/height()`), matching the Python path.

### `src/jarvis_export.h`
- `#include "camera.h"`.
- `ExportConfig`: add `bool telecentric`, `bool scale_10x`,
  `std::vector<CameraParams> camera_params`, `std::vector<int> image_width/image_height`
  (optional pre-resolved dims from the unified path).
- New helper `resolve_export_dims()`: per camera, dims priority =
  `cfg.image_width[ci]` (>0) → `camera_params[ci].image_width` (>0) →
  telecentric: open video via `FrameReader`; else: read `<cam>.yaml`.
- Both `export_jarvis_dataset` overloads: replace the hard-coded `<cam>.yaml` dim read
  with `resolve_export_dims(...)`.
- `write_calibration_yamls`: when `telecentric`, write
  `{ image_width, image_height, projectionMatrix (with [0:2,0:3]*=0.1 if scale_10x), scale }`
  from `camera_params[ci].projection_mat`; else keep the existing pinhole path.
  (Node name **must** be exactly `projectionMatrix` — that's what JARVIS's
  `Camera.get_mat_from_file(..., 'projectionMatrix')` reads.)
- The annotation-JSON path is **unchanged** (already reads `<cam>.csv` 2D labels; structure
  already matches the Python exporter: `category_id:1`, no `area`, Y-flip, 0-based ids).

### `src/gui/jarvis_export_window.h` (legacy "JARVIS Export Tool" — the one in the screenshot)
- `JarvisExportState`: add `bool scale_10x = true;`.
- Add a **"Scale 10× (fly/telecentric)"** checkbox.
- Config build: set `jcfg.telecentric = pm.telecentric; jcfg.scale_10x = state.scale_10x;
  jcfg.camera_params = pm.camera_params;`.

### `src/export_formats.h` + `src/gui/export_window.h` (unified Export window)
- `ExportFormats::ExportConfig`: add `bool telecentric`, `bool scale_10x`.
- `export_window.h`: add checkbox to `ExportWindowState`/UI; set
  `ecfg.telecentric = pm.telecentric; ecfg.scale_10x = state.scale_10x;`.
- `export_formats.h::export_jarvis()`: forward `telecentric`, `scale_10x`, `camera_params`,
  `image_width`, `image_height` into the `JarvisExport::ExportConfig` (currently dropped).

### Build & verify
- `cmake --build release -j`.
- Export Proboscis; check `calib_params/<trial>/<cam>.yaml` has `projectionMatrix` + `scale:10`,
  `annotations/instances_{train,val}.json`, and JPEGs under `train/`,`val/`.
- Structural diff vs the `dltv2` Python exporter on the same session.

### Watch-items
- **Frame offset:** C++ applies `detect_negative_pts_offset`; the Python fly path applies
  none. Internal consistency holds (annotation `file_name` and JPEG name both use the RED
  frame number), but confirm decoded image content matches labels.
- Confirm `opencv_yaml::YamlWriter` emits a `projectionMatrix` node that `cv2.FileStorage`
  reads back as a 3×4 `mat()`.

---

## Goal 2 — CLI training   [ENV SET UP + SMOKE-TESTED ✓]

Branch `dlt_fly_20260331`. Env `jarvis` (`~/miniconda3`; `conda` not on PATH →
`source ~/miniconda3/etc/profile.d/conda.sh`). **Working recipe on this box (fresh Ubuntu 24):**

```bash
conda create -n jarvis python=3.9 -y -c conda-forge --override-channels   # skips Anaconda ToS
conda activate jarvis
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
cd /home/rob/src/JARVIS-HybridNet && pip install -e .
pip install numpy==1.23.5 opencv-python==4.9.0.80 altair==4.2.2   # override bad transitive pins
```

Why not the README env: README's `pytorch=1.10.1 cudatoolkit=11.3` has no sm_89 kernels →
fails on the RTX 4000 Ada. torch 2.0.1/cu118 works and still runs the torch-1.10-era code.
Re-pins needed because deps pull numpy 2 (breaks imgaug/torch), opencv-python 5 (needs numpy 2),
altair 5 (breaks streamlit 1.11's `import streamlit.cli`).

**GPU safety (orange capture box):** 8× A16 = capture (GPUs 0-3,5-8, never touch); 1× RTX 4000
Ada = free training GPU (GPU 4). Pin every jarvis command with
`CUDA_VISIBLE_DEVICES=GPU-ce2647f2-0d5a-b7da-073b-cfd4fc97240f`.

```bash
GPU=GPU-ce2647f2-0d5a-b7da-073b-cfd4fc97240f
# create-project: grid-spacing suggestion comes out 0 for ×10 fly data (divide-by-zero if
# accepted) → force 1mm. Feed stdin: yes (2D bbox) / no,1 (grid) / yes (3D bbox):
printf 'yes\nno\n1\nyes\n' | CUDA_VISIBLE_DEVICES=$GPU jarvis-local create-project \
    --dataset2d <export> --dataset3d <export> Proboscis
CUDA_VISIBLE_DEVICES=$GPU jarvis-local train centerDetect   --num_epochs 50  Proboscis
CUDA_VISIBLE_DEVICES=$GPU jarvis-local train keypointDetect --num_epochs 100 Proboscis
CUDA_VISIBLE_DEVICES=$GPU jarvis-local train hybridNet --num_epochs 50 \
    --weights_keypoint_detect latest Proboscis
```

Smoke test (1 epoch each) PASSED on 2026-07-16: centerDetect val 4.4 px, keypointDetect val
4.9 px, hybridNet val 1.88 "mm" (÷10 → ~0.19 mm). Project config: 192 px 2D box, 4 joints,
7 cams, ROI_CUBE_SIZE 8, GRID_SPACING 1 (8³ grid — coarse; revisit for final quality).
Weights → `projects/Proboscis/models/{CenterDetect,KeypointDetect,HybridNet}/`.
`--pretrained_weights fly50_Dec15` available on this branch. Predicted 3D is ×10 → divide by 10.

---

## Goal 3 — In-red JARVIS training & project integration   [DESIGN — decisions deferred]

New settings: `jarvis_repo_path` (default `/home/rob/src/JARVIS-HybridNet`), `jarvis_conda_env`
(default `jarvis`). Reuse red's `popen` + detached `std::thread` + atomic Job struct pattern
(see the conversion buttons in `src/gui/jarvis_predict_window.h`).

New `src/gui/jarvis_training_window.h`:
- **Export → Project:** `jarvis-local create-project --dataset3d <export> <name>`.
  Open decision: the interactive grid-spacing prompt — either add non-interactive
  `--grid_spacing/--bbox_size` flags to the fork (cleaner) or script stdin from red.
- **Projects list:** scan `<jarvis_repo>/projects/*/config.yaml`; show trained-net status
  (presence in `models/{CenterDetect,KeypointDetect,HybridNet}`); "Load" sets active.
- **Train:** buttons per net (+ "train all"), epoch inputs, pretrain dropdown (`fly50_Dec15`),
  stdout streamed to a log panel, progress from parsed epoch lines, Cancel via `pclose`.
- **After training:** hand off to existing `.pth`→TensorRT conversion + register a
  `pm.jarvis_models` entry, closing the loop into red's predict window.

Phasing: P1 launch-training + log; P2 project list/load/status; P3 create-project from export
(needs the fork flag decision); P4 auto-register trained model for inference.
