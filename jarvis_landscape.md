# JARVIS Integration Landscape: Closing the Active Learning Loop

## The Vision

RED today is a manual annotation tool. The goal is to transform it into a **semi-automated annotation platform** where the user annotates a small seed set, trains a model, imports predictions, corrects the worst ones, and retrains — with each iteration requiring less manual work. This is the active learning loop that DeepLabCut, SLEAP, and every modern pose estimation tool provides.

RED's unique advantage is **native multi-view 3D** — calibrated cameras, DLT triangulation, and reprojection error are all built in. Combined with JARVIS-HybridNet's volumetric 3D inference, we can build an active learning loop that exploits multi-view geometry in ways no competitor does.

## Current State

### What works today

```
RED (manual annotation) → jarvis_export.h → JARVIS dataset (COCO JSON + JPEGs + YAMLs)
                                                    ↓
                                              jarvis train (external)
                                                    ↓
                                              jarvis predict → data3D.csv
                                                    ↓
                                        jarvis2red3d.py (Python script)
                                                    ↓
                                        predictions/<timestamp>/ (RED label CSVs)
                                                    ↓
                                        [manually load in RED — awkward]
```

### What's missing

1. **No native prediction import** — `jarvis2red3d.py` is a standalone Python script, not integrated into RED's GUI
2. **No confidence visualization** — JARVIS outputs per-keypoint confidence scores (0-1), but they are discarded during import. `KeyPoints2D::confidence` and `KeyPoints3D::confidence` fields exist in the struct but are never populated
3. **No frame flagging** — no way to identify which predicted frames need human correction
4. **No accept/correct workflow** — user cannot review predictions and selectively accept or fix them
5. **No training launch** — must switch to terminal to run `jarvis train`
6. **No cluster submission** — training happens wherever the user manually runs the command
7. **No iteration tracking** — no dashboard showing annotation progress across active learning rounds

## Competitor Analysis

### DeepLabCut — Gold Standard Active Learning

DeepLabCut has the most mature iteration workflow:
- `extract_outlier_frames()` with four methods: `jump` (temporal discontinuity), `uncertain` (low heatmap confidence), `fitting` (state-space model deviation), `arima` (SARIMAX anomaly detection)
- `refine_labels()` GUI shows predictions overlaid with red edges for low confidence
- `merge_datasets()` combines corrected frames with original training data
- Iteration number auto-increments in config; datasets accumulate across iterations
- Typically 2-4 refinement iterations reach satisfactory accuracy
- Initial training: 50-200 frames; each refinement: 10-50 corrected frames

**Limitation**: 2D only. 3D via Anipose triangulation as post-processing — no cross-view reasoning during training.

### SLEAP — Unified GUI

SLEAP integrates everything in one application:
- Frame selection via image feature clustering (PCA + k-means) for diverse initial frames
- Train, predict, and proofread all within the GUI
- "Labeling Suggestions" panel recommends frames to annotate
- Predictions can be accepted/corrected in place
- Achieves >800 FPS at 1024x1024 (2D, lightweight architectures)

**Limitation**: 2D only. No multi-view or 3D support.

### DANNCE — Volumetric 3D

DANNCE operates in 3D voxel space (similar concept to JARVIS HybridNet):
- Constructs 3D volumetric representations from multi-view images
- Trains 3D CNN directly on voxel features
- ~10 FPS inference from 6 cameras on Titan V
- s-DANNCE extension for multi-animal with graph neural networks

**Limitation**: No active learning. Linear workflow: label → train → predict.

### Facebook Research — Multi-View Active Learning

Key paper: Feng et al., "Rethinking the Data Annotation Process for Multi-view 3D Pose Estimation with Active Learning and Self-Training" (2023):
- **Multi-view geometric consistency**: select frames where 2D predictions from different cameras are geometrically inconsistent (high triangulation/reprojection error)
- **Pseudo-label self-training**: confident predictions become pseudo-labels without human annotation
- **Result**: 60% reduction in turnaround time, 80% reduction in annotation cost on CMU Panoptic

**This is directly applicable to RED** — we already compute reprojection error during labeling.

### What RED + JARVIS Can Do That No One Else Does

| Capability | DLC | SLEAP | DANNCE | JARVIS+RED |
|-----------|-----|-------|--------|------------|
| Native multi-view 3D training | No | No | Yes | Yes |
| Calibrated camera geometry | Via Anipose | No | Yes | Yes (built-in) |
| Cross-view 3D reasoning in network | No | No | Yes | Yes (HybridNet) |
| Active learning loop | Yes | Yes | No | **Planned** |
| GPU-accelerated calibration | No | No | No | Yes (Metal) |
| Real-time 3D triangulation during labeling | No | No | No | Yes |
| Reprojection error for frame flagging | No | No | No | **Planned** |

## Architecture: The Full Integration

### The Active Learning Loop (Target State)

```
┌─────────────────────────────────────────────────────────────┐
│                          RED App                             │
│                                                              │
│  1. ANNOTATE ──→ 2. EXPORT ──→ 3. TRAIN ──→ 4. PREDICT     │
│       ↑                          (local or      │            │
│       │                           cluster)      ↓            │
│  7. RE-EXPORT ← 6. CORRECT ← 5. IMPORT + FLAG              │
│       │                                                      │
│       └──→ (iterate until convergence)                       │
└─────────────────────────────────────────────────────────────┘
```

Each step should be launchable from within RED. The user should never need to open a terminal or switch tools.

### Compute Model: Local Mac + Cluster

Training and inference must support two execution environments:

**Local (Mac with Apple Silicon)**:
- Good for: small datasets (<500 frames), quick iteration, prototyping
- JARVIS fork needs PyTorch with MPS (Metal Performance Shaders) backend
- Johnson Lab fork already at PyTorch 2.5.1 which has mature MPS support
- Expected training time: CenterDetect ~30 min, KeypointDetect ~1 hr, HybridNet ~2 hrs on M2 Max
- Prediction: minutes per video (vs seconds on CUDA GPU)

**Cluster (HHMI Janelia HPC with NVIDIA GPUs)**:
- Good for: large datasets, production training, batch prediction across many recordings
- Submit via SSH + SLURM (or Janelia's job scheduler)
- RED spawns SSH command, monitors job status, pulls results when done
- Training time: 10-30 min total on A100

**Architecture for dual compute**:

```cpp
struct TrainingJob {
    enum Target { LOCAL, CLUSTER };
    Target target;
    std::string project_name;
    std::string dataset_path;

    // Local
    pid_t local_pid = 0;              // subprocess PID

    // Cluster
    std::string cluster_host;          // e.g. "login1.janelia.org"
    std::string cluster_job_id;        // SLURM job ID
    std::string remote_project_path;   // /nrs/johnson/jarvis_projects/...

    // Progress (parsed from stdout or log file)
    int current_epoch = 0;
    int total_epochs = 0;
    std::string current_stage;         // "CenterDetect", "KeypointDetect", "HybridNet"
    float current_loss = 0;

    // Status
    enum Status { PENDING, RUNNING, COMPLETED, FAILED };
    Status status = PENDING;
};
```

**Local execution**:
```bash
# RED spawns this as a subprocess
cd /path/to/jarvis_project && jarvis train all my_project 2>&1 | tee train.log
```

**Cluster execution**:
```bash
# RED runs via SSH
ssh login1.janelia.org "sbatch /path/to/jarvis_train.sh my_project"
# Then polls:
ssh login1.janelia.org "squeue -j <job_id> -h -o '%T'"
# When done, copies results:
rsync -az login1.janelia.org:/path/to/predictions/ ./predictions/
```

Alternatively, use a shared filesystem (NFS/Lustre) so no file copying is needed — the project directory is the same path on both local and cluster. Janelia's `/nrs/` filesystem is mounted on both workstations and cluster nodes, making this the natural approach.

### Data Flow for Each Step

#### Step 1: Annotate
- Manual annotation in RED's labeling tool (existing)
- 10-50 initial frames, selected for pose diversity

#### Step 2: Export
- `jarvis_export.h` (existing) → COCO JSON + YAMLs + JPEGs
- Output to shared filesystem if using cluster

#### Step 3: Train
- Launch from RED UI: "Train JARVIS Model" button
- Choose: Local Mac or Cluster
- Local: `subprocess` with stdout pipe for progress
- Cluster: SSH + SLURM submit, poll for completion
- Display: progress bar, current epoch/loss, ETA

#### Step 4: Predict
- Launch from RED UI: "Run Predictions" button
- Same local/cluster choice
- `jarvis predict predict3D` on unlabeled video
- Output: `data3D.csv` with per-keypoint confidences

#### Step 5: Import + Flag
**This is the key new functionality.** Import JARVIS predictions and identify frames needing correction:

1. **Parse `data3D.csv`**: Read [x, y, z, confidence] per keypoint per frame
2. **Project to 2D**: For each camera, project 3D predictions using calibration → 2D overlay
3. **Populate confidence fields**: Store per-keypoint confidence in existing `KeyPoints2D::confidence` / `KeyPoints3D::confidence`
4. **Flag frames** using multiple criteria:
   - **Low confidence**: any keypoint below threshold (e.g., 0.5)
   - **High reprojection error**: triangulate predicted 3D → reproject to all cameras → if residual > N px, flag
   - **Temporal discontinuity**: if a keypoint moves > N mm between consecutive frames
   - **Anatomical implausibility**: if bone lengths deviate > 2σ from mean (requires skeleton edge lengths)
5. **Present flagged frames**: sorted by severity, navigable with "Next Uncertain Frame" button

#### Step 6: Correct
- Flagged frames shown with confidence color coding (green=high, red=low)
- User can:
  - **Accept** a prediction (becomes a manual label)
  - **Correct** a keypoint (drag to right position)
  - **Skip** (leave for next iteration)
- Accepted predictions are as valuable as manual labels for retraining

#### Step 7: Re-export
- Export expanded dataset (original labels + accepted predictions + corrections)
- Return to Step 3

### Smart Initial Frame Selection

For the first annotation round (before any model exists), help the user choose diverse frames:

1. Decode every Nth frame from each camera (e.g., every 100th frame)
2. Extract simple features: mean brightness, edge density (Laplacian variance), optical flow magnitude
3. K-means clustering → select 1 frame per cluster
4. Present as "Suggested frames to annotate" in the Labeling Tool

This is what SLEAP does and it significantly reduces the number of frames needed for initial training.

## Prediction Import: Detailed Design

### Reading JARVIS data3D.csv

```
Header row 0: Snout,,,,EarL,,,,EarR,,,,Tail,,,
Header row 1: x,y,z,confidence,x,y,z,confidence,x,y,z,confidence,x,y,z,confidence
Data row:     15.6,136.1,10.2,0.95,10.9,119.2,31.9,0.87,...
NaN row:      NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,...  (detection failed)
```

- 4 columns per keypoint: x, y, z, confidence
- Frame ID is implicit (row index, starting from 0 after 2 header rows)
- NaN rows = detection failed entirely (skip)
- Individual NaN keypoints within a detected frame = that keypoint was occluded

### Coordinate Transforms

```
JARVIS 3D output (world coords) ──→ RED 3D (world coords, same frame)
                                         │
                                    Project via K, R, T, dist
                                         │
                                         ↓
                                    Image 2D (top-left origin)
                                         │
                                    Y-flip: y_red = img_h - y_image
                                         │
                                         ↓
                                    RED 2D CSV (ImPlot coords)
```

### Output Format

Write to `<project>/predictions/<timestamp>/`:
- `keypoints3d.csv` — same format as manual labels (frame_id, node_idx, x, y, z, ...)
- `<camera>.csv` — same format as manual labels (frame_id, node_idx, x, y, ...)
- `confidence.csv` — frame_id, conf_0, conf_1, ..., conf_N (companion file)
- `metadata.json` — model name, threshold, frame range, avg confidence

Since the output matches `load_keypoints()` format exactly, existing load code works unchanged. Confidence is a sidecar file.

### Merge Strategy

When the user accepts predictions and saves:
- Accepted predictions go into the standard `labeled_data/` folder
- They become indistinguishable from manual labels (by design — they are human-verified)
- The JARVIS export sees them as regular training data
- Confidence.csv is not re-exported (it was for display only)

## In-App Training: Detailed Design

### UI: Training Panel

```
┌─ JARVIS Training ──────────────────────────────┐
│                                                  │
│  Project: rat_pose_v2                            │
│  Dataset: jarvis_export/2026_03_08_10_30_00      │
│  Frames:  120 train / 14 val                     │
│                                                  │
│  ┌─ Target ───────────────────────────────────┐  │
│  │ ○ Local (this Mac)                         │  │
│  │ ● Cluster (login1.janelia.org)             │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  Epochs: CenterDetect [50] KeypointDetect [100]  │
│          HybridNet [30]                          │
│  Model size: [medium ▼]                          │
│                                                  │
│  [ Start Training ]                              │
│                                                  │
│  ┌─ Progress ─────────────────────────────────┐  │
│  │ Stage: KeypointDetect (2/3)                │  │
│  │ Epoch: 67/100  Loss: 0.0034                │  │
│  │ ████████████████████░░░░░  67%             │  │
│  │ ETA: 23 min                                │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  [ Cancel ]                                      │
└──────────────────────────────────────────────────┘
```

### Progress Monitoring

**Local**: Parse stdout from the subprocess. JARVIS prints epoch progress lines like:
```
Epoch 67/100 | Loss: 0.0034 | LR: 0.00089
```

**Cluster**: Two options:
1. **Poll log file via SSH**: `ssh host "tail -1 /path/to/train.log"` every 10 seconds
2. **Shared filesystem**: If project is on NFS (`/nrs/`), read the log file directly — no SSH needed for monitoring

### JARVIS Config Generation

RED generates `config.yaml` for the JARVIS project based on UI settings:

```yaml
CENTERDETECT:
  IMAGE_SIZE: 320
  MODEL_SIZE: medium
  BATCH_SIZE: 8
  NUM_EPOCHS: 50
KEYPOINTDETECT:
  MODEL_SIZE: medium
  BOUNDING_BOX_SIZE: 320
  BATCH_SIZE: 8
  NUM_EPOCHS: 100
HYBRIDNET:
  BATCH_SIZE: 1
  NUM_EPOCHS: 30
  MODE: 3D_only
```

## In-App Inference: Two-Tier Architecture

### Tier 1: Fast 2D Preview (CoreML on Mac)

For interactive use during annotation — show predicted keypoints as the user scrubs through video:

1. Export JARVIS CenterDetect + KeypointDetect to CoreML via `coremltools`
2. Load `.mlmodelc` in RED at startup (if model exists)
3. Feed decoded CVPixelBuffer directly to CoreML (zero-copy on unified memory)
4. Run CenterDetect → crop → KeypointDetect per camera
5. Triangulate 2D predictions using `red_math.h` DLT
6. Display as semi-transparent overlay (distinct from manual labels)

**Performance estimate**: ~20ms per camera on M2. For 16 cameras: ~80ms if batched in groups of 4. Gives ~12 Hz prediction updates — sufficient for interactive feedback.

**Why CoreML not ONNX Runtime**: RED is macOS-only. CoreML accepts CVPixelBuffer directly (no GPU→CPU→GPU roundtrip). ONNX Runtime's CoreML EP adds an abstraction layer with known issues (silent FP16 conversion, no dynamic shapes) while providing no cross-platform benefit.

### Tier 2: Full 3D HybridNet (Python, local or cluster)

For production predictions — maximum accuracy using the full volumetric 3D pipeline:

1. Run `jarvis predict predict3D` as external process (same local/cluster pattern as training)
2. Import results via the prediction import pipeline described above
3. This is the output that goes into the active learning correction loop

**Why not export HybridNet to CoreML**: The 3D stage uses custom volumetric operations (multi-view feature aggregation into a voxel grid, 3D convolutions) that are difficult to export. The 2D stages export cleanly; the 3D stage almost certainly would not without significant rework. And the accuracy benefit of running full HybridNet justifies the latency of a batch job.

## JARVIS Fork Development (rob_dev branch)

The Johnson Lab fork at `github.com/JohnsonLabJanelia/JARVIS-HybridNet` has 7 branches:

| Branch | Purpose | Status |
|--------|---------|--------|
| `master` | Upstream tracking + NumPy 2.0 fix | Stable, PyTorch 2.5.1 |
| `dlt` / `dltv2` | Telecentric lens DLT projection | Ratan Othayoth, same commit |
| `dlt_fly` | Fly-specific DLT variant | Ratan Othayoth |
| `mouse3d` | Frame variation augmentation | Active |
| `datamerge` | Dataset merging changes | Minimal |
| `diff_reso` | Different camera resolution support | 7 files modified |

### Planned `rob_dev` branch

Base on `master`, cherry-pick from `dltv2` for telecentric lens support. Changes needed:

1. **Fix multi-dataset camera count** (upstream issue #8): Make `Dataset3D.__init__` handle variable `num_cameras` per frameset with padding/masking in the reprojection layer. This blocks multi-rig training workflows.

2. **MPS (Metal) backend support**: PyTorch 2.5.1 has mature MPS support. Verify JARVIS trains on `device='mps'`. Known issues: some 3D conv ops may fall back to CPU. Profile and fix.

3. **Structured progress output**: Add `--json-progress` flag to training commands that outputs machine-parseable progress (epoch, loss, stage) for RED to consume.

4. **ONNX/CoreML export script**: `tools/export_coreml.py` that takes trained weights and produces `.mlmodelc` files for the 2D stages.

5. **Batch predict2D** (upstream issue #6): Currently loops over frames. Batch for 3-5x speedup.

6. **Fine-tune mode**: `jarvis train all --finetune --weights latest` that uses 10x lower LR and trains fewer epochs. For active learning iteration rounds.

## Implementation Roadmap

### Phase 1: Prediction Import + Confidence Visualization (1-2 weeks)

Close the loop at the most basic level:
- `src/jarvis_import.h`: read data3D.csv, project to 2D, write RED CSVs
- `src/gui/jarvis_import_window.h`: import dialog with confidence threshold slider
- Extend `load_keypoints()` to read companion confidence.csv
- Color-code keypoints by confidence in viewport
- "Next Uncertain Frame" navigation

### Phase 2: Training Launch (1-2 weeks)

- Training panel UI with local/cluster toggle
- Local: spawn subprocess, parse stdout for progress
- Cluster: SSH + SLURM submit, poll log file for progress
- Config generation from UI settings
- "Run Predictions" button (same local/cluster pattern)

### Phase 3: Active Learning Dashboard (1 week)

- Iteration tracking: how many frames labeled (manual vs accepted)
- Per-iteration validation error (from `jarvis analyze`)
- Frame flagging criteria: confidence, reprojection error, temporal jumps
- "Suggested frames" queue sorted by uncertainty

### Phase 4: CoreML 2D Preview (2-3 weeks)

- Export script for CenterDetect + KeypointDetect to CoreML
- CoreML inference in RED with CVPixelBuffer input
- DLT triangulation of 2D predictions for 3D overlay
- Background thread, ~12 Hz update rate for 16 cameras

### Phase 5: JARVIS Fork Improvements (ongoing)

- Fix multi-dataset camera counts (#8)
- MPS backend validation
- Structured progress output
- Fine-tune mode
- Batch predict2D

## Key Technical Decisions

### Why not integrate JARVIS as a Python library via pybind11?

Embedding Python in a C++ app adds enormous complexity (GIL, package management, virtual environments). Instead, RED spawns JARVIS as a subprocess and communicates via files (CSV, JSON, YAML) and stdout. This keeps the two codebases independent and lets each use its natural tooling.

### Why separate confidence.csv instead of extending the CSV format?

The existing `load_keypoints()` function is battle-tested and used by both manual labels and prediction import. Adding a column would break backward compatibility. A companion file is cleaner: if it exists, load confidence; if not, assume 1.0.

### Why not use JARVIS's 2D predictions directly?

JARVIS `predict3D` produces 3D output, not per-camera 2D. For the import workflow we need 2D overlays, so we project the 3D predictions back to each camera. This is more robust than running `predict2D` per camera because the 3D predictions incorporate cross-view reasoning from HybridNet.

### Why local Mac training matters

Even though cluster GPUs are faster, local training enables:
- Quick iteration without network latency or queue wait times
- Working offline (travel, home)
- Prototyping with small datasets before committing to a full cluster job
- MPS on M2 Max is surprisingly capable for small models

## References

- [DeepLabCut active learning workflow](https://deeplabcut.github.io/DeepLabCut/docs/standardDeepLabCut_UserGuide.html)
- [SLEAP labeling suggestions](https://sleap.ai/)
- [Facebook multi-view active learning (Feng et al., 2023)](https://arxiv.org/abs/2112.13709)
- [JARVIS-HybridNet](https://github.com/JARVIS-MoCap/JARVIS-HybridNet)
- [Anipose multi-camera triangulation](https://anipose.readthedocs.io/en/latest/)
- [DANNCE volumetric 3D](https://github.com/spoonsso/dannce)
- [DeepLabCut-live real-time inference](https://elifesciences.org/articles/61909)
- [CoreML execution provider](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
- [SuperAnimal foundation models](https://www.nature.com/articles/s41467-024-48792-2)
- [Active transfer learning for pose estimation (WACV 2024)](https://arxiv.org/abs/2311.05041)
