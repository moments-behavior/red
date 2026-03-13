# RED Technology Roadmap: From Labeling Tool to Behavioral Neuroscience Platform

*Johnson Lab, HHMI Janelia Research Campus -- March 2026*

This roadmap synthesizes cutting-edge developments across animal pose estimation,
behavioral analysis, and neuroscience tooling, with concrete recommendations for
RED integration. Each section covers the state of the art, what RED should build,
and the expected impact.

---

## Executive Summary

RED is already the only tool that unifies GPU-accelerated multi-camera calibration,
real-time video playback, 3D keypoint annotation, and neural network inference in a
single native desktop application. To become the default platform for multi-camera
behavioral neuroscience, RED should pursue three strategic thrusts:

1. **Close the annotation loop** -- active learning, foundation model bootstrapping,
   and point-tracker propagation to make labeling 10-50x more efficient
2. **Extend downstream** -- multi-animal tracking, behavior classification, and
   neural recording alignment to keep researchers inside RED longer
3. **Enable live experiments** -- real-time 3D pose streaming for closed-loop
   optogenetics, robotics, and adaptive paradigms

The result: a platform where a researcher calibrates cameras, labels a few dozen
frames, trains a model, runs real-time 3D tracking, classifies behaviors, and
correlates with neural recordings -- all from one application on a MacBook.

---

## 1. Foundation Models for Pose Estimation

### State of the Art

**SuperAnimal** (Ye et al., Nature Communications 2024) provides zero-shot pose
estimation across 45+ species via unified keypoint spaces and gradient masking to
prevent catastrophic forgetting during multi-dataset training. Fine-tuning requires
10-100x less labeled data than training from scratch. Integrated into DeepLabCut 3.0
with SuperAnimal-Quadruped and SuperAnimal-TopViewMouse pretrained models.

**ViTPose / ViTPose++** (Xu et al., CVPR 2023 / TPAMI 2023) establishes Vision
Transformers as universal pose estimation backbones. Simple architecture: ViT encoder
+ lightweight decoder. Scales from ViT-S (fast) to ViT-H (highest accuracy). The
APT-36K benchmark (NeurIPS 2022) evaluates across 30 animal species.

**RTMPose** (CVPR 2023) achieves real-time multi-person pose estimation: 75.8% AP on
COCO at 90+ FPS on CPU, 430+ FPS on GPU, 70+ FPS on mobile (Snapdragon 865).
Whole-body 67.0% AP at 130+ FPS. Optimized for deployment via TensorRT/ONNX/CoreML.

### What RED Should Build

**Priority: HIGH -- Near-term (1-2 months)**

- **SuperAnimal bootstrap import.** Before any manual labeling, offer a "Bootstrap
  with SuperAnimal" button that runs SuperAnimal predictions on a frame subset (via
  Python subprocess or pre-exported CoreML model) and imports results as editable
  draft labels. Map SuperAnimal's unified keypoint space to RED's skeleton definition
  via a configurable keypoint mapping table.

- **RTMPose as lightweight inference backbone.** RTMPose's speed (430+ FPS on GPU)
  makes it ideal for RED's real-time prediction overlay. Export RTMPose-m or RTMPose-l
  to CoreML for Apple Silicon. This could replace or complement the current
  JARVIS CenterDetect+KeypointDetect pipeline with a single-stage model.

- **ViTPose for high-accuracy offline prediction.** Offer ViTPose-L/H as an accuracy-
  optimized option for final prediction passes, exported to CoreML. The larger model
  size (~300M params for ViT-H) is acceptable for batch processing.

### Expected Impact

Eliminates the cold-start problem. A new lab can install RED, open videos, run
SuperAnimal, and get reasonable 3D pose estimates within minutes -- before labeling
a single frame. This alone would attract labs currently using DeepLabCut, because
DLC requires its own Python environment and training pipeline to access SuperAnimal.

---

## 2. Self-Supervised and Semi-Supervised Pose Estimation

### State of the Art

**Lightning Pose** (Biderman et al., Nature Methods 2024) introduces three
semi-supervised losses for multi-view pose estimation:
- Pose PCA loss: enforces predictions within a learned low-dimensional subspace
  (r=0.91 correlation with actual error -- strongest known signal)
- Multi-view PCA loss: penalizes deviation from learned 3D hyperplane after
  epipolar triangulation (r=0.88 correlation with error)
- Temporal difference loss: penalizes frame-to-frame jumps (r=0.26, weakest)

**SelfPose3d** (2024) achieves fully self-supervised multi-person 3D pose
estimation using only calibrated multi-view images and off-the-shelf 2D detectors.
No ground-truth 2D or 3D annotations required. Uses differentiable rendering of
2D Gaussian heatmaps with adaptive attention to handle pseudo-label noise.

**MVT (Multi-View Transformer)** (arXiv 2510.09903, 2025) concatenates all camera
views into a single transformer sequence with learnable view encodings. Patch
masking curriculum (10% to 50%) forces cross-view information propagation. Combined
with a 3D triangulation loss, achieves +10.8 AP25 improvement on 3-camera setups.

### What RED Should Build

**Priority: HIGH -- Core active learning signal (1 month)**

- **Reprojection error as free uncertainty signal.** RED already computes DLT
  triangulation across 16 cameras. After running inference on all frames, triangulate
  predicted 2D keypoints to 3D and compute per-keypoint reprojection error. This
  gives RED an uncertainty signal as good as model ensembles (r~0.88) for zero
  additional compute cost. No other single-view tool can match this.

- **Pose PCA scoring.** Fit PCA on the labeled 3D poses. Score each predicted frame
  by its Mahalanobis distance from the learned subspace. Combined with reprojection
  error, this provides the two strongest known frame-ranking signals (r=0.91 and
  r=0.88) without any ensemble training.

- **Semi-supervised loss export.** When exporting training data, include multi-view
  geometry (camera matrices, frame correspondences) so that Lightning Pose or
  similar methods can use RED's calibration for semi-supervised training. RED's
  calibration files become training signal.

### Expected Impact

RED's multi-camera geometry provides frame-ranking signals that match or exceed
ensemble methods, for free. This is RED's single most defensible advantage for
active learning. A Nature Methods paper can demonstrate this quantitatively:
"reprojection error from 16 calibrated cameras provides uncertainty estimates
equivalent to 5-model ensembles at 1/5 the compute cost."

---

## 3. Real-Time 3D Tracking for Live Experiments

### State of the Art

**DeepLabCut-Live** (eLife 2021) provides real-time 2D pose estimation with <15ms
latency for closed-loop optogenetics. Supports PyTorch/TensorFlow models with
dynamic cropping and custom processor callbacks. No built-in 3D capability.

**RTMPose** achieves 430+ FPS on GPU, 90+ FPS on CPU -- more than sufficient for
real-time multi-camera tracking at typical behavioral neuroscience frame rates
(30-200 fps).

**BlueBerry** (bioRxiv 2025) combines wireless optogenetics with real-time
multi-animal pose tracking for behavior-contingent stimulation.

**AnimalRTPose** (Neural Networks 2025) achieves 476-1111 FPS for 2D animal pose
estimation, purpose-built for real-time applications.

### What RED Should Build

**Priority: HIGH -- Differentiating capability (2-3 months)**

- **Live prediction overlay during playback.** Run CoreML inference on each decoded
  frame during video playback, displaying predicted 2D keypoints and triangulated
  3D positions in real-time. Current CoreML latency (~16ms/camera) is already fast
  enough for 30fps playback across 16 cameras with pipelining.

- **Real-time 3D pose streaming via ZMQ/shared memory.** Accept live camera feeds
  (RTSP, USB, GigE Vision) or connect to frame grabber APIs. Run RTMPose per
  camera, triangulate to 3D via Eigen DLT, and stream 3D poses at <50ms latency.
  Publish on a ZMQ socket for consumption by:
  - Closed-loop optogenetics controllers
  - Universal Robots cobot path planners (RED already has UR calibration)
  - Open Ephys for synchronized neural+behavioral recording
  - Custom experiment control scripts

- **Quantized models for latency.** Use coremltools post-training int8 quantization
  on EfficientTrack/RTMPose backbone. Target: <100ms total for 16-camera 3D pose
  (decode + infer + triangulate).

### Expected Impact

No existing tool provides real-time calibrated 3D multi-camera pose tracking on
consumer hardware. DeepLabCut-Live does 2D only. DANNCE does 3D but requires GPU
servers and is not real-time. This capability enables a new class of experiments:
real-time behavior-contingent manipulation with calibrated 3D pose feedback. For
labs with UR cobots (increasingly common at Janelia), RED becomes the control
backbone.

---

## 4. Behavior Classification from Pose

### State of the Art

**B-SOiD** (Hsu & Bhatt, Nature Communications 2021) discovers behavioral clusters
from pose data using UMAP dimensionality reduction + HDBSCAN clustering. Unsupervised:
finds behavioral categories without predefined labels. Works with DLC/SLEAP output.

**Keypoint-MoSeq** (Weinreb et al., Nature Methods 2024) parses sub-second
behavioral syllables from keypoint data using autoregressive hidden Markov models.
Discovers stereotyped movement patterns and transition structure ("behavioral grammar")
without supervision. Published as a Nature Methods paper.

**CEBRA** (Nature 2023) jointly embeds behavioral and neural data into consistent
latent spaces using self-supervised contrastive learning. Enables decoding of neural
activity from behavior and vice versa. Supports arbitrary time series inputs.

**AmadeusGPT** (Adaptive Motor Control Lab) uses LLMs to translate natural language
queries about behavior into executable analysis code. "How much time does the mouse
spend grooming?" generates the computation automatically from pose keypoint files.

**MoSeq** (Drosophila/mouse) uses depth video to discover behavioral syllables via
autoregressive HMMs. Keypoint-MoSeq extends this to work with 2D pose data directly.

**SimBA** (Nature Methods 2019) provides a GUI for supervised behavioral classification
from 2D pose using random forests and other classifiers.

### What RED Should Build

**Priority: MEDIUM -- Natural downstream extension (2-3 months)**

- **3D pose export in B-SOiD/Keypoint-MoSeq/CEBRA formats.** Export calibrated 3D
  joint positions + velocities + angles as time series CSV/HDF5. Include metadata
  (skeleton topology, joint names, coordinate system) for plug-and-play import.
  This is low effort (~200 LOC per format) with high impact.

- **Built-in behavioral annotation layer.** Add frame-range labels (e.g., "grooming"
  from frame 1200-1350, "locomotion" from 1350-1500) alongside keypoint annotations.
  Store in AnnotationMap. Display as colored bands on the transport bar timeline.
  Export as ethogram CSV.

- **Lightweight behavior embedding visualization.** Compute PCA or UMAP on 3D joint
  angle time series. Display as a 2D scatter in an ImPlot panel, colored by behavior
  label or time. Researchers can visually verify that behavioral clusters correspond
  to meaningful categories.

- **LLM-powered behavior query (longer-term).** Integrate an LLM API (local or
  cloud) that accepts natural language queries about the 3D pose data, similar to
  AmadeusGPT. "Show me all frames where the mouse rears on hind legs" generates a
  filter on 3D joint angles. This would be a compelling demo feature.

### Expected Impact

RED produces calibrated 3D pose trajectories -- the ideal input for behavioral
analysis tools. But today, researchers must export from RED, write conversion
scripts, and import into separate Python tools. Native export + visualization keeps
researchers in RED and positions it as the entry point to the behavioral analysis
pipeline. The 3D pose advantage is real: view-invariant features from calibrated
multi-camera tracking are more robust than 2D features from single cameras.

---

## 5. Multi-Animal Tracking with Identity Assignment

### State of the Art

**Multi-animal DeepLabCut** (maDLC, Lauer et al., Nature Methods 2022) adds identity
prediction networks and part affinity fields for multi-animal 2D tracking. Identity
assignment uses appearance + spatial features.

**SLEAP** is designed from the ground up for multi-animal pose with both top-down
(detect then estimate) and bottom-up (detect all joints then group) strategies.
Flow-shift tracking associates poses across frames using temporal context + optical
flow. Fast: up to 600+ FPS batch, <10ms realtime.

**DREEM** (Relates Every Entity's Motion) uses Global Tracking Transformers for
multi-object identity tracking in biological video. Transformer-based architecture
maintains identity across occlusions. Exports .slp files compatible with SLEAP.

**Social-DANNCE** (s-DANNCE, Cell 2025) extends volumetric 3D pose to interacting
animals with 6.7x precision improvement during social interactions.

**SBeA** (Nature Machine Intelligence 2024) achieves few-shot multi-animal 3D pose
from ~400 annotated frames with >90% identity recognition accuracy via zero-shot
identity transfer.

**vmTracking** (PLOS Biology 2025) uses virtual markers to resolve identity in
crowded multi-animal environments with minimal manual corrections.

### What RED Should Build

**Priority: HIGH -- Field requirement for social neuroscience (3-4 months)**

- **N instances per frame in AnnotationMap.** Extend the data model from 1 skeleton
  per frame to N, each with a persistent integer ID. UI for creating/deleting/
  switching animal instances with keyboard shortcuts. This is the prerequisite for
  everything else in multi-animal support.

- **Cross-camera identity linking via epipolar geometry.** When multiple animals are
  detected in multiple views, use calibrated epipolar geometry to match detections
  across cameras. RED's 16-camera setup makes this dramatically easier than
  single-camera approaches: an animal occluded in camera 3 is visible in cameras
  1, 2, 4-16. Rank candidate matches by geometric consistency (triangulation
  residual).

- **Temporal identity propagation.** Use simple nearest-neighbor matching on
  triangulated 3D positions to propagate IDs across frames. When 3D distance
  between consecutive frames exceeds a threshold (indicating potential swap or
  occlusion), flag for human review. The 3D trajectory constraint is much stronger
  than 2D tracking.

- **Multi-animal export.** Support DANNCE .mat (per-animal 3D landmarks), SLEAP .slp
  (via sleap-io), COCO multi-instance, and DLC multi-animal H5 formats.

### Expected Impact

Social neuroscience (pair housing, social defeat, mating, aggression) is the
fastest-growing application area for pose estimation. Labs that need multi-animal
tracking currently must use DLC or SLEAP, which lack RED's multi-camera 3D
advantage. RED's geometric identity assignment (using 16-camera triangulation to
disambiguate animals) would be a unique capability that no single-camera tool can
replicate. This is likely the single most important feature for expanding RED's
user base.

---

## 6. Integration with Neural Recording Systems

### State of the Art

**Open Ephys** provides three Python modules for neural recording integration:
- Analysis module: loads Binary/OpenEphys/NWB formats with common interface
- Control module: remote GUI automation for programmatic data acquisition
- Streaming module: real-time data reception via callbacks

**CEBRA** (Nature 2023) jointly embeds neural and behavioral data using self-
supervised contrastive learning. Requires synchronized time series of neural
activity and behavioral variables (e.g., 3D pose).

**Facemap** (Nature Neuroscience 2024) predicts neural activity from mouse facial
movements using deep learning. Demonstrates that orofacial behavior is a strong
predictor of cortical activity.

**Neuropixels** probes record from hundreds to thousands of neurons simultaneously.
Synchronization with behavioral data requires shared TTL triggers or network time
protocol alignment.

### What RED Should Build

**Priority: MEDIUM -- Completes the neuroscience workflow (2 months)**

- **TTL/trigger event export.** Export frame timestamps as TTL-compatible event files
  (Open Ephys binary, NWB events) for post-hoc alignment with neural recordings.
  Include camera trigger timestamps, behavioral event markers (from behavior
  annotation layer), and pose-derived events (e.g., "rearing onset").

- **NWB export for 3D pose data.** Export triangulated 3D keypoint trajectories in
  Neurodata Without Borders (NWB) format using the ndx-pose extension. This is
  the emerging standard for sharing behavioral data alongside neural recordings.
  DANDI archive requires NWB format for data sharing.

- **CEBRA-compatible export.** Export 3D pose time series (joint positions, velocities,
  angles) as NumPy arrays with timestamps aligned to neural recording clocks.
  Include metadata for CEBRA's auxiliary variable format.

- **Real-time pose streaming to Open Ephys.** Via RED's ZMQ pose streaming (see
  Section 3), publish 3D pose data that Open Ephys can receive through its streaming
  module. Enables real-time neural-behavioral co-visualization.

### Expected Impact

The neuroscience community increasingly demands synchronized neural and behavioral
data in standardized formats. NWB+DANDI compliance is becoming a requirement for
NIH-funded research. RED producing NWB-formatted 3D pose data directly would make
it the preferred annotation tool for any lab that also records neural activity --
which is the majority of behavioral neuroscience labs.

---

## 7. Cloud-Based Training Pipelines

### State of the Art

**DeepLabCut 3.0** migrated to PyTorch backend, enabling standard distributed
training workflows via PyTorch Lightning.

**Lightning Pose** uses PyTorch Lightning for cloud-native training with multi-GPU
support, experiment tracking, and checkpoint management.

**SLEAP** provides integrated training via its GUI with real-time loss curve
visualization over ZeroMQ.

Labs without local GPU servers increasingly use cloud compute: AWS (p3/p4 instances),
Google Cloud (TPU pods), Lambda Labs, and Janelia's internal HPC cluster.

### What RED Should Build

**Priority: MEDIUM -- Reduces friction for training (1-2 months)**

- **One-click "Export and Train" workflow.** RED exports COCO/DLC/JARVIS format,
  packages with a training script (PyTorch Lightning-based), and optionally uploads
  to a configured cloud endpoint (S3 bucket, SSH server, or Janelia HPC). Display
  training progress in RED via WebSocket or log polling.

- **Training recipe presets.** Bundle tested training configurations:
  - "Quick" (RTMPose-s, 50 epochs, ~15 min on single GPU)
  - "Standard" (RTMPose-m, 200 epochs, ~1 hour)
  - "High accuracy" (ViTPose-L, 300 epochs, ~4 hours)
  - "Fine-tune SuperAnimal" (SuperAnimal-Quadruped, 50 epochs from pretrained)

- **Model zoo integration.** Download pretrained models (SuperAnimal, RTMPose,
  ViTPose) directly from RED's UI. Auto-convert to CoreML for local inference.
  Maintain a curated list of models with species, keypoint count, and accuracy.

- **Distributed training support.** Generate Slurm job scripts for HPC clusters.
  Support multi-GPU training via PyTorch DDP. Include Weights & Biases or MLflow
  integration for experiment tracking.

### Expected Impact

Training is currently the biggest friction point in the RED workflow: users must
leave RED, set up a Python environment, configure training, wait, convert the model,
and import back. Each step loses users. A streamlined export-train-import cycle
within RED would be a major usability win.

---

## 8. Active Learning Strategies

### State of the Art

**DeepLabCut** uses three outlier detection methods: uncertainty (heatmap likelihood),
jump (frame-to-frame displacement), and fitting (ARIMA time series model). Initial
frame selection via k-means clustering on downsampled images.

**SLEAP** has the most sophisticated suggestion system: BRISK + HOG feature
extraction, PCA, k-means clustering for visually distinctive frame selection.
Prediction score sorting for model-uncertainty-based suggestion.

**Lightning Pose** demonstrates that Pose PCA loss (r=0.91) and Multi-view PCA loss
(r=0.88) are the strongest known uncertainty signals for pose estimation, beating
ensemble disagreement in correlation with true error while being cheaper to compute.

**MVT paper** (arXiv 2510.09903, 2025) establishes quality+diversity filtering as
the best practice: select top-N frames by uncertainty, then k-means on 3D poses to
ensure diversity. Random selection significantly degrades performance vs. this
combined strategy.

**Prodigy** demonstrates the power of binary accept/reject workflows: show a
prediction, user clicks "accept" or drags to fix. Streaming EMA for prioritization
without loading all scores into memory.

**PosePAL** (arXiv 2025) uses CoTracker3 to propagate sparse annotations across
video, achieving 81.6% accuracy from only 6 annotated frames in a 60-frame clip.

### What RED Should Build

**Priority: CRITICAL -- RED's primary differentiator (2-3 months)**

This is the single most important feature set for RED's publication and adoption.

- **Phase 1: Geometric uncertainty scoring (2 weeks).** After running inference on
  all frames, compute per-frame composite score:
  ```
  score = w1 * reprojection_error + w2 * heatmap_entropy + w3 * pca_deviation
  ```
  where reprojection_error comes from triangulating 2D predictions across 16
  cameras, heatmap_entropy from the inference output, and pca_deviation from a
  PCA model fit on labeled 3D poses. This gives RED uncertainty estimates
  equivalent to 5-model ensembles at 1/5 the cost.

- **Phase 2: Quality+diversity frame selection (1 week).** Implement the MVT
  paper's two-stage filtering: (1) select top-N frames by composite uncertainty,
  (2) k-means on triangulated 3D poses, select frame nearest each cluster center.
  Present ranked queue in labeling UI.

- **Phase 3: Accept/fix workflow (1 week).** Prodigy-inspired binary workflow:
  navigate to suggested frame, predicted keypoints shown as translucent overlay,
  user clicks "Accept All" or drags individual keypoints to correct. Promoted
  from Predicted to Manual in AnnotationMap.

- **Phase 4: Point tracker propagation (2 weeks).** Integrate CoTracker3 (or
  TAPIR) to propagate keypoint annotations across video clips. User labels frame N,
  point tracker fills frames N+1 through N+K, user reviews and corrects. Combined
  with active learning, this could reduce annotation from 200 frames to ~20 manually
  labeled frames + review.

- **Phase 5: Dashboard with stopping criterion (2 weeks).** ImPlot dashboard
  showing accuracy vs. labeled frames curve, per-keypoint error, uncertainty
  heatmap over timeline. Automatic stopping criterion when improvement drops below
  threshold for two consecutive rounds.

### Expected Impact

This is RED's Nature Methods paper. The key claim: "Multi-camera geometric
consistency provides free uncertainty signals (r=0.88-0.91) that match ensemble
methods, enabling 10x reduction in labeling effort for 3D pose estimation." No
single-view tool can make this claim. The benchmarks practically write themselves:
learning curves comparing RED's geometry-aware active learning vs. DLC's outlier
detection vs. SLEAP's feature-based suggestions vs. random sampling.

---

## 9. Transformer-Based Pose Estimation

### State of the Art

**ViTPose** (CVPR 2023) demonstrates that a plain Vision Transformer with minimal
modifications outperforms specialized pose architectures. Key insight: large-scale
pretraining (MAE on ImageNet) + simple decoder is sufficient. ViTPose-H achieves
state-of-the-art on COCO, AP-10K (animal), and other benchmarks.

**RTMPose** (CVPR 2023) optimizes the deployment pipeline: SimCC coordinate
classification head (instead of heatmap regression), knowledge distillation from
large to small models, and deployment-aware architecture search. Achieves industrial-
grade speed (430+ FPS on GTX 1660 Ti) with competitive accuracy.

**Multi-View Transformers** for pose embed all camera views as tokens in a single
transformer, enabling learned cross-view attention. The MVT paper shows this
outperforms separate per-view processing + triangulation.

### What RED Should Build

**Priority: MEDIUM -- Model infrastructure (1-2 months)**

- **Model-agnostic inference interface.** Abstract RED's inference pipeline behind a
  common interface: `predict(image) -> heatmaps + keypoints + confidence`. Support
  pluggable backends:
  - CoreML (Apple Silicon, current JARVIS models)
  - ONNX Runtime (cross-platform, any PyTorch model)
  - TensorRT (Linux NVIDIA, highest throughput)

- **RTMPose CoreML export pipeline.** Script to convert RTMPose models (from MMPose)
  to CoreML via torch -> ONNX -> CoreML. Include in RED's model zoo. RTMPose-m
  offers the best speed/accuracy tradeoff for interactive use.

- **ViTPose for accuracy-critical workflows.** Offer ViTPose-L as a "high accuracy"
  option for final prediction passes. The larger model is acceptable for batch
  processing where latency is not critical.

- **Multi-view transformer integration (longer-term).** When MVT-style models
  mature, support loading a single model that consumes all 16 camera views
  simultaneously. This eliminates the need for separate per-camera inference +
  triangulation. RED's calibration YAML files would be consumed directly as
  geometric priors by the model.

### Expected Impact

Model-agnostic inference makes RED future-proof. As new architectures emerge (and
they will -- the field moves fast), RED can adopt them without architectural changes.
RTMPose on CoreML would give RED the fastest interactive pose prediction of any tool
in the field.

---

## 10. 3D Reconstruction (NeRF, Gaussian Splatting) for Behavioral Arenas

### State of the Art

**3D Gaussian Splatting** (Kerbl et al., SIGGRAPH 2023) represents scenes as
collections of 3D Gaussians, enabling real-time novel-view synthesis at 100+ FPS
at 1080p. Training requires only multi-view images + camera poses. Key advantages
over NeRF: explicit representation (no neural network at render time), real-time
rendering, and editable scene geometry.

**Nerfstudio** (Berkeley 2022) provides a modular framework for neural radiance
field development, supporting multiple methods including NeRF, Instant-NGP, and
Gaussian splatting variants.

**Dynamic 3D Gaussians** and **4D Gaussian Splatting** extend the approach to
dynamic scenes, tracking Gaussian movements over time -- directly relevant to
tracking animals in 3D arenas.

### What RED Should Build

**Priority: LOW -- Experimental/future (3-6 months)**

- **Arena reconstruction from calibration images.** RED's calibration pipeline
  already captures multi-view images of the behavioral arena (ChArUco boards in
  the scene). Use these same images to build a 3D Gaussian Splatting model of the
  empty arena. This provides:
  - A 3D visualization of the recording environment
  - Novel viewpoint synthesis for checking calibration quality
  - A geometric prior for pose estimation (animals vs. background)

- **Background model for foreground segmentation.** A Gaussian Splatting model of
  the empty arena can render the expected background from any camera viewpoint.
  Difference between the rendered background and the actual frame gives a clean
  foreground mask -- superior to static background subtraction because it handles
  lighting changes and multi-view consistency.

- **4D pose visualization.** Render 3D animal trajectories (from RED's triangulated
  poses) overlaid on the Gaussian Splatting arena model. Enable free-viewpoint
  replay of behavior from any angle, not just the fixed camera views. This would
  be a compelling visualization for publications and presentations.

### Expected Impact

This is a "wow factor" feature rather than a core need. The arena reconstruction
has practical value for calibration verification and background subtraction. The
4D visualization would make RED demos and paper figures dramatically more compelling.
However, the engineering effort is significant and the Gaussian Splatting ecosystem
is still maturing. Recommend as a post-publication stretch goal.

---

## Strategic Priorities: What Would Make RED the Default Tool?

### For Labs Currently Using DeepLabCut

DLC's strengths: large community, SuperAnimal foundation models, multi-animal
support, extensive documentation. DLC's weaknesses: Python/Qt GUI is slow for
multi-camera work, no built-in calibration, no real-time playback, 3D requires
separate Anipose step.

**To attract DLC users, RED needs:**
1. SuperAnimal import (bootstrap labeling without training)
2. DLC project import/export (seamless migration)
3. Active learning that demonstrably beats DLC's outlier detection
4. Multi-animal support (table stakes for social neuroscience)

### For Labs Currently Using SLEAP

SLEAP's strengths: best-in-class multi-animal tracking, integrated training GUI,
active learning suggestions, 600+ FPS inference. SLEAP's weaknesses: no native
3D, no calibration, limited to 2D, Python/Qt GUI.

**To attract SLEAP users, RED needs:**
1. Multi-animal support with identity tracking
2. SLEAP format import/export (.slp via sleap-io)
3. Active learning with visual suggestion queue
4. Proof that 3D multi-camera tracking outperforms 2D for their species

### For Labs Building New Rigs

These labs have no tool lock-in and evaluate on capability + ease of setup.

**To capture new rig installations, RED needs:**
1. Zero-friction install (Homebrew -- already done)
2. End-to-end workflow demo: calibrate -> label -> train -> predict in one session
3. Documentation and tutorials (video walkthroughs)
4. Support for common camera hardware (FLIR, Basler, Allied Vision)

### What a Nature Methods Paper Would Emphasize

1. **Geometric active learning.** Quantitative demonstration that multi-camera
   reprojection error provides uncertainty estimates equivalent to model ensembles,
   enabling 10x reduction in labeling effort. Learning curves on multiple datasets
   (rat, mouse, fly) comparing RED vs. DLC vs. SLEAP vs. random sampling.

2. **Integrated pipeline.** Time-to-result comparison: raw video to trained 3D pose
   model. RED (single application) vs. DLC+Anipose+custom scripts (fragmented).

3. **Real-time capability.** Demonstrate real-time 3D pose overlay during video
   playback on consumer MacBook. No other tool achieves this.

4. **Calibration accuracy.** RED's 0.447px reprojection error vs. Anipose and
   multiview_calib baselines.

5. **Accessibility.** Homebrew install, no Python, no CUDA drivers. Inference on
   Apple Silicon Neural Engine.

---

## Implementation Timeline

### Q2 2026: Publication-Ready Core
- Active learning Phase 1-3 (geometric uncertainty + frame selection + accept/fix)
- SuperAnimal bootstrap import
- RTMPose CoreML export and integration
- Active learning dashboard (ImPlot)
- Benchmarking suite for paper figures
- NWB export for 3D poses

### Q3 2026: Multi-Animal and Training Integration
- Multi-instance annotation data model
- Cross-camera identity linking
- One-click export-and-train workflow
- Training recipe presets
- Point tracker propagation (CoTracker3)
- Behavior annotation layer

### Q4 2026: Live Experiments and Downstream Analysis
- Real-time 3D pose streaming (ZMQ)
- Open Ephys integration
- B-SOiD / Keypoint-MoSeq / CEBRA export
- Behavior embedding visualization
- Model zoo with download UI
- SLEAP/DLC project import

### 2027: Advanced Capabilities
- Multi-view transformer inference
- Arena Gaussian Splatting reconstruction
- LLM-powered behavior queries
- Physics-informed pose refinement (MuJoCo)
- 128+ camera scalability

---

## References

### Foundation Models
- Ye et al. (2024). SuperAnimal pretrained pose estimation models. *Nature Communications*.
- Xu et al. (2023). ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation. *CVPR*.
- Jiang et al. (2023). RTMPose: Real-Time Multi-Person Pose Estimation. *arXiv*.

### Semi-Supervised Methods
- Biderman et al. (2024). Lightning Pose. *Nature Methods*.
- SelfPose3d (2024). Self-supervised multi-person 3D pose from calibrated multi-view.
- Aharon et al. (2025). MVT: Uncertainty-aware multi-view animal pose estimation. *arXiv 2510.09903*.

### Multi-Animal Tracking
- Lauer et al. (2022). Multi-animal DeepLabCut. *Nature Methods*.
- Pereira et al. (2022). SLEAP. *Nature Methods*.
- DREEM: Global Tracking Transformers for biological multi-object tracking.
- Social-DANNCE (2025). *Cell*.
- SBeA (2024). *Nature Machine Intelligence*.

### Behavioral Analysis
- Hsu & Bhatt (2021). B-SOiD. *Nature Communications*.
- Weinreb et al. (2024). Keypoint-MoSeq. *Nature Methods*.
- CEBRA (2023). *Nature*.
- AmadeusGPT. Adaptive Motor Control Lab.

### Real-Time Tracking
- DeepLabCut-Live (2021). *eLife*.
- AnimalRTPose (2025). *Neural Networks*.

### 3D Reconstruction
- Kerbl et al. (2023). 3D Gaussian Splatting. *SIGGRAPH*.
- Nerfstudio (2022). Berkeley.

### Neural Recording Integration
- Open Ephys Python Tools.
- Facemap (2024). *Nature Neuroscience*.
- NWB ndx-pose extension.

### Point Tracking
- TAPIR / TAP-Net. Google DeepMind.
- CoTracker3. Meta AI Research.
- PosePAL (2025). *arXiv*.
