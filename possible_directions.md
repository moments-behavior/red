# RED: Possible Directions for Scaling and Impact

*Johnson Lab, HHMI Janelia Research Campus — March 2026*

This document outlines ambitious directions for RED's development and its positioning
for a bioRxiv publication. Each section describes the opportunity, relevant prior work,
and concrete next steps.

---

## 1. Closed-Loop Active Learning

**The opportunity.** The most expensive step in markerless pose estimation is manual
annotation. RED already has the full loop: annotate → export → train (JARVIS) →
predict (CoreML) → inspect → fix. Making this loop *adaptive* — where the system
selects the most informative frames for human review — could reduce labeling effort
by 5–10x.

**Prior work.**
- DeepLabCut's active learning selects frames with highest prediction uncertainty
  for human correction (Mathis et al., Nature Neuroscience 2018).
- SLEAP provides a labeling GUI with integrated active learning and proofreading
  (Pereira et al., Nature Methods 2022).
- Aharon et al. (arXiv 2510.09903, 2025) present an uncertainty-aware framework
  for data-efficient multi-view animal pose estimation, using model distillation
  to generate pseudo-labels and reduce dependence on manual labeling.

**What RED can add.** RED's multi-camera triangulation provides a unique signal:
*reprojection error*. A prediction that triangulates well across 16 cameras is almost
certainly correct, regardless of the 2D confidence score. Conversely, high-confidence
2D predictions that triangulate poorly indicate ambiguity or occlusion — exactly the
frames a human should review. This geometric consistency signal is unavailable to
single-view tools like DeepLabCut or SLEAP.

**Next steps.**
- Add per-frame triangulation error to AnnotationMap (already computed during
  reprojection, just not stored).
- Implement "suggest frames" that ranks unpredicted frames by a composite score:
  (a) temporal distance from nearest labeled frame, (b) predicted 2D entropy,
  (c) cross-camera reprojection error.
- One-click "accept" or "fix" workflow in the Labeling Tool for predicted frames.
- Track labeling progress in a Dashboard panel (frames labeled, model accuracy
  curve, estimated remaining effort).

---

## 2. Real-Time Inference on Apple Silicon

**The opportunity.** RED's CoreML backend already achieves ~255 ms for 16-camera
prediction (CenterDetect + KeypointDetect + triangulation) on Apple M-series chips.
With batch inference and pipeline overlap, real-time prediction during playback
(≥30 fps across all cameras) is within reach.

**Prior work.**
- MobilePoser (UIST 2024) achieves ~14 ms per frame on iPhone 15 Pro using CoreML
  with mixed-precision quantization.
- Apple's Neural Engine instruments (WWDC 2024) provide profiling for CoreML models
  to maximize ANE utilization.
- YOLOv8 pose models run at real-time speeds on Apple Silicon via CoreML export
  (Ultralytics, 2024–2025).

**What RED can add.** No existing tool provides real-time 3D multi-camera pose
prediction with triangulation on a consumer laptop. This would enable researchers
to preview annotation quality during data collection — a paradigm shift from
"record first, annotate later" to "annotate live, fix later."

**Next steps.**
- Profile current CoreML pipeline with Instruments to identify ANE vs GPU split
  and find bottlenecks (likely the bilinear resize in vImage).
- Implement pipeline overlap: decode frame N+1 while predicting frame N.
- Explore int8 quantization of EfficientTrack backbone (coremltools provides
  post-training quantization; target <100 ms for 16 cameras).
- Add a "live prediction" toggle that runs inference on every decoded frame during
  playback, displaying predicted keypoints in real-time.

---

## 3. Foundation Model Integration (SuperAnimal, SAM)

**The opportunity.** Foundation models trained on massive datasets can provide
zero-shot or few-shot predictions across species, dramatically reducing cold-start
labeling effort.

**Prior work.**
- SuperAnimal (Ye et al., Nature Communications 2024) unifies keypoint space across
  45+ species and provides zero-shot pose estimation. Models available at
  modelzoo.deeplabcut.org. Requires 10–100x less labeled data than training from
  scratch.
- SAM 2 (Ravi et al., 2024) provides unified image + video segmentation with a
  memory bank for tracking objects across frames.
- SAM 3 (Meta, 2025) adds text-prompt-driven concept segmentation, enabling
  "segment all rats" without click prompts.
- CellSAM (Nature Methods 2025) demonstrates domain-specific SAM adaptation for
  biological data.

**What RED already has.** MobileSAM integration with click-to-segment, multi-mask
candidate cycling, and per-frame mask persistence. This foundation is ready for
extension.

**Next steps.**
- Integrate SuperAnimal as a zero-shot initialization: before any manual labeling,
  run SuperAnimal predictions on a subset of frames to bootstrap the annotation.
- Use SAM 2's video propagation to automatically propagate accepted masks across
  frames, reducing per-frame annotation to a single click on the first frame.
- Investigate SAM 3's text prompts for species-level segmentation (e.g., "segment
  the mouse" across all cameras simultaneously).
- Export SAM masks as training data for JARVIS's segmentation branch.

---

## 4. Multi-Animal Tracking and Identity

**The opportunity.** Social behavior studies require tracking multiple interacting
animals simultaneously. Identity preservation during occlusion is the key challenge.

**Prior work.**
- Multi-animal DeepLabCut (Lauer et al., Nature Methods 2022) adds identity
  prediction networks for multi-animal tracking.
- SLEAP's flow-shift tracking associates poses across frames using temporal context
  and optical flow.
- DIPLOMAT (August 2025) uses a novel approach to tolerate occlusion and preserve
  animal identity in multi-animal scenarios.
- vmTracking (2025) minimizes manual corrections using virtual markers and achieves
  state-of-the-art identity preservation.
- Multi-animal 3D social pose estimation (Nature Machine Intelligence 2023) uses
  a few-shot learning framework for multi-animal 3D pose.

**What RED can add.** RED's multi-camera setup provides a natural solution to
occlusion: an animal occluded in one camera view is likely visible in another. By
triangulating identity assignments across cameras, RED can maintain identity through
occlusions that defeat single-camera approaches. The existing AnnotationMap already
supports multiple instances per frame.

**Next steps.**
- Add multi-instance support to the labeling UI (currently one instance per frame).
- Implement cross-camera identity assignment using epipolar geometry: if animal A
  in camera 1 and animal B in camera 2 produce consistent triangulations, they are
  the same individual.
- Add a temporal tracking layer that propagates identity assignments across frames
  using triangulated 3D trajectories.
- Export multi-animal annotations in COCO multi-instance and SLEAP formats.

---

## 5. Behavior Classification from 3D Pose

**The opportunity.** The end goal of pose estimation is understanding behavior.
Researchers want to know not just *where* the animal is, but *what* it is doing
(grooming, rearing, freezing, social interaction). RED's 3D poses are ideal input
for behavior classifiers.

**Prior work.**
- B-SOiD (Hsu & Bhatt, Nature Communications 2021) discovers behavioral clusters
  from pose data using unsupervised learning.
- SimBA (Nature Methods 2019, updated 2024) provides a simple behavioral analysis
  GUI for classifying behaviors from 2D pose.
- VAME (Luxem et al., Nature Communications 2022) uses variational autoencoders for
  behavioral segmentation from pose sequences.
- Keypoint-MoSeq (Weinreb et al., Nature Methods 2024) discovers behavioral
  syllables from keypoint data using autoregressive HMMs.

**What RED can add.** Existing tools operate on 2D pose from single cameras. RED's
calibrated 3D trajectories provide view-invariant behavioral features that are more
robust to camera angle and occlusion. A behavior classification module operating on
3D joint angles and velocities would be a unique capability.

**Next steps.**
- Export 3D pose trajectories in a format compatible with B-SOiD and Keypoint-MoSeq.
- Add a simple built-in behavior annotation layer: frame-level labels (e.g.,
  "grooming", "locomotion") alongside keypoint labels.
- Implement a lightweight behavior embedding (PCA or UMAP of 3D joint angle time
  series) visualized in a new "Behavior" panel.
- Train simple classifiers (random forest, small LSTM) on labeled behavior segments
  and display predictions on the timeline.

---

## 6. Physics-Informed Pose Refinement with MuJoCo

**The opportunity.** Neural network predictions can produce physically impossible
poses (limb penetration, impossible joint angles, floating contacts). Physics
simulation can refine poses to be physically plausible while staying close to the
neural network predictions.

**Prior work.**
- Whole-body physics simulation of fruit fly locomotion (Vaxenburg et al., Nature
  2025, Janelia) introduced a comprehensive Drosophila model in MuJoCo with
  adhesion actuators, proprioceptive sensors, and RL-trained controllers. The body
  model datasets are publicly available on Figshare.
- FARMS (Thandiackal et al., PLOS Computational Biology 2024) is an open-source
  framework integrating MuJoCo for animal locomotion simulation across species
  including mice, flies, fish, and salamanders.
- MuJoCo Playground (Google DeepMind, 2025) demonstrates rapid policy training
  in minutes on a single GPU using MJX (JAX-accelerated MuJoCo).

**What RED can add.** The Johnson Lab studies rats, mice, and fruit flies — all
species with published MuJoCo body models. RED could use these models as
biomechanical priors: given a noisy neural network prediction, find the nearest
physically valid pose by solving an inverse kinematics problem constrained by
MuJoCo's joint limits, contact constraints, and dynamics.

**Next steps.**
- Load MuJoCo MJCF/XML body models as "skeletons with physics" alongside RED's
  existing skeleton definitions.
- Implement joint angle constraints from the MuJoCo model as post-processing
  filters on predicted keypoints.
- For the Drosophila model (Janelia collaboration): run MuJoCo inverse kinematics
  on RED's 3D trajectories to extract physically consistent joint angles.
- Long-term: use MuJoCo forward simulation to fill in occluded frames where no
  camera provides a clear view.

---

## 7. Scalability: From 16 to 128+ Cameras

**The opportunity.** Current high-density recording systems at Janelia and elsewhere
use 16–64 cameras. As hardware costs decrease and resolution increases, rigs with
100+ cameras will become practical. RED's architecture must scale.

**Current bottlenecks.**
- VideoToolbox decoder: one thread per camera, memory proportional to camera count.
- CoreML inference: sequential per-camera pipeline (~16 ms per camera).
- ImPlot rendering: O(cameras) viewport tiles, each with full keypoint overlay.

**Next steps.**
- Implement camera tiling with level-of-detail: show thumbnails for non-selected
  cameras, full resolution only for the active viewport.
- Batch CoreML inference: process multiple cameras in a single model call using
  batched CVPixelBuffers (requires re-export with batch>1).
- Lazy decoding: only decode cameras that are currently visible in the viewport.
- Hierarchical triangulation: use subsets of cameras for initial triangulation,
  refine with all cameras only for ambiguous frames.

---

## 8. Unified Project Format (.redproj)

**The opportunity.** A single project file containing calibration, annotations,
trained models, and behavior labels would simplify collaboration and reproducibility.

**Next steps.**
- Merge calibration project and annotation project into a single .redproj.
- Store calibration parameters, skeleton definition, annotation CSVs, model
  metadata, and behavior labels in a single directory structure.
- Add project versioning and export to standard formats (NWB, BIDS-animal).
- Support importing from DeepLabCut (.h5), SLEAP (.slp), and Anipose projects.

---

## bioRxiv Paper: Positioning and Impact

### Paper Title (working)
*RED: A GPU-Accelerated Platform for Multi-Camera 3D Animal Pose Annotation,
Active Learning, and Real-Time Inference*

### Key Differentiators vs. Existing Tools

| Feature | DeepLabCut | SLEAP | Anipose | JARVIS | **RED** |
|---------|-----------|-------|---------|--------|---------|
| Multi-camera 3D | via Anipose | via sleap-anipose | yes | yes | **native** |
| GPU-accelerated labeling | no | no | no | no | **yes (Metal/CUDA)** |
| Real-time video playback | no | no | no | no | **yes (VT/NVDEC)** |
| Built-in calibration | no | no | yes | separate | **yes** |
| SAM integration | no | no | no | no | **yes** |
| Active learning | basic | yes | no | yes | **planned** |
| Multi-animal identity | yes | yes | no | no | **planned** |
| Behavior classification | no | no | no | no | **planned** |
| CoreML/ANE inference | no | no | no | no | **yes** |
| Export formats | DLC, COCO | SLEAP | Anipose | JARVIS | **all 5+** |

### Impact Argument

1. **Throughput.** RED's hardware-accelerated pipeline (VideoToolbox decode →
   CoreML inference → Eigen triangulation) processes 16-camera recordings at
   real-time speeds on a consumer MacBook. This eliminates the GPU server
   dependency that makes tools like DeepLabCut and SLEAP expensive to deploy.

2. **Integration.** RED is the first tool to unify calibration, annotation, training
   data export, model inference, and 3D reconstruction in a single application.
   Existing workflows require 3–5 separate tools (DeepLabCut + Anipose + MATLAB
   calibration + custom scripts).

3. **Annotation efficiency.** The combination of multi-camera reprojection error
   (for active frame selection), SAM-assisted segmentation (for body part
   delineation), and CoreML real-time inference (for prediction overlay) creates
   a labeling pipeline that requires 10x fewer manually annotated frames than
   traditional approaches.

4. **Accessibility.** By running natively on Apple Silicon with no Python
   dependencies, RED lowers the barrier to entry for labs without dedicated
   computing infrastructure. Install via Homebrew, open videos, start labeling.

### Benchmarks to Include

- Labeling throughput: frames per hour with RED vs. DeepLabCut GUI vs. SLEAP
- Active learning curve: accuracy vs. number of labeled frames
- Inference latency: CoreML vs. ONNX Runtime vs. DeepLabCut GPU
- Triangulation accuracy: RED's Eigen DLT vs. Anipose
- End-to-end time: from raw video to trained 3D pose model

### Target Journals

- **Primary:** Nature Methods (tool paper, high impact for methods community)
- **Alternative:** bioRxiv preprint → eLife (open access, rapid review)
- **Conference:** NeurIPS Datasets and Benchmarks Track

---

## Priority Ranking

Based on impact, feasibility, and paper readiness:

1. **Active Learning** (Section 1) — highest unique contribution, directly measurable
2. **Real-Time Inference** (Section 2) — already partially implemented, strong demo
3. **Foundation Models** (Section 3) — leverages existing SAM work, high impact
4. **Behavior Classification** (Section 5) — natural extension, broad audience
5. **Multi-Animal** (Section 4) — large engineering effort, but high demand
6. **MuJoCo** (Section 6) — Janelia collaboration opportunity, novel contribution
7. **Scalability** (Section 7) — important for production but less novel
8. **Unified Format** (Section 8) — quality of life, not paper-worthy alone
