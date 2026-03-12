# RED — Landscape Analysis and Development Roadmap

*March 2026 | Johnson Lab, HHMI Janelia Research Campus*

---

## 1. What RED Is Today

RED is a **GPU-accelerated multi-camera 3D keypoint annotation and calibration platform** for behavioral neuroscience. Written in C++ with ImGui, it runs natively on macOS (Apple Silicon with Metal) and Linux (NVIDIA with CUDA). RED bridges the gap between raw synchronized multi-camera video and labeled datasets ready for training 3D pose estimation models.

### Current Capabilities

- **Real-time playback** of 16+ synchronized high-speed camera streams at native resolution (3208x2200 @ 180 fps on Apple Silicon)
- **Built-in camera calibration**: ChArUco detection with GPU-accelerated adaptive thresholding, Zhang's intrinsic calibration, experimental PnP-based extrinsic pipeline with GNC bundle adjustment (0.447 px reprojection accuracy on 16-camera rig), laser spot refinement
- **3D keypoint annotation** with live DLT triangulation across all camera views, reprojection error visualization, and per-keypoint status tracking
- **Skeleton system** with built-in presets for rats, mice, and fruit flies, plus custom JSON skeleton support
- **JARVIS/COCO export** for training JARVIS-HybridNet, plus Python exporters for DeepLabCut and YOLO formats
- **Calibration quality dashboard** with ImPlot per-camera error charts, and a 3D calibration viewer (ImPlot3D) for inspecting camera frustums and 3D point clouds

### Technical Differentiators

1. **Native C++ performance**: ImGui/Metal/OpenGL stack provides instant response for frame-by-frame annotation — no Python GUI latency
2. **Integrated calibration pipeline**: Only tool that combines ChArUco detection through bundle adjustment with annotation in one application
3. **High camera count**: Architecture designed for 16+ cameras with per-camera decoder threads, circular PBO buffers, and GPU texture pipelines
4. **Zero OpenCV on macOS**: All math in Eigen + Ceres (red_math.h, intrinsic_calibration.h, aruco_detect.h), all I/O in custom headers
5. **Telecentric lens support** (dltv2 branch): Unique among calibration tools — enables macro-scale imaging of small animals

---

## 2. The Competitive Landscape

### Tool Comparison (March 2026)

| Tool | 3D Multi-Cam | Multi-Animal | ID Tracking | Calibration | Active? | GitHub Stars |
|------|-------------|-------------|-------------|-------------|---------|------|
| **DeepLabCut** | 2-cam built-in; Anipose for 3+ | Yes (maDLC) | Partial | Via Anipose | Very active | 5,536 |
| **SLEAP** | Via sleap-anipose | Yes (core strength) | Yes (flow-shift) | Via Anipose | Very active | 562 |
| **DANNCE/s-DANNCE** | Yes (volumetric 3D CNN) | Yes | Yes | Checkerboard + L-frame | Moderate | 247 |
| **Anipose** | Yes (triangulation toolkit) | Via upstream | Via upstream | ChArUco + iterative BA | Moderate | 432 |
| **JARVIS-HybridNet** | Yes (hybrid 2D/3D CNN) | No | No | ChArUco | Stale (2023) | 36 |
| **Lightning Pose** | Yes (multi-view losses) | No | No | Standard | Very active | 289 |
| **SBeA** | Yes | Yes (few-shot) | Yes (zero-shot >90%) | N/A | New (2024) | — |
| **MMPose/ViTPose** | Not for animals | Yes | Limited | Not included | Active | 7,404 |
| **RED** | **Yes (16+ cameras)** | **No (planned)** | **No (planned)** | **Yes (built-in, 0.447 px)** | **Active** | — |

### Where RED Fits

RED does not compete with DeepLabCut, SLEAP, or DANNCE on the **inference and training** side. It competes on the **annotation and calibration** side, where existing tools are weakest:

- **DLC and SLEAP** treat each camera view independently in their labeling GUIs. Neither can smoothly play 16 synchronized high-speed camera streams.
- **JARVIS-AnnotationTool** is the closest comparator (multi-view annotation projection, integrated calibration), but is unmaintained since December 2023.
- **Anipose** provides excellent calibration but is a command-line Python tool, not an interactive annotation application.
- No existing tool integrates camera calibration, multi-view video annotation, and training data export in a single native desktop application.

### Key Gaps RED Fills

1. **GPU-accelerated multi-camera annotation** at scale (16+ cameras, real-time playback)
2. **Calibration and labeling unified** in one application (no tool-switching)
3. **Export compatibility** without lock-in (JARVIS, COCO, DLC, YOLO formats)
4. **Native desktop performance** for annotation UX (C++/ImGui vs Python/Qt)
5. **Telecentric lens support** for macro imaging (unique in the field)

---

## 3. Recent Developments in the Field (2024-2026)

### Foundation Models for Animal Pose

**SuperAnimal** (Ye et al., Nature Communications 2024) provides panoptic pose estimation models trained on 45+ species with zero-shot inference capability. Fine-tuning requires 10-100x less labeled data than training from scratch. This fundamentally changes the annotation economics: instead of labeling 200+ frames from scratch, users can import SuperAnimal predictions as draft labels and correct only the errors.

**ViTPose++** (TPAMI 2023) establishes Vision Transformers as universal pose estimation backbones. The APT-36K benchmark (NeurIPS 2022) provides inter-species evaluation across 30 animal species.

**ADPT** (Anti-Drift Pose Tracker, eLife 2025) addresses the practical problem of keypoint tracking drift in long videos using transformer-based attention mechanisms.

### Multi-View Fusion with Transformers

**MV-SSM** (CVPR 2025) introduces state-space models (Mamba) for multi-view pose fusion, achieving +10.8 AP25 improvement on challenging 3-camera setups. **Multiple View Geometry Transformers** (CVPR 2024) inject geometric priors into attention for early cross-view fusion. These approaches could eventually consume RED's calibration YAML files directly as geometric priors.

### Semi-Supervised 3D Pose

**Lightning Pose** (Nature Methods 2024) uses temporal smoothness, multi-view consistency, and low-dimensional subspace losses to learn from unlabeled multi-camera video. **SelfPose3d** (2024) requires zero ground-truth poses — only calibrated multi-view images and an off-the-shelf 2D detector. RED records exactly the synchronized multi-camera data these methods need.

### Multi-Animal 3D Tracking

**Social-DANNCE** (Cell 2025) extends volumetric 3D pose estimation to interacting animals with 6.7x precision improvement during social interactions. **SBeA** (Nature Machine Intelligence 2024) achieves few-shot multi-animal 3D pose from ~400 annotated frames with >90% identity recognition accuracy. **3D-MuPPET** (IJCV 2024) tracks up to 10 pigeons in 3D. **vmTracking** (PLOS Biology 2025) uses virtual markers to resolve identity in crowded environments.

The field is converging on a solution: volumetric or triangulation-based 3D reasoning, combined with graph neural networks for multi-animal identity, trained on minimal labeled data via self-supervised or few-shot approaches.

### Behavioral Phenotyping from 3D Pose

**Keypoint-MoSeq** (Nature Methods 2024) parses sub-second behavioral syllables from keypoint data. **CEBRA** (Nature 2023) jointly embeds behavioral and neural data into consistent latent spaces. Both consume 3D pose time series as input — exactly what RED's pipeline produces.

### Annotation Efficiency Innovations

**PosePAL** (arXiv 2025) uses CoTracker3 (a general-purpose point tracker) to propagate sparse annotations across video, achieving 81.6% accuracy from only 6 annotated frames in a 60-frame clip. JARVIS AnnotationTool projects annotations from 2+ camera views to remaining cameras via triangulation and reprojection. These approaches could reduce RED's per-frame annotation cost by 5-10x.

### Real-Time and Closed-Loop Applications

**AnimalRTPose** (Neural Networks 2025) achieves 476-1111 FPS for 2D pose estimation. **DeepLabCut-Live** enables <15ms latency for closed-loop optogenetic experiments. **BlueBerry** (bioRxiv 2025) combines wireless optogenetics with real-time multi-animal pose tracking. RED's GPU-accelerated video decode pipeline and cobot integration (UR3e/UR5e/UR10e) position it for closed-loop robot-animal interaction experiments.

---

## 4. Development Roadmap

### Phase A — Close the Active Learning Loop (Near-term, High Impact)

**A1. Multi-view annotation projection**
Label a keypoint in 2+ camera views, auto-fill the remaining cameras via triangulation + reprojection. RED already has the calibration parameters and triangulation math (red_math.h). This is JARVIS AnnotationTool's strongest feature, and RED has all the infrastructure to implement it.
- *Impact*: 5-10x reduction in per-frame annotation time for 16-camera rigs
- *Effort*: Medium (~200 LOC in gui_keypoints.h)

**A2. Import model predictions for inspection**
Load JARVIS-HybridNet predictions (JSON with per-keypoint 2D coordinates and confidence) into RED for visualization alongside manual labels. Display prediction-vs-label comparison, per-keypoint confidence coloring, and flag low-confidence frames for review.
- *Impact*: Closes the annotate→train→predict→inspect loop without leaving RED
- *Effort*: Medium (~300 LOC for JSON import + overlay rendering)

**A3. Uncertainty-guided frame selection**
After importing predictions, automatically rank unlabeled frames by model uncertainty (low confidence, high reprojection error when triangulated across views, disagreement between views). Present a queue of "frames most worth labeling" to the annotator.
- *Impact*: Reduces total annotation burden by 50%+ by targeting the most informative frames
- *Effort*: Medium (~200 LOC for scoring + UI queue)

**A4. Import SuperAnimal/ViTPose zero-shot predictions as draft labels**
Run SuperAnimal or ViTPose on RED's exported frames (externally), then import the 2D predictions as editable draft labels. The annotator corrects rather than annotates from scratch.
- *Impact*: Dramatically reduces cold-start annotation cost
- *Effort*: Low (~100 LOC for CSV/JSON prediction import)

### Phase B — Multi-Animal Support (Medium-term, Field Requirement)

**B1. N skeletons per frame with persistent identity labels**
Extend the annotation data model from 1 skeleton per frame to N, each with a stable ID that persists across frames and camera views. UI for creating/deleting/switching animal instances.
- *Impact*: Enables RED for the dominant use case in social neuroscience
- *Effort*: High (data model + UI + export format changes)

**B2. Cross-view identity linking via epipolar geometry**
When multiple animals are detected in multiple views, use calibrated epipolar geometry to match which detection in camera A corresponds to which in camera B. Present candidates ranked by geometric consistency.
- *Impact*: Solves the multi-animal multi-view association problem during annotation
- *Effort*: High (~500 LOC using existing red_math.h)

**B3. Temporal identity propagation**
Use optical flow or simple nearest-neighbor tracking to propagate animal IDs from labeled frames to adjacent unlabeled frames. Allow the user to correct ID swaps.
- *Impact*: Maintains identity across time without labeling every frame
- *Effort*: Medium

**B4. Multi-animal export to DANNCE/SBeA/SLEAP formats**
Export multi-animal annotations in the formats expected by s-DANNCE (.mat with per-animal 3D landmarks), SBeA (multi-animal COCO), and SLEAP (.slp via sleap-io).
- *Impact*: RED becomes the annotation front-end for the entire multi-animal 3D ecosystem
- *Effort*: Low-Medium per format

### Phase C — Training and Behavioral Analysis Integration (Longer-term, Full Platform)

**C1. Subprocess training via JARVIS-HybridNet**
Launch JARVIS training as a subprocess from within RED, with progress reporting (loss curves, validation metrics) displayed in an ImGui window. Auto-import the trained model's predictions when training completes.
- *Impact*: Eliminates context-switching between annotation and training
- *Effort*: Medium

**C2. Export to behavioral analysis tools**
Export 3D pose time series in Keypoint-MoSeq and CEBRA input formats. Optionally import behavioral syllable boundaries back into RED for visualization overlaid on video.
- *Impact*: Completes the pose-to-phenotype pipeline
- *Effort*: Low per format

**C3. Real-time 3D pose streaming**
Add a real-time inference path: 2D pose estimation per camera (via SLEAP, DeepLabCut-Live, or AnimalRTPose running as external process) feeding into RED's triangulation engine, streaming 3D poses via ZMQ or shared memory. Display predicted poses overlaid on live video. Stream to cobot controller for closed-loop experiments.
- *Impact*: Enables a new class of experiments: real-time behavior-contingent manipulation
- *Effort*: High (real-time architecture changes, IPC protocol)

**C4. Point tracker integration for dense annotation**
Integrate CoTracker3 or similar point tracking to propagate sparse keypoint annotations across video clips. Annotator labels every Nth frame; point tracker fills in the rest; annotator reviews and corrects.
- *Impact*: 5-10x annotation speed improvement per video
- *Effort*: Medium (external model integration + review UI)

### Phase D — Platform Maturity (Ongoing)

**D1. Wire up 3D calibration viewer**
The ImPlot3D-based viewer (`calib_viewer_window.h`) is implemented but not yet connected to the Calibration Tool UI. Add "Show 3D" button after calibration completes.

**D2. Calibration visualization enhancements**
Add board position visualization (colored by frame), camera FOV overlap heatmap, and per-camera uncertainty ellipsoids to the 3D viewer.

**D3. Landing page with recent projects**
When RED opens with no project loaded, show a landing page with recently opened projects (calibration and annotation), create-new buttons, and quick-start documentation.

**D4. Rig Setup panel**
Visualize the camera rig in 3D with camera labels, world frame, robot bases, and arena geometry. Allow interactive adjustment of world registration points.

---

## 5. Preprint Framing

### Contribution Statement

RED is an open-source, GPU-accelerated platform for multi-camera 3D keypoint annotation that integrates camera calibration, synchronized video playback, and training data export in a single native desktop application. We present:

1. **An integrated calibration and annotation workflow** that eliminates the fragmented multi-tool pipeline currently used in behavioral neuroscience (calibrate with Anipose → label with DLC/SLEAP → triangulate with Anipose → train with JARVIS)
2. **A novel experimental calibration pipeline** using PnP-based initialization with graduated non-convexity bundle adjustment, achieving 0.447 px on images (5.7% better than the multiview_calib baseline) and 0.586 px on video (28% better than baseline)
3. **GPU-accelerated ChArUco detection** using Metal compute shaders with separable box filter adaptive thresholding, processing 16 cameras at 64 images/sec (15.3 ms/frame) with zero OpenCV dependency
4. **Real-time playback of 16 synchronized high-speed camera streams** at native resolution, enabling efficient frame-by-frame annotation with live 3D triangulation feedback
5. **Export compatibility** with multiple training backends (JARVIS-HybridNet, DeepLabCut, YOLO) without lock-in

### Target Venue

bioRxiv preprint → Nature Methods (tool paper), or eLife (tool paper track). The calibration pipeline innovations and GPU-accelerated detection could also fit a computer vision venue (CVPR/ECCV workshop on animal behavior).

### Key Comparisons to Include

| Metric | RED | JARVIS AnnotationTool | DLC GUI | SLEAP GUI |
|--------|-----|----------------------|---------|-----------|
| Max cameras | 16+ tested | 10+ | 2 (built-in 3D) | 2+ (via sleap-anipose) |
| Built-in calibration | Yes (ChArUco+BA) | Yes (ChArUco) | No | No |
| Video playback | Real-time GPU | Frame extraction | Frame extraction | Frame extraction |
| Live triangulation | Yes | Yes | No | No |
| Runtime language | C++/Metal | Python/Qt | Python/Qt | Python/Qt |
| OpenCV dependency (macOS) | None | Full | Full | Partial |
| Calibration accuracy | 0.447 px (16 cam) | Not reported | N/A | N/A |
| Active development | Yes (2026) | No (last 2023) | Yes | Yes |

---

## 6. The Big Picture

RED sits at a critical junction in the behavioral neuroscience pipeline:

```
Hardware (cameras, rigs, lights)
    ↓
Camera Calibration ←──── RED (built-in, 0.447 px)
    ↓
Synchronized Video Recording
    ↓
3D Keypoint Annotation ←──── RED (multi-camera, real-time playback)
    ↓
Training Data Export ←──── RED (JARVIS/COCO/DLC/YOLO)
    ↓
Model Training (JARVIS-HybridNet, DLC, SLEAP, Lightning Pose)
    ↓
3D Pose Prediction
    ↓
Behavioral Analysis (Keypoint-MoSeq, CEBRA, B-SOiD)
    ↓
Scientific Discovery (neural correlates, disease models, drug effects)
```

Today, RED covers the top three steps well and the fourth partially. The roadmap extends RED's reach downward through the pipeline: importing predictions for active learning (Phase A), supporting multi-animal workflows (Phase B), integrating training and behavioral analysis (Phase C), and eventually enabling real-time closed-loop experiments (Phase C3).

The field is moving toward **foundation models** (SuperAnimal, ViTPose++) that require minimal labeled data, **self-supervised methods** (Lightning Pose, SelfPose3d) that learn from unlabeled multi-camera video, and **behavioral embedding** (CEBRA, Keypoint-MoSeq) that extracts scientific meaning from 3D poses. RED's unique position — owning the calibration, the multi-camera video, and the annotation quality — makes it the natural hub for this emerging ecosystem.

The vision: a researcher records behavior with a calibrated multi-camera rig, opens RED, imports SuperAnimal draft predictions, corrects a few dozen frames, exports to JARVIS-HybridNet for fine-tuning, loads predictions back for quality inspection, exports 3D poses to Keypoint-MoSeq for behavioral syllable discovery, and correlates with neural recordings — all from a single application that plays synchronized high-speed video in real time with GPU-accelerated everything.

---

## References

### Multi-Camera Calibration
- multiview_calib (CVLAB EPFL): https://github.com/cvlab-epfl/multiview_calib
- Anipose (Karashchuk et al., Cell Reports 2021): https://anipose.readthedocs.io/
- mrcal (NASA JPL): https://mrcal.secretsauce.net/

### Pose Estimation
- DeepLabCut (Mathis et al., Nature Neuroscience 2018): https://github.com/DeepLabCut/DeepLabCut
- SLEAP (Pereira et al., Nature Methods 2022): https://sleap.ai/
- DANNCE (Dunn et al., Nature Methods 2021): https://github.com/spoonsso/dannce
- JARVIS-HybridNet: https://github.com/JARVIS-MoCap/JARVIS-HybridNet
- Lightning Pose (Biderman et al., Nature Methods 2024): https://github.com/paninski-lab/lightning-pose

### Foundation Models
- SuperAnimal (Ye et al., Nature Communications 2024)
- ViTPose++ (Xu et al., TPAMI 2023): https://github.com/ViTAE-Transformer/ViTPose
- ADPT (eLife 2025): https://elifesciences.org/articles/95709
- STEP (arXiv 2025): https://arxiv.org/abs/2503.13344

### Multi-Animal 3D
- s-DANNCE (Cell 2025): https://github.com/tqxli/sdannce
- SBeA (Nature Machine Intelligence 2024): https://github.com/YNCris/SBeA_release
- 3D-MuPPET (IJCV 2024): https://github.com/alexhang212/3D-MuPPET
- PAIR-R24M (Marshall et al., NeurIPS 2021)
- vmTracking (PLOS Biology 2025)

### Behavioral Analysis
- Keypoint-MoSeq (Nature Methods 2024)
- CEBRA (Nature 2023): https://cebra.ai/
- B-SOiD: https://github.com/YttriLab/B-SOID

### Annotation Efficiency
- PosePAL / CoTracker3 (arXiv 2025): https://github.com/Zhuoyang-Pan/PosePAL
- Active Learning for Multi-View 3D Pose (Gong et al., arXiv 2021)

### Real-Time and Closed-Loop
- AnimalRTPose (Neural Networks 2025)
- DeepLabCut-Live (eLife 2021)
- BlueBerry wireless optogenetics (bioRxiv 2025)
