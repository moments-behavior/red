# RED UI Overhaul Plan

Phased refactoring plan for RED's UI architecture, informed by ImHex patterns
and the project's growth trajectory. Each phase is independent and incremental.

See `UI_STYLE_GUIDE.md` for naming conventions, panel categories, and content patterns.

## Current State (updated 2026-03-07)

- `src/red.cpp`: **3,779 LOC** (down from 6,450). Main loop + 4 inline windows.
- `src/gui/`: **5,453 LOC** across 15 files. 10 windows extracted as state struct + draw function.
- Infrastructure: DeferredQueue, drawPanel(), PopupStack, ToastQueue, ProjectHandlerRegistry
- Architectural style: structs + free functions (no class hierarchies)
- Algorithms (red_math.h, aruco_detect.h, calibration_pipeline.h) well-isolated
- No circular dependencies

### What remains inline in red.cpp

| Section | Lines | LOC | Priority |
|---------|-------|-----|----------|
| Camera viewport loop | 1423-2959 | **1,537** | High — largest section |
| Labeling Tool window | 3076-3534 | **458** | High — straightforward extraction |
| Navigator window | 593-883 | **291** | Medium — includes menu bar |
| File dialog handlers | 909-1065 | **157** | Medium — 9 dialog blocks |
| Frames in Buffer window | 1110-1156 | **47** | Low — small |
| Laser detection viz | 1290-1420 | **130** | Low — macOS-only, tied to render |

### Duplicated patterns in red.cpp

| Pattern | Occurrences | Fix |
|---------|-------------|-----|
| "Ensure keypoints exist" (malloc + allocate + insert) | 7 | Helper function |
| Coordinate clamping to frame bounds | ~30 | `clamp_to_frame()` helper |
| Project load sequence (load_json + setup + on_project_loaded) | 3 | `load_project_from_path()` helper |
| Static variables as hidden state (bbox drag, OBB state) | 15 | Move into state structs |

---

## Phase 1: Extract Remaining Windows _(in progress)_

**Goal:** Get red.cpp under 2,000 lines by extracting the 4 remaining inline windows.
Same state-struct + draw-function pattern used by all 10 existing extracted windows.

### 1a. Camera Viewport — `gui/camera_viewport.h` _(highest impact)_

The 1,537-line camera loop contains 5 distinct concerns that can be separated:

| New function | What it does | ~LOC |
|-------------|-------------|------|
| `UploadCameraTexture()` | Platform-specific texture upload (Metal/CUDA/PBO) | 150 |
| `HandleKeypointInteraction()` | W/A/D/E/Q key labeling, keypoint creation | 90 |
| `HandleBboxInteraction()` | Shift-drag creation, MyDragRect, deletion, bbox keypoints | 600 |
| `HandleObbInteraction()` | 3-point OBB construction, drag, deletion | 490 |
| `DrawTransportControls()` | Rewind/step/play/pause/FF buttons, frame slider | 110 |

**State to bundle:**
```cpp
struct BboxInteractionState {
    std::vector<std::string> class_names = {"Class_1"};
    std::vector<ImVec4> class_colors;
    int current_class = 0;
    int current_id = 0;
    bool show_ids = false;
    int hovered_cam = -1, hovered_idx = -1;
    float hovered_confidence = 0.0f;
    int hovered_class = -1, hovered_id = -1;
};

struct ObbInteractionState {
    int hovered_cam = -1, hovered_idx = -1;
    float hovered_confidence = 0.0f;
    int hovered_class = -1, hovered_id = -1;
    // Move the 6 static variables (lines 2521-2527) here
};
```

The per-camera loop stays in red.cpp but becomes ~50 lines calling these functions.

### 1b. Labeling Tool — `gui/labeling_tool_window.h`

458 lines following the standard extraction pattern:
```cpp
struct LabelingToolState { bool show = true; /* ... */ };
inline void DrawLabelingToolWindow(LabelingToolState &state, ...);
```

Contains: save/load label buttons, frame navigation, YOLO model selection,
YOLO inference trigger, label statistics.

### 1c. Navigator — `gui/navigator_window.h`

291 lines. The menu bar + playback controls + display controls.
```cpp
struct NavigatorState { /* brightness, contrast, playback settings */ };
inline void DrawNavigatorWindow(NavigatorState &state, ...);
```

### 1d. Helper functions — `src/app_helpers.h`

Eliminate duplicated patterns:
```cpp
// Ensure keypoints exist for current frame (replaces 7 copies)
inline KeyPoints *ensure_keypoints(std::map<u32, KeyPoints *> &map,
                                   int frame, RenderScene *scene,
                                   SkeletonContext *skeleton);

// Clamp point to camera frame bounds (replaces ~30 copies)
inline ImVec2 clamp_to_frame(double x, double y, int width, int height);

// Load a .redproj and run on_project_loaded (replaces 3 copies)
inline void load_project_from_path(const std::filesystem::path &path,
                                   ProjectManager &pm, ...);
```

**Target:** red.cpp drops from 3,779 to ~1,500-2,000 lines after Phase 1.

---

## Phase 2: AppContext + Panel Registry

**Goal:** Reduce parameter passing and centralize draw dispatch.
No class hierarchy — stays struct + free function.

### 2a. AppContext struct — `src/app_context.h`

Many draw functions take 8-15 parameters. Bundle shared references:
```cpp
struct AppContext {
    ProjectManager &pm;
    PlaybackState &ps;
    RenderScene *scene;
    DecoderContext *dc_context;
    PopupStack &popups;
    ToastQueue &toasts;
    DeferredQueue &deferred;
    SkeletonContext &skeleton;
    std::map<u32, KeyPoints *> &keypoints_map;
    // ... other shared state
};
```

Draw functions simplify from:
```cpp
DrawCalibrationToolWindow(state, pm, ps, scene, dc_context, settings, ...);
// to:
DrawCalibrationToolWindow(state, ctx);
```

### 2b. Panel Registry — `src/gui/panel_registry.h`

Flat registry of draw functions (no virtual dispatch):
```cpp
struct PanelEntry {
    std::string name;
    bool *show;                    // pointer to state's show flag
    std::function<void()> draw;    // bound draw call
    bool show_in_menu = true;      // appear in View menu?
};

struct PanelRegistry {
    std::vector<PanelEntry> panels;
    void add(PanelEntry entry);
    void drawAll();                // replaces 15 manual draw calls in main loop
    void drawViewMenu();           // auto-generates View > toggle panel items
};
```

Registration at startup:
```cpp
panels.add({"Help", &show_help, [&]{ DrawHelpWindow(show_help); }});
panels.add({"Calibration Tool", &calib_state.show,
    [&]{ DrawCalibrationToolWindow(calib_state, ctx); }});
```

Main loop becomes:
```cpp
panels.drawAll();
drawToasts(toasts);
drawPopups(popups);
```

Adding a new window = one .h file + one `panels.add()` call. No editing menus or the main loop.

---

## Phase 3: State Bundling

**Goal:** Reduce the ~60 loose variables in main() to a handful of state structs.

| Struct | Variables absorbed | ~Count |
|--------|-------------------|--------|
| `BboxInteractionState` | class_names, colors, current_class/id, hovered_bbox_* | 15 |
| `ObbInteractionState` | hovered_obb_*, drag statics | 10 |
| `DisplayState` | brightness, contrast, pivot_midgray | 3 |
| `YoloState` | detection, model_path, auto_labeling, processed_frames, thresholds | 10 |
| `AppLayout` | project_ini_path, main_loop_running, switch_ini lambda | 5 |

Each struct is defined near where it's used (in the relevant gui/ header).
The `AppContext` struct (Phase 2a) holds references to these.

---

## Phase 4: Enhanced Settings

**Goal:** Tunable parameters are configurable and persistent.

Extend existing `user_settings.h` with auto-UI generation:
```cpp
struct SettingDef {
    std::string category;   // "Display", "Calibration", "Export"
    std::string key;
    std::string label;      // UI display name
    enum Type { Bool, Int, Float, String, Path } type;
};

inline void drawSettingsPanel(UserSettings &settings,
                              const std::vector<SettingDef> &defs);
```

**Categories:**
- **Display:** contrast/brightness defaults, playback speed, show grid
- **Calibration:** ArUco dictionary, threshold window sizes, BA iterations, RANSAC tolerance
- **Export:** JARVIS/YOLO default parameters
- **Paths:** default project root, default media root (already exists)

Settings are defined where they're used (each subsystem registers its settings),
and the Settings panel auto-generates the UI from the registry.

---

## Phase 5: Library Separation _(deferred)_

**Goal:** Faster incremental builds, reusable components.

| Library | Contents | Dependencies |
|---------|----------|-------------|
| `libred_core` | types, camera, project, skeleton, red_math, opencv_yaml_io | Eigen |
| `libred_decode` | FFmpegDemuxer, vt_async_decoder, NvDecoder, decoder | FFmpeg, VT/CUDA |
| `libred_calib` | calibration_pipeline, aruco_detect, intrinsic_calibration, aruco_metal | Eigen, Ceres, Metal |
| `red` (app) | UI views, main loop | links above + ImGui, GLFW |

**Status:** Defer until build times become painful. Currently full rebuild is ~30s.

---

## Future Features (planned panels from UI Style Guide)

These are new features, not refactoring. Each requires its own design work.

| Panel | Category | Purpose | Prerequisite |
|-------|----------|---------|-------------|
| Landing Page | — | Recent projects, create new | Phase 2 (panel registry) |
| Dashboard | Workspace | Active learning cycle progress metrics | JARVIS prediction loading |
| Predictions | Workspace | Loaded JARVIS predictions with confidence viz | JARVIS import feature |
| Quality | Workspace | Per-keypoint confidence plots, error histograms | Predictions panel |
| Rig Setup | Workspace | World frame, robot bases, camera extrinsics tree | Robot integration |
| View Menu | — | Toggle workspace panel visibility | Phase 2b (panel registry) |

---

## Guiding Principles

1. **Structs + free functions** — no class hierarchies, no virtual dispatch.
   A class is acceptable when it owns a mutex (DeferredQueue).
2. **Build gradually** — each phase delivers value independently.
3. **No behavior changes during refactoring** — app works the same, code is just organized better.
4. **Don't over-engineer** — RED is not ImHex. No dynamic plugins, no localization,
   no event system. Direct function calls are fine at this scale.
5. **Algorithms stay clean** — red_math.h, aruco_detect.h, calibration_pipeline.h
   are already well-isolated. Don't touch them during UI refactoring.
6. **Test after each change** — `cmake --build release` all targets + `test_gui` + `test_pipeline_run` + manual smoke test.

---

## Progress Log

| Date | What Changed |
|------|-------------|
| 2026-03-07 | Created UI_STYLE_GUIDE.md, renamed File Browser -> Navigator |
| 2026-03-07 | Split gui.h into 6 domain files + forwarding header |
| 2026-03-07 | Extracted 6 windows from red.cpp (Help, Skeleton, YOLO, JARVIS, Annotation, Calibration) |
| 2026-03-07 | Added infrastructure: DeferredQueue, drawPanel(), PopupStack, ToastQueue, ProjectHandlerRegistry |
| 2026-03-07 | Migrated laser_load_pending to DeferredQueue, show_error to PopupStack |
| 2026-03-07 | Consolidated duplicated post-load sequence into on_project_loaded() |
| 2026-03-07 | Converted 5 windows to use drawPanel() (Help, JARVIS, YOLO, Skeleton, Annotation) |
| 2026-03-07 | Added toast notification on Ctrl+S save |
| 2026-03-07 | Updated this plan to reflect current state and revised architecture |
