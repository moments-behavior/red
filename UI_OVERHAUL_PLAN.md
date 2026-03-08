# RED UI Overhaul Plan

Phased refactoring plan for RED's UI architecture, informed by ImHex patterns
and the project's growth trajectory. Each phase is independent and incremental.

See `UI_STYLE_GUIDE.md` for naming conventions, panel categories, and content patterns.

## Current State (updated 2026-03-07)

- `src/red.cpp`: **1,818 LOC** (down from 3,779 after tool removal + extraction).
- `src/gui/`: extracted GUI modules. 7 windows as state struct + draw function.
- Infrastructure: DeferredQueue, drawPanel(), PopupStack, ToastQueue, ProjectHandlerRegistry
- Architectural style: structs + free functions (no class hierarchies)
- Algorithms (red_math.h, aruco_detect.h, calibration_pipeline.h) well-isolated
- No circular dependencies
- Unused tools removed: bbox, OBB, YOLO, skeleton creator, spreadsheet, reprojection
  (documented in old_tools.md)

### What remains inline in red.cpp

| Section | ~LOC | Notes |
|---------|------|-------|
| Navigator window (menu bar + playback + display) | 279 | Tightly coupled to main() locals |
| Camera viewport loop (upload + plot + labeling keys + transport) | 487 | Platform-specific, deep main loop integration |
| File dialog handlers (9 dialogs) | 148 | Opened by Navigator, results modify main() locals |
| Laser detection viz (macOS) | 132 | GPU dispatch, tied to render pipeline |
| Frame buffer navigation | 86 | Small, self-contained |
| Initialization + cleanup + playback sync | ~686 | Application skeleton |

---

## Phase 1: Extract Remaining Windows _(complete)_

**Result:** red.cpp dropped from 3,779 to **1,818 lines**.

### What was done

1. **Tool removal** — Deleted 11 files (~5,600 LOC) of unused tools: bbox labeling,
   OBB labeling, YOLO inference/export, skeleton creator, spreadsheet, reprojection.
   Documented in `old_tools.md`.

2. **Skeleton/save simplification** — Removed BoundingBox, OrientedBoundingBox, RectState,
   OBBState structs. Simplified allocate/free/copy_keypoints, load_keypoints, save logic.
   skeleton.h: 161→81, skeleton.cpp: 899→405, gui_save_load.h: 1301→352,
   keypoints_table.h: 508→127.

3. **Labeling Tool extraction** — `gui/labeling_tool_window.h` with LabelingToolState +
   DrawLabelingToolWindow(). Includes file dialog handlers for keypoints folder and
   load-from-selected.

### What was NOT extracted (and why)

- **Navigator** — Tightly coupled to main() locals (pm, ps, scene, dc_context,
  calib_state, annot_state, user_settings). Extraction would require a massive
  parameter list or context struct (see Phase 2a) without reducing complexity.
- **Camera viewport** — Platform-specific frame upload intertwined with ImPlot
  context and labeling keys. Better to extract after AppContext (Phase 2a).
- **File dialogs** — Opened by Navigator menu, results modify main() locals.
  Will naturally move when Navigator is extracted.

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

**Goal:** Reduce the remaining loose variables in main() to a handful of state structs.

After tool removal, main() has ~25 local variables (down from ~60).
The biggest remaining candidates:

| Struct | Variables absorbed | ~Count |
|--------|-------------------|--------|
| `DisplayState` | brightness, contrast, pivot_midgray | 3 |
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
| 2026-03-07 | Removed 11 unused tool files (~5,600 LOC): bbox, OBB, YOLO, skeleton creator, spreadsheet, reprojection |
| 2026-03-07 | Simplified skeleton.h/cpp (removed BoundingBox, OBB structs), gui_save_load.h, keypoints_table.h |
| 2026-03-07 | Extracted Labeling Tool → gui/labeling_tool_window.h |
| 2026-03-07 | red.cpp: 3,779 → 1,818 LOC. Phase 1 complete. |
