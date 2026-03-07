# RED UI Style Guide

Reference document for all UI decisions in RED. Consult this before adding or
modifying any window, panel, dialog, or menu.

## 1. Application Architecture

### Two-Level Navigation

**Landing Page** (no project loaded):
- Recent .redproj files (calibration or annotation — same file type)
- "Create Annotation Project" and "Create Calibration Project" buttons
- Clean and simple. No clutter.

**Project Workspace** (project loaded):
- Navigator panel (left), camera viewports (center), tool panels (docked around viewports)
- Panel configuration depends on project content (calibration data, annotations, predictions)
- Layout persists per-project via ImGui .ini

### Unified .redproj Format

All projects use a single .redproj file type (like .blend in Blender). A project may
contain calibration data, annotation data, or both. The UI adapts based on what the
project contains — not what "type" it was created as.

---

## 2. Panel Classification

Every UI surface in RED falls into one of these categories:

| Category | Behavior | Docking | Close Button | Examples |
|----------|----------|---------|-------------|----------|
| **Anchor** | Always visible when project is open | Fixed (left sidebar) | No | Navigator |
| **Viewport** | Displays media, user-arrangeable | Yes (center grid) | No | Camera views |
| **Workspace panel** | Context-dependent tools | Yes (dock anywhere) | Optional | Calibration, Labeling, Keypoints, Frames, Predictions |
| **Tool window** | On-demand, transient | Floating | Yes | JARVIS Export, YOLO Export, Skeleton Creator, Spreadsheet |
| **Dialog** | Temporary form, opens and closes | Floating, no dock | Yes | Create Project, Create Calibration |
| **Modal popup** | Blocks interaction until dismissed | Centered overlay | N/A | Error messages, destructive confirmations |
| **File picker** | OS-style file selection | Modal | N/A | ImGuiFileDialog calls |

### Rules

- **Anchor panels** never float, never close. They are the persistent navigation hub.
- **Workspace panels** start docked in the default layout. Users can undock, rearrange,
  or merge them into docking tabs. They appear/disappear based on project content.
- **Tool windows** are floating by default. They are opened from menus and closed
  when done. Do not add them to the default dock layout.
- **Dialogs** are temporary. They collect input and disappear. Never dock them.
- **Modal popups** are for destructive or irreversible actions ONLY. Do not use modals
  for settings or configuration. No default button on destructive modals unless
  undo/recovery exists.

---

## 3. Named Panels

### Current

| Panel Name | Category | Purpose |
|------------|----------|---------|
| **Navigator** | Anchor | Menu bar, project info, playback controls, buffer settings |
| **Calibration** | Workspace | Calibration pipeline controls (ChArUco detect, intrinsics, BA) |
| **Labeling Tool** | Workspace | Keypoint annotation, triangulation, save controls |
| **Keypoints** | Workspace | Keypoint list, bounding box classes, color assignment |
| **Frames in the buffer** | Workspace | Frame buffer browser, frame selection |
| **Help Menu** | Workspace | Keyboard shortcuts reference |
| **JARVIS Export Tool** | Tool window | JARVIS/COCO format export |
| **YOLO Export Tool** | Tool window | YOLO format export |
| **Skeleton Creator** | Tool window | Interactive skeleton graph editor |
| **Spreadsheet** | Tool window | CSV/TSV data viewer |
| **Reprojection** | Tool window | Reprojection error visualization |

### Planned

| Panel Name | Category | Purpose |
|------------|----------|---------|
| **Dashboard** | Workspace | Annotation progress tracking (see Section 8) |
| **Predictions** | Workspace | Loaded JARVIS predictions, confidence visualization |
| **Quality** | Workspace | Per-keypoint confidence plots, error histograms |
| **Rig Setup** | Workspace | World frame, robot bases, camera extrinsics (future) |

### Naming Conventions

- Panel names are **1-2 words**, title case, no "Tool" or "Window" suffix
  (exception: existing "Labeling Tool" keeps its name for continuity).
- Menu items that open tool windows use the exact panel name.
- Use **nouns**, not verbs (e.g., "Calibration" not "Calibrate",
  "Predictions" not "View Predictions").

---

## 4. Layout and Docking

### Default Layout Structure

```
┌──────────────────────────────────────────────────────┐
│  Navigator  │          Camera View 1  │  Camera View 2│
│  (anchor)   │          (viewport)     │  (viewport)   │
│             ├─────────────────────────┼───────────────┤
│  Calibration│          Camera View 3  │  Camera View 4│
│  (dock tab) │          (viewport)     │  (viewport)   │
├─────────────┤                         │               │
│  Frames     │  Labeling Tool          │               │
│             │                         │               │
├─────────────┴─────────────────────────┴───────────────┤
│  Keypoints                                             │
└────────────────────────────────────────────────────────┘
```

- **Left column**: Navigator (anchor), workspace panels docked below or as tabs
- **Center/right**: Camera viewports in a 2x2 grid (user can resize, add more)
- **Bottom**: Keypoints or other data panels

### Docking Tabs vs BeginTabBar

Use **ImGui docking tabs** (automatic when windows share a dock node) for:
- Top-level panels that users should freely rearrange
- Navigator + Calibration + Help (all dock into the same node as tabs)

Use **BeginTabBar / BeginTabItem** (hardcoded) for:
- Fixed subcategories WITHIN a single panel (e.g., tabs inside a properties inspector)
- Never for top-level windows

### .ini Persistence

- **Per-project**: Each .redproj folder has `imgui_layout.ini`
- **Factory default**: `default_imgui_layout.ini` shipped with the app
- **New projects** get a copy of the factory default
- **"Reset Layout"** option in Settings menu (copies factory default, reloads)
- **Every new panel** must be added to `default_imgui_layout.ini` with a DockId

### Adding a New Panel Checklist

1. Choose its category (anchor / workspace / tool / dialog / modal)
2. Add `ImGui::Begin("Panel Name", ...)` with appropriate flags
3. If workspace panel: add entry to `default_imgui_layout.ini` with a DockId
4. If tool window: no ini entry needed (floats by default)
5. Add show/hide toggle to appropriate menu (View menu for workspace panels,
   Tools menu for tool windows)
6. Update this style guide's panel table

---

## 5. Menu Organization

The Navigator's menu bar is the primary entry point for all actions:

| Menu | Contents |
|------|----------|
| **File** | Open Video(s), Open Images, Create Project, Load Project |
| **Annotate** | Create Annotation Project, Load Annotation Project |
| **Calibrate** | Create Calibration Project, Load Calibration Project |
| **Tools** | Skeleton Creator, YOLO Export, JARVIS Export, Spreadsheet, Reprojection |
| **Settings** | Default Project Root, Default Media Root, Reset Layout |

### Rules

- **File** menu: media and project I/O only
- **Annotate / Calibrate**: project-type-specific actions
- **Tools**: opens tool windows (floating, transient)
- **Settings**: app-wide preferences
- Future: **View** menu for toggling workspace panel visibility

---

## 6. Dialog and Form Patterns

### Creation Dialogs

Used for: Create Project, Create Annotation, Create Calibration

- Floating window, `ImGuiWindowFlags_NoCollapse`
- Close button (`&show_flag`)
- **3-column table** layout: Label | Input Field | Action Button
- Path fields with "Browse..." button that opens ImGuiFileDialog
- "Create" button at the bottom, disabled until required fields are filled
- Error text in red above the Create button

### File Dialogs

- Always modal (`ImGuiFileDialogFlags_Modal`)
- Use descriptive dialog IDs: `"ChooseCalibRootDir"`, `"LoadAnnotProject"`
- Filter by relevant extensions

### Settings and Properties

- Non-modal. Changes take effect immediately.
- Use collapsible headers (`ImGui::CollapsingHeader`) for grouping.
- Sliders for continuous values, checkboxes for booleans, dropdowns for 3+ options.

---

## 7. Content Patterns

### Progress Indicators

For background operations (calibration pipeline, export, future training):
- Show progress bar with fraction text: `"Phase 1: 520/1328 images (62 img/s)"`
- Disable controls while running
- Show elapsed time and ETA when possible
- Status text below the progress bar

### Status Text

- Normal status: default text color
- Errors: `ImVec4(1.0f, 0.3f, 0.3f, 1.0f)` (red)
- Success/completion: `ImVec4(0.3f, 1.0f, 0.3f, 1.0f)` (green)
- Place status text at the bottom of its section, after a `Separator()`

### Collapsible Sections

- Use `CollapsingHeader` with `ImGuiTreeNodeFlags_DefaultOpen` for primary sections
- Use `CollapsingHeader` without DefaultOpen for advanced/secondary sections
- `ImGui::Indent()` / `Unindent()` inside each section for visual grouping

### Tables

- **Form tables**: 2-3 columns (Label | Control | Action), `SizingStretchProp`
- **Data tables**: full `ImGuiTableFlags_Borders | Sortable | Resizable`
- Label column: fixed width (~170px), text with `AlignTextToFramePadding()`

---

## 8. Dashboard (Planned)

The Dashboard tracks annotation projects through the active learning cycle:

### Metrics to Display

| Metric | Source | Visualization |
|--------|--------|---------------|
| Frames labeled | Annotation project CSV | Count + progress bar |
| Annotator (per frame) | Annotation project metadata | Name tag per frame |
| Model loss | JARVIS training log | Line chart over epochs |
| Mean prediction confidence | JARVIS prediction output | Scalar + trend |
| Per-keypoint confidence | JARVIS prediction output | Bar chart or heatmap |
| Flagged frames | User-marked during inspection | Count + list |

### Active Learning Cycle

```
Annotate ──→ Export ──→ Train ──→ Load Predictions ──→ Inspect
    ^                                                      │
    └──────────── Fix flagged frames ←─────────────────────┘
```

The Dashboard should help users understand where they are in this cycle and what
to do next. Build up gradually — start with frame counts and annotation progress,
add prediction metrics as JARVIS integration matures.

---

## 9. Future: Coordinate Frames and Rig Setup

When RED adds robot workspace calibration, coordinate frames become first-class objects:

- **World frame**: the rig's origin (defined during extrinsic calibration)
- **Camera frames**: one per camera (from calibration)
- **Robot base frames**: 6-DoF pose of each UR robot base in world coordinates
- **Object frames**: detected object poses (ArUco markers, etc.)

The **Rig Setup** panel will show a tree of these frames, similar to RoboDK's
Station Tree. Selecting a frame shows its 6-DoF pose in a properties sub-panel.

Terminology (following robotics conventions):
- "Frame" for a named coordinate system
- "Pose" for a 6-DoF position + orientation
- "Extrinsics" for camera-to-world transforms
- "Intrinsics" for camera internal parameters

---

## 10. Build Gradually

Do not implement features before they are needed. This guide describes the target
architecture. When adding new functionality:

1. Check this guide for the correct panel category and naming
2. Implement the minimum viable version
3. Use the existing patterns (collapsible sections, 3-column tables, progress bars)
4. Update this guide if new patterns emerge
5. Keep the landing page simple — complexity should live in the project workspace
