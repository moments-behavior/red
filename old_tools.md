# Removed Tools Documentation

Last commit with all tools present: `4dc58c4` (branch `rob_ui_overhaul`)

## Tools Removed

### 1. Bounding Box (AABB) Labeling
- **Purpose:** Multi-class 2D bounding box annotation with triangulation support
- **Files:**
  - `src/gui/gui_bbox.h` (472 LOC) — MyDragRect, bbox triangulation, bbox keypoint plotting
  - ~640 lines in `src/red.cpp` — bbox creation via Shift+drag, hover detection, keyboard shortcuts (F/O delete, Z/X class switch, C/V id switch, N new class)
  - ~440 lines in `src/keypoints_table.h` — bbox class management UI, per-bbox keypoint table
- **Structs:** `BoundingBox`, `RectState` enum (in skeleton.h)
- **Globals:** `user_active_bbox_idx` (global.h)

### 2. Oriented Bounding Box (OBB) Labeling
- **Purpose:** 3-click rotated bounding box annotation for YOLO-OBB export
- **Files:**
  - `src/gui/gui_obb.h` (381 LOC) — OBB drawing, hit testing, property calculation
  - ~510 lines in `src/red.cpp` — 3-point OBB construction, drag handles, hover/keyboard interaction
- **Structs:** `OrientedBoundingBox`, `OBBState` enum (in skeleton.h)

### 3. YOLO Inference (Linux-only)
- **Purpose:** LibTorch-based auto-labeling via YOLO object detection
- **Files:**
  - `src/yolo_torch.h` (65 LOC) — YoloBBox/YoloPrediction structs, runYoloInference decl
  - `src/yolo_torch.cpp` (306 LOC) — LibTorch inference, NMS, bbox conversion
  - ~130 lines in `src/red.cpp` — auto-YOLO detection block (frame stepping + inference)
  - ~230 lines in `src/red.cpp` — YOLO UI in Labeling Tool (model selection, sliders, run button)
- **Globals:** `g_yolo_class_map`, `g_reverse_yolo_class_map`, `next_class_id`, `confidence_threshold`, `iou_threshold`, `yolo_bboxes`, `yolo_predictions`, `yolo_model_path`, `yolo_drag_boxes`, `yolo_active_bbox_idx` (global.h/cpp)

### 4. YOLO Export Tool
- **Purpose:** Export bbox/OBB/pose annotations as YOLO training datasets
- **Files:**
  - `src/gui/yolo_export_window.h` (396 LOC) — YoloExportState + DrawYoloExportWindow
  - `src/yolo_export.h` (147 LOC) — export function declarations
  - `src/yolo_export.cpp` (1745 LOC) — YOLO format conversion and file writing
- **CMake:** Excluded from macOS build via `list(REMOVE_ITEM)`

### 5. Skeleton Creator
- **Purpose:** Visual node+edge editor for creating custom skeleton JSON files
- **Files:**
  - `src/gui/skeleton_creator_window.h` (509 LOC) — SkeletonCreatorState + DrawSkeletonCreatorWindow
- **Structs:** `SkeletonCreatorNode`, `SkeletonCreatorEdge` (in skeleton.h) — kept for now since they're harmless

### 6. Spreadsheet / LiveTable
- **Purpose:** Embedded CSV editor with frame-seek on Shift+Click
- **Files:**
  - `src/live_table.h` (546 LOC) — LiveTable struct + DrawLiveTable
- **Note:** Consider reimplementing as a lightweight labeled-frame browser later.

### 7. Reprojection Error Tool
- **Purpose:** Per-keypoint/camera error bar charts for calibration validation
- **Files:**
  - `src/reprojection_tool.h` (314 LOC) — ReprojectionTool struct + DrawReprojectionWindow
- **Note:** Reimplement inside Calibration Tool later. The `reprojection()` function in gui_keypoints.h is the triangulation function and was NOT removed.

## Skeleton Presets Removed
- `Rat4Box` — rat 4-keypoint + bounding box
- `Rat4Box3Ball` — rat 4-keypoint + bbox + 3 ball keypoints
- `BoundingBox` (SP_BBOX) — bbox-only mode
- `OrientedBoundingBox` (SP_OBB) — OBB-only mode
- `Simple BBox+Skeleton` (SP_SIMPLE_BBOX_SKELETON) — bbox + 2-point skeleton

## Save/Load Functions Removed
- `save_bboxes()` — bbox CSV export
- `save_bbox_keypoints()` — per-bbox keypoint CSV export
- `save_obb()` — OBB CSV export
- `load_bboxes()` — bbox CSV import
- `load_bbox_keypoints()` — per-bbox keypoint CSV import
- `load_obb()` — OBB CSV import
- `load_keypoints_depreciated()` — old format keypoint loader
