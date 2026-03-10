# Annotation Data Model Refactor: Final Implementation Plan

**Branch:** `rob_annotation_v2` (off `rob_ui_overhaul`)
**Status:** Architecture finalized, ready for implementation
**Date:** 2026-03-10

---

## Table of Contents

1. [Design Summary](#1-design-summary)
2. [Final Struct Definitions](#2-final-struct-definitions)
3. [Final CSV Format](#3-final-csv-format)
4. [Phase A: New Data Model + Save/Load + Converter](#4-phase-a)
5. [Phase B: Switch All Consumers](#5-phase-b)
6. [Phase C: Delete Old Code](#6-phase-c)
7. [Test Plan](#7-test-plan)
8. [Open Questions](#8-open-questions)
9. [File Inventory](#9-file-inventory)

---

## 1. Design Summary

### What Changes

The current annotation system has a dual-model problem: the old `KeyPoints*`
raw-pointer system coexists with the newer `AnnotationMap` via a fragile bridge
(`refresh_keypoints_in_amap`, `migrate_keypoints_map`). The refactor:

1. Replaces `KeyPoints*` / `KeyPoints2D` / `KeyPoints3D` raw-pointer bags with
   a clean struct-of-arrays-per-keypoint model (`Keypoint2D`, `Keypoint3D`).
2. Flattens the `InstanceAnnotation` indirection (single instance per frame,
   inline, not wrapped in a vector).
3. Moves cold data (bbox/OBB/mask) behind `unique_ptr<CameraExtras>`.
4. Eliminates `std::vector<bool>` everywhere.
5. Writes a new v2 CSV format with column headers, inline confidence, and source
   flags.
6. Removes all bridge/migration code between the two systems.
7. Removes the `confidence.csv` sidecar pattern from jarvis_import.h.

### What Does Not Change

- `std::map<u32, FrameAnnotation>` stays (pointer stability for ImPlot::DragPoint).
- `annotations.json` sidecar stays (for bbox/OBB/mask extended data).
- `AnnotationConfig` in `project.h` stays unchanged.
- Old `KeyPoints*` system will be fully deleted, not maintained.
- JARVIS export continues to read from CSV on disk (not from AnnotationMap
  in-memory) for the file-based path. The COCO/YOLO/DLC exporters continue
  reading from AnnotationMap.

### Backward Compatibility

None required. A one-time converter (`convert_v1_to_v2`) reads old-format CSVs
and writes v2 CSVs. The converter runs automatically when old-format files are
detected during load.

---

## 2. Final Struct Definitions

All structs live in `src/annotation.h`.

### 2.1 Core Keypoint Types

```cpp
// Sentinel for unlabeled coordinates (matches existing CSV convention)
static constexpr double UNLABELED = 1E7;

// Source of a keypoint label
enum class LabelSource : uint8_t {
    Manual    = 0,  // labeled by human in GUI
    Predicted = 1,  // from ML model (JARVIS, ONNX, etc.)
    Imported  = 2,  // from external tool (DLC, SLEAP, etc.)
};

struct Keypoint2D {
    double x = UNLABELED;
    double y = UNLABELED;
    bool   labeled = false;
    float  confidence = 0.0f;   // 0.0 = manual, >0 = model confidence
    LabelSource source = LabelSource::Manual;
};

struct Keypoint3D {
    double x = UNLABELED;
    double y = UNLABELED;
    double z = UNLABELED;
    bool   triangulated = false;
    float  confidence = 0.0f;
};
```

### 2.2 Cold Data (bbox/OBB/mask)

```cpp
struct CameraExtras {
    // Axis-aligned bounding box
    double bbox_x = 0, bbox_y = 0, bbox_w = 0, bbox_h = 0;
    bool has_bbox = false;

    // Oriented bounding box
    double obb_cx = 0, obb_cy = 0, obb_w = 0, obb_h = 0, obb_angle = 0;
    bool has_obb = false;

    // Segmentation mask polygons
    std::vector<std::vector<tuple_d>> mask_polygons;
    bool has_mask = false;
};
```

### 2.3 Per-Camera Annotation

```cpp
struct CameraAnnotation {
    std::vector<Keypoint2D> keypoints;  // [num_nodes]
    u32 active_id = 0;                  // UI state, not persisted

    // Cold data — null for 99%+ of frames
    std::unique_ptr<CameraExtras> extras;

    // Convenience accessors for bbox/obb/mask (create extras on demand)
    CameraExtras &get_extras() {
        if (!extras) extras = std::make_unique<CameraExtras>();
        return *extras;
    }
    bool has_bbox() const { return extras && extras->has_bbox; }
    bool has_obb()  const { return extras && extras->has_obb; }
    bool has_mask() const { return extras && extras->has_mask; }
};
```

### 2.4 Per-Frame Annotation

```cpp
struct FrameAnnotation {
    u32 frame_number = 0;
    int instance_id = 0;    // object identity (single-instance for now)
    int category_id = 0;    // class index

    // 3D keypoints (triangulated from multi-view)
    std::vector<Keypoint3D> kp3d;           // [num_nodes]

    // Per-camera 2D annotations
    std::vector<CameraAnnotation> cameras;  // [num_cameras]
};

using AnnotationMap = std::map<u32, FrameAnnotation>;
```

### 2.5 Key Differences from Current Code

| Current (annotation.h)              | New                                      |
|--------------------------------------|------------------------------------------|
| `Camera2D` with parallel vectors    | `CameraAnnotation` with `Keypoint2D` AoS |
| `std::vector<bool> kp_labeled`      | `Keypoint2D::labeled` (inline bool)      |
| `std::vector<float> kp_confidence`  | `Keypoint2D::confidence` (inline float)  |
| `std::vector<tuple_d> keypoints`    | `Keypoint2D::x, y` (inline doubles)     |
| `InstanceAnnotation` in a vector    | Fields inlined into `FrameAnnotation`    |
| `std::vector<bool> kp3d_triangulated` | `Keypoint3D::triangulated` (inline bool) |
| `std::vector<float> kp3d_confidence` | `Keypoint3D::confidence` (inline float)  |
| `std::vector<triple_d> kp3d`        | `Keypoint3D::x,y,z` (inline doubles)   |
| `bbox3d_center, bbox3d_size, has_bbox3d` | Removed (never written by UI)       |
| bbox/obb/mask inline in Camera2D    | Behind `unique_ptr<CameraExtras>`        |
| No source tracking                   | `LabelSource` enum per keypoint          |

---

## 3. Final CSV Format

### 3.1 v2 2D CSV (per camera)

```
#red_csv v2
#skeleton Rat4
frame,x0,y0,c0,s0,x1,y1,c1,s1,x2,y2,c2,s2
5995,1681.6,1596.5,,,1703.46,1658.13,0.92,P,,,,,
6001,100.5,200.3,,,150.2,300.4,0.87,P,200.1,400.6,,
```

**Rules:**
- Line 1: `#red_csv v2` (magic + version)
- Line 2: `#skeleton <name>`
- Line 3: column header row — `frame` then repeating `x<i>,y<i>,c<i>,s<i>` for
  `i` in `[0, num_nodes)`
- Data rows: frame number, then 4 columns per keypoint
- **Unlabeled keypoint:** all 4 cells empty (`,,,`) — no sentinel in file
- **Manual label:** x,y filled, c and s empty
- **Predicted label:** x,y filled, c = confidence float, s = `P`
- **Imported label:** x,y filled, c = confidence (or empty), s = `I`
- Coordinates are in ImPlot space (Y=0 at bottom, matching current convention)

### 3.2 v2 3D CSV

```
#red_csv v2
#skeleton Rat4
frame,x0,y0,z0,c0,x1,y1,z1,c1,x2,y2,z2,c2
5995,12.34,56.78,90.12,,23.45,67.89,01.23,0.95,,,,,
```

**Rules:**
- Same magic/skeleton header
- Column header: `frame` then `x<i>,y<i>,z<i>,c<i>` per keypoint
- Unlabeled (not triangulated): all 4 cells empty
- Triangulated: x,y,z filled, c = confidence (or empty)
- No source flag on 3D (always derived from 2D)

### 3.3 v1 Detection (for converter)

Old format is detected by absence of `#red_csv` magic line. First line is
skeleton name only (e.g. `Rat4`). Data rows: `frame,node_idx,x,y,node_idx,x,y,...`
with `1E7` sentinel for unlabeled.

### 3.4 Parser Design (strtod pointer-arithmetic)

```cpp
// Parses one comma-delimited double from `ptr`, advancing past the comma.
// Returns true if a value was read, false if the cell was empty.
inline bool parse_csv_double(const char *&ptr, double &out) {
    while (*ptr == ' ') ++ptr;           // skip whitespace
    if (*ptr == ',' || *ptr == '\n' || *ptr == '\0') {
        if (*ptr == ',') ++ptr;
        return false;                    // empty cell
    }
    char *end;
    out = strtod(ptr, &end);
    ptr = end;
    if (*ptr == ',') ++ptr;
    return true;
}

// Parses one comma-delimited char from `ptr`, advancing past the comma.
// Returns '\0' if empty.
inline char parse_csv_char(const char *&ptr) {
    while (*ptr == ' ') ++ptr;
    char ch = '\0';
    if (*ptr != ',' && *ptr != '\n' && *ptr != '\0') {
        ch = *ptr++;
    }
    if (*ptr == ',') ++ptr;
    return ch;
}
```

---

## 4. Phase A: New Data Model + Save/Load + Converter

### Goal
New structs compile, new CSV save/load works, old-format converter works.
Old `KeyPoints*` system still exists and is still used by all consumers.
The two systems coexist but do NOT bridge (no `refresh_keypoints_in_amap`).

### 4.1 File: `src/annotation.h` — Rewrite

**Order of operations:**
1. Rename current `annotation.h` to `annotation_v1.h` (temporary, deleted in Phase C).
2. Write new `annotation.h` with final structs from Section 2.
3. Include `annotation_v1.h` from `annotation.h` to keep bridge functions
   compiling during the transition.

**New contents of `annotation.h`:**

```cpp
#pragma once
#include "types.h"
#include "json.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>

// ── Forward compat: include old types for bridge during Phase A/B ──
// #include "annotation_v1.h"  // uncomment during transition, remove in Phase C

static constexpr double UNLABELED = 1E7;

enum class LabelSource : uint8_t {
    Manual    = 0,
    Predicted = 1,
    Imported  = 2,
};

struct Keypoint2D { ... };  // as in Section 2.1
struct Keypoint3D { ... };  // as in Section 2.1
struct CameraExtras { ... }; // as in Section 2.2
struct CameraAnnotation { ... }; // as in Section 2.3
struct FrameAnnotation { ... }; // as in Section 2.4
using AnnotationMap = std::map<u32, FrameAnnotation>;

// ── Helpers ──

inline FrameAnnotation make_frame(u32 frame, int num_nodes, int num_cameras,
                                   int instance_id = 0, int category_id = 0) {
    FrameAnnotation fa;
    fa.frame_number = frame;
    fa.instance_id = instance_id;
    fa.category_id = category_id;
    fa.kp3d.resize(num_nodes);
    fa.cameras.resize(num_cameras);
    for (auto &cam : fa.cameras)
        cam.keypoints.resize(num_nodes);
    return fa;
}

inline FrameAnnotation &get_or_create_frame(AnnotationMap &amap, u32 frame,
                                             int num_nodes, int num_cameras) {
    auto it = amap.find(frame);
    if (it != amap.end()) return it->second;
    auto [ins_it, _] = amap.emplace(frame, make_frame(frame, num_nodes, num_cameras));
    return ins_it->second;
}

inline bool frame_has_any_labels(const FrameAnnotation &fa) {
    for (const auto &cam : fa.cameras)
        for (const auto &kp : cam.keypoints)
            if (kp.labeled) return true;
    return false;
}

inline bool frame_is_complete(const FrameAnnotation &fa) {
    // All 2D keypoints labeled in all cameras, and all 3D triangulated
    for (const auto &cam : fa.cameras)
        for (const auto &kp : cam.keypoints)
            if (!kp.labeled) return false;
    for (const auto &kp3d : fa.kp3d)
        if (!kp3d.triangulated) return false;
    return true;
}
```

### 4.2 File: `src/annotation_csv.h` — NEW file (save/load/convert)

This is the only new file created. Contains all CSV I/O.

**Key function signatures:**

```cpp
#pragma once
#include "annotation.h"
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace AnnotationCSV {

// ── strtod parser primitives ──
inline bool parse_csv_double(const char *&ptr, double &out);
inline char parse_csv_char(const char *&ptr);

// ── v2 Save ──

// Save 2D keypoints for one camera to v2 CSV format.
// Writes to: <folder>/<camera_name>.csv
// Y-coordinates are stored in ImPlot space (no flip here; flip at export time).
inline bool save_2d_csv(const std::string &path,
                         const std::string &skeleton_name,
                         const AnnotationMap &amap,
                         int cam_idx,
                         int num_nodes);

// Save 3D keypoints to v2 CSV format.
// Writes to: <folder>/keypoints3d.csv
inline bool save_3d_csv(const std::string &path,
                         const std::string &skeleton_name,
                         const AnnotationMap &amap,
                         int num_nodes);

// Save all CSVs (3D + per-camera 2D) to a timestamped subfolder.
// Creates <root>/<YYYY_MM_DD_HH_MM_SS>/ and writes all files.
// Returns the created folder path, or empty string on error.
inline std::string save_all(const std::string &root_dir,
                             const std::string &skeleton_name,
                             const AnnotationMap &amap,
                             int num_cameras,
                             int num_nodes,
                             const std::vector<std::string> &camera_names,
                             std::string *error = nullptr);

// ── v2 Load ──

// Load a single 2D CSV (v2 format) into the AnnotationMap.
// Creates FrameAnnotation entries as needed.
inline bool load_2d_csv(const std::string &path,
                         AnnotationMap &amap,
                         int cam_idx,
                         int num_nodes,
                         int num_cameras,
                         std::string &error);

// Load 3D CSV (v2 format) into the AnnotationMap.
inline bool load_3d_csv(const std::string &path,
                         AnnotationMap &amap,
                         int num_nodes,
                         int num_cameras,
                         std::string &error);

// Load all CSVs from a folder (auto-detects v1 or v2).
// Returns 0 on success, 1 on error.
inline int load_all(const std::string &folder,
                     AnnotationMap &amap,
                     const std::string &skeleton_name,
                     int num_nodes,
                     int num_cameras,
                     const std::vector<std::string> &camera_names,
                     std::string &error);

// ── v1 Converter ──

// Detect if a folder contains v1-format CSVs (no #red_csv header).
inline bool is_v1_format(const std::string &folder,
                          const std::vector<std::string> &camera_names);

// Convert v1 CSVs in `src_folder` to v2 format, writing to `dst_folder`.
// Reads old format (skeleton name header, frame,node,x,y,... with 1E7 sentinel),
// produces new format (v2 headers, empty-cell unlabeled, inline confidence).
// Also converts confidence.csv sidecar if present.
inline bool convert_v1_to_v2(const std::string &src_folder,
                              const std::string &dst_folder,
                              const std::string &skeleton_name,
                              int num_nodes,
                              const std::vector<std::string> &camera_names,
                              std::string *error = nullptr);

// ── find_most_recent_labels (moved from gui_save_load.h) ──
inline int find_most_recent_labels(const std::string &root_dir,
                                    std::string &most_recent_folder,
                                    std::string &error);

} // namespace AnnotationCSV
```

**Implementation details for `save_2d_csv`:**

```cpp
inline bool save_2d_csv(const std::string &path,
                         const std::string &skeleton_name,
                         const AnnotationMap &amap,
                         int cam_idx,
                         int num_nodes) {
    std::ofstream f(path);
    if (!f) return false;

    // Header
    f << "#red_csv v2\n";
    f << "#skeleton " << skeleton_name << "\n";
    f << "frame";
    for (int k = 0; k < num_nodes; ++k)
        f << ",x" << k << ",y" << k << ",c" << k << ",s" << k;
    f << "\n";

    // Data rows
    for (const auto &[fnum, fa] : amap) {
        if (cam_idx >= (int)fa.cameras.size()) continue;
        const auto &cam = fa.cameras[cam_idx];

        f << fnum;
        for (int k = 0; k < num_nodes; ++k) {
            if (k < (int)cam.keypoints.size() && cam.keypoints[k].labeled) {
                const auto &kp = cam.keypoints[k];
                f << "," << kp.x << "," << kp.y << ",";
                // Confidence: omit if manual (0.0)
                if (kp.confidence > 0.0f)
                    f << kp.confidence;
                f << ",";
                // Source: omit if manual
                if (kp.source == LabelSource::Predicted) f << "P";
                else if (kp.source == LabelSource::Imported) f << "I";
            } else {
                f << ",,,,";  // 4 empty cells
            }
        }
        f << "\n";
    }
    return true;
}
```

**Implementation details for `load_2d_csv` (strtod parser):**

```cpp
inline bool load_2d_csv(const std::string &path,
                         AnnotationMap &amap,
                         int cam_idx,
                         int num_nodes,
                         int num_cameras,
                         std::string &error) {
    std::ifstream fin(path);
    if (!fin) { error = "Failed to open: " + path; return false; }

    std::string line;
    int line_num = 0;
    bool is_v2 = false;

    while (std::getline(fin, line)) {
        line_num++;
        if (line.empty()) continue;

        // Skip comment/header lines
        if (line[0] == '#') {
            if (line.find("#red_csv v2") != std::string::npos) is_v2 = true;
            continue;
        }
        // Skip column header row (starts with "frame,")
        if (line_num <= 3 && line.find("frame,") == 0) continue;

        const char *ptr = line.c_str();

        // Parse frame number
        char *end;
        u32 frame = (u32)strtoul(ptr, &end, 10);
        ptr = end;
        if (*ptr == ',') ++ptr;

        // Get or create frame
        auto &fa = get_or_create_frame(amap, frame, num_nodes, num_cameras);
        if (cam_idx >= (int)fa.cameras.size()) continue;
        auto &cam = fa.cameras[cam_idx];

        // Parse num_nodes groups of (x, y, c, s)
        for (int k = 0; k < num_nodes && *ptr; ++k) {
            double x, y;
            bool has_x = parse_csv_double(ptr, x);
            bool has_y = parse_csv_double(ptr, y);
            double conf;
            bool has_conf = parse_csv_double(ptr, conf);
            char src_ch = parse_csv_char(ptr);

            if (has_x && has_y) {
                cam.keypoints[k].x = x;
                cam.keypoints[k].y = y;
                cam.keypoints[k].labeled = true;
                cam.keypoints[k].confidence = has_conf ? (float)conf : 0.0f;
                if (src_ch == 'P') cam.keypoints[k].source = LabelSource::Predicted;
                else if (src_ch == 'I') cam.keypoints[k].source = LabelSource::Imported;
                else cam.keypoints[k].source = LabelSource::Manual;
            }
            // else: keypoint stays at default (unlabeled)
        }
    }
    return true;
}
```

**Implementation details for `convert_v1_to_v2`:**

```cpp
inline bool convert_v1_to_v2(const std::string &src_folder,
                              const std::string &dst_folder,
                              const std::string &skeleton_name,
                              int num_nodes,
                              const std::vector<std::string> &camera_names,
                              std::string *error) {
    namespace fs = std::filesystem;
    int num_cameras = (int)camera_names.size();

    // 1. Load old-format into a temporary AnnotationMap using a v1 reader
    AnnotationMap amap;
    // Read 3D CSV (old format)
    // ... (reuse old parsing logic adapted for AnnotationMap)

    // 2. Read confidence.csv sidecar if present, merge into amap
    // confidence.csv: frame_id,kp0,kp1,...
    std::string conf_path = src_folder + "/confidence.csv";
    if (fs::exists(conf_path)) {
        // Parse and set confidence + source=Predicted on all keypoints
        // for frames found in confidence.csv
    }

    // 3. Write v2 CSVs to dst_folder
    fs::create_directories(dst_folder);
    save_3d_csv(dst_folder + "/keypoints3d.csv", skeleton_name, amap, num_nodes);
    for (int c = 0; c < num_cameras; ++c)
        save_2d_csv(dst_folder + "/" + camera_names[c] + ".csv",
                    skeleton_name, amap, c, num_nodes);

    // 4. Copy annotations.json if present
    std::string json_src = src_folder + "/annotations.json";
    if (fs::exists(json_src))
        fs::copy_file(json_src, dst_folder + "/annotations.json");

    return true;
}
```

### 4.3 File: `src/annotation.h` — JSON persistence update

The `annotations_to_json` / `annotations_from_json` functions are updated to
match the new struct layout (no `instances` vector, `CameraExtras` behind
unique_ptr):

```cpp
inline nlohmann::json annotations_to_json(const AnnotationMap &amap) {
    nlohmann::json root;
    root["version"] = 2;
    nlohmann::json frames_arr = nlohmann::json::array();

    for (const auto &[fnum, fa] : amap) {
        bool has_extended = false;
        for (const auto &cam : fa.cameras) {
            if (cam.has_bbox() || cam.has_obb() || cam.has_mask()) {
                has_extended = true; break;
            }
        }
        if (!has_extended) continue;

        nlohmann::json jf;
        jf["frame"] = fnum;
        jf["instance_id"] = fa.instance_id;
        jf["category_id"] = fa.category_id;

        nlohmann::json cams = nlohmann::json::array();
        for (size_t c = 0; c < fa.cameras.size(); ++c) {
            const auto &cam = fa.cameras[c];
            if (!cam.extras) continue;
            const auto &ext = *cam.extras;

            nlohmann::json jc;
            jc["cam"] = (int)c;
            if (ext.has_bbox)
                jc["bbox"] = {ext.bbox_x, ext.bbox_y, ext.bbox_w, ext.bbox_h};
            if (ext.has_obb)
                jc["obb"] = {ext.obb_cx, ext.obb_cy, ext.obb_w, ext.obb_h, ext.obb_angle};
            if (ext.has_mask) {
                nlohmann::json polys = nlohmann::json::array();
                for (const auto &poly : ext.mask_polygons) {
                    nlohmann::json pts = nlohmann::json::array();
                    for (const auto &pt : poly)
                        pts.push_back({pt.x, pt.y});
                    polys.push_back(pts);
                }
                jc["mask"] = polys;
            }
            if (jc.size() > 1) cams.push_back(jc);
        }
        if (!cams.empty()) jf["cameras"] = cams;
        frames_arr.push_back(jf);
    }
    root["frames"] = frames_arr;
    return root;
}

inline void annotations_from_json(const nlohmann::json &root, AnnotationMap &amap) {
    if (!root.contains("frames")) return;
    for (const auto &jf : root["frames"]) {
        u32 fnum = jf["frame"].get<u32>();
        auto it = amap.find(fnum);
        if (it == amap.end()) continue;
        auto &fa = it->second;

        if (jf.contains("cameras")) {
            for (const auto &jc : jf["cameras"]) {
                int c = jc["cam"].get<int>();
                if (c >= (int)fa.cameras.size()) continue;
                auto &ext = fa.cameras[c].get_extras();

                if (jc.contains("bbox")) {
                    auto &b = jc["bbox"];
                    ext.bbox_x = b[0]; ext.bbox_y = b[1];
                    ext.bbox_w = b[2]; ext.bbox_h = b[3];
                    ext.has_bbox = true;
                }
                if (jc.contains("obb")) {
                    auto &o = jc["obb"];
                    ext.obb_cx = o[0]; ext.obb_cy = o[1];
                    ext.obb_w = o[2]; ext.obb_h = o[3]; ext.obb_angle = o[4];
                    ext.has_obb = true;
                }
                if (jc.contains("mask")) {
                    ext.mask_polygons.clear();
                    for (const auto &jpoly : jc["mask"]) {
                        std::vector<tuple_d> poly;
                        for (const auto &jpt : jpoly)
                            poly.push_back({jpt[0].get<double>(), jpt[1].get<double>()});
                        ext.mask_polygons.push_back(std::move(poly));
                    }
                    ext.has_mask = !ext.mask_polygons.empty();
                }
            }
        }
    }
}
```

### 4.4 Files Modified in Phase A (summary)

| File | Change |
|------|--------|
| `src/annotation.h` | Rewrite with new structs (Section 2), keep old types via `annotation_v1.h` include during transition |
| `src/annotation_v1.h` | NEW — copy of current `annotation.h` content (temporary) |
| `src/annotation_csv.h` | NEW — all CSV I/O (save, load, convert, find_most_recent_labels) |
| `tests/test_annotation.cpp` | Add 6 new tests (Section 7), old tests still compile against v1 types |

### 4.5 Exact Order of Operations (Phase A)

1. Copy `src/annotation.h` -> `src/annotation_v1.h`.
2. Add `#include "annotation_v1.h"` guard so old code still compiles.
3. Write new structs in `src/annotation.h` above the v1 include.
4. Update `make_frame` (was `make_instance`), `get_or_create_frame`, `frame_has_any_labels` to use new structs.
5. Update JSON persistence functions to use new struct layout.
6. Create `src/annotation_csv.h` with `parse_csv_double`, `parse_csv_char`, `save_2d_csv`, `save_3d_csv`, `save_all`, `load_2d_csv`, `load_3d_csv`, `load_all`, `is_v1_format`, `convert_v1_to_v2`, `find_most_recent_labels`.
7. Write 6 new tests (Section 7).
8. Verify all tests pass: both old bridge tests (against v1 types) and new tests.
9. Commit.

---

## 5. Phase B: Switch All Consumers

### Goal
Every file that touches `KeyPoints*`, `Camera2D`, or `InstanceAnnotation` is
switched to use `FrameAnnotation` / `CameraAnnotation` / `Keypoint2D` /
`Keypoint3D`. The old `keypoints_map` is removed from `AppContext`. After this
phase, the v1 types are dead code.

### 5.1 File: `src/app_context.h`

**Changes:**
- Remove `std::map<u32, KeyPoints *> &keypoints_map;`
- Remove `bool &annotations_dirty;`
- `AnnotationMap &annotations;` becomes the sole data path
- Remove `#include "gui/gui_save_load.h"` (save/load moves to annotation_csv.h)
- Add `#include "annotation_csv.h"`

```cpp
struct AppContext {
    // Core project/scene
    ProjectManager &pm;
    PlaybackState &ps;
    RenderScene *scene;
    DecoderContext *dc_context;

    // Skeleton
    SkeletonContext &skeleton;
    std::map<std::string, SkeletonPrimitive> &skeleton_map;

    // Annotations (sole data path — no more keypoints_map)
    AnnotationMap &annotations;

    // UI infrastructure
    PopupStack &popups;
    ToastQueue &toasts;
    DeferredQueue &deferred;

    // ... rest unchanged ...
};
```

**`on_project_loaded` changes:**
```cpp
inline void on_project_loaded(AppContext &ctx, ...) {
    switch_ini_to_project(ctx);
    load_videos(...);
    if (print_metadata_fn) print_metadata_fn();

    std::string label_err, most_recent_folder;
    if (!AnnotationCSV::find_most_recent_labels(ctx.pm.keypoints_root_folder,
                                                 most_recent_folder, label_err)) {
        // Auto-detect v1 and convert
        if (AnnotationCSV::is_v1_format(most_recent_folder, ctx.pm.camera_names)) {
            std::string converted = most_recent_folder + "_v2";
            std::string conv_err;
            if (!AnnotationCSV::convert_v1_to_v2(most_recent_folder, converted,
                    ctx.skeleton.name, ctx.skeleton.num_nodes,
                    ctx.pm.camera_names, &conv_err)) {
                ctx.popups.pushError("v1 conversion failed: " + conv_err);
            } else {
                most_recent_folder = converted;
            }
        }

        int ret = AnnotationCSV::load_all(most_recent_folder, ctx.annotations,
                                           ctx.skeleton.name,
                                           ctx.skeleton.num_nodes,
                                           ctx.scene->num_cams,
                                           ctx.pm.camera_names, label_err);
        if (ret != 0) {
            ctx.annotations.clear();
            ctx.popups.pushError(label_err);
        } else {
            load_annotations_json(ctx.annotations, most_recent_folder);
        }
    }
    if (print_summary_fn) print_summary_fn(most_recent_folder);
}
```

### 5.2 File: `src/red.cpp`

**Changes:**
- Remove `std::map<u32, KeyPoints *> keypoints_map;`
- Remove `bool annotations_dirty = false;`
- Remove `keypoints_map` from `AppContext` initializer
- Remove `annotations_dirty` from `AppContext` initializer
- `keypoints_find` becomes: `bool keypoints_find = annotations.count(current_frame_num) > 0;`
- All keypoint editing code (B key, W key, A/D keys, Q/E keys, R key, F key,
  Backspace key) rewired to use `AnnotationMap`

**Keypoint editing (W key — label keypoint):**

```cpp
// Old:
keypoints_map[current_frame_num]->kp2d[j][*kp].position = {mouse.x, mouse.y};
keypoints_map[current_frame_num]->kp2d[j][*kp].is_labeled = true;

// New:
auto &fa = get_or_create_frame(annotations, current_frame_num,
                                skeleton.num_nodes, scene->num_cams);
auto &kp2d = fa.cameras[j].keypoints[fa.cameras[j].active_id];
kp2d.x = mouse.x;
kp2d.y = mouse.y;
kp2d.labeled = true;
kp2d.source = LabelSource::Manual;
kp2d.confidence = 0.0f;
```

**B key (create frame entry):**
```cpp
// Old:
KeyPoints *keypoints = (KeyPoints *)malloc(sizeof(KeyPoints));
allocate_keypoints(keypoints, scene, &skeleton);
keypoints_map[current_frame_num] = keypoints;

// New:
get_or_create_frame(annotations, current_frame_num,
                     skeleton.num_nodes, scene->num_cams);
```

**Backspace (delete all keypoints on frame):**
```cpp
// Old:
free_keypoints(keypoints_map[current_frame_num], scene);
keypoints_map.erase(current_frame_num);

// New:
annotations.erase(current_frame_num);
```

**active_id navigation (A/D/Q/E keys):**
```cpp
// Old:
u32 *kp = &(keypoints_map[current_frame_num]->active_id[j]);

// New:
u32 &active = annotations.at(current_frame_num).cameras[j].active_id;
```

**R key (delete single keypoint) and F key (delete from all views):**
```cpp
// Old:
keypoints->kp2d[view_idx][node].position = {1E7, 1E7};
keypoints->kp2d[view_idx][node].is_labeled = false;

// New:
auto &kp = fa.cameras[view_idx].keypoints[node];
kp.x = UNLABELED;
kp.y = UNLABELED;
kp.labeled = false;
kp.confidence = 0.0f;
kp.source = LabelSource::Manual;
```

### 5.3 File: `src/gui/gui_keypoints.h`

**`gui_plot_keypoints` — complete rewrite of signature and body:**

```cpp
static void gui_plot_keypoints(FrameAnnotation &fa, SkeletonContext *skeleton,
                                int view_idx, int num_cams) {
    auto &cam = fa.cameras[view_idx];
    float pt_size = 6.0f;
    for (u32 node = 0; node < skeleton->num_nodes; node++) {
        auto &kp = cam.keypoints[node];
        if (kp.labeled) {
            ImVec4 node_color;
            if (cam.active_id == node) {
                node_color = (ImVec4)ImColor::HSV(0.8, 1.0f, 1.0f);
                node_color.w = 0.9;
                pt_size = 8.0f;
            } else {
                node_color = skeleton->node_colors.at(node);
                node_color.w = 0.9;
                pt_size = 6.0f;
            }
            int id = skeleton->num_nodes * view_idx + node;
            static bool drag_point_clicked, drag_point_hovered, drag_point_modified;
            drag_point_modified = ImPlot::DragPoint(
                id, &kp.x, &kp.y, node_color,
                pt_size, ImPlotDragToolFlags_None,
                &drag_point_clicked, &drag_point_hovered);
            if (drag_point_modified) {
                fa.kp3d[node].triangulated = false;
            }
            if (drag_point_hovered) {
                if (fa.kp3d[node].triangulated) {
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(2);
                    oss << "(" << fa.kp3d[node].x << ", "
                        << fa.kp3d[node].y << ", "
                        << fa.kp3d[node].z << ")";
                    // ... tooltip rendering ...
                }
                if (ImGui::IsKeyPressed(ImGuiKey_R, false)) {
                    kp.x = UNLABELED; kp.y = UNLABELED;
                    kp.labeled = false;
                    cam.active_id = node;
                }
                if (ImGui::IsKeyPressed(ImGuiKey_F, false)) {
                    for (int ci = 0; ci < num_cams; ci++) {
                        auto &ckp = fa.cameras[ci].keypoints[node];
                        ckp.x = UNLABELED; ckp.y = UNLABELED;
                        ckp.labeled = false;
                        fa.cameras[ci].active_id = node;
                    }
                }
            }
            if (drag_point_clicked) {
                cam.active_id = node;
            }
        }
    }
    // Edge drawing
    for (u32 edge = 0; edge < skeleton->num_edges; edge++) {
        auto [a, b] = skeleton->edges[edge];
        if (cam.keypoints[a].labeled && cam.keypoints[b].labeled) {
            double xs[2]{cam.keypoints[a].x, cam.keypoints[b].x};
            double ys[2]{cam.keypoints[a].y, cam.keypoints[b].y};
            ImPlot::PlotLine("##line", xs, ys, 2);
        }
    }
}
```

**Note on pointer stability for ImPlot::DragPoint:** The `&kp.x` and `&kp.y`
pointers are stable because:
- `AnnotationMap` is `std::map` (node-based, pointer-stable on insert/erase).
- `CameraAnnotation::keypoints` is a `std::vector<Keypoint2D>` whose size is
  fixed at creation and never resized.
- `FrameAnnotation::cameras` is likewise fixed-size.
- So `&fa.cameras[j].keypoints[node].x` is stable for the lifetime of the
  FrameAnnotation entry in the map.

**`reprojection` — updated signature:**

```cpp
static void reprojection(FrameAnnotation &fa, SkeletonContext *skeleton,
                          std::vector<CameraParams> camera_params,
                          RenderScene *scene) {
    for (u32 node = 0; node < skeleton->num_nodes; node++) {
        u32 num_views_labeled = 0;
        for (u32 vi = 0; vi < scene->num_cams; vi++)
            if (fa.cameras[vi].keypoints[node].labeled) num_views_labeled++;

        if (num_views_labeled >= 2) {
            std::vector<Eigen::Vector2d> undist_pts;
            std::vector<Eigen::Matrix<double, 3, 4>> proj_mats;
            for (u32 vi = 0; vi < scene->num_cams; vi++) {
                auto &kp = fa.cameras[vi].keypoints[node];
                if (!kp.labeled) continue;
                Eigen::Vector2d pt(kp.x, (double)scene->image_height[vi] - kp.y);
                auto pt_undist = red_math::undistortPoint(
                    pt, camera_params[vi].k, camera_params[vi].dist_coeffs);
                undist_pts.push_back(pt_undist);
                proj_mats.push_back(camera_params[vi].projection_mat);
            }
            Eigen::Vector3d pt3d = red_math::triangulatePoints(undist_pts, proj_mats);
            fa.kp3d[node].x = pt3d(0);
            fa.kp3d[node].y = pt3d(1);
            fa.kp3d[node].z = pt3d(2);
            fa.kp3d[node].triangulated = true;

            // Reproject to all cameras
            for (u32 vi = 0; vi < scene->num_cams; vi++) {
                // ... (same logic, writing to fa.cameras[vi].keypoints[node])
            }
        }
    }
}
```

### 5.4 File: `src/gui/labeling_tool_window.h`

**Changes:**
- Replace all `keypoints_map` references with `ctx.annotations`
- `has_any_labels(kp, skeleton, scene)` -> `frame_has_any_labels(fa)`
- Save calls `AnnotationCSV::save_all()` instead of old `save_keypoints()`
- "Copy Prev" copies FrameAnnotation instead of malloc/copy_keypoints
- Remove `refresh_keypoints_in_amap` call
- Remove `annotations_dirty` usage

```cpp
inline void DrawLabelingToolWindow(
    LabelingToolState &state, AppContext &ctx,
    int current_frame_num, bool keypoints_find) {

    auto &amap = ctx.annotations;
    auto &skeleton = ctx.skeleton;
    auto *scene = ctx.scene;
    // ...

    // Prev/Next labeled frame search
    auto next_labeled_it = amap.end();
    for (auto it = amap.upper_bound(current_frame_num);
         it != amap.end(); ++it) {
        if (frame_has_any_labels(it->second)) {
            next_labeled_it = it; break;
        }
    }
    // ... (same pattern for prev)

    // Save
    if (state.save_requested) {
        std::string err;
        std::string saved = AnnotationCSV::save_all(
            pm.keypoints_root_folder, skeleton.name, amap,
            scene->num_cams, skeleton.num_nodes,
            pm.camera_names, &err);
        if (saved.empty()) {
            toasts.pushError("Save failed: " + err);
        } else {
            save_annotations_json(amap, saved);
            toasts.pushSuccess("Labels saved");
        }
        state.last_saved = time(NULL);
    }

    // Copy Prev
    if (ImGui::Button("Copy Prev")) {
        auto &prev_fa = amap.at(prev_frame);
        auto &curr_fa = get_or_create_frame(amap, current_frame_num,
                                             skeleton.num_nodes, scene->num_cams);
        // Deep copy keypoints (not extras/active_id)
        curr_fa.kp3d = prev_fa.kp3d;
        for (int c = 0; c < (int)prev_fa.cameras.size(); ++c)
            curr_fa.cameras[c].keypoints = prev_fa.cameras[c].keypoints;
    }
}
```

### 5.5 File: `src/keypoints_table.h`

**Changes — complete rewrite to use AnnotationMap:**

```cpp
inline void DrawKeypointsWindow(AppContext &ctx, int current_frame_num) {
    auto &amap = ctx.annotations;
    auto &skeleton = ctx.skeleton;
    auto *scene = ctx.scene;
    auto &is_view_focused = ctx.is_view_focused;

    bool keypoints_find = amap.count(current_frame_num) > 0;

    // ... table setup unchanged ...

    // Cell rendering
    if (keypoints_find) {
        const auto &fa = amap.at(current_frame_num);
        const auto &cam = fa.cameras[row];
        if (cam.active_id == column - 1) {
            node_color = (ImVec4)ImColor::HSV(0.8f, 1.0f, 1.0f);
        } else if (cam.keypoints[column - 1].labeled) {
            node_color = skeleton.node_colors[column - 1];
            node_color.w = 0.9f;
        }
        if (fa.kp3d[column - 1].triangulated) {
            ImGui::TextColored(ImVec4(1,1,1,1), "T");
        }
    }
}
```

### 5.6 File: `src/gui/export_window.h`

**Changes:**
- Remove `refresh_keypoints_in_amap` call
- Remove `annotations_dirty` check
- `DrawExportWindow` takes `AnnotationMap &amap` (already does, but
  the sync code is removed)

```cpp
inline void DrawExportWindow(ExportWindowState &state, AppContext &ctx,
                              AnnotationMap &amap) {
    // Remove these lines:
    // if (ctx.annotations_dirty) {
    //     refresh_keypoints_in_amap(amap, ctx.keypoints_map, ...);
    //     ctx.annotations_dirty = false;
    // }

    // Rest of function unchanged — it already reads from amap
}
```

### 5.7 File: `src/export_formats.h`

**Changes:**
- All functions that read `inst.cameras[ci].kp_labeled[k]` change to
  `cam.keypoints[k].labeled`
- All functions that read `inst.cameras[ci].keypoints[k].x` change to
  `cam.keypoints[k].x`
- `inst.cameras` -> `fa.cameras` (no more `instances` vector)
- `cam.has_bbox` -> `cam.has_bbox()` (method, not field)
- `cam.has_obb` -> `cam.has_obb()` (method, not field)
- `cam.has_mask` -> `cam.has_mask()` (method, not field)
- `cam.bbox_x` -> `cam.extras->bbox_x` (only when has_bbox() is true)
- `cam.mask_polygons` -> `cam.extras->mask_polygons`

**Example — `build_coco_json` inner loop:**
```cpp
for (u32 frame : frames) {
    auto it = amap.find(frame);
    if (it == amap.end()) continue;
    const auto &fa = it->second;

    // ... image entry ...

    for (int cam_idx_for_inst = 0; cam_idx_for_inst < 1; ++cam_idx_for_inst) {
        // Single instance: fa directly
        if (cam_idx >= (int)fa.cameras.size()) continue;
        const auto &cam = fa.cameras[cam_idx];

        int num_visible = 0;
        for (const auto &kp : cam.keypoints)
            if (kp.labeled) ++num_visible;
        if (num_visible == 0) continue;

        nlohmann::json kp_flat = nlohmann::json::array();
        double x_min = 1e9, x_max = -1e9, y_min = 1e9, y_max = -1e9;
        for (const auto &kp : cam.keypoints) {
            if (kp.labeled) {
                double x = kp.x;
                double y = img_h - kp.y;
                kp_flat.push_back(x); kp_flat.push_back(y); kp_flat.push_back(2);
                // bbox calc...
            } else {
                kp_flat.push_back(0); kp_flat.push_back(0); kp_flat.push_back(0);
            }
        }

        // Bbox from extras or keypoints
        double bx, by, bw, bh;
        if (cam.has_bbox()) {
            bx = cam.extras->bbox_x; by = cam.extras->bbox_y;
            bw = cam.extras->bbox_w; bh = cam.extras->bbox_h;
        } else {
            // derive from keypoints...
        }

        // Mask
        nlohmann::json seg = nlohmann::json::array();
        if (cam.has_mask()) {
            for (const auto &poly : cam.extras->mask_polygons) { ... }
        }
        // ...
    }
}
```

### 5.8 File: `src/jarvis_export.h`

**Changes:**
- `read_csv_2d` and `read_csv_3d` updated to handle v2 CSV format
  (skip `#red_csv` and `#skeleton` headers, parse column header row)
- The JARVIS exporter reads CSVs from disk (not AnnotationMap), so the
  main change is CSV format detection. The simplest approach: detect `#red_csv`
  header and skip the first 3 lines instead of 1.

```cpp
inline std::map<int, std::vector<std::vector<double>>>
read_csv_2d(const std::string &path, int img_height) {
    // ...
    std::string line;
    // Detect v2: skip all lines starting with # or "frame,"
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;
        if (line.find("frame,") == 0) continue;
        break; // first data line
    }
    // Process first data line, then continue with rest
    // Parse format: frame,x,y,c,s,x,y,c,s,...
    // Group into x,y pairs (skip c,s columns)
    // ...
}
```

### 5.9 File: `src/jarvis_import.h`

**Changes:**
- `write_prediction_csvs` writes v2 format (not v1)
- Remove `confidence.csv` sidecar — confidence is inline in v2 CSVs
- Remove `load_confidence` function entirely
- `LabelSource::Predicted` is set for all imported keypoints

```cpp
inline bool
write_prediction_csvs(const std::string &output_folder,
                      const std::string &skeleton_name,
                      const std::map<int, Prediction3D> &preds,
                      const std::vector<CameraParams> &cameras,
                      const std::vector<std::string> &camera_names,
                      int img_height,
                      std::string *error = nullptr) {
    // Write v2 format keypoints3d.csv
    {
        std::ofstream f(output_folder + "/keypoints3d.csv");
        f << "#red_csv v2\n";
        f << "#skeleton " << skeleton_name << "\n";
        // Column headers
        int nj = preds.empty() ? 0 : (int)preds.begin()->second.positions.size();
        f << "frame";
        for (int j = 0; j < nj; j++)
            f << ",x" << j << ",y" << j << ",z" << j << ",c" << j;
        f << "\n";
        // Data
        for (const auto &[fid, pred] : preds) {
            f << fid;
            for (int j = 0; j < (int)pred.positions.size(); j++) {
                f << "," << pred.positions[j].x()
                  << "," << pred.positions[j].y()
                  << "," << pred.positions[j].z()
                  << "," << pred.confidences[j];
            }
            f << "\n";
        }
    }

    // Per-camera 2D CSVs (v2 format with confidence + source=P)
    for (int c = 0; c < (int)cameras.size(); c++) {
        auto pts2d_map = project_to_camera(preds, cameras[c]);
        std::ofstream f(output_folder + "/" + camera_names[c] + ".csv");
        f << "#red_csv v2\n";
        f << "#skeleton " << skeleton_name << "\n";
        int nj = preds.empty() ? 0 : (int)preds.begin()->second.positions.size();
        f << "frame";
        for (int j = 0; j < nj; j++)
            f << ",x" << j << ",y" << j << ",c" << j << ",s" << j;
        f << "\n";
        for (const auto &[fid, pts2d] : pts2d_map) {
            const auto &pred = preds.at(fid);
            f << fid;
            for (int j = 0; j < (int)pts2d.size(); j++) {
                double x = pts2d[j].x();
                double y = (double)img_height - pts2d[j].y();
                f << "," << x << "," << y << "," << pred.confidences[j] << ",P";
            }
            f << "\n";
        }
    }

    // NO confidence.csv sidecar — confidence is inline in v2 CSVs

    return true;
}
```

### 5.10 File: `src/gui/jarvis_import_window.h`

**Changes:**
- "Load into RED" uses `AnnotationCSV::load_all` instead of `load_keypoints` +
  `load_confidence`
- References `ctx.annotations` instead of `ctx.keypoints_map`

```cpp
if (ImGui::Button("Load into RED")) {
    std::string err;
    int ret = AnnotationCSV::load_all(
        state.result.output_folder, ctx.annotations,
        skeleton.name, skeleton.num_nodes,
        scene->num_cams, ctx.pm.camera_names, err);
    if (ret == 0) {
        load_annotations_json(ctx.annotations, state.result.output_folder);
        ImGui::TextColored(ImVec4(0,1,0,1), "Predictions loaded!");
    } else {
        state.result.error = "Load failed: " + err;
    }
}
```

### 5.11 File: `src/gui/bbox_tool.h`

**Changes:**
- `it->second.instances[i]` -> direct access on `it->second`
- `inst.cameras[cam_idx]` -> `fa.cameras[cam_idx]`
- `cam.has_bbox` -> `cam.has_bbox()`
- `cam.bbox_x` -> `cam.extras->bbox_x`
- `make_instance(...)` -> `make_frame(...)` in get_or_create_frame
- `get_or_create_frame` signature unchanged (returns `FrameAnnotation &`)
- Remove instance iteration loop (single instance)

### 5.12 File: `src/gui/obb_tool.h`

Same changes as bbox_tool.h — replace `instances[i]` iteration with direct
`fa.cameras[cam_idx]` access, use `cam.get_extras()` for write, `cam.has_obb()`
for read.

### 5.13 File: `src/gui/sam_tool.h`

Same pattern — replace `fa.instances.empty()` check with direct camera access,
store mask in `cam.get_extras().mask_polygons`.

### 5.14 File: `src/skeleton.h` and `src/skeleton.cpp`

**skeleton.h changes:**
- `KeyPoints2D`, `KeyPoints3D`, `KeyPoints` structs: KEEP during Phase B
  (deleted in Phase C). They are still referenced by skeleton.cpp functions.
- `has_any_labels(const KeyPoints*, ...)`: KEEP during Phase B.
- Add new overload: `has_labeled_frames(const AnnotationMap &)`.

**skeleton.cpp changes:**
- `allocate_keypoints`, `free_keypoints`, `free_all_keypoints`,
  `copy_keypoints`, `cleanup_skeleton_data`: KEEP during Phase B (not called
  anymore, deleted in Phase C).
- Add: `bool has_labeled_frames(const AnnotationMap &amap)` — walks amap
  checking `frame_has_any_labels`.

### 5.15 File: `src/gui/gui_save_load.h`

**Changes:**
- `save_keypoints()`: KEEP during Phase B for v1 converter reference, but
  no longer called from any live code path.
- `load_keypoints()`, `load_2d_keypoints()`, `find_most_recent_labels()`: moved
  to `annotation_csv.h`. The old versions stay in gui_save_load.h during
  Phase B (deleted in Phase C).

### 5.16 Files Modified in Phase B (summary)

| File | Change |
|------|--------|
| `src/app_context.h` | Remove keypoints_map, annotations_dirty; update on_project_loaded |
| `src/red.cpp` | Remove keypoints_map local; rewrite all editing to use AnnotationMap |
| `src/gui/gui_keypoints.h` | Rewrite gui_plot_keypoints + reprojection signatures |
| `src/gui/labeling_tool_window.h` | Rewrite to use AnnotationMap for save/load/navigate |
| `src/keypoints_table.h` | Rewrite to use AnnotationMap |
| `src/gui/export_window.h` | Remove bridge sync code |
| `src/export_formats.h` | Update field access (no instances vector, extras accessor) |
| `src/jarvis_export.h` | Update CSV readers to handle v2 format |
| `src/jarvis_import.h` | Write v2 format, remove confidence.csv, remove load_confidence |
| `src/gui/jarvis_import_window.h` | Use AnnotationCSV::load_all |
| `src/gui/bbox_tool.h` | Remove instances iteration, use extras accessor |
| `src/gui/obb_tool.h` | Remove instances iteration, use extras accessor |
| `src/gui/sam_tool.h` | Remove instances iteration, use extras accessor |
| `src/skeleton.h` | Add new has_labeled_frames overload |
| `src/skeleton.cpp` | Add new has_labeled_frames implementation |

### 5.17 Exact Order of Operations (Phase B)

1. Update `src/annotation.h`: remove v1 types, finalize new structs only.
2. Update `src/app_context.h`: remove keypoints_map, annotations_dirty.
3. Update `src/red.cpp`: remove keypoints_map local, rewrite all editing code.
4. Update `src/gui/gui_keypoints.h`: new signatures for gui_plot_keypoints, reprojection.
5. Update `src/gui/labeling_tool_window.h`: AnnotationMap-based save/load/navigate.
6. Update `src/keypoints_table.h`: AnnotationMap-based cell rendering.
7. Update `src/gui/export_window.h`: remove bridge sync.
8. Update `src/export_formats.h`: remove instances indirection.
9. Update `src/jarvis_export.h`: v2 CSV reader support.
10. Update `src/jarvis_import.h`: v2 CSV writer, remove confidence.csv.
11. Update `src/gui/jarvis_import_window.h`: use AnnotationCSV::load_all.
12. Update `src/gui/bbox_tool.h`, `obb_tool.h`, `sam_tool.h`: remove instances.
13. Update `src/skeleton.h/cpp`: add new overloads.
14. Build and fix all compile errors.
15. Run all tests, fix failures.
16. Manual test: load project, label keypoints, save, reload, export JARVIS.
17. Commit.

---

## 6. Phase C: Delete Old Code

### Goal
Remove all dead code from the v1 system. Clean, minimal codebase.

### 6.1 Files to Delete

| File | Reason |
|------|--------|
| `src/annotation_v1.h` | Temporary bridge file from Phase A |

### 6.2 Code to Remove from Existing Files

**`src/skeleton.h`:**
- Delete `KeyPoints2D` struct (lines 12-18)
- Delete `KeyPoints3D` struct (lines 20-24)
- Delete `KeyPoints` struct (lines 26-30)
- Delete `has_labeled_frames(const std::map<u32, KeyPoints *> &, ...)` overload
- Delete `allocate_keypoints` declaration
- Delete `free_keypoints` declaration
- Delete `free_all_keypoints` declaration
- Delete `cleanup_skeleton_data` declaration
- Delete `has_any_labels(const KeyPoints *, ...)` declaration
- Delete `copy_keypoints` declaration

**`src/skeleton.cpp`:**
- Delete `allocate_keypoints` implementation (~30 lines of malloc)
- Delete `free_keypoints` implementation
- Delete `free_all_keypoints` implementation
- Delete `cleanup_skeleton_data` implementation
- Delete `has_any_labels(const KeyPoints *, ...)` implementation
- Delete `copy_keypoints` implementation
- Delete `has_labeled_frames(const std::map<u32, KeyPoints *> &, ...)` implementation

**`src/gui/gui_save_load.h`:**
- Delete `save_keypoints()` function (entire thing, ~65 lines)
- Delete `load_2d_keypoints()` function (~80 lines)
- Delete `load_keypoints()` function (~100 lines)
- Delete `find_most_recent_labels()` (moved to annotation_csv.h)
- Delete `current_date_time()` helper (replaced by AnnotationCSV internal)
- If file is now empty, delete the file entirely.

**`src/annotation.h`:**
- Delete all bridge/migration functions:
  - `instance_from_keypoints()`
  - `migrate_keypoints_map()`
  - `refresh_keypoints_in_amap()`
- Delete any remaining `Camera2D` or `InstanceAnnotation` references

**`src/jarvis_import.h`:**
- Delete `load_confidence()` function (confidence is inline in v2 CSV)

**`src/types.h`:**
- Delete `YoloExportMode` enum (unused since bbox/OBB tools reference
  `ExportFormats::Format` instead)

### 6.3 Tests to Delete

From `tests/test_annotation.cpp`, delete these 13 bridge/migration tests:

1. `test_migration_roundtrip` — tests KeyPoints* -> AnnotationMap migration
2. `test_bridge_keypoints_and_extended_roundtrip` — tests bridge sync
3. `test_refresh_empty_keypoints_map` — tests refresh_keypoints_in_amap
4. `test_refresh_null_keypoints_skipped` — tests null KeyPoints* handling
5. `test_refresh_preserves_bbox_on_update` — tests bridge preserving extras
6. `test_refresh_adds_new_frames` — tests bridge adding frames
7. `test_refresh_removes_stale_frames` — tests bridge removing frames
8. `test_refresh_empty_instances_creates_new` — tests bridge creating instances
9. `test_migrate_null_keypoints_skipped` — tests migration null handling
10. `test_migrate_multi_frame` — tests multi-frame migration
11. `test_migrate_empty_map` — tests empty map migration
12. `test_save_load_keypoints_roundtrip` — tests old save_keypoints + load_keypoints
13. `test_save_load_with_extended_data` — tests old save + load + annotations.json

### 6.4 Exact Order of Operations (Phase C)

1. Delete `src/annotation_v1.h`.
2. Remove bridge functions from `src/annotation.h`.
3. Remove old KeyPoints* types and functions from `src/skeleton.h` and `src/skeleton.cpp`.
4. Remove old save/load functions from `src/gui/gui_save_load.h` (or delete file).
5. Remove `load_confidence` from `src/jarvis_import.h`.
6. Remove `YoloExportMode` from `src/types.h`.
7. Delete 13 bridge tests from `tests/test_annotation.cpp`.
8. Build, verify no compile errors.
9. Run all remaining tests, verify pass.
10. Commit.

---

## 7. Test Plan

### 7.1 New Tests (added in Phase A, updated in Phase B)

All tests live in `tests/test_annotation.cpp`.

**Test 1: `test_save_amap_csv_roundtrip`**
```
Purpose: Verify v2 CSV save + load is lossless for all Keypoint2D fields.
Setup:   Create AnnotationMap with 3 frames, 4 keypoints, 2 cameras.
         Mix of manual, predicted (conf=0.92, source=P), and unlabeled.
Action:  save_all() to temp dir, then load_all() into fresh AnnotationMap.
Verify:  All x,y values match within 1e-6.
         All labeled flags match.
         All confidence values match within 1e-6.
         All source flags match (Manual/Predicted/Imported).
         All unlabeled keypoints have default values.
```

**Test 2: `test_save_amap_csv_unlabeled`**
```
Purpose: Verify that fully-unlabeled frames produce correct empty-cell CSV.
Setup:   Create AnnotationMap with 1 frame, all keypoints unlabeled.
Action:  save_2d_csv(), read file content as string.
Verify:  No "1E7" or "1e7" sentinel appears in file.
         Each keypoint group is ",,,,".
         Reload produces identical AnnotationMap.
```

**Test 3: `test_load_old_format_csv`**
```
Purpose: Verify one-time v1 -> v2 converter produces correct output.
Setup:   Write a v1-format CSV file:
         - First line: "Rat4"
         - Data: "100,0,500.5,600.3,1,1E7,1E7"
Action:  is_v1_format() returns true.
         convert_v1_to_v2() writes v2 files.
         load_all() reads v2 files.
Verify:  Frame 100, keypoint 0: x=500.5, y=600.3, labeled=true, source=Manual.
         Frame 100, keypoint 1: labeled=false (was 1E7 sentinel).
         v2 file starts with "#red_csv v2".
```

**Test 4: `test_jarvis_json_from_amap`**
```
Purpose: Verify JARVIS annotation JSON generated from AnnotationMap matches
         expected schema.
Setup:   Create AnnotationMap with known keypoints.
         Write v2 CSVs with save_all().
         Create matching calibration YAMLs.
Action:  Call JarvisExport::generate_annotation_json (reading CSVs from disk).
Verify:  JSON has correct images, annotations, keypoints arrays.
         Keypoint x,y values match after Y-flip.
         bbox computed correctly from keypoints + margin.
```

**Test 5: `test_jarvis_json_partial_labels`**
```
Purpose: Verify JARVIS export handles partially-labeled frames correctly.
Setup:   Frame with keypoints 0,1 labeled in camera 0, keypoint 2 unlabeled.
Action:  Generate annotation JSON.
Verify:  Frame is included in images array.
         Annotation is NOT included (JARVIS requires all keypoints labeled).
         No crash or assertion failure.
```

**Test 6: `test_jarvis_json_matches_csv_path` (golden comparison)**
```
Purpose: Verify new code produces identical JARVIS JSON to old code.
         Run BEFORE deleting old code (Phase B, before Phase C).
Setup:   Create identical data in both KeyPoints* map and AnnotationMap.
         Write old-format CSVs with save_keypoints().
         Write new-format CSVs with save_all().
Action:  Generate JARVIS JSON from both CSV sets.
Verify:  Both JSONs are structurally identical (same images, annotations,
         keypoints values). Allow floating-point tolerance.
Note:    This test uses v1 types and is deleted in Phase C along with
         the other bridge tests.
```

### 7.2 Tests That Need Updating (Phase B)

These existing tests reference the old data model and need field access updates:

- `test_make_instance` -> `test_make_frame` (new struct names)
- `test_make_instance_zero_dims` -> `test_make_frame_zero_dims`
- `test_get_or_create_frame` (update field access)
- `test_frame_has_any_labels` (update field access)
- `test_json_roundtrip_bbox` (use extras accessor)
- `test_json_roundtrip_mask` (use extras accessor)
- `test_json_roundtrip_bbox3d` -> DELETE (bbox3d removed)
- `test_build_coco_json` (update field access)
- `test_build_coco_json_with_explicit_bbox` (use extras accessor)
- `test_build_coco_json_with_mask` (use extras accessor)
- All YOLO/DLC export tests (update field access)
- All OBB tests (unchanged — they test pure geometry)
- All SAM tests (unchanged — they test pure geometry/inference)
- `test_annotation_config_*` (unchanged — AnnotationConfig is unchanged)
- `test_frame_has_any_labels_multi_instance` -> DELETE (no multi-instance)
- `test_make_instance_sizes_match` -> `test_make_frame_sizes_match`

### 7.3 Tests to Delete (Phase C)

See Section 6.3 — 13 bridge tests plus `test_json_roundtrip_bbox3d` and
`test_frame_has_any_labels_multi_instance` (15 total).

### 7.4 Test Build

The test binary `test_annotation` is built by CMake. Ensure it includes:
```cmake
target_sources(test_annotation PRIVATE tests/test_annotation.cpp)
target_include_directories(test_annotation PRIVATE src src/gui)
```

No new dependencies. The `annotation_csv.h` is header-only and included
transitively through `annotation.h` or directly.

---

## 8. Open Questions

### 8.1 Resolved

**Q: Should `last_position` and `last_is_labeled` be kept?**
A: No. These fields on `KeyPoints2D` are only used by the reprojection function
to restore pre-reprojection state. The reprojection function will be rewritten
to not need undo state. If undo is needed later, it should be a proper undo
stack, not per-field shadow copies.

**Q: Should the JARVIS exporter read from AnnotationMap or CSV?**
A: It continues reading from CSV on disk. This avoids a large rewrite of
`jarvis_export.h` and ensures the export matches exactly what was saved. The
CSV readers in `jarvis_export.h` are updated to handle v2 format headers.

**Q: Multi-instance: should we support it?**
A: No (YAGNI). The `instance_id` and `category_id` fields are kept as
placeholders on `FrameAnnotation` for forward compatibility, but there is no
`vector<Instance>` indirection. The bbox/OBB/mask tools write to the single
frame's cameras. If multi-instance is needed in the future, it can be added by
wrapping `FrameAnnotation` contents in a named struct.

**Q: Should `active_id` be persisted in CSV?**
A: No. It is UI state only (which keypoint is currently selected for editing).
It defaults to 0 on load.

### 8.2 Remaining

**Q: Should the v1 converter run in-place or create a sibling folder?**
A: Creates a sibling folder (`<timestamp>_v2`). This is non-destructive. The
original v1 folder is untouched. The user can delete it manually.

**Q: Should `gui_save_load.h` be deleted entirely in Phase C?**
A: Yes, if all functions have been moved to `annotation_csv.h`. The
`current_date_time()` helper is replaced by the timestamp logic in
`AnnotationCSV::save_all()`.

---

## 9. File Inventory

### Complete list of files touched across all phases

| File | Phase A | Phase B | Phase C | Action |
|------|---------|---------|---------|--------|
| `src/annotation.h` | Rewrite | Finalize | Remove bridge | Core data model |
| `src/annotation_v1.h` | Create | Keep | Delete | Temporary bridge |
| `src/annotation_csv.h` | Create | Keep | Keep | New save/load/convert |
| `src/app_context.h` | — | Rewrite | — | Remove keypoints_map |
| `src/red.cpp` | — | Rewrite editing | — | Use AnnotationMap |
| `src/gui/gui_keypoints.h` | — | Rewrite | — | New signatures |
| `src/gui/gui_save_load.h` | — | Deprecate | Delete | Replaced by annotation_csv.h |
| `src/gui/labeling_tool_window.h` | — | Rewrite | — | AnnotationMap save/load |
| `src/keypoints_table.h` | — | Rewrite | — | AnnotationMap rendering |
| `src/gui/export_window.h` | — | Remove sync | — | Drop bridge |
| `src/export_formats.h` | — | Update access | — | No instances vector |
| `src/jarvis_export.h` | — | Update CSV reader | — | Handle v2 format |
| `src/jarvis_import.h` | — | v2 writer | Remove load_confidence | Inline confidence |
| `src/gui/jarvis_import_window.h` | — | Use load_all | — | New load path |
| `src/gui/bbox_tool.h` | — | Update access | — | No instances |
| `src/gui/obb_tool.h` | — | Update access | — | No instances |
| `src/gui/sam_tool.h` | — | Update access | — | No instances |
| `src/skeleton.h` | — | Add overload | Remove old types | KeyPoints* gone |
| `src/skeleton.cpp` | — | Add overload | Remove old funcs | malloc/free gone |
| `src/types.h` | — | — | Remove YoloExportMode | Cleanup |
| `tests/test_annotation.cpp` | Add 6 tests | Update tests | Delete 15 tests | Full cycle |

### New files (only 2)

1. `src/annotation_csv.h` — all CSV I/O (permanent)
2. `src/annotation_v1.h` — temporary bridge (deleted in Phase C)

### Lines of code estimate

| Component | Estimated LOC |
|-----------|---------------|
| New `annotation.h` structs + helpers | ~120 |
| `annotation_csv.h` (save/load/convert) | ~350 |
| `annotation_v1.h` (temporary copy) | ~200 (deleted in Phase C) |
| Phase B consumer rewrites | ~400 net change (mostly mechanical) |
| New tests | ~250 |
| Deleted code (Phase C) | -800 to -1000 |
| **Net change** | **~-200 to -400 lines** |

---

## End of Plan

This document is the single source of truth for the annotation refactor. Each
phase is independently committable and testable. The converter ensures zero data
loss for existing projects. The final codebase will be smaller, have no
`std::vector<bool>`, no raw malloc/free, no bridge code, and a clean CSV format
with inline confidence and pandas-compatible headers.
