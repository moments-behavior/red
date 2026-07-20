#pragma once
// annotation.h — Unified instance-based annotation data model (v2)
//
// Flat per-frame model: each FrameAnnotation has per-camera 2D keypoints,
// 3D keypoints, and optional extras (bbox, OBB, mask) behind unique_ptr.
//
// Replaces the v1 model that used InstanceAnnotation and flat Camera2D.

#include "types.h"
#include "json.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <tuple>

// ── Sentinel value for "unlabeled" (matches existing CSV convention) ──
static constexpr double UNLABELED = 1E7;

// ── Label provenance ──
enum class LabelSource : int {
    Manual    = 0,
    Predicted = 1,
    Imported  = 2
};

// ── Per-keypoint 2D annotation ──
struct Keypoint2D {
    double x = UNLABELED;
    double y = UNLABELED;
    bool   labeled    = false;
    float  confidence = 0.0f;
    LabelSource source = LabelSource::Manual;
};

// ── 3D label provenance ──
// Tracks where a Keypoint3D's values came from. Combined with `reviewed`,
// this drives the active-learning loop: predictions need review, approved or
// edited points become training-quality. See dev_docs / claude_refresher for
// the state-transition table.
enum class Kp3DSource : int {
    None         = 0,  // no 3D values yet
    Triangulated = 1,  // DLT-solved from 2D labels
    HybridNet    = 2,  // direct 3D prediction by HybridNet
    Manual       = 3,  // user placed/edited 3D directly in the viewer
    Imported     = 4,  // external CSV/JSON import
};

// ── Per-keypoint 3D annotation ──
struct Keypoint3D {
    double x = UNLABELED;
    double y = UNLABELED;
    double z = UNLABELED;
    bool   triangulated = false;             // legacy presence flag, kept in sync with source != None
    Kp3DSource source = Kp3DSource::None;    // immediate provenance of the values
    bool   reviewed = false;                 // user signed off (approved or edited)
    float  confidence   = 0.0f;

    // Setter helpers keep `triangulated` (legacy bool) and `source` in sync.
    // Migrate new write sites to these; legacy reads of `.triangulated` keep
    // working unchanged.
    void set_triangulated(float conf = 1.0f) {
        source = Kp3DSource::Triangulated;
        triangulated = true;
        confidence = conf;
    }
    void set_hybridnet(float conf) {
        source = Kp3DSource::HybridNet;
        triangulated = true;
        reviewed = false;  // freshly predicted; awaits user review
        confidence = conf;
    }
    void set_manual() {
        source = Kp3DSource::Manual;
        triangulated = true;
        reviewed = true;   // user-placed = implicitly reviewed
        confidence = 1.0f;
    }
    void set_imported(float conf = 1.0f) {
        source = Kp3DSource::Imported;
        triangulated = true;
        reviewed = false;
        confidence = conf;
    }
    void approve() { reviewed = true; }      // accept current values without changing them
    void clear() {
        source = Kp3DSource::None;
        triangulated = false;
        reviewed = false;
        confidence = 0.0f;
    }
};

// ── Optional per-camera extras (bbox, OBB, mask) ──
// Allocated on demand via unique_ptr in CameraAnnotation to keep the
// common keypoint-only case lightweight.
struct CameraExtras {
    // Axis-aligned bounding box
    double bbox_x = 0, bbox_y = 0, bbox_w = 0, bbox_h = 0;
    bool has_bbox = false;

    // Oriented bounding box
    double obb_cx = 0, obb_cy = 0, obb_w = 0, obb_h = 0, obb_angle = 0;
    bool has_obb = false;

    // Segmentation mask as polygon contours
    std::vector<std::vector<tuple_d>> mask_polygons;
    bool has_mask = false;
};

// ── Per-camera annotation for one frame ──
struct CameraAnnotation {
    std::vector<Keypoint2D> keypoints;   // [num_nodes]
    u32 active_id = 0;                   // UI state: selected keypoint index

    // Extras (bbox/OBB/mask) — lazily allocated
    std::unique_ptr<CameraExtras> extras;

    // Default + move constructors work. Copy must deep-copy extras.
    CameraAnnotation() = default;
    CameraAnnotation(CameraAnnotation &&) = default;
    CameraAnnotation &operator=(CameraAnnotation &&) = default;
    CameraAnnotation(const CameraAnnotation &o)
        : keypoints(o.keypoints), active_id(o.active_id),
          extras(o.extras ? std::make_unique<CameraExtras>(*o.extras) : nullptr) {}
    CameraAnnotation &operator=(const CameraAnnotation &o) {
        if (this != &o) {
            keypoints = o.keypoints;
            active_id = o.active_id;
            extras = o.extras ? std::make_unique<CameraExtras>(*o.extras) : nullptr;
        }
        return *this;
    }

    // Get-or-create accessor for extras
    CameraExtras &get_extras() {
        if (!extras) extras = std::make_unique<CameraExtras>();
        return *extras;
    }
    const CameraExtras &get_extras() const {
        static const CameraExtras empty;
        if (!extras) return empty;
        return *extras;
    }

    // Convenience queries
    bool has_bbox() const { return extras && extras->has_bbox; }
    bool has_obb()  const { return extras && extras->has_obb;  }
    bool has_mask() const { return extras && extras->has_mask;  }
};

// ── Single-view midline constraint ──
// Lets the user solve 3D for a midline structure (e.g. the 4-keypoint
// proboscis) from ONE side camera plus a 2-click line drawn in a top/line
// camera. The keypoints are labeled only in `keypoint_camera_id`; the line's
// two endpoints (ImPlot coords, y-up, matching Keypoint2D) live in
// `line_camera_id`. The line fixes the plane the midline lies in; each side ray
// is intersected with that plane to recover 3D. See red_math midline helpers.
// Small + by-value so FrameAnnotation stays trivially copyable.
struct MidlineConstraint {
    int  keypoint_camera_id = -1;   // side camera the keypoints are labeled in
    int  line_camera_id     = -1;   // camera the 2-click line is drawn in
    double p1x = 0, p1y = 0;        // line endpoint 1 (ImPlot coords, y-up)
    double p2x = 0, p2y = 0;        // line endpoint 2
    bool force_vertical = false;    // false: true preimage plane (default,
                                    //  ~3× more accurate); true: extrude the
                                    //  footprint along world up (regularized)
    bool has_line = false;          // both endpoints placed
};

// ── All annotations for one frame ──
struct FrameAnnotation {
    u32 frame_number = 0;
    int instance_id  = 0;   // object identity (for multi-animal tracking)
    int category_id  = 0;   // class index

    // Optional single-view midline solve constraint for this frame.
    MidlineConstraint midline;

    // Set when a predicted frame is promoted from the prediction store into the
    // Labeling Tool for manual correction. Surfaces the frame in the Labeling
    // Tool's "Needs Improvement" section and protects it from being overwritten
    // by a later Batch Predict. Cleared when the user marks it fixed.
    bool needs_improvement = false;

    // 3D keypoints (triangulated from multi-view)
    std::vector<Keypoint3D> kp3d;         // [num_nodes]

    // Per-camera 2D annotations
    std::vector<CameraAnnotation> cameras; // [num_cameras]
};

// ── The main annotation container ──
using AnnotationMap = std::map<u32, FrameAnnotation>;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

// Allocate a FrameAnnotation with the right sizes for keypoints
inline FrameAnnotation make_frame(int num_nodes, int num_cameras, u32 frame_number = 0,
                                  int instance_id = 0, int category_id = 0) {
    FrameAnnotation fa;
    fa.frame_number = frame_number;
    fa.instance_id  = instance_id;
    fa.category_id  = category_id;

    fa.kp3d.resize(num_nodes);  // defaults: UNLABELED, triangulated=false, confidence=0

    fa.cameras.resize(num_cameras);
    for (auto &cam : fa.cameras)
        cam.keypoints.resize(num_nodes); // defaults: UNLABELED, labeled=false, confidence=0, Manual

    return fa;
}

// Get-or-create a FrameAnnotation with default sizes
inline FrameAnnotation &get_or_create_frame(AnnotationMap &amap, u32 frame,
                                             int num_nodes, int num_cameras) {
    auto it = amap.find(frame);
    if (it != amap.end()) return it->second;
    FrameAnnotation &fa = amap[frame];
    fa = make_frame(num_nodes, num_cameras, frame);
    return fa;
}

// Check if the frame has any annotation data (keypoints, masks, or bboxes)
inline bool frame_has_any_labels(const FrameAnnotation &fa) {
    for (const auto &cam : fa.cameras) {
        for (const auto &kp : cam.keypoints)
            if (kp.labeled) return true;
        if (cam.has_mask() || cam.has_bbox() || cam.has_obb()) return true;
    }
    return false;
}

// Check if any keypoint in the frame is labeled (any camera)
inline bool frame_has_any_keypoints(const FrameAnnotation &fa) {
    for (const auto &cam : fa.cameras)
        for (const auto &kp : cam.keypoints)
            if (kp.labeled) return true;
    return false;
}

// Check if the frame has anything a user manually provided or is actively
// correcting: a hand-placed 2D keypoint, a hand-edited 3D keypoint, or a
// promoted frame awaiting correction. Used to gate destructive operations
// (e.g. switching skeletons, gui/switch_skeleton_window.h) that re-index
// every keypoint by node position and would silently corrupt this data.
inline bool frame_has_any_manual_labels(const FrameAnnotation &fa) {
    if (fa.needs_improvement) return true;
    for (const auto &kp3 : fa.kp3d)
        if (kp3.source == Kp3DSource::Manual) return true;
    for (const auto &cam : fa.cameras)
        for (const auto &kp : cam.keypoints)
            if (kp.labeled && kp.source == LabelSource::Manual) return true;
    return false;
}

// Whole-project version of frame_has_any_manual_labels.
inline bool project_has_any_manual_labels(const AnnotationMap &amap) {
    for (const auto &[frame, fa] : amap)
        if (frame_has_any_manual_labels(fa)) return true;
    return false;
}

// Check if any camera has a mask on this frame
inline bool frame_has_any_masks(const FrameAnnotation &fa) {
    for (const auto &cam : fa.cameras)
        if (cam.has_mask()) return true;
    return false;
}

// Check if all keypoints on all cameras are labeled
inline bool frame_is_complete(const FrameAnnotation &fa) {
    if (fa.cameras.empty()) return false;
    for (const auto &cam : fa.cameras)
        for (const auto &kp : cam.keypoints)
            if (!kp.labeled) return false;
    return true;
}

// Check if all 3D keypoints are triangulated
inline bool frame_is_fully_triangulated(const FrameAnnotation &fa, int num_nodes) {
    for (int k = 0; k < num_nodes; ++k)
        if (k >= (int)fa.kp3d.size() || !fa.kp3d[k].triangulated)
            return false;
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// JSON persistence for extended annotations (bbox, OBB, mask)
//
// Saved alongside the CSV keypoint files as `annotations.json`.
// Only writes entries that have extras data (bbox/obb/mask) — keypoints
// continue to use the existing CSV format for backward compatibility.
// ═══════════════════════════════════════════════════════════════════════════

inline nlohmann::json annotations_to_json(const AnnotationMap &amap) {
    nlohmann::json root;
    root["version"] = 2;
    nlohmann::json frames_arr = nlohmann::json::array();

    for (const auto &[fnum, fa] : amap) {
        // Serialize frames that carry extended (extras) data OR a needs-fix flag
        // OR a single-view midline constraint.
        bool has_extended = fa.needs_improvement || fa.midline.has_line;
        for (const auto &cam : fa.cameras) {
            if (cam.has_bbox() || cam.has_obb() || cam.has_mask()) {
                has_extended = true;
                break;
            }
        }
        if (!has_extended) continue;

        nlohmann::json jf;
        jf["frame"] = fnum;
        jf["instance_id"] = fa.instance_id;
        jf["category_id"] = fa.category_id;
        if (fa.needs_improvement) jf["needs_improvement"] = true;

        if (fa.midline.has_line) {
            const auto &m = fa.midline;
            jf["midline"] = {
                {"keypoint_camera_id", m.keypoint_camera_id},
                {"line_camera_id", m.line_camera_id},
                {"p1", {m.p1x, m.p1y}},
                {"p2", {m.p2x, m.p2y}},
                {"force_vertical", m.force_vertical},
            };
        }

        nlohmann::json cams = nlohmann::json::array();
        for (size_t c = 0; c < fa.cameras.size(); ++c) {
            const auto &cam = fa.cameras[c];
            if (!cam.extras) continue;
            const auto &ext = *cam.extras;

            nlohmann::json jc;
            jc["cam"] = (int)c;

            if (ext.has_bbox) {
                jc["bbox"] = {ext.bbox_x, ext.bbox_y, ext.bbox_w, ext.bbox_h};
            }
            if (ext.has_obb) {
                jc["obb"] = {ext.obb_cx, ext.obb_cy, ext.obb_w, ext.obb_h, ext.obb_angle};
            }
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

            if (jc.size() > 1) // more than just "cam"
                cams.push_back(jc);
        }

        if (!cams.empty())
            jf["cameras"] = cams;

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
        if (it == amap.end()) continue; // only augment existing frames

        auto &fa = it->second;

        // Read instance/category IDs if present
        if (jf.contains("instance_id"))
            fa.instance_id = jf["instance_id"].get<int>();
        if (jf.contains("category_id"))
            fa.category_id = jf["category_id"].get<int>();
        if (jf.contains("needs_improvement"))
            fa.needs_improvement = jf["needs_improvement"].get<bool>();

        if (jf.contains("midline")) {
            const auto &jm = jf["midline"];
            auto &m = fa.midline;
            m.keypoint_camera_id = jm.value("keypoint_camera_id", -1);
            m.line_camera_id = jm.value("line_camera_id", -1);
            if (jm.contains("p1")) { m.p1x = jm["p1"][0]; m.p1y = jm["p1"][1]; }
            if (jm.contains("p2")) { m.p2x = jm["p2"][0]; m.p2y = jm["p2"][1]; }
            m.force_vertical = jm.value("force_vertical", false);
            m.has_line = true;
        }

        if (!jf.contains("cameras")) continue;

        for (const auto &jc : jf["cameras"]) {
            int c = jc["cam"].get<int>();
            if (c < 0 || c >= (int)fa.cameras.size()) continue;
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

// Save extended annotations to a JSON file alongside keypoint CSVs
inline bool save_annotations_json(const AnnotationMap &amap, const std::string &folder) {
    auto j = annotations_to_json(amap);
    if (j["frames"].empty()) return true; // nothing to save
    std::ofstream f(folder + "/annotations.json");
    if (!f) return false;
    f << j.dump(2);
    return true;
}

// Load extended annotations from JSON (call after loading keypoint CSVs)
inline bool load_annotations_json(AnnotationMap &amap, const std::string &folder) {
    std::string path = folder + "/annotations.json";
    if (!std::filesystem::exists(path)) return true; // no extended data, ok
    try {
        std::ifstream f(path);
        nlohmann::json j;
        f >> j;
        annotations_from_json(j, amap);
        return true;
    } catch (...) {
        return false;
    }
}

