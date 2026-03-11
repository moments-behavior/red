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

// ── Per-keypoint 3D annotation (triangulated) ──
struct Keypoint3D {
    double x = UNLABELED;
    double y = UNLABELED;
    double z = UNLABELED;
    bool   triangulated = false;
    float  confidence   = 0.0f;
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

// ── All annotations for one frame ──
struct FrameAnnotation {
    u32 frame_number = 0;
    int instance_id  = 0;   // object identity (for multi-animal tracking)
    int category_id  = 0;   // class index

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
        // Only serialize frames that have extended (extras) data
        bool has_extended = false;
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

