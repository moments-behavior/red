#pragma once
// annotation.h — Unified instance-based annotation data model (v2)
//
// Flat per-frame model: each FrameAnnotation has per-camera 2D keypoints,
// 3D keypoints, and optional extras (bbox, OBB, mask) behind unique_ptr.
//
// Replaces the v1 model that used InstanceAnnotation and flat Camera2D.

#include "types.h"
#include "json.hpp"
#include <algorithm>
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
// Axis-aligned bounding box, image coords (top-left origin, y-down).
// Each box carries its own class + instance so one camera view can hold
// several objects of different classes.
struct BBox {
    double x = 0, y = 0, w = 0, h = 0;  // top-left corner + size
    int category_id = 0;                // index into AnnotationConfig::class_names
    int instance_id = 0;
};

// Oriented bounding box, image coords (y-down; angle in radians).
struct OBB {
    double cx = 0, cy = 0, w = 0, h = 0, angle = 0;
    int category_id = 0;
    int instance_id = 0;
};

struct CameraExtras {
    std::vector<BBox> bboxes;
    std::vector<OBB>  obbs;

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
    bool has_bbox() const { return extras && !extras->bboxes.empty(); }
    bool has_obb()  const { return extras && !extras->obbs.empty();   }
    bool has_mask() const { return extras && extras->has_mask;  }
};

// ── All annotations for one frame ──
struct FrameAnnotation {
    u32 frame_number = 0;
    int instance_id  = 0;   // keypoint skeleton identity (boxes carry their own)
    int category_id  = 0;   // keypoint skeleton class (boxes carry their own)

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

// camera_names (optional) stamps each camera entry with "cam_name" so a load
// with a different camera set (e.g. after excluding a camera) maps by name
// rather than by index. Cameras in `excluded` are not written.
inline nlohmann::json annotations_to_json(
    const AnnotationMap &amap,
    const std::vector<std::string> &camera_names = {},
    const std::vector<std::string> &excluded = {}) {
    nlohmann::json root;
    root["version"] = 3;  // v3: per-camera "bboxes"/"obbs" lists (v2: single "bbox"/"obb")
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
            const bool named = c < camera_names.size();
            if (named && std::find(excluded.begin(), excluded.end(),
                                   camera_names[c]) != excluded.end())
                continue;

            nlohmann::json jc;
            jc["cam"] = (int)c;
            if (named) jc["cam_name"] = camera_names[c];

            if (!ext.bboxes.empty()) {
                nlohmann::json arr = nlohmann::json::array();
                for (const auto &b : ext.bboxes)
                    arr.push_back({{"xywh", {b.x, b.y, b.w, b.h}},
                                   {"category_id", b.category_id},
                                   {"instance_id", b.instance_id}});
                jc["bboxes"] = arr;
            }
            if (!ext.obbs.empty()) {
                nlohmann::json arr = nlohmann::json::array();
                for (const auto &o : ext.obbs)
                    arr.push_back({{"cxcywha", {o.cx, o.cy, o.w, o.h, o.angle}},
                                   {"category_id", o.category_id},
                                   {"instance_id", o.instance_id}});
                jc["obbs"] = arr;
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

            if (jc.size() > (named ? 2u : 1u)) // more than the cam id
                cams.push_back(jc);
        }

        if (!cams.empty())
            jf["cameras"] = cams;

        frames_arr.push_back(jf);
    }

    root["frames"] = frames_arr;
    return root;
}

inline void annotations_from_json(const nlohmann::json &root, AnnotationMap &amap,
                                  const std::vector<std::string> &camera_names = {}) {
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
            if (jc.contains("cam_name") && !camera_names.empty()) {
                auto it_n = std::find(camera_names.begin(), camera_names.end(),
                                      jc["cam_name"].get<std::string>());
                if (it_n == camera_names.end()) continue;  // camera not loaded
                c = (int)(it_n - camera_names.begin());
            }
            if (c < 0 || c >= (int)fa.cameras.size()) continue;
            auto &ext = fa.cameras[c].get_extras();

            if (jc.contains("bboxes")) {
                ext.bboxes.clear();
                for (const auto &jb : jc["bboxes"]) {
                    const auto &v = jb["xywh"];
                    BBox b;
                    b.x = v[0]; b.y = v[1]; b.w = v[2]; b.h = v[3];
                    b.category_id = jb.value("category_id", 0);
                    b.instance_id = jb.value("instance_id", 0);
                    ext.bboxes.push_back(b);
                }
            } else if (jc.contains("bbox")) {
                // v2: one box per camera, class/instance stored on the frame
                const auto &v = jc["bbox"];
                BBox b;
                b.x = v[0]; b.y = v[1]; b.w = v[2]; b.h = v[3];
                b.category_id = fa.category_id;
                b.instance_id = fa.instance_id;
                ext.bboxes = {b};
            }
            if (jc.contains("obbs")) {
                ext.obbs.clear();
                for (const auto &jo : jc["obbs"]) {
                    const auto &v = jo["cxcywha"];
                    OBB o;
                    o.cx = v[0]; o.cy = v[1]; o.w = v[2]; o.h = v[3]; o.angle = v[4];
                    o.category_id = jo.value("category_id", 0);
                    o.instance_id = jo.value("instance_id", 0);
                    ext.obbs.push_back(o);
                }
            } else if (jc.contains("obb")) {
                const auto &v = jc["obb"];
                OBB o;
                o.cx = v[0]; o.cy = v[1]; o.w = v[2]; o.h = v[3]; o.angle = v[4];
                o.category_id = fa.category_id;
                o.instance_id = fa.instance_id;
                ext.obbs = {o};
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
inline bool save_annotations_json(const AnnotationMap &amap, const std::string &folder,
                                  const std::vector<std::string> &camera_names = {},
                                  const std::vector<std::string> &excluded = {}) {
    auto j = annotations_to_json(amap, camera_names, excluded);
    if (j["frames"].empty()) return true; // nothing to save
    std::ofstream f(folder + "/annotations.json");
    if (!f) return false;
    f << j.dump(2);
    return true;
}

// Load extended annotations from JSON (call after loading keypoint CSVs)
inline bool load_annotations_json(AnnotationMap &amap, const std::string &folder,
                                  const std::vector<std::string> &camera_names = {}) {
    std::string path = folder + "/annotations.json";
    if (!std::filesystem::exists(path)) return true; // no extended data, ok
    try {
        std::ifstream f(path);
        nlohmann::json j;
        f >> j;
        annotations_from_json(j, amap, camera_names);
        return true;
    } catch (...) {
        return false;
    }
}

