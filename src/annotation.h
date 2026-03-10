#pragma once
// annotation.h — Unified instance-based annotation data model
//
// Replaces the flat KeyPoints* bags with a richer model where each frame
// contains instances (objects), and each instance can carry any combination
// of keypoints, bounding boxes, oriented bounding boxes, and masks.
//
// Backward-compatible: existing keypoint-only projects migrate trivially
// (one instance per frame, keypoints filled in, everything else empty).

#include "types.h"
#include "render.h"
#include "skeleton.h"
#include <map>
#include <string>
#include <vector>

// ── Sentinel value for "unlabeled" (matches existing CSV convention) ──
static constexpr double UNLABELED = 1E7;

// ── Per-camera 2D annotations for one instance ──
struct Camera2D {
    // Keypoints
    std::vector<tuple_d> keypoints;       // [num_nodes], UNLABELED if not set
    std::vector<bool>    kp_labeled;      // [num_nodes]
    std::vector<float>   kp_confidence;   // [num_nodes], 0 = manual, >0 = predicted

    // Axis-aligned bounding box (optional)
    double bbox_x = 0, bbox_y = 0, bbox_w = 0, bbox_h = 0;
    bool has_bbox = false;

    // Oriented bounding box (optional)
    double obb_cx = 0, obb_cy = 0, obb_w = 0, obb_h = 0, obb_angle = 0;
    bool has_obb = false;

    // Segmentation mask as polygon contours (optional)
    std::vector<std::vector<tuple_d>> mask_polygons;
    bool has_mask = false;

    // Active keypoint selection (UI state, not persisted)
    u32 active_id = 0;
};

// ── One annotated instance (object) within a frame ──
struct InstanceAnnotation {
    int instance_id = 0;    // object identity (for multi-animal tracking)
    int category_id = 0;    // class index into AnnotationConfig::class_names

    // 3D keypoints (triangulated from multi-view)
    std::vector<triple_d> kp3d;           // [num_nodes], UNLABELED sentinel
    std::vector<bool>     kp3d_triangulated; // [num_nodes]
    std::vector<float>    kp3d_confidence;   // [num_nodes]

    // 3D bounding box (optional, from multi-view triangulation)
    triple_d bbox3d_center = {0, 0, 0};
    triple_d bbox3d_size   = {0, 0, 0};
    bool has_bbox3d = false;

    // Per-camera 2D annotations
    std::vector<Camera2D> cameras;        // [num_cameras]
};

// ── All annotations for one frame ──
struct FrameAnnotation {
    u32 frame_number = 0;
    std::vector<InstanceAnnotation> instances;  // usually 1 (single animal)
};

// ── The main annotation container (replaces std::map<u32, KeyPoints*>) ──
using AnnotationMap = std::map<u32, FrameAnnotation>;

// AnnotationConfig lives in project.h (project-level annotation capabilities)

// ── Helpers ──

// Allocate a new InstanceAnnotation with the right sizes for keypoints
inline InstanceAnnotation make_instance(int num_nodes, int num_cameras,
                                        int instance_id = 0, int category_id = 0) {
    InstanceAnnotation inst;
    inst.instance_id = instance_id;
    inst.category_id = category_id;

    inst.kp3d.resize(num_nodes, {UNLABELED, UNLABELED, UNLABELED});
    inst.kp3d_triangulated.resize(num_nodes, false);
    inst.kp3d_confidence.resize(num_nodes, 0.0f);

    inst.cameras.resize(num_cameras);
    for (auto &cam : inst.cameras) {
        cam.keypoints.resize(num_nodes, {UNLABELED, UNLABELED});
        cam.kp_labeled.resize(num_nodes, false);
        cam.kp_confidence.resize(num_nodes, 0.0f);
    }
    return inst;
}

// Get-or-create a FrameAnnotation with a default single instance
inline FrameAnnotation &get_or_create_frame(AnnotationMap &amap, u32 frame,
                                             int num_nodes, int num_cameras) {
    auto it = amap.find(frame);
    if (it != amap.end()) return it->second;
    FrameAnnotation &fa = amap[frame];
    fa.frame_number = frame;
    fa.instances.push_back(make_instance(num_nodes, num_cameras));
    return fa;
}

// Check if any keypoint in the frame is labeled (any instance, any camera)
inline bool frame_has_any_labels(const FrameAnnotation &fa) {
    for (const auto &inst : fa.instances)
        for (const auto &cam : inst.cameras)
            for (size_t k = 0; k < cam.kp_labeled.size(); ++k)
                if (cam.kp_labeled[k]) return true;
    return false;
}

// Check if all keypoints are labeled across all cameras (fully labeled frame)
inline bool frame_fully_labeled(const FrameAnnotation &fa, int num_nodes,
                                 int num_cameras) {
    for (const auto &inst : fa.instances) {
        for (int c = 0; c < num_cameras && c < (int)inst.cameras.size(); ++c)
            for (int k = 0; k < num_nodes && k < (int)inst.cameras[c].kp_labeled.size(); ++k)
                if (!inst.cameras[c].kp_labeled[k]) return false;
    }
    return true;
}

// ── Migration from old KeyPoints* system ──

// Convert a single KeyPoints* to InstanceAnnotation
inline InstanceAnnotation instance_from_keypoints(const KeyPoints *kp,
                                                   const SkeletonContext &skel,
                                                   const RenderScene *scene) {
    int nn = skel.num_nodes;
    int nc = (int)scene->num_cams;
    InstanceAnnotation inst = make_instance(nn, nc);

    if (kp->kp3d) {
        for (int k = 0; k < nn; ++k) {
            inst.kp3d[k] = kp->kp3d[k].position;
            inst.kp3d_triangulated[k] = kp->kp3d[k].is_triangulated;
            inst.kp3d_confidence[k] = kp->kp3d[k].confidence;
        }
    }
    if (kp->kp2d) {
        for (int c = 0; c < nc; ++c) {
            inst.cameras[c].active_id = kp->active_id[c];
            for (int k = 0; k < nn; ++k) {
                inst.cameras[c].keypoints[k] = kp->kp2d[c][k].position;
                inst.cameras[c].kp_labeled[k] = kp->kp2d[c][k].is_labeled;
                inst.cameras[c].kp_confidence[k] = kp->kp2d[c][k].confidence;
            }
        }
    }
    return inst;
}

// Migrate entire keypoints_map to AnnotationMap
inline AnnotationMap migrate_keypoints_map(const std::map<u32, KeyPoints *> &km,
                                            const SkeletonContext &skel,
                                            const RenderScene *scene) {
    AnnotationMap amap;
    for (const auto &[frame, kp] : km) {
        if (!kp) continue;
        FrameAnnotation fa;
        fa.frame_number = frame;
        fa.instances.push_back(instance_from_keypoints(kp, skel, scene));
        amap[frame] = std::move(fa);
    }
    return amap;
}

// Refresh keypoint data from keypoints_map into an existing AnnotationMap,
// preserving bbox/OBB/mask data that annotation tools wrote.
// Call this after save_keypoints() to keep the two systems in sync.
inline void refresh_keypoints_in_amap(AnnotationMap &amap,
                                       const std::map<u32, KeyPoints *> &km,
                                       const SkeletonContext &skel,
                                       const RenderScene *scene) {
    int nn = skel.num_nodes;
    int nc = (int)scene->num_cams;
    for (const auto &[frame, kp] : km) {
        if (!kp) continue;
        auto it = amap.find(frame);
        if (it == amap.end()) {
            FrameAnnotation fa;
            fa.frame_number = frame;
            fa.instances.push_back(instance_from_keypoints(kp, skel, scene));
            amap[frame] = std::move(fa);
        } else {
            if (it->second.instances.empty())
                it->second.instances.push_back(instance_from_keypoints(kp, skel, scene));
            else {
                auto &inst = it->second.instances[0];
                if (kp->kp3d)
                    for (int k = 0; k < nn; ++k) {
                        inst.kp3d[k] = kp->kp3d[k].position;
                        inst.kp3d_triangulated[k] = kp->kp3d[k].is_triangulated;
                        inst.kp3d_confidence[k] = kp->kp3d[k].confidence;
                    }
                if (kp->kp2d)
                    for (int c = 0; c < nc; ++c) {
                        inst.cameras[c].active_id = kp->active_id[c];
                        for (int k = 0; k < nn; ++k) {
                            inst.cameras[c].keypoints[k] = kp->kp2d[c][k].position;
                            inst.cameras[c].kp_labeled[k] = kp->kp2d[c][k].is_labeled;
                            inst.cameras[c].kp_confidence[k] = kp->kp2d[c][k].confidence;
                        }
                    }
            }
        }
    }
    // Remove frames from amap that no longer exist in keypoints_map
    std::vector<u32> to_remove;
    for (const auto &[f, _] : amap)
        if (km.find(f) == km.end()) to_remove.push_back(f);
    for (u32 f : to_remove) amap.erase(f);
}

// Write back AnnotationMap to old KeyPoints* system (for backward compat with
// existing code paths that still use keypoints_map directly)
inline void sync_to_keypoints_map(const AnnotationMap &amap,
                                   std::map<u32, KeyPoints *> &km,
                                   const SkeletonContext &skel,
                                   RenderScene *scene) {
    // Only syncs keypoint data from instance 0 (single-animal compat)
    for (const auto &[frame, fa] : amap) {
        if (fa.instances.empty()) continue;
        const auto &inst = fa.instances[0];

        auto it = km.find(frame);
        KeyPoints *kp;
        if (it == km.end()) {
            kp = (KeyPoints *)malloc(sizeof(KeyPoints));
            allocate_keypoints(kp, scene, const_cast<SkeletonContext *>(&skel));
            km[frame] = kp;
        } else {
            kp = it->second;
        }

        int nn = skel.num_nodes;
        int nc = (int)scene->num_cams;

        if (kp->kp3d) {
            for (int k = 0; k < nn && k < (int)inst.kp3d.size(); ++k) {
                kp->kp3d[k].position = inst.kp3d[k];
                kp->kp3d[k].is_triangulated = inst.kp3d_triangulated[k];
                kp->kp3d[k].confidence = inst.kp3d_confidence[k];
            }
        }
        if (kp->kp2d) {
            for (int c = 0; c < nc && c < (int)inst.cameras.size(); ++c) {
                kp->active_id[c] = inst.cameras[c].active_id;
                for (int k = 0; k < nn && k < (int)inst.cameras[c].keypoints.size(); ++k) {
                    kp->kp2d[c][k].position = inst.cameras[c].keypoints[k];
                    kp->kp2d[c][k].is_labeled = inst.cameras[c].kp_labeled[k];
                    kp->kp2d[c][k].confidence = inst.cameras[c].kp_confidence[k];
                }
            }
        }
    }

    // Remove frames from km that are no longer in amap
    std::vector<u32> to_remove;
    for (const auto &[frame, kp] : km)
        if (amap.find(frame) == amap.end())
            to_remove.push_back(frame);
    for (u32 f : to_remove) {
        free_keypoints(km[f], scene);
        km.erase(f);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// JSON persistence for extended annotations (bbox, OBB, mask)
//
// Saved alongside the CSV keypoint files as `annotations.json`.
// Only writes entries that have bbox/obb/mask data — keypoints continue
// to use the existing CSV format for backward compatibility.
// ═══════════════════════════════════════════════════════════════════════════

#include "json.hpp"

inline nlohmann::json annotations_to_json(const AnnotationMap &amap) {
    nlohmann::json root;
    root["version"] = 1;
    nlohmann::json frames_arr = nlohmann::json::array();

    for (const auto &[fnum, fa] : amap) {
        bool has_extended = false;
        for (const auto &inst : fa.instances) {
            if (inst.has_bbox3d) { has_extended = true; break; }
            for (const auto &cam : inst.cameras) {
                if (cam.has_bbox || cam.has_obb || cam.has_mask) {
                    has_extended = true; break;
                }
            }
            if (has_extended) break;
        }
        if (!has_extended) continue;

        nlohmann::json jf;
        jf["frame"] = fnum;
        nlohmann::json insts = nlohmann::json::array();

        for (const auto &inst : fa.instances) {
            nlohmann::json ji;
            ji["instance_id"] = inst.instance_id;
            ji["category_id"] = inst.category_id;

            if (inst.has_bbox3d) {
                ji["bbox3d_center"] = {inst.bbox3d_center.x, inst.bbox3d_center.y, inst.bbox3d_center.z};
                ji["bbox3d_size"] = {inst.bbox3d_size.x, inst.bbox3d_size.y, inst.bbox3d_size.z};
            }

            nlohmann::json cams = nlohmann::json::array();
            for (size_t c = 0; c < inst.cameras.size(); ++c) {
                const auto &cam = inst.cameras[c];
                nlohmann::json jc;
                jc["cam"] = (int)c;

                if (cam.has_bbox) {
                    jc["bbox"] = {cam.bbox_x, cam.bbox_y, cam.bbox_w, cam.bbox_h};
                }
                if (cam.has_obb) {
                    jc["obb"] = {cam.obb_cx, cam.obb_cy, cam.obb_w, cam.obb_h, cam.obb_angle};
                }
                if (cam.has_mask) {
                    nlohmann::json polys = nlohmann::json::array();
                    for (const auto &poly : cam.mask_polygons) {
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
                ji["cameras"] = cams;

            insts.push_back(ji);
        }

        jf["instances"] = insts;
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

        for (const auto &ji : jf["instances"]) {
            int inst_id = ji.value("instance_id", 0);
            int cat_id = ji.value("category_id", 0);

            // Find matching instance
            InstanceAnnotation *target = nullptr;
            for (auto &inst : it->second.instances) {
                if (inst.instance_id == inst_id && inst.category_id == cat_id) {
                    target = &inst;
                    break;
                }
            }
            if (!target) continue;

            if (ji.contains("bbox3d_center") && ji.contains("bbox3d_size")) {
                auto &c = ji["bbox3d_center"];
                target->bbox3d_center = {c[0].get<double>(), c[1].get<double>(), c[2].get<double>()};
                auto &s = ji["bbox3d_size"];
                target->bbox3d_size = {s[0].get<double>(), s[1].get<double>(), s[2].get<double>()};
                target->has_bbox3d = true;
            }

            if (ji.contains("cameras")) {
                for (const auto &jc : ji["cameras"]) {
                    int c = jc["cam"].get<int>();
                    if (c >= (int)target->cameras.size()) continue;
                    auto &cam = target->cameras[c];

                    if (jc.contains("bbox")) {
                        auto &b = jc["bbox"];
                        cam.bbox_x = b[0]; cam.bbox_y = b[1];
                        cam.bbox_w = b[2]; cam.bbox_h = b[3];
                        cam.has_bbox = true;
                    }
                    if (jc.contains("obb")) {
                        auto &o = jc["obb"];
                        cam.obb_cx = o[0]; cam.obb_cy = o[1];
                        cam.obb_w = o[2]; cam.obb_h = o[3]; cam.obb_angle = o[4];
                        cam.has_obb = true;
                    }
                    if (jc.contains("mask")) {
                        cam.mask_polygons.clear();
                        for (const auto &jpoly : jc["mask"]) {
                            std::vector<tuple_d> poly;
                            for (const auto &jpt : jpoly)
                                poly.push_back({jpt[0].get<double>(), jpt[1].get<double>()});
                            cam.mask_polygons.push_back(std::move(poly));
                        }
                        cam.has_mask = !cam.mask_polygons.empty();
                    }
                }
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

// end of annotation.h
