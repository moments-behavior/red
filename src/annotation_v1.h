// annotation_v1.h — TEMPORARY bridge for old KeyPoints* ↔ new AnnotationMap.
// DELETE this file in Phase C.
//
// These functions let old code (that still uses KeyPoints*) interoperate with
// the new AnnotationMap during the transition. The old Camera2D / InstanceAnnotation
// types are gone — these functions bridge directly from KeyPoints* to the new model.

#pragma once

#include "types.h"
#include "skeleton.h"
#include "render.h"
#include <map>
#include <vector>

// Forward-declared from annotation.h (already included before this file)
// struct Keypoint2D; struct Keypoint3D; struct CameraAnnotation;
// struct FrameAnnotation; using AnnotationMap;

// ── Convert KeyPoints* → new FrameAnnotation ──

inline FrameAnnotation frame_from_keypoints(const KeyPoints *kp,
                                              const SkeletonContext &skel,
                                              const RenderScene *scene) {
    int nn = skel.num_nodes;
    int nc = (int)scene->num_cams;
    FrameAnnotation fa = make_frame(nn, nc, 0);

    if (kp->kp3d) {
        for (int k = 0; k < nn; ++k) {
            fa.kp3d[k].x = kp->kp3d[k].position.x;
            fa.kp3d[k].y = kp->kp3d[k].position.y;
            fa.kp3d[k].z = kp->kp3d[k].position.z;
            fa.kp3d[k].triangulated = kp->kp3d[k].is_triangulated;
            fa.kp3d[k].confidence = kp->kp3d[k].confidence;
        }
    }
    if (kp->kp2d) {
        for (int c = 0; c < nc; ++c) {
            fa.cameras[c].active_id = kp->active_id[c];
            for (int k = 0; k < nn; ++k) {
                fa.cameras[c].keypoints[k].x = kp->kp2d[c][k].position.x;
                fa.cameras[c].keypoints[k].y = kp->kp2d[c][k].position.y;
                fa.cameras[c].keypoints[k].labeled = kp->kp2d[c][k].is_labeled;
                fa.cameras[c].keypoints[k].confidence = kp->kp2d[c][k].confidence;
            }
        }
    }
    return fa;
}

// ── Migrate entire keypoints_map → new AnnotationMap ──

inline AnnotationMap migrate_keypoints_map(const std::map<u32, KeyPoints *> &km,
                                            const SkeletonContext &skel,
                                            const RenderScene *scene) {
    AnnotationMap amap;
    for (const auto &[frame, kp] : km) {
        if (!kp) continue;
        FrameAnnotation fa = frame_from_keypoints(kp, skel, scene);
        fa.frame_number = frame;
        amap[frame] = std::move(fa);
    }
    return amap;
}

// ── Refresh: sync keypoints_map edits into existing AnnotationMap ──
// Preserves extras (bbox/obb/mask) that annotation tools wrote.

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
            FrameAnnotation fa = frame_from_keypoints(kp, skel, scene);
            fa.frame_number = frame;
            amap[frame] = std::move(fa);
        } else {
            auto &fa = it->second;
            if (kp->kp3d)
                for (int k = 0; k < nn; ++k) {
                    fa.kp3d[k].x = kp->kp3d[k].position.x;
                    fa.kp3d[k].y = kp->kp3d[k].position.y;
                    fa.kp3d[k].z = kp->kp3d[k].position.z;
                    fa.kp3d[k].triangulated = kp->kp3d[k].is_triangulated;
                    fa.kp3d[k].confidence = kp->kp3d[k].confidence;
                }
            if (kp->kp2d)
                for (int c = 0; c < nc; ++c) {
                    fa.cameras[c].active_id = kp->active_id[c];
                    for (int k = 0; k < nn; ++k) {
                        fa.cameras[c].keypoints[k].x = kp->kp2d[c][k].position.x;
                        fa.cameras[c].keypoints[k].y = kp->kp2d[c][k].position.y;
                        fa.cameras[c].keypoints[k].labeled = kp->kp2d[c][k].is_labeled;
                        fa.cameras[c].keypoints[k].confidence = kp->kp2d[c][k].confidence;
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

// ── Populate KeyPoints* from AnnotationMap (reverse bridge for Phase B transition) ──

inline void populate_keypoints_from_amap(std::map<u32, KeyPoints *> &km,
                                          const AnnotationMap &amap,
                                          SkeletonContext *skeleton,
                                          RenderScene *scene) {
    for (const auto &[frame, fa] : amap) {
        if (km.find(frame) == km.end()) {
            KeyPoints *kp = (KeyPoints *)malloc(sizeof(KeyPoints));
            allocate_keypoints(kp, scene, skeleton);
            km[frame] = kp;
        }
        KeyPoints *kp = km[frame];
        int nn = skeleton->num_nodes;
        int nc = (int)scene->num_cams;

        for (int k = 0; k < nn; ++k) {
            kp->kp3d[k].position.x = fa.kp3d[k].x;
            kp->kp3d[k].position.y = fa.kp3d[k].y;
            kp->kp3d[k].position.z = fa.kp3d[k].z;
            kp->kp3d[k].is_triangulated = fa.kp3d[k].triangulated;
            kp->kp3d[k].confidence = fa.kp3d[k].confidence;
        }
        for (int c = 0; c < nc; ++c) {
            kp->active_id[c] = fa.cameras[c].active_id;
            for (int k = 0; k < nn; ++k) {
                kp->kp2d[c][k].position.x = fa.cameras[c].keypoints[k].x;
                kp->kp2d[c][k].position.y = fa.cameras[c].keypoints[k].y;
                kp->kp2d[c][k].is_labeled = fa.cameras[c].keypoints[k].labeled;
                kp->kp2d[c][k].confidence = fa.cameras[c].keypoints[k].confidence;
            }
        }
    }
}

// ── Legacy compatibility aliases ──
// Old code used instance_from_keypoints() — redirect to new function
inline FrameAnnotation instance_from_keypoints(const KeyPoints *kp,
                                                const SkeletonContext &skel,
                                                const RenderScene *scene) {
    return frame_from_keypoints(kp, skel, scene);
}

// end of annotation_v1.h
