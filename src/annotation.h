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
struct Keypoint2D {
    double x = UNLABELED;
    double y = UNLABELED;   // bottom-origin; see above
    float  confidence = 0.0f;

    // `exist` is presence: are there coordinates here at all.
    //
    // Two independent questions, so two axes:
    //
    //   AUTHORSHIP -- who decided this keypoint belongs here. `manual` and
    //   `predicted` are mutually exclusive; the setters keep them so.
    //
    //   COORDINATE ORIGIN -- `reprojected` says the numbers currently stored
    //   came from this frame's 3D. It is independent of authorship: a point
    //   YOU placed and then refreshed with T is both manual and reprojected,
    //   and must still export as `visible`. Collapsing the two is what made
    //   a T refresh erase the record that you had placed a point at all.
    // Renamed from `exist` deliberately: it answers only "are there
    // coordinates here", and the readers that matter want a second question
    // answered too -- see usable() below. The rename turned all ~70 call
    // sites into compile errors so each could be re-decided rather than
    // silently keeping the old meaning.
    bool has_pos     = false;
    // Who put this point here. Three states, held exclusive by the type
    // rather than by the setters remembering to clear each other.
    //
    //   Derived    red computed it by reprojecting this frame's 3D
    //   Manual     a person clicked it in this camera
    //   Predicted  a model produced it directly in this view
    //
    // `Derived` rather than `Unknown`: set_reprojected() is the only way a
    // point gets a position without an author, so for anything with has_pos
    // this state says exactly where the numbers came from. It is also the
    // default, which on a point with no position means nothing at all -- the
    // same way x and y mean nothing there.
    enum class Author : unsigned char { Derived, Manual, Predicted };
    Author author = Author::Derived;

    bool is_manual() const { return author == Author::Manual; }
    bool is_predicted() const { return author == Author::Predicted; }
    bool reprojected = false;   // the stored numbers came from the 3D

    // Independent of the above: an assessment that the point is not visible
    // here. It has no usable coordinates, so `exist` is false while this is
    // true -- but it keeps its AUTHOR, because deciding a part is hidden is
    // itself something a person or a model did. Without that, every occluded
    // row bucketed as tracked and a single one split a hand-labelled session
    // in two on export. This is tailcycle's keypoints.pq `missing` status.
    // Can you see the part in this image? Three answers, so one field with
    // three values rather than a pair of booleans that could contradict each
    // other. This is the format's own shape: vis2d is an int8, not a flag.
    //
    // Its own axis, separate from authorship and from where the coordinates
    // came from. Red used to answer it with `manual`, which is why importing
    // a tracked session marked every machine-made point hand-placed -- the
    // only way to get `visible` back out -- and then split the session in two
    // on export.
    //
    //   Unknown   nobody has judged this view      -> `projected`
    //   Observed  someone says the part is visible -> `visible`
    //   Occluded  someone says it is not           -> `missing`
    enum class Vis : unsigned char { Unknown, Observed, Occluded };
    Vis vis = Vis::Unknown;

    bool is_occluded() const { return vis == Vis::Occluded; }
    bool is_observed() const { return vis == Vis::Observed; }

    // There is a position here, and it is one you can act on: not an
    // assessment that the part is hidden in this view. Triangulation,
    // drawing and export all want this rather than has_pos -- an occluded
    // point keeps its coordinates (the overlay draws the cross from them)
    // but must not feed a solve or export as visible.
    bool usable() const { return has_pos && vis != Vis::Occluded; }

    // Occlusion is ONE flag and touches nothing else. It used to clear
    // has_pos too, which said the point had no coordinates while x/y sat
    // right there holding them -- and x/y is what the cross is drawn from.
    // Presence and visibility are separate questions; this answers only the
    // second. Authorship and coordinate origin are likewise untouched: a
    // point you placed is still yours after you judge it hidden, and one
    // whose numbers came from a solve still came from a solve.
    void set_occluded() { vis = Vis::Occluded; }

    // Back to what it was before. Merging the two booleans cost one thing: an
    // Observed that M overwrote cannot be read back. It is reconstructible
    // though -- placing a point IS an observation, so a manual point returns
    // to Observed and anything else to Unknown, which is what it was.
    void clear_occluded() {
        vis = is_manual() ? Vis::Observed : Vis::Unknown;
    }

    // Authorship. A fresh placement also resets the coordinate origin: these
    // numbers came from the click, not from a solve.
    void set_manual() {
        has_pos = true; author = Author::Manual; vis = Vis::Observed;
        reprojected = false;
    }
    void set_predicted(float conf = 0.0f) {
        has_pos = true; author = Author::Predicted; confidence = conf;
        reprojected = false; vis = Vis::Unknown;
    }

    // Coordinate origin only -- authorship is deliberately left alone, so a
    // refreshed hand label stays manual.
    void set_reprojected() {
        has_pos = true; reprojected = true;
        if (vis == Vis::Occluded) vis = Vis::Unknown;
    }
    void clear() { *this = Keypoint2D{}; }
};

inline bool keypoint2d_assessed(const Keypoint2D &kp) {
    return kp.has_pos || kp.is_occluded();
}


// ── 3D label provenance ──
// Tracks where a Keypoint3D's values came from.
// Values 2 (HybridNet) and 3 (Manual) were removed: nothing ever produced
// them. red has no UI for placing a 3D point directly, and HybridNet output
// arrives through set_predicted(). The numbering is left alone so the two are
// not silently reused -- the 3D origin flags are in-memory only and never serialised,
// but a reader comparing this against older code should see the gap.
struct Keypoint3D {
    double x = UNLABELED;
    double y = UNLABELED;
    double z = UNLABELED;
    float  confidence = 0.0f;

    // Is there a position here. No `usable()` counterpart, and no rename to
    // has_pos: the 2D split exists because an occluded point keeps its
    // coordinates while ceasing to be usable, and the 3D layer has no
    // occlusion to model. Should it ever gain one -- points3d.pq does carry a
    // `missing` status (§8) that red cannot currently express -- this becomes
    // has_pos and the pair comes with it.
    bool exist = false;
    // Where this 3D point came from. Same shape as Keypoint2D::author, with
    // the values the 3D layer actually has -- red cannot hand-place a 3D
    // point, so `Manual` has no counterpart here.
    //
    //   Unknown       not established
    //   Triangulated  DLT-solved from 2D, here or by whoever made the dataset
    //   Predicted     a model produced it
    enum class Origin : unsigned char { Unknown, Triangulated, Predicted };
    Origin origin = Origin::Unknown;

    bool is_triangulated() const { return origin == Origin::Triangulated; }
    bool is_predicted() const { return origin == Origin::Predicted; }

    void set_triangulated(float conf = 1.0f) {
        exist = true; origin = Origin::Triangulated; confidence = conf;
    }
    void set_predicted(float conf = 1.0f) {
        exist = true; origin = Origin::Predicted; confidence = conf;
    }
    void clear() { *this = Keypoint3D{}; }
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
};

// ── Per-camera annotation for one frame ──
struct CameraAnnotation {
    std::vector<Keypoint2D> keypoints;   // [num_nodes]
    u32 active_id = 0;                   // UI state: selected keypoint index

    // Extras (bbox/OBB) — lazily allocated
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
// ── The main annotation container ──
//
// A frame holds one FrameAnnotation per animal. FrameAnnotation was always the
// per-animal record -- it carries instance_id -- it was simply stored one to a
// frame, so a second animal had nowhere to go.
//
// Order is the instance order; instance_id is the identity that travels with
// the animal across frames and views, and is what a tailcycle session's
// `animal_id` maps onto. Index and identity are deliberately not the same
// thing: an animal that appears late must keep its id.
using FrameInstances = std::vector<FrameAnnotation>;
using AnnotationMap = std::map<u32, FrameInstances>;

// INVARIANT: a frame present in the map has at least one instance. Callers use
// front() for "the animal being labelled"; an empty vector would make that
// undefined. get_or_create_frame maintains this -- prefer it to amap[frame],
// which default-constructs an empty list.

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

// ── Instance lookup ──
//
// Most of red works on one animal at a time, so these take an instance and
// default to the first. A caller that means "the animal being labelled" passes
// the active instance; a caller that means "every animal" iterates.

// The instance with this id, or nullptr. Prefer this to indexing: the id is
// stable across frames, the index is not.
inline FrameAnnotation *find_instance(FrameInstances &fis, int instance_id) {
    for (auto &fa : fis)
        if (fa.instance_id == instance_id) return &fa;
    return nullptr;
}
inline const FrameAnnotation *find_instance(const FrameInstances &fis, int instance_id) {
    for (const auto &fa : fis)
        if (fa.instance_id == instance_id) return &fa;
    return nullptr;
}

// The frame's instances, or an empty list if the frame has none.
inline const FrameInstances &instances_at(const AnnotationMap &amap, u32 frame) {
    static const FrameInstances empty;
    const auto it = amap.find(frame);
    return it == amap.end() ? empty : it->second;
}

// Get-or-create a FrameAnnotation with default sizes
// Get-or-create one animal's annotation for a frame. Defaults to instance 0,
// which is every existing project and every caller that predates multi-animal.
inline FrameAnnotation &get_or_create_frame(AnnotationMap &amap, u32 frame,
                                            int num_nodes, int num_cameras,
                                            int instance_id = 0) {
    FrameInstances &fis = amap[frame];
    if (FrameAnnotation *fa = find_instance(fis, instance_id)) return *fa;
    fis.push_back(make_frame(num_nodes, num_cameras, frame, instance_id));
    return fis.back();
}

// The animal being edited, clamped into range. A frame may hold fewer
// instances than the UI's index -- switching frames must not put the editor
// out of bounds, and silently editing the wrong animal would be worse than
// falling back to the first.
inline FrameAnnotation &instance_or_first(FrameInstances &fis, int index) {
    if (index > 0 && index < (int)fis.size()) return fis[(size_t)index];
    return fis.front();
}
inline const FrameAnnotation &instance_or_first(const FrameInstances &fis, int index) {
    if (index > 0 && index < (int)fis.size()) return fis[(size_t)index];
    return fis.front();
}

// Whole-frame versions of the per-animal predicates below: true when ANY
// animal in the frame qualifies.
inline bool frame_has_any_labels(const FrameAnnotation &fa);
inline bool frame_has_any_keypoints(const FrameAnnotation &fa);
inline bool frame_has_any_manual_labels(const FrameAnnotation &fa);

inline bool any_instance_has_labels(const FrameInstances &fis) {
    for (const auto &fa : fis) if (frame_has_any_labels(fa)) return true;
    return false;
}
inline bool any_instance_has_keypoints(const FrameInstances &fis) {
    for (const auto &fa : fis) if (frame_has_any_keypoints(fa)) return true;
    return false;
}
inline bool any_instance_has_manual_labels(const FrameInstances &fis) {
    for (const auto &fa : fis) if (frame_has_any_manual_labels(fa)) return true;
    return false;
}

// Check if the frame has any annotation data (keypoints or bboxes)
inline bool frame_has_any_labels(const FrameAnnotation &fa) {
    for (const auto &cam : fa.cameras) {
        for (const auto &kp : cam.keypoints)
            if (keypoint2d_assessed(kp)) return true;
        if (cam.has_bbox() || cam.has_obb()) return true;
    }
    return false;
}

// Check if any keypoint in the frame is assessed (placed or explicitly
// occluded) in any camera.
inline bool frame_has_any_keypoints(const FrameAnnotation &fa) {
    for (const auto &cam : fa.cameras)
        for (const auto &kp : cam.keypoints)
            if (keypoint2d_assessed(kp)) return true;
    return false;
}

// Check if the frame has anything a user manually provided or is actively
// correcting: a hand-placed 2D keypoint, a hand-edited 3D keypoint, or a
// promoted frame awaiting correction. Used to gate destructive operations
// (e.g. switching skeletons, gui/switch_skeleton_window.h) that re-index
// every keypoint by node position and would silently corrupt this data.
inline bool frame_has_any_manual_labels(const FrameAnnotation &fa) {
    if (fa.needs_improvement) return true;
    // No 3D check: red has no way to hand-place a 3D point, so any frame with
    // hand-made data is caught by the 2D pass below or by needs_improvement.
    for (const auto &cam : fa.cameras)
        for (const auto &kp : cam.keypoints)
            // Same split bucket_2d makes. An occlusion is hand-made unless
            // a model claimed it, so it is `!predicted` here rather than
            // `manual` -- marking a never-placed node hidden sets no author
            // at all, and that frame is still your work.
            if ((kp.is_occluded() && !kp.is_predicted()) ||
                (kp.usable() && kp.is_manual())) return true;
    return false;
}

// Whole-project version of frame_has_any_manual_labels.
inline bool project_has_any_manual_labels(const AnnotationMap &amap) {
    for (const auto &[frame, fis] : amap)
        if (any_instance_has_manual_labels(fis)) return true;
    return false;
}

// Check if all keypoints on all cameras are assessed (visible or explicitly
// occluded).
inline bool frame_is_complete(const FrameAnnotation &fa) {
    if (fa.cameras.empty()) return false;
    for (const auto &cam : fa.cameras)
        for (const auto &kp : cam.keypoints)
            if (!keypoint2d_assessed(kp)) return false;
    return true;
}

// Check if all keypoints that are visible in at least one camera are
// triangulated. A point assessed as missing in every camera has no 3D point to
// triangulate and therefore does not make a frame incomplete.
inline bool frame_is_fully_triangulated(const FrameAnnotation &fa, int num_nodes) {
    for (int k = 0; k < num_nodes; ++k) {
        bool visible = false;
        for (const auto &cam : fa.cameras)
            if (k < (int)cam.keypoints.size() && cam.keypoints[k].usable()) {
                visible = true;
                break;
            }
        if (visible && (k >= (int)fa.kp3d.size() || !fa.kp3d[k].exist))
            return false;
    }
    return true;
}

// How far along a frame's keypoint labelling is.
//
// Shared by the Labeling Tool's timeline and the Frame Buffer list, which used
// to classify independently and disagree: the Frame Buffer lumped
// Triangulated and Untriangulated into one "partial", so the same frame read
// as two different states depending on which panel you looked at.
//
// is_2d / has_skeleton / num_cams are passed rather than a ProjectManager so
// this stays where the predicates it builds on already live.
enum class KpProgress {
    None = 0,        // no keypoints on this frame at all
    Untriangulated,  // some placed keypoint has no 3D yet
    Triangulated,    // every placed keypoint is triangulated, not all placed
    // Every keypoint placed in EVERY camera, and triangulated.
    //
    // Read it as "nothing is missing from any view", not as "this frame is
    // done": a keypoint the animal's own body hides from one camera can never
    // be placed there, so a frame that is finished as far as anyone can take
    // it still sits in Triangulated below. The legend's tooltip says so.
    Complete,
};

inline KpProgress frame_kp_progress(const FrameAnnotation &fa, int num_nodes,
                                    int num_cams, bool is_2d,
                                    bool has_skeleton) {
    if (!frame_has_any_keypoints(fa)) return KpProgress::None;
    if (!has_skeleton) return KpProgress::Untriangulated;
    if (is_2d)
        return frame_is_complete(fa) ? KpProgress::Complete
                                     : KpProgress::Untriangulated;
    if (num_cams > 1 && frame_is_complete(fa) &&
        frame_is_fully_triangulated(fa, num_nodes))
        return KpProgress::Complete;
    // Triangulated iff every placed node (labeled in >=1 camera) is
    // triangulated. Triangulated implies placed, so this means the placed and
    // triangulated sets coincide.
    int placed = 0, placed_untriangulated = 0;
    for (int n = 0; n < num_nodes; ++n) {
        bool node_placed = false;
        for (const auto &cam : fa.cameras)
            if (n < (int)cam.keypoints.size() &&
                keypoint2d_assessed(cam.keypoints[n])) {
                node_placed = true;
                break;
            }
        bool node_tri = n < (int)fa.kp3d.size() && fa.kp3d[n].exist;
        bool node_visible = false;
        for (const auto &cam : fa.cameras)
            if (n < (int)cam.keypoints.size() && cam.keypoints[n].usable()) {
                node_visible = true;
                break;
            }
        if (node_placed) {
            ++placed;
            // A node missing in every camera has no 3D observation to solve.
            if (node_visible && !node_tri) ++placed_untriangulated;
        }
    }
    if (placed > 0 && placed_untriangulated == 0) return KpProgress::Triangulated;
    return KpProgress::Untriangulated;
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

    for (const auto &[fnum, fis] : amap)
      for (const auto &fa : fis) {
        // Serialize frames that carry extended (extras) data OR a needs-fix flag
        // OR a single-view midline constraint.
        bool has_extended = fa.needs_improvement || fa.midline.has_line;
        for (const auto &cam : fa.cameras) {
            if (cam.has_bbox() || cam.has_obb()) {
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

        // Match the instance this record belongs to. A v2 file written before
        // multi-animal has one record per frame with instance_id 0, which is
        // also what the CSV loader created, so it lands on the right one.
        const int inst = jf.contains("instance_id") ? jf["instance_id"].get<int>() : 0;
        FrameAnnotation *fap = find_instance(it->second, inst);
        if (!fap) continue;
        auto &fa = *fap;
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

