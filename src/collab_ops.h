#pragma once
// collab_ops.h -- the operation model for collaborative annotation.
//
// An op is one durable, attributable change to one addressable object. Ops are
// the only thing that crosses the network; the AnnotationMap is rebuilt from
// them locally. Two properties make eventual sync work:
//
//   * COMMUTATIVE + IDEMPOTENT. Applying a set of ops in any order, any number
//     of times, yields the same AnnotationMap. Replaying a log is a no-op, and
//     peers that received ops in different orders still converge.
//   * TOTALLY ORDERED per object. Each op carries a Lamport clock; the winner
//     for an object is the highest (lamport, peer). The peer id breaks ties
//     deterministically so every replica picks the SAME winner -- using wall
//     clock time here would not, since peer clocks disagree (the pump-events
//     work already documents seconds of drift on this rig).
//
// Nothing is ever discarded. Losing ops stay in the log; the History view
// shows them and "restore" emits a NEW op rather than rewriting the past.
//
// Positional indexing: fa.cameras[i] tracks pm.camera_names[i], and
// keypoints[n] tracks skeleton node n. An op naming node 7 is meaningless to a
// peer on a different skeleton, so RoomBinding below gates this at join time,
// and apply() defensively drops out-of-range ops rather than resizing.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "annotation.h"
#include "json.hpp"

namespace collab {

// =========================================================================
// Room binding
// =========================================================================

// The shape every peer in a room must agree on. Verified on join; a mismatch
// is a hard failure with a readable reason, the same stance
// AnnotationCSV::load_all takes on a skeleton mismatch and JarvisMerge takes
// on keypoint_names. Silently merging across skeletons would scramble labels
// in a way no user could later untangle.
struct RoomBinding {
    std::string skeleton_name;
    int num_nodes = 0;
    std::vector<std::string> camera_names;

    bool compatible_with(const RoomBinding &o, std::string *why = nullptr) const {
        if (skeleton_name != o.skeleton_name) {
            if (why)
                *why = "skeleton mismatch: room uses '" + skeleton_name +
                       "', this project uses '" + o.skeleton_name + "'";
            return false;
        }
        if (num_nodes != o.num_nodes) {
            if (why)
                *why = "keypoint count mismatch: room has " +
                       std::to_string(num_nodes) + ", this project has " +
                       std::to_string(o.num_nodes);
            return false;
        }
        if (camera_names != o.camera_names) {
            if (why)
                *why = "camera list mismatch: annotations are indexed by "
                       "camera position, so the names and their order must "
                       "match exactly";
            return false;
        }
        return true;
    }
};

inline void to_json(nlohmann::json &j, const RoomBinding &b) {
    j = nlohmann::json{{"skeleton_name", b.skeleton_name},
                       {"num_nodes", b.num_nodes},
                       {"camera_names", b.camera_names}};
}

inline void from_json(const nlohmann::json &j, RoomBinding &b) {
    b.skeleton_name = j.value("skeleton_name", std::string{});
    b.num_nodes = j.value("num_nodes", 0);
    b.camera_names = j.value("camera_names", std::vector<std::string>{});
}

// =========================================================================
// Op kinds
// =========================================================================

// "Clear" is deliberately not a separate kind: clearing a keypoint is a Set
// with labeled=false. One kind per object type keeps LWW uniform -- there is
// never a question of whether a Clear beats a Set, only which is newer.
enum class OpKind : int {
    FrameCreate    = 1,
    FrameDelete    = 2,
    FrameFlags     = 3,   // instance_id, category_id, needs_improvement
    Kp2dSet        = 10,
    Kp3dSet        = 11,
    BboxSet        = 20,
    ObbSet         = 21,
    MaskSet        = 22,
    MidlineSet     = 23,
    CommentPost    = 30,
    CommentResolve = 31,
};

inline const char *op_kind_name(OpKind k) {
    switch (k) {
        case OpKind::FrameCreate:    return "FrameCreate";
        case OpKind::FrameDelete:    return "FrameDelete";
        case OpKind::FrameFlags:     return "FrameFlags";
        case OpKind::Kp2dSet:        return "Kp2dSet";
        case OpKind::Kp3dSet:        return "Kp3dSet";
        case OpKind::BboxSet:        return "BboxSet";
        case OpKind::ObbSet:         return "ObbSet";
        case OpKind::MaskSet:        return "MaskSet";
        case OpKind::MidlineSet:     return "MidlineSet";
        case OpKind::CommentPost:    return "CommentPost";
        case OpKind::CommentResolve: return "CommentResolve";
    }
    return "Unknown";
}

inline bool op_kind_valid(int k) {
    switch (static_cast<OpKind>(k)) {
        case OpKind::FrameCreate: case OpKind::FrameDelete:
        case OpKind::FrameFlags:
        case OpKind::Kp2dSet: case OpKind::Kp3dSet:
        case OpKind::BboxSet: case OpKind::ObbSet: case OpKind::MaskSet:
        case OpKind::MidlineSet:
        case OpKind::CommentPost: case OpKind::CommentResolve:
            return true;
    }
    return false;
}

// =========================================================================
// Op
// =========================================================================

struct Op {
    std::string peer;      // stable random id for the authoring machine
    uint64_t    seq = 0;   // per-peer monotonic; (peer, seq) is globally unique
    uint64_t    lamport = 0;  // the ordering authority
    int64_t     wall_ms = 0;  // display only -- NEVER used for ordering
    OpKind      kind = OpKind::FrameCreate;
    uint32_t    frame = 0;
    int16_t     camera = -1;  // -1 for 3D / frame-level
    int16_t     node = -1;    // -1 for non-keypoint
    std::string obj_id;       // comments only
    nlohmann::json payload;

    // Display name of the author, carried along so History and Comments read
    // sensibly without a peer-directory lookup.
    std::string author;

    bool operator==(const Op &o) const {
        return peer == o.peer && seq == o.seq;
    }
};

// Strictly newer? Lamport first, peer id as the deterministic tiebreak.
inline bool op_newer(const Op &a, const Op &b) {
    if (a.lamport != b.lamport) return a.lamport > b.lamport;
    return a.peer > b.peer;
}

// =========================================================================
// Object addressing
// =========================================================================

// Which object an op targets. Two ops with the same key are in conflict and
// LWW picks between them; different keys never conflict.
struct ObjKey {
    int         cls = 0;      // an OpKind-derived class, not the kind itself
    uint32_t    frame = 0;
    int16_t     camera = -1;
    int16_t     node = -1;
    std::string id;

    bool operator<(const ObjKey &o) const {
        if (cls != o.cls) return cls < o.cls;
        if (frame != o.frame) return frame < o.frame;
        if (camera != o.camera) return camera < o.camera;
        if (node != o.node) return node < o.node;
        return id < o.id;
    }
    bool operator==(const ObjKey &o) const {
        return cls == o.cls && frame == o.frame && camera == o.camera &&
               node == o.node && id == o.id;
    }
};

// FrameCreate and FrameDelete address the SAME object -- "does this frame
// exist" -- so they compete under LWW rather than both applying.
// CommentPost and CommentResolve likewise both address the comment.
inline int obj_class_of(OpKind k) {
    switch (k) {
        case OpKind::FrameCreate:
        case OpKind::FrameDelete:    return 1;   // frame existence
        case OpKind::FrameFlags:     return 2;
        case OpKind::Kp2dSet:        return 3;
        case OpKind::Kp3dSet:        return 4;
        case OpKind::BboxSet:        return 5;
        case OpKind::ObbSet:         return 6;
        case OpKind::MaskSet:        return 7;
        case OpKind::MidlineSet:     return 8;
        case OpKind::CommentPost:    return 9;
        // Resolve addresses a DIFFERENT object than the comment body: sharing
        // a class would make "resolve" and "edit text" compete under LWW, and
        // resolving a thread would silently blank its text.
        case OpKind::CommentResolve: return 10;
    }
    return 0;
}

inline ObjKey key_of(const Op &op) {
    ObjKey k;
    k.cls = obj_class_of(op.kind);
    k.frame = op.frame;
    k.camera = op.camera;
    k.node = op.node;
    k.id = op.obj_id;
    return k;
}

// =========================================================================
// Lamport clock
// =========================================================================

struct LamportClock {
    uint64_t value = 0;

    uint64_t tick() { return ++value; }

    // Called for every op received, so a local edit made after seeing a remote
    // edit is ordered strictly after it.
    void observe(uint64_t remote) {
        if (remote > value) value = remote;
    }
};

// =========================================================================
// JSON
// =========================================================================

inline void to_json(nlohmann::json &j, const Op &op) {
    j = nlohmann::json{{"peer", op.peer},
                       {"seq", op.seq},
                       {"lamport", op.lamport},
                       {"wall_ms", op.wall_ms},
                       {"kind", static_cast<int>(op.kind)},
                       {"frame", op.frame},
                       {"payload", op.payload}};
    if (op.camera >= 0) j["camera"] = op.camera;
    if (op.node >= 0) j["node"] = op.node;
    if (!op.obj_id.empty()) j["obj_id"] = op.obj_id;
    if (!op.author.empty()) j["author"] = op.author;
}

// Returns false on anything structurally unusable. Every op arriving from the
// network or from a possibly-torn log goes through this, so it never throws
// and never trusts a field's presence.
inline bool op_from_json(const nlohmann::json &j, Op &op) {
    if (!j.is_object()) return false;

    const int kind = j.value("kind", -1);
    if (!op_kind_valid(kind)) return false;

    op.kind = static_cast<OpKind>(kind);
    op.peer = j.value("peer", std::string{});
    if (op.peer.empty()) return false;

    op.seq = j.value("seq", (uint64_t)0);
    op.lamport = j.value("lamport", (uint64_t)0);
    if (op.lamport == 0) return false;  // 0 means "unset"; ordering would break
    op.wall_ms = j.value("wall_ms", (int64_t)0);
    op.frame = j.value("frame", (uint32_t)0);
    op.camera = static_cast<int16_t>(j.value("camera", -1));
    op.node = static_cast<int16_t>(j.value("node", -1));
    op.obj_id = j.value("obj_id", std::string{});
    op.author = j.value("author", std::string{});

    if (j.contains("payload")) op.payload = j.at("payload");
    else op.payload = nlohmann::json::object();

    // Comment ops are addressed by id; without one they are unaddressable.
    if ((op.kind == OpKind::CommentPost ||
         op.kind == OpKind::CommentResolve) && op.obj_id.empty())
        return false;

    return true;
}

// =========================================================================
// Payload encoding for each annotation type
// =========================================================================

// Exact comparison is what we want: any change to a stored double is a change
// the user made and must be replicated. NaN needs the explicit case because
// NaN != NaN would otherwise emit an op on every diff pass forever.
inline bool same_double(double a, double b) {
    if (std::isnan(a) && std::isnan(b)) return true;
    return a == b;
}

inline nlohmann::json kp2d_payload(const Keypoint2D &k) {
    return nlohmann::json{{"x", k.x},
                          {"y", k.y},
                          {"labeled", k.labeled},
                          {"confidence", k.confidence},
                          {"source", static_cast<int>(k.source)}};
}

inline Keypoint2D kp2d_from_payload(const nlohmann::json &p) {
    Keypoint2D k;
    k.x = p.value("x", UNLABELED);
    k.y = p.value("y", UNLABELED);
    k.labeled = p.value("labeled", false);
    k.confidence = p.value("confidence", 0.0f);
    k.source = static_cast<LabelSource>(
        p.value("source", static_cast<int>(LabelSource::Manual)));
    return k;
}

inline bool kp2d_equal(const Keypoint2D &a, const Keypoint2D &b) {
    return same_double(a.x, b.x) && same_double(a.y, b.y) &&
           a.labeled == b.labeled && a.confidence == b.confidence &&
           a.source == b.source;
}

inline nlohmann::json kp3d_payload(const Keypoint3D &k) {
    return nlohmann::json{{"x", k.x},
                          {"y", k.y},
                          {"z", k.z},
                          {"triangulated", k.triangulated},
                          {"source", static_cast<int>(k.source)},
                          {"reviewed", k.reviewed},
                          {"confidence", k.confidence}};
}

inline Keypoint3D kp3d_from_payload(const nlohmann::json &p) {
    Keypoint3D k;
    k.x = p.value("x", UNLABELED);
    k.y = p.value("y", UNLABELED);
    k.z = p.value("z", UNLABELED);
    k.triangulated = p.value("triangulated", false);
    k.source = static_cast<Kp3DSource>(
        p.value("source", static_cast<int>(Kp3DSource::None)));
    k.reviewed = p.value("reviewed", false);
    k.confidence = p.value("confidence", 0.0f);
    return k;
}

inline bool kp3d_equal(const Keypoint3D &a, const Keypoint3D &b) {
    return same_double(a.x, b.x) && same_double(a.y, b.y) &&
           same_double(a.z, b.z) && a.triangulated == b.triangulated &&
           a.source == b.source && a.reviewed == b.reviewed &&
           a.confidence == b.confidence;
}

inline nlohmann::json bbox_payload(const CameraAnnotation &c) {
    if (!c.has_bbox()) return nlohmann::json{{"has", false}};
    const CameraExtras &e = c.get_extras();
    return nlohmann::json{{"has", true}, {"x", e.bbox_x}, {"y", e.bbox_y},
                          {"w", e.bbox_w}, {"h", e.bbox_h}};
}

inline nlohmann::json obb_payload(const CameraAnnotation &c) {
    if (!c.has_obb()) return nlohmann::json{{"has", false}};
    const CameraExtras &e = c.get_extras();
    return nlohmann::json{{"has", true}, {"cx", e.obb_cx}, {"cy", e.obb_cy},
                          {"w", e.obb_w}, {"h", e.obb_h},
                          {"angle", e.obb_angle}};
}

inline nlohmann::json mask_payload(const CameraAnnotation &c) {
    if (!c.has_mask()) return nlohmann::json{{"has", false}};
    const CameraExtras &e = c.get_extras();
    nlohmann::json polys = nlohmann::json::array();
    for (const auto &poly : e.mask_polygons) {
        nlohmann::json pts = nlohmann::json::array();
        for (const auto &pt : poly) pts.push_back({pt.x, pt.y});
        polys.push_back(pts);
    }
    return nlohmann::json{{"has", true}, {"polygons", polys}};
}

inline nlohmann::json midline_payload(const MidlineConstraint &m) {
    return nlohmann::json{{"keypoint_camera_id", m.keypoint_camera_id},
                          {"line_camera_id", m.line_camera_id},
                          {"p1x", m.p1x}, {"p1y", m.p1y},
                          {"p2x", m.p2x}, {"p2y", m.p2y},
                          {"force_vertical", m.force_vertical},
                          {"has_line", m.has_line}};
}

inline MidlineConstraint midline_from_payload(const nlohmann::json &p) {
    MidlineConstraint m;
    m.keypoint_camera_id = p.value("keypoint_camera_id", -1);
    m.line_camera_id = p.value("line_camera_id", -1);
    m.p1x = p.value("p1x", 0.0);
    m.p1y = p.value("p1y", 0.0);
    m.p2x = p.value("p2x", 0.0);
    m.p2y = p.value("p2y", 0.0);
    m.force_vertical = p.value("force_vertical", false);
    m.has_line = p.value("has_line", false);
    return m;
}

inline bool midline_equal(const MidlineConstraint &a,
                          const MidlineConstraint &b) {
    return a.keypoint_camera_id == b.keypoint_camera_id &&
           a.line_camera_id == b.line_camera_id && same_double(a.p1x, b.p1x) &&
           same_double(a.p1y, b.p1y) && same_double(a.p2x, b.p2x) &&
           same_double(a.p2y, b.p2y) && a.force_vertical == b.force_vertical &&
           a.has_line == b.has_line;
}

inline nlohmann::json flags_payload(const FrameAnnotation &fa) {
    return nlohmann::json{{"instance_id", fa.instance_id},
                          {"category_id", fa.category_id},
                          {"needs_improvement", fa.needs_improvement}};
}

// =========================================================================
// Comments
// =========================================================================

struct Comment {
    std::string id;
    std::string author;       // peer id
    std::string author_name;  // display name at post time
    std::string text;
    int64_t     created_ms = 0;
    uint32_t    frame = 0;
    int16_t     camera = -1;  // -1 = not tied to one view
    int16_t     node = -1;    // -1 = not tied to one keypoint
    bool        resolved = false;
};

using CommentStore = std::map<std::string, Comment>;

// =========================================================================
// Op construction
// =========================================================================

// Allocates the per-peer sequence and the Lamport stamp for locally authored
// ops. One factory per project session; its state is persisted alongside the
// log so sequence numbers never repeat across restarts.
struct OpFactory {
    std::string  peer;
    std::string  author;
    uint64_t     next_seq = 1;
    LamportClock clock;

    Op make(OpKind kind, uint32_t frame, int camera, int node,
            nlohmann::json payload, const std::string &obj_id = std::string{}) {
        Op op;
        op.peer = peer;
        op.author = author;
        op.seq = next_seq++;
        op.lamport = clock.tick();
        op.wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
        op.kind = kind;
        op.frame = frame;
        op.camera = static_cast<int16_t>(camera);
        op.node = static_cast<int16_t>(node);
        op.obj_id = obj_id;
        op.payload = std::move(payload);
        return op;
    }
};

// =========================================================================
// Diff: derive ops from what changed between two AnnotationMaps
// =========================================================================
//
// Edits are made by direct field mutation at ~40 call sites, and
// ImPlot::DragPoint writes straight into the model's doubles through a
// pointer. Instrumenting every write site would be invasive and would still
// miss the drag path. Diffing a shadow copy catches every mutation regardless
// of how it was made, at the cost of one comparison pass per tick.

namespace detail {

// Emits ops for every field of `after` that differs from `before`. A null
// `before` means the frame is new, in which case it is compared against a
// freshly-made default frame so only non-empty content produces ops.
inline void diff_frame(const FrameAnnotation *before,
                       const FrameAnnotation &after, OpFactory &f,
                       std::vector<Op> &out) {
    const uint32_t frame = after.frame_number;

    FrameAnnotation blank;
    if (!before) {
        blank = make_frame(static_cast<int>(after.kp3d.size()),
                           static_cast<int>(after.cameras.size()), frame,
                           /*instance_id=*/0, /*category_id=*/0);
        before = &blank;
    }

    // Frame-level flags
    if (before->instance_id != after.instance_id ||
        before->category_id != after.category_id ||
        before->needs_improvement != after.needs_improvement)
        out.push_back(f.make(OpKind::FrameFlags, frame, -1, -1,
                             flags_payload(after)));

    // 3D keypoints
    const size_t n3 = after.kp3d.size();
    for (size_t n = 0; n < n3; ++n) {
        const bool had = n < before->kp3d.size();
        if (had && kp3d_equal(before->kp3d[n], after.kp3d[n])) continue;
        out.push_back(f.make(OpKind::Kp3dSet, frame, -1, static_cast<int>(n),
                             kp3d_payload(after.kp3d[n])));
    }

    // Per-camera 2D keypoints and extras
    const size_t ncam = after.cameras.size();
    for (size_t c = 0; c < ncam; ++c) {
        const CameraAnnotation &ca = after.cameras[c];
        const CameraAnnotation *cb =
            c < before->cameras.size() ? &before->cameras[c] : nullptr;

        for (size_t n = 0; n < ca.keypoints.size(); ++n) {
            const bool had = cb && n < cb->keypoints.size();
            if (had && kp2d_equal(cb->keypoints[n], ca.keypoints[n])) continue;
            out.push_back(f.make(OpKind::Kp2dSet, frame, static_cast<int>(c),
                                 static_cast<int>(n),
                                 kp2d_payload(ca.keypoints[n])));
        }

        // NOTE: CameraAnnotation::active_id is UI selection state, not
        // annotation data. It is deliberately never diffed -- syncing it would
        // yank a collaborator's selected keypoint out from under them.

        const nlohmann::json bb = bbox_payload(ca);
        if (!cb || bbox_payload(*cb) != bb)
            out.push_back(f.make(OpKind::BboxSet, frame, static_cast<int>(c),
                                 -1, bb));

        const nlohmann::json ob = obb_payload(ca);
        if (!cb || obb_payload(*cb) != ob)
            out.push_back(f.make(OpKind::ObbSet, frame, static_cast<int>(c),
                                 -1, ob));

        const nlohmann::json mk = mask_payload(ca);
        if (!cb || mask_payload(*cb) != mk)
            out.push_back(f.make(OpKind::MaskSet, frame, static_cast<int>(c),
                                 -1, mk));
    }

    if (!midline_equal(before->midline, after.midline))
        out.push_back(f.make(OpKind::MidlineSet, frame, -1, -1,
                             midline_payload(after.midline)));
}

}  // namespace detail

// Produces the ops that carry `before` to `after`. An empty result means
// nothing changed, which is the common case on most ticks.
inline std::vector<Op> diff(const AnnotationMap &before,
                            const AnnotationMap &after, OpFactory &f) {
    std::vector<Op> out;

    for (const auto &kv : after) {
        const auto it = before.find(kv.first);
        if (it == before.end()) {
            out.push_back(f.make(OpKind::FrameCreate, kv.first, -1, -1,
                                 nlohmann::json::object()));
            detail::diff_frame(nullptr, kv.second, f, out);
        } else {
            detail::diff_frame(&it->second, kv.second, f, out);
        }
    }

    for (const auto &kv : before) {
        if (after.find(kv.first) == after.end())
            out.push_back(f.make(OpKind::FrameDelete, kv.first, -1, -1,
                                 nlohmann::json::object()));
    }

    return out;
}

// =========================================================================
// LWW state
// =========================================================================

// The current winner for every object seen so far. Held across calls so that
// incrementally merging a batch of remote ops gives the same result as
// replaying the whole log: without it, a remote op OLDER than an already
// applied local edit would wrongly overwrite it.
struct LwwState {
    struct Winner {
        uint64_t    lamport = 0;
        std::string peer;
        OpKind      kind = OpKind::FrameCreate;
    };

    std::map<ObjKey, Winner> winners;

    bool beats(const ObjKey &k, const Op &op) const {
        const auto it = winners.find(k);
        if (it == winners.end()) return true;
        if (op.lamport != it->second.lamport)
            return op.lamport > it->second.lamport;
        return op.peer > it->second.peer;
    }

    void record(const ObjKey &k, const Op &op) {
        Winner w;
        w.lamport = op.lamport;
        w.peer = op.peer;
        w.kind = op.kind;
        winners[k] = w;
    }

    const Winner *find(const ObjKey &k) const {
        const auto it = winners.find(k);
        return it == winners.end() ? nullptr : &it->second;
    }
};

inline ObjKey frame_existence_key(uint32_t frame) {
    ObjKey k;
    k.cls = 1;
    k.frame = frame;
    return k;
}

// =========================================================================
// Apply
// =========================================================================

struct ApplyStats {
    int applied = 0;
    int superseded = 0;  // lost LWW against a newer edit; kept in the log
    int rejected = 0;    // out of range for this skeleton/camera set
    int deleted_frames = 0;
    int comments = 0;
};

namespace detail {

inline void apply_extras(CameraAnnotation &ca, const Op &op) {
    const nlohmann::json &p = op.payload;
    const bool has = p.value("has", false);

    switch (op.kind) {
        case OpKind::BboxSet: {
            if (!has) {
                if (ca.extras) ca.extras->has_bbox = false;
                return;
            }
            CameraExtras &e = ca.get_extras();
            e.bbox_x = p.value("x", 0.0);
            e.bbox_y = p.value("y", 0.0);
            e.bbox_w = p.value("w", 0.0);
            e.bbox_h = p.value("h", 0.0);
            e.has_bbox = true;
            return;
        }
        case OpKind::ObbSet: {
            if (!has) {
                if (ca.extras) ca.extras->has_obb = false;
                return;
            }
            CameraExtras &e = ca.get_extras();
            e.obb_cx = p.value("cx", 0.0);
            e.obb_cy = p.value("cy", 0.0);
            e.obb_w = p.value("w", 0.0);
            e.obb_h = p.value("h", 0.0);
            e.obb_angle = p.value("angle", 0.0);
            e.has_obb = true;
            return;
        }
        case OpKind::MaskSet: {
            if (!has) {
                if (ca.extras) {
                    ca.extras->has_mask = false;
                    ca.extras->mask_polygons.clear();
                }
                return;
            }
            CameraExtras &e = ca.get_extras();
            e.mask_polygons.clear();
            if (p.contains("polygons") && p["polygons"].is_array()) {
                for (const auto &poly : p["polygons"]) {
                    if (!poly.is_array()) continue;
                    std::vector<tuple_d> pts;
                    for (const auto &pt : poly) {
                        if (!pt.is_array() || pt.size() < 2) continue;
                        pts.push_back(tuple_d{pt[0].get<double>(),
                                              pt[1].get<double>()});
                    }
                    e.mask_polygons.push_back(std::move(pts));
                }
            }
            e.has_mask = !e.mask_polygons.empty();
            return;
        }
        default:
            return;
    }
}

}  // namespace detail

// Merges `ops` into `amap`, resolving conflicts against `state`.
//
// Safe to call with ops already applied (they lose LWW and are counted as
// superseded), with ops in any order, and with ops from any number of peers.
// A full replay is this function over the whole log with a fresh state and an
// empty map.
inline ApplyStats apply_ops(AnnotationMap &amap, const std::vector<Op> &ops,
                            int num_nodes, int num_cameras, LwwState &state,
                            CommentStore *comments = nullptr,
                            LamportClock *clock = nullptr) {
    ApplyStats st;

    // Pass 1 -- frame existence. Create and delete compete for one object, so
    // this settles whether the frame is there before any content lands.
    for (const Op &op : ops) {
        if (clock) clock->observe(op.lamport);
        if (op.kind != OpKind::FrameCreate && op.kind != OpKind::FrameDelete)
            continue;

        const ObjKey k = key_of(op);
        if (!state.beats(k, op)) {
            ++st.superseded;
            continue;
        }
        state.record(k, op);

        if (op.kind == OpKind::FrameDelete) {
            if (amap.erase(op.frame) > 0) ++st.deleted_frames;
        } else {
            get_or_create_frame(amap, op.frame, num_nodes, num_cameras);
        }
        ++st.applied;
    }

    // Pass 2 -- content.
    for (const Op &op : ops) {
        if (op.kind == OpKind::FrameCreate || op.kind == OpKind::FrameDelete)
            continue;

        const ObjKey k = key_of(op);
        if (!state.beats(k, op)) {
            ++st.superseded;
            continue;
        }

        // Comments live outside the AnnotationMap.
        if (op.kind == OpKind::CommentPost) {
            state.record(k, op);
            if (comments) {
                Comment &c = (*comments)[op.obj_id];
                c.id = op.obj_id;
                c.author = op.peer;
                c.author_name = op.author;
                c.text = op.payload.value("text", std::string{});
                c.created_ms = op.wall_ms;
                c.frame = op.frame;
                c.camera = op.camera;
                c.node = op.node;
                ++st.comments;
            }
            ++st.applied;
            continue;
        }
        if (op.kind == OpKind::CommentResolve) {
            state.record(k, op);
            if (comments) {
                // Resolving a thread nobody has seen yet is legal: the post
                // may arrive in a later batch, so hold the flag on a stub.
                Comment &c = (*comments)[op.obj_id];
                c.id = op.obj_id;
                c.resolved = op.payload.value("resolved", true);
            }
            ++st.applied;
            continue;
        }

        // A frame deleted more recently than this edit swallows it. An edit
        // NEWER than the delete resurrects the frame instead -- the author
        // made that change after the deletion was already ordered, so
        // discarding it would lose work that add-wins semantics should keep.
        const LwwState::Winner *ex = state.find(frame_existence_key(op.frame));
        if (ex && ex->kind == OpKind::FrameDelete) {
            const bool newer = (op.lamport != ex->lamport)
                                   ? (op.lamport > ex->lamport)
                                   : (op.peer > ex->peer);
            if (!newer) {
                ++st.superseded;
                continue;
            }
        }

        FrameAnnotation &fa =
            get_or_create_frame(amap, op.frame, num_nodes, num_cameras);

        // Defensive bounds check. RoomBinding should have prevented a peer
        // with a different skeleton from ever joining, but a malformed or
        // stale op must not index out of range.
        const int cam = op.camera;
        const int node = op.node;
        if (cam >= 0 && static_cast<size_t>(cam) >= fa.cameras.size()) {
            ++st.rejected;
            continue;
        }
        if (node >= 0) {
            const size_t limit = (op.kind == OpKind::Kp3dSet)
                                     ? fa.kp3d.size()
                                     : (cam >= 0 ? fa.cameras[cam].keypoints.size()
                                                 : 0);
            if (static_cast<size_t>(node) >= limit) {
                ++st.rejected;
                continue;
            }
        }

        switch (op.kind) {
            case OpKind::FrameFlags:
                fa.instance_id = op.payload.value("instance_id", fa.instance_id);
                fa.category_id = op.payload.value("category_id", fa.category_id);
                fa.needs_improvement =
                    op.payload.value("needs_improvement", fa.needs_improvement);
                break;
            case OpKind::Kp3dSet:
                if (node < 0) { ++st.rejected; continue; }
                fa.kp3d[node] = kp3d_from_payload(op.payload);
                break;
            case OpKind::Kp2dSet:
                if (cam < 0 || node < 0) { ++st.rejected; continue; }
                fa.cameras[cam].keypoints[node] = kp2d_from_payload(op.payload);
                break;
            case OpKind::BboxSet:
            case OpKind::ObbSet:
            case OpKind::MaskSet:
                if (cam < 0) { ++st.rejected; continue; }
                detail::apply_extras(fa.cameras[cam], op);
                break;
            case OpKind::MidlineSet:
                fa.midline = midline_from_payload(op.payload);
                break;
            default:
                ++st.rejected;
                continue;
        }

        state.record(k, op);
        ++st.applied;
    }

    return st;
}

// Convenience for a cold replay of a whole log.
inline ApplyStats replay(AnnotationMap &amap, const std::vector<Op> &ops,
                         int num_nodes, int num_cameras,
                         CommentStore *comments = nullptr) {
    LwwState state;
    amap.clear();
    if (comments) comments->clear();
    return apply_ops(amap, ops, num_nodes, num_cameras, state, comments,
                     nullptr);
}

}  // namespace collab
