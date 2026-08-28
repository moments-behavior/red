// test_collab_ops.cpp -- merge semantics for src/collab_ops.h
//
// The load-bearing property is CONVERGENCE: peers that receive the same ops in
// different orders must end up with byte-identical annotations. Everything
// else here (idempotency, tiebreaks, delete/resurrect) is a facet of that.

#include "test_framework.h"

#include "collab_ops.h"

#include <algorithm>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace collab;

static const int kNodes = 5;
static const int kCams = 3;

// ── comparison ──

static bool cam_equal(const CameraAnnotation &a, const CameraAnnotation &b) {
    if (a.keypoints.size() != b.keypoints.size()) return false;
    for (size_t i = 0; i < a.keypoints.size(); ++i)
        if (!kp2d_equal(a.keypoints[i], b.keypoints[i])) return false;
    if (a.has_bbox() != b.has_bbox()) return false;
    if (a.has_obb() != b.has_obb()) return false;
    if (a.has_mask() != b.has_mask()) return false;
    if (a.has_bbox()) {
        const auto &x = a.get_extras();
        const auto &y = b.get_extras();
        if (x.bbox_x != y.bbox_x || x.bbox_y != y.bbox_y ||
            x.bbox_w != y.bbox_w || x.bbox_h != y.bbox_h)
            return false;
    }
    if (a.has_obb()) {
        const auto &x = a.get_extras();
        const auto &y = b.get_extras();
        if (x.obb_cx != y.obb_cx || x.obb_cy != y.obb_cy || x.obb_w != y.obb_w ||
            x.obb_h != y.obb_h || x.obb_angle != y.obb_angle)
            return false;
    }
    if (a.has_mask()) {
        const auto &x = a.get_extras().mask_polygons;
        const auto &y = b.get_extras().mask_polygons;
        if (x.size() != y.size()) return false;
        for (size_t i = 0; i < x.size(); ++i) {
            if (x[i].size() != y[i].size()) return false;
            for (size_t k = 0; k < x[i].size(); ++k)
                if (x[i][k].x != y[i][k].x || x[i][k].y != y[i][k].y)
                    return false;
        }
    }
    return true;
}

static bool frame_equal(const FrameAnnotation &a, const FrameAnnotation &b) {
    if (a.frame_number != b.frame_number) return false;
    if (a.instance_id != b.instance_id) return false;
    if (a.category_id != b.category_id) return false;
    if (a.needs_improvement != b.needs_improvement) return false;
    if (!midline_equal(a.midline, b.midline)) return false;
    if (a.kp3d.size() != b.kp3d.size()) return false;
    for (size_t i = 0; i < a.kp3d.size(); ++i)
        if (!kp3d_equal(a.kp3d[i], b.kp3d[i])) return false;
    if (a.cameras.size() != b.cameras.size()) return false;
    for (size_t i = 0; i < a.cameras.size(); ++i)
        if (!cam_equal(a.cameras[i], b.cameras[i])) return false;
    return true;
}

static bool amap_equal(const AnnotationMap &a, const AnnotationMap &b) {
    if (a.size() != b.size()) return false;
    auto ia = a.begin();
    auto ib = b.begin();
    for (; ia != a.end(); ++ia, ++ib) {
        if (ia->first != ib->first) return false;
        if (!frame_equal(ia->second, ib->second)) return false;
    }
    return true;
}

// ── fixtures ──

static OpFactory factory(const std::string &peer) {
    OpFactory f;
    f.peer = peer;
    f.author = peer + "-name";
    return f;
}

static void place_kp(AnnotationMap &m, uint32_t frame, int cam, int node,
                     double x, double y) {
    FrameAnnotation &fa = get_or_create_frame(m, frame, kNodes, kCams);
    fa.cameras[cam].keypoints[node].x = x;
    fa.cameras[cam].keypoints[node].y = y;
    fa.cameras[cam].keypoints[node].labeled = true;
}

// ── diff / apply round trip ──

static void test_diff_apply_roundtrip() {
    AnnotationMap before;
    AnnotationMap after;
    place_kp(after, 10, 0, 2, 101.5, 88.25);
    place_kp(after, 10, 1, 2, 99.0, 90.0);
    place_kp(after, 42, 2, 0, 5.0, 6.0);
    after[10].needs_improvement = true;
    after[10].kp3d[2].x = 1.0;
    after[10].kp3d[2].y = 2.0;
    after[10].kp3d[2].z = 3.0;
    after[10].kp3d[2].set_manual();

    OpFactory f = factory("peerA");
    const std::vector<Op> ops = diff(before, after, f);
    EXPECT_TRUE(!ops.empty());

    AnnotationMap rebuilt;
    replay(rebuilt, ops, kNodes, kCams);
    EXPECT_TRUE(amap_equal(rebuilt, after));
}

// Diffing an unchanged map must emit nothing -- this runs every second, so a
// spurious op would flood the log forever.
static void test_diff_no_change_is_empty() {
    AnnotationMap m;
    place_kp(m, 1, 0, 0, 1.0, 2.0);
    OpFactory f = factory("peerA");
    EXPECT_TRUE(diff(m, m, f).empty());

    AnnotationMap copy = m;
    EXPECT_TRUE(diff(m, copy, f).empty());
}

// active_id is UI selection state. Syncing it would yank a collaborator's
// cursor around, so a change to it alone must produce no ops at all.
static void test_active_id_not_synced() {
    AnnotationMap before;
    place_kp(before, 7, 0, 0, 1.0, 2.0);
    AnnotationMap after = before;
    after[7].cameras[0].active_id = 3;
    after[7].cameras[1].active_id = 4;

    OpFactory f = factory("peerA");
    EXPECT_TRUE(diff(before, after, f).empty());
}

// ── convergence ──

// The core guarantee. Same op set, different arrival orders, identical result.
static void test_convergence_under_reordering() {
    OpFactory fa = factory("aaa");
    OpFactory fb = factory("bbb");
    OpFactory fc = factory("ccc");

    std::vector<Op> all;

    {   // peer A labels frame 10
        AnnotationMap b, a;
        place_kp(a, 10, 0, 1, 10.0, 20.0);
        place_kp(a, 10, 1, 1, 11.0, 21.0);
        for (auto &op : diff(b, a, fa)) all.push_back(op);
    }
    {   // peer B labels frame 20 and part of frame 10
        AnnotationMap b, a;
        place_kp(a, 20, 2, 3, 30.0, 40.0);
        place_kp(a, 10, 2, 4, 12.0, 22.0);
        for (auto &op : diff(b, a, fb)) all.push_back(op);
    }
    {   // peer C edits the SAME keypoint peer A touched -- a real conflict
        AnnotationMap b, a;
        place_kp(a, 10, 0, 1, 999.0, 888.0);
        for (auto &op : diff(b, a, fc)) all.push_back(op);
    }

    AnnotationMap reference;
    replay(reference, all, kNodes, kCams);

    std::mt19937 rng(12345);
    for (int trial = 0; trial < 40; ++trial) {
        std::vector<Op> shuffled = all;
        std::shuffle(shuffled.begin(), shuffled.end(), rng);

        AnnotationMap replica;
        replay(replica, shuffled, kNodes, kCams);
        EXPECT_TRUE(amap_equal(replica, reference));
    }
}

// Delivering ops in several batches must match delivering them all at once.
// This is the incremental-merge path, and it is where a stateless LWW would
// silently regress an already-applied newer edit.
static void test_convergence_across_batches() {
    OpFactory fa = factory("aaa");
    OpFactory fb = factory("bbb");

    std::vector<Op> batch1, batch2;
    {
        AnnotationMap b, a;
        place_kp(a, 5, 0, 0, 1.0, 1.0);
        batch1 = diff(b, a, fa);
    }
    {
        AnnotationMap b, a;
        place_kp(a, 5, 0, 0, 2.0, 2.0);
        batch2 = diff(b, a, fb);
    }

    std::vector<Op> all = batch1;
    all.insert(all.end(), batch2.begin(), batch2.end());
    AnnotationMap one_shot;
    replay(one_shot, all, kNodes, kCams);

    // Newest first, then the older batch -- the older one must NOT win.
    AnnotationMap incremental;
    LwwState state;
    apply_ops(incremental, batch2, kNodes, kCams, state);
    apply_ops(incremental, batch1, kNodes, kCams, state);
    EXPECT_TRUE(amap_equal(incremental, one_shot));

    // And the other direction.
    AnnotationMap incremental2;
    LwwState state2;
    apply_ops(incremental2, batch1, kNodes, kCams, state2);
    apply_ops(incremental2, batch2, kNodes, kCams, state2);
    EXPECT_TRUE(amap_equal(incremental2, one_shot));
}

// Replaying a log twice must change nothing.
static void test_idempotency() {
    OpFactory f = factory("aaa");
    AnnotationMap b, a;
    place_kp(a, 1, 0, 0, 3.0, 4.0);
    place_kp(a, 2, 1, 1, 5.0, 6.0);
    const std::vector<Op> ops = diff(b, a, f);

    AnnotationMap once;
    LwwState s1;
    apply_ops(once, ops, kNodes, kCams, s1);

    AnnotationMap twice;
    LwwState s2;
    const ApplyStats first = apply_ops(twice, ops, kNodes, kCams, s2);
    const ApplyStats second = apply_ops(twice, ops, kNodes, kCams, s2);

    EXPECT_TRUE(amap_equal(once, twice));
    EXPECT_TRUE(first.applied > 0);
    EXPECT_EQ(second.applied, 0);           // nothing new landed
    EXPECT_EQ(second.superseded, first.applied);
}

// ── LWW details ──

// Equal Lamport stamps are broken by peer id, deterministically, so every
// replica picks the same winner.
static void test_lww_tiebreak_by_peer() {
    Op lo, hi;
    lo.peer = "aaa"; lo.lamport = 7; lo.kind = OpKind::Kp2dSet;
    lo.frame = 1; lo.camera = 0; lo.node = 0;
    lo.payload = kp2d_payload([]{ Keypoint2D k; k.x = 1; k.y = 1;
                                  k.labeled = true; return k; }());
    hi = lo;
    hi.peer = "zzz";
    hi.payload = kp2d_payload([]{ Keypoint2D k; k.x = 2; k.y = 2;
                                  k.labeled = true; return k; }());

    EXPECT_TRUE(op_newer(hi, lo));
    EXPECT_FALSE(op_newer(lo, hi));

    for (int order = 0; order < 2; ++order) {
        std::vector<Op> ops = order ? std::vector<Op>{hi, lo}
                                    : std::vector<Op>{lo, hi};
        AnnotationMap m;
        replay(m, ops, kNodes, kCams);
        EXPECT_EQ(m[1].cameras[0].keypoints[0].x, 2.0);  // "zzz" always wins
    }
}

// A higher Lamport stamp beats a lower one regardless of arrival order and
// regardless of wall-clock time, which is never consulted.
static void test_lww_lamport_dominates_wall_clock() {
    Op older, newer;
    older.peer = "aaa"; older.lamport = 5; older.wall_ms = 9999999;
    older.kind = OpKind::Kp2dSet; older.frame = 1;
    older.camera = 0; older.node = 0;
    older.payload = kp2d_payload([]{ Keypoint2D k; k.x = 11; k.labeled = true;
                                     return k; }());

    newer = older;
    newer.lamport = 6;
    newer.wall_ms = 1;  // a badly-skewed clock, deliberately older
    newer.payload = kp2d_payload([]{ Keypoint2D k; k.x = 22; k.labeled = true;
                                     return k; }());

    AnnotationMap m;
    replay(m, {newer, older}, kNodes, kCams);
    EXPECT_EQ(m[1].cameras[0].keypoints[0].x, 22.0);
}

// A Lamport clock that has observed a remote op must issue strictly later
// stamps, so a reply to an edit always orders after it.
static void test_lamport_clock_observe() {
    LamportClock c;
    EXPECT_EQ(c.tick(), (uint64_t)1);
    c.observe(100);
    EXPECT_EQ(c.tick(), (uint64_t)101);
    c.observe(5);  // an older stamp must not move it backwards
    EXPECT_EQ(c.tick(), (uint64_t)102);
}

// ── delete vs edit ──

// A delete that is newer than an edit removes the frame.
static void test_delete_beats_older_edit() {
    OpFactory fa = factory("aaa");
    AnnotationMap b, a;
    place_kp(a, 9, 0, 0, 1.0, 2.0);
    std::vector<Op> ops = diff(b, a, fa);

    OpFactory fb = factory("bbb");
    fb.clock.observe(fa.clock.value);
    ops.push_back(fb.make(OpKind::FrameDelete, 9, -1, -1,
                          nlohmann::json::object()));

    AnnotationMap m;
    replay(m, ops, kNodes, kCams);
    EXPECT_TRUE(m.find(9) == m.end());

    // Order of arrival must not matter.
    std::reverse(ops.begin(), ops.end());
    AnnotationMap m2;
    replay(m2, ops, kNodes, kCams);
    EXPECT_TRUE(m2.find(9) == m2.end());
}

// An edit made AFTER a delete resurrects the frame: the author acted on it
// later, so add-wins keeps their work rather than silently dropping it.
static void test_edit_newer_than_delete_resurrects() {
    OpFactory fa = factory("aaa");
    AnnotationMap b, a;
    place_kp(a, 9, 0, 0, 1.0, 2.0);
    std::vector<Op> ops = diff(b, a, fa);

    OpFactory fb = factory("bbb");
    fb.clock.observe(fa.clock.value);
    ops.push_back(fb.make(OpKind::FrameDelete, 9, -1, -1,
                          nlohmann::json::object()));

    // Peer C edits after seeing the delete.
    OpFactory fc = factory("ccc");
    fc.clock.observe(fb.clock.value);
    AnnotationMap b2, a2;
    place_kp(a2, 9, 1, 3, 77.0, 88.0);
    for (auto &op : diff(b2, a2, fc))
        if (op.kind != OpKind::FrameCreate) ops.push_back(op);

    AnnotationMap m;
    replay(m, ops, kNodes, kCams);
    EXPECT_TRUE(m.find(9) != m.end());
    EXPECT_EQ(m[9].cameras[1].keypoints[3].x, 77.0);
    // The pre-delete content stays gone.
    EXPECT_FALSE(m[9].cameras[0].keypoints[0].labeled);

    std::mt19937 rng(999);
    for (int t = 0; t < 20; ++t) {
        std::vector<Op> sh = ops;
        std::shuffle(sh.begin(), sh.end(), rng);
        AnnotationMap m2;
        replay(m2, sh, kNodes, kCams);
        EXPECT_TRUE(amap_equal(m2, m));
    }
}

// ── extras ──

static void test_extras_roundtrip() {
    AnnotationMap before, after;
    FrameAnnotation &fa = get_or_create_frame(after, 3, kNodes, kCams);
    CameraExtras &e = fa.cameras[1].get_extras();
    e.bbox_x = 1; e.bbox_y = 2; e.bbox_w = 30; e.bbox_h = 40;
    e.has_bbox = true;
    e.obb_cx = 5; e.obb_cy = 6; e.obb_w = 7; e.obb_h = 8; e.obb_angle = 0.5;
    e.has_obb = true;
    e.mask_polygons.push_back({tuple_d{1.0, 2.0}, tuple_d{3.0, 4.0},
                               tuple_d{5.0, 6.0}});
    e.has_mask = true;
    fa.midline.keypoint_camera_id = 0;
    fa.midline.line_camera_id = 2;
    fa.midline.p1x = 1; fa.midline.p1y = 2;
    fa.midline.p2x = 3; fa.midline.p2y = 4;
    fa.midline.has_line = true;

    OpFactory f = factory("aaa");
    const std::vector<Op> ops = diff(before, after, f);

    AnnotationMap rebuilt;
    replay(rebuilt, ops, kNodes, kCams);
    EXPECT_TRUE(amap_equal(rebuilt, after));
    EXPECT_TRUE(rebuilt[3].cameras[1].has_mask());
    EXPECT_EQ(rebuilt[3].cameras[1].get_extras().mask_polygons[0].size(),
              (size_t)3);
}

// Clearing a bbox must replicate as a clear, not be lost as "no change".
static void test_extras_clear_replicates() {
    AnnotationMap before;
    FrameAnnotation &fb = get_or_create_frame(before, 3, kNodes, kCams);
    CameraExtras &eb = fb.cameras[0].get_extras();
    eb.bbox_x = 1; eb.bbox_w = 10; eb.has_bbox = true;

    AnnotationMap after = before;
    after[3].cameras[0].get_extras().has_bbox = false;

    OpFactory f = factory("aaa");
    const std::vector<Op> ops = diff(before, after, f);
    EXPECT_TRUE(!ops.empty());

    AnnotationMap m = before;
    LwwState s;
    apply_ops(m, ops, kNodes, kCams, s);
    EXPECT_FALSE(m[3].cameras[0].has_bbox());
}

// ── robustness ──

// An op naming a camera or node this project does not have must be rejected,
// not applied out of bounds. RoomBinding should prevent it, but a stale or
// malformed op must never corrupt memory.
static void test_out_of_range_rejected() {
    Op bad_cam;
    bad_cam.peer = "aaa"; bad_cam.lamport = 1; bad_cam.kind = OpKind::Kp2dSet;
    bad_cam.frame = 1; bad_cam.camera = 99; bad_cam.node = 0;
    bad_cam.payload = kp2d_payload(Keypoint2D{});

    Op bad_node = bad_cam;
    bad_node.camera = 0;
    bad_node.node = 99;
    bad_node.lamport = 2;

    Op bad_kp3d;
    bad_kp3d.peer = "aaa"; bad_kp3d.lamport = 3; bad_kp3d.kind = OpKind::Kp3dSet;
    bad_kp3d.frame = 1; bad_kp3d.camera = -1; bad_kp3d.node = 99;
    bad_kp3d.payload = kp3d_payload(Keypoint3D{});

    AnnotationMap m;
    LwwState s;
    const ApplyStats st =
        apply_ops(m, {bad_cam, bad_node, bad_kp3d}, kNodes, kCams, s);
    EXPECT_EQ(st.rejected, 3);
    EXPECT_EQ(st.applied, 0);
}

static void test_room_binding_gate() {
    RoomBinding room;
    room.skeleton_name = "Mouse22";
    room.num_nodes = 22;
    room.camera_names = {"cam1", "cam2"};

    RoomBinding ok = room;
    std::string why;
    EXPECT_TRUE(room.compatible_with(ok, &why));

    RoomBinding wrong_skel = room;
    wrong_skel.skeleton_name = "Fly50";
    EXPECT_FALSE(room.compatible_with(wrong_skel, &why));
    EXPECT_TRUE(why.find("skeleton") != std::string::npos);

    RoomBinding wrong_n = room;
    wrong_n.num_nodes = 21;
    EXPECT_FALSE(room.compatible_with(wrong_n, &why));

    // Camera ORDER matters: annotations index cameras positionally.
    RoomBinding reordered = room;
    reordered.camera_names = {"cam2", "cam1"};
    EXPECT_FALSE(room.compatible_with(reordered, &why));
    EXPECT_TRUE(why.find("camera") != std::string::npos);
}

// ── JSON ──

static void test_op_json_roundtrip() {
    OpFactory f = factory("peerX");
    Op op = f.make(OpKind::Kp2dSet, 42, 1, 3,
                   kp2d_payload([]{ Keypoint2D k; k.x = 1.5; k.y = 2.5;
                                    k.labeled = true; k.confidence = 0.9f;
                                    k.source = LabelSource::Predicted;
                                    return k; }()));

    nlohmann::json j;
    to_json(j, op);

    Op back;
    EXPECT_TRUE(op_from_json(j, back));
    EXPECT_STR_EQ(back.peer, op.peer);
    EXPECT_EQ(back.seq, op.seq);
    EXPECT_EQ(back.lamport, op.lamport);
    EXPECT_TRUE(back.kind == op.kind);
    EXPECT_EQ(back.frame, op.frame);
    EXPECT_EQ(back.camera, op.camera);
    EXPECT_EQ(back.node, op.node);
    EXPECT_STR_EQ(back.author, op.author);
    EXPECT_TRUE(back.payload == op.payload);
}

// Malformed ops arrive from the network and from torn log lines. They must be
// refused, never half-parsed into something applyable.
static void test_op_json_rejects_malformed() {
    Op out;
    EXPECT_FALSE(op_from_json(nlohmann::json::array(), out));
    EXPECT_FALSE(op_from_json(nlohmann::json::object(), out));

    nlohmann::json j;
    j["kind"] = 9999;  // not a real kind
    j["peer"] = "a";
    j["lamport"] = 1;
    EXPECT_FALSE(op_from_json(j, out));

    j["kind"] = static_cast<int>(OpKind::Kp2dSet);
    j["peer"] = "";  // unattributable
    EXPECT_FALSE(op_from_json(j, out));

    j["peer"] = "a";
    j["lamport"] = 0;  // unset stamp would break ordering
    EXPECT_FALSE(op_from_json(j, out));

    j["lamport"] = 1;
    EXPECT_TRUE(op_from_json(j, out));

    // A comment op without an id is unaddressable.
    nlohmann::json c;
    c["kind"] = static_cast<int>(OpKind::CommentPost);
    c["peer"] = "a";
    c["lamport"] = 1;
    EXPECT_FALSE(op_from_json(c, out));
    c["obj_id"] = "abc";
    EXPECT_TRUE(op_from_json(c, out));
}

// ── comments ──

static void test_comments() {
    OpFactory f = factory("aaa");
    CommentStore store;
    AnnotationMap m;
    LwwState s;

    nlohmann::json body;
    body["text"] = "this antenna tip looks off by a few px";
    const Op post = f.make(OpKind::CommentPost, 100, 1, 4, body, "c-1");

    apply_ops(m, {post}, kNodes, kCams, s, &store);
    EXPECT_EQ(store.size(), (size_t)1);
    EXPECT_STR_EQ(store["c-1"].text,
                  "this antenna tip looks off by a few px");
    EXPECT_EQ(store["c-1"].frame, (uint32_t)100);
    EXPECT_EQ(store["c-1"].camera, (int16_t)1);
    EXPECT_EQ(store["c-1"].node, (int16_t)4);
    EXPECT_FALSE(store["c-1"].resolved);
    EXPECT_STR_EQ(store["c-1"].author_name, "aaa-name");

    nlohmann::json res;
    res["resolved"] = true;
    const Op resolve = f.make(OpKind::CommentResolve, 100, 1, 4, res, "c-1");
    apply_ops(m, {resolve}, kNodes, kCams, s, &store);
    EXPECT_TRUE(store["c-1"].resolved);
    // Resolving must not blank the body -- they are separate objects.
    EXPECT_STR_EQ(store["c-1"].text,
                  "this antenna tip looks off by a few px");

    // Out-of-order delivery: a resolve arriving before its post.
    CommentStore store2;
    AnnotationMap m2;
    LwwState s2;
    apply_ops(m2, {resolve, post}, kNodes, kCams, s2, &store2);
    EXPECT_TRUE(store2["c-1"].resolved);
    EXPECT_STR_EQ(store2["c-1"].text,
                  "this antenna tip looks off by a few px");
}

int main() {
    test_diff_apply_roundtrip();
    test_diff_no_change_is_empty();
    test_active_id_not_synced();
    test_convergence_under_reordering();
    test_convergence_across_batches();
    test_idempotency();
    test_lww_tiebreak_by_peer();
    test_lww_lamport_dominates_wall_clock();
    test_lamport_clock_observe();
    test_delete_beats_older_edit();
    test_edit_newer_than_delete_resurrects();
    test_extras_roundtrip();
    test_extras_clear_replicates();
    test_out_of_range_rejected();
    test_room_binding_gate();
    test_op_json_roundtrip();
    test_op_json_rejects_malformed();
    test_comments();

    std::printf("test_collab_ops: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
