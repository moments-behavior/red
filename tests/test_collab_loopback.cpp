// test_collab_loopback.cpp -- end-to-end test of client <-> relay.
//
// Runs a real relay in-process on an ephemeral port and drives it with real
// RelaySession clients. This covers everything the pure-logic tests cannot:
// the handshake, auth rejection, the room binding gate, op fan-out between two
// peers, manifest exchange, and resumable blob transfer.
//
// It cannot cover NAT traversal or real latency -- that is what the manual
// two-machine pass in the plan is for.

#include "test_framework.h"

#include "collab_client.h"
#include "collab_ops.h"
#include "relay/relay_core.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace collab;

static const std::string kPsk = "a-long-enough-shared-secret-for-tests";
static const int kNodes = 4;
static const int kCams = 2;

// ── fixture ──

struct RelayFixture {
    fs::path dir;
    std::atomic<bool> stop{false};
    relay::Relay r;
    std::thread th;
    uint16_t port = 0;

    bool start() {
        relay::g_quiet = true;
        dir = fs::temp_directory_path() /
              ("red_collab_test_" + random_hex(8));
        fs::create_directories(dir);

        const fs::path secrets = dir / "rooms.json";
        nlohmann::json j;
        j["rooms"]["rig-a"]["psk"] = kPsk;
        j["rooms"]["rig-b"]["psk"] = kPsk;
        write_file_atomic(secrets, j.dump(2), nullptr);

        std::string err;
        if (!r.init(0, dir / "data", secrets, 200, &err)) {
            std::fprintf(stderr, "relay init failed: %s\n", err.c_str());
            return false;
        }
        port = r.bound_port();
        if (port == 0) {
            std::fprintf(stderr, "relay did not report a bound port\n");
            return false;
        }
        th = std::thread([this] { r.run(stop); });
        return true;
    }

    void shutdown() {
        stop.store(true);
        if (th.joinable()) th.join();
        std::error_code ec;
        fs::remove_all(dir, ec);
    }

    RelayConfig config(const std::string &room = "rig-a") const {
        RelayConfig c;
        c.host = "127.0.0.1";
        c.port = port;
        c.room = room;
        c.psk = kPsk;
        c.timeout_ms = 5000;
        return c;
    }
};

static RoomBinding standard_binding() {
    RoomBinding b;
    b.skeleton_name = "TestSkel";
    b.num_nodes = kNodes;
    b.camera_names = {"cam1", "cam2"};
    return b;
}

static OpFactory factory(const std::string &peer) {
    OpFactory f;
    f.peer = peer;
    f.author = peer;
    return f;
}

static void place_kp(AnnotationMap &m, uint32_t frame, int cam, int node,
                     double x, double y) {
    FrameAnnotation &fa = get_or_create_frame(m, frame, kNodes, kCams);
    fa.cameras[cam].keypoints[node].x = x;
    fa.cameras[cam].keypoints[node].y = y;
    fa.cameras[cam].keypoints[node].labeled = true;
}

// ── handshake ──

static void test_connect_and_auth(RelayFixture &fx) {
    RelaySession s;
    std::string err;
    EXPECT_TRUE(s.connect(fx.config(), "peer-alice", "Alice",
                          standard_binding(), &err));
    if (!err.empty()) std::fprintf(stderr, "  connect err: %s\n", err.c_str());
    EXPECT_TRUE(s.connected());
    s.bye();
}

// A wrong shared secret must be refused, with a reason a human can act on.
static void test_wrong_psk_rejected(RelayFixture &fx) {
    RelayConfig bad = fx.config();
    bad.psk = "this-is-not-the-right-secret-at-all";

    RelaySession s;
    std::string err;
    EXPECT_FALSE(s.connect(bad, "peer-mallory", "Mallory", standard_binding(),
                           &err));
    EXPECT_TRUE(err.find("authentication") != std::string::npos);
    EXPECT_FALSE(s.connected());
}

static void test_unknown_room_rejected(RelayFixture &fx) {
    RelayConfig bad = fx.config("no-such-room");
    RelaySession s;
    std::string err;
    EXPECT_FALSE(s.connect(bad, "peer-x", "X", standard_binding(), &err));
    EXPECT_TRUE(err.find("no such room") != std::string::npos);
}

// The load-bearing safety gate: annotations are indexed positionally, so a
// peer whose skeleton or camera list differs must be refused at join rather
// than allowed to scramble labels.
static void test_binding_mismatch_rejected(RelayFixture &fx) {
    RelaySession first;
    std::string err;
    EXPECT_TRUE(first.connect(fx.config("rig-b"), "peer-1", "One",
                              standard_binding(), &err));
    first.bye();

    RoomBinding wrong_skeleton = standard_binding();
    wrong_skeleton.skeleton_name = "SomethingElse";
    RelaySession s2;
    EXPECT_FALSE(s2.connect(fx.config("rig-b"), "peer-2", "Two",
                            wrong_skeleton, &err));
    EXPECT_TRUE(err.find("skeleton") != std::string::npos);

    RoomBinding wrong_cams = standard_binding();
    wrong_cams.camera_names = {"cam2", "cam1"};  // reordered
    RelaySession s3;
    EXPECT_FALSE(s3.connect(fx.config("rig-b"), "peer-3", "Three", wrong_cams,
                            &err));
    EXPECT_TRUE(err.find("camera") != std::string::npos);

    // A matching peer still gets in.
    RelaySession s4;
    EXPECT_TRUE(s4.connect(fx.config("rig-b"), "peer-4", "Four",
                           standard_binding(), &err));
    s4.bye();
}

// ── op fan-out ──

// An edit on A must reach B through the relay and land in B's AnnotationMap.
static void test_ops_reach_other_peer(RelayFixture &fx) {
    std::string err;

    RelaySession a;
    EXPECT_TRUE(a.connect(fx.config(), "peer-alice", "Alice",
                          standard_binding(), &err));

    OpFactory fa = factory("peer-alice");
    AnnotationMap before, after;
    place_kp(after, 1042, 0, 2, 101.5, 88.25);
    place_kp(after, 1042, 1, 2, 99.0, 90.0);
    const std::vector<Op> ops = diff(before, after, fa);
    EXPECT_TRUE(!ops.empty());

    int accepted = 0, rejected = 0;
    uint64_t seq = 0;
    EXPECT_TRUE(a.push_ops(ops, accepted, rejected, seq, &err));
    EXPECT_EQ(accepted, (int)ops.size());
    EXPECT_EQ(rejected, 0);
    a.bye();

    RelaySession b;
    EXPECT_TRUE(b.connect(fx.config(), "peer-bob", "Bob", standard_binding(),
                          &err));

    std::vector<Op> got;
    uint64_t high = 0;
    bool more = false;
    int malformed = 0;
    EXPECT_TRUE(b.pull_ops(0, got, high, more, malformed, &err));
    EXPECT_EQ(malformed, 0);
    EXPECT_EQ(got.size(), ops.size());

    AnnotationMap bob;
    LwwState state;
    apply_ops(bob, got, kNodes, kCams, state);
    EXPECT_TRUE(bob.find(1042) != bob.end());
    EXPECT_EQ(bob[1042].cameras[0].keypoints[2].x, 101.5);
    EXPECT_EQ(bob[1042].cameras[1].keypoints[2].y, 90.0);
    b.bye();
}

// Pulling from a cursor must return only what is new -- this is what keeps a
// routine sync cheap once a project has a long history.
static void test_incremental_pull(RelayFixture &fx) {
    std::string err;
    RelaySession a;
    EXPECT_TRUE(a.connect(fx.config(), "peer-alice", "Alice",
                          standard_binding(), &err));

    std::vector<Op> got;
    uint64_t high_before = 0;
    bool more = false;
    int malformed = 0;
    EXPECT_TRUE(a.pull_ops(0, got, high_before, more, malformed, &err));

    OpFactory fa = factory("peer-alice");
    fa.next_seq = 5000;  // continue past what this peer already pushed
    AnnotationMap before, after;
    place_kp(after, 7, 0, 1, 5.0, 6.0);
    const std::vector<Op> ops = diff(before, after, fa);
    int acc = 0, rej = 0;
    uint64_t seq = 0;
    EXPECT_TRUE(a.push_ops(ops, acc, rej, seq, &err));

    std::vector<Op> fresh;
    uint64_t high_after = 0;
    EXPECT_TRUE(a.pull_ops(high_before, fresh, high_after, more, malformed,
                           &err));
    EXPECT_EQ(fresh.size(), ops.size());
    EXPECT_TRUE(high_after > high_before);

    // Pulling again from the new cursor yields nothing.
    std::vector<Op> none;
    uint64_t h = 0;
    EXPECT_TRUE(a.pull_ops(high_after, none, h, more, malformed, &err));
    EXPECT_EQ(none.size(), (size_t)0);
    a.bye();
}

// A peer must not be able to push ops attributed to someone else.
static void test_forged_authorship_rejected(RelayFixture &fx) {
    std::string err;
    RelaySession a;
    EXPECT_TRUE(a.connect(fx.config(), "peer-alice", "Alice",
                          standard_binding(), &err));

    OpFactory impostor = factory("peer-bob");   // claims to be Bob
    impostor.next_seq = 90000;
    AnnotationMap before, after;
    place_kp(after, 55, 0, 0, 1.0, 1.0);
    const std::vector<Op> ops = diff(before, after, impostor);

    int accepted = 0, rejected = 0;
    uint64_t seq = 0;
    EXPECT_TRUE(a.push_ops(ops, accepted, rejected, seq, &err));
    EXPECT_EQ(accepted, 0);
    EXPECT_EQ(rejected, (int)ops.size());
    a.bye();
}

// A client advances its sent-cursor only after a whole sync succeeds, so a
// push that lands before a later step fails gets re-pushed. The relay must
// drop the repeat instead of accumulating a second copy in its log forever.
static void test_duplicate_push_is_deduped(RelayFixture &fx) {
    std::string err;
    RelaySession a;
    EXPECT_TRUE(a.connect(fx.config(), "peer-dupe", "Dupe", standard_binding(),
                          &err));

    OpFactory f = factory("peer-dupe");
    AnnotationMap before, after;
    place_kp(after, 4242, 0, 1, 7.0, 8.0);
    const std::vector<Op> ops = diff(before, after, f);
    EXPECT_TRUE(!ops.empty());

    int acc = 0, rej = 0, dup = 0;
    uint64_t seq = 0;
    EXPECT_TRUE(a.push_ops(ops, acc, rej, seq, &err, &dup));
    EXPECT_EQ(acc, (int)ops.size());
    EXPECT_EQ(dup, 0);

    // The same ops again -- exactly what a retry after a failed pull sends.
    int acc2 = 0, rej2 = 0, dup2 = 0;
    uint64_t seq2 = 0;
    EXPECT_TRUE(a.push_ops(ops, acc2, rej2, seq2, &err, &dup2));
    EXPECT_EQ(acc2, 0);
    EXPECT_EQ(rej2, 0);
    EXPECT_EQ(dup2, (int)ops.size());
    EXPECT_EQ(seq2, seq);   // the relay log did not grow

    // A genuinely new op from the same peer is still accepted.
    AnnotationMap b2, a2;
    place_kp(a2, 4243, 0, 1, 9.0, 10.0);
    const std::vector<Op> more = diff(b2, a2, f);
    int acc3 = 0, rej3 = 0, dup3 = 0;
    uint64_t seq3 = 0;
    EXPECT_TRUE(a.push_ops(more, acc3, rej3, seq3, &err, &dup3));
    EXPECT_EQ(acc3, (int)more.size());
    EXPECT_EQ(dup3, 0);
    EXPECT_TRUE(seq3 > seq2);

    a.bye();
}

// Two peers editing the same keypoint while disconnected must converge on the
// same value once both have synced -- the whole point of the feature.
static void test_two_peer_conflict_converges(RelayFixture &fx) {
    std::string err;
    const std::string room = "rig-a";

    // Both start from the relay's current state.
    RelaySession a, b;
    EXPECT_TRUE(a.connect(fx.config(room), "peer-aaa", "A", standard_binding(),
                          &err));
    EXPECT_TRUE(b.connect(fx.config(room), "peer-zzz", "Z", standard_binding(),
                          &err));

    std::vector<Op> base;
    uint64_t cursor_a = 0, cursor_b = 0;
    bool more = false;
    int mal = 0;
    EXPECT_TRUE(a.pull_ops(0, base, cursor_a, more, mal, &err));
    cursor_b = cursor_a;

    // Offline, both edit frame 3000 / cam 0 / node 1 with the SAME Lamport
    // stamp, so only the peer-id tiebreak separates them.
    OpFactory fa = factory("peer-aaa");
    fa.next_seq = 10000;
    fa.clock.value = 500;
    OpFactory fb = factory("peer-zzz");
    fb.next_seq = 10000;
    fb.clock.value = 500;

    AnnotationMap ba, aa;
    place_kp(aa, 3000, 0, 1, 111.0, 111.0);
    const std::vector<Op> ops_a = diff(ba, aa, fa);

    AnnotationMap bb, ab;
    place_kp(ab, 3000, 0, 1, 222.0, 222.0);
    const std::vector<Op> ops_b = diff(bb, ab, fb);

    int acc = 0, rej = 0;
    uint64_t seq = 0;
    EXPECT_TRUE(a.push_ops(ops_a, acc, rej, seq, &err));
    EXPECT_TRUE(b.push_ops(ops_b, acc, rej, seq, &err));

    // Each side pulls everything and applies it on top of its own edits.
    std::vector<Op> in_a, in_b;
    uint64_t ha = 0, hb = 0;
    EXPECT_TRUE(a.pull_ops(cursor_a, in_a, ha, more, mal, &err));
    EXPECT_TRUE(b.pull_ops(cursor_b, in_b, hb, more, mal, &err));

    AnnotationMap map_a, map_b;
    LwwState st_a, st_b;
    apply_ops(map_a, ops_a, kNodes, kCams, st_a);
    apply_ops(map_a, in_a, kNodes, kCams, st_a);
    apply_ops(map_b, ops_b, kNodes, kCams, st_b);
    apply_ops(map_b, in_b, kNodes, kCams, st_b);

    // "peer-zzz" > "peer-aaa", so Z's value wins on both replicas.
    EXPECT_EQ(map_a[3000].cameras[0].keypoints[1].x, 222.0);
    EXPECT_EQ(map_b[3000].cameras[0].keypoints[1].x, 222.0);
    EXPECT_EQ(map_a[3000].cameras[0].keypoints[1].x,
              map_b[3000].cameras[0].keypoints[1].x);

    a.bye();
    b.bye();
}

// ── presence ──

static void test_presence(RelayFixture &fx) {
    std::string err;
    RelaySession a, b;
    EXPECT_TRUE(a.connect(fx.config(), "peer-alice", "Alice",
                          standard_binding(), &err));
    EXPECT_TRUE(b.connect(fx.config(), "peer-bob", "Bob", standard_binding(),
                          &err));

    std::vector<PeerPresence> peers;
    EXPECT_TRUE(a.presence(1042, peers, &err));
    EXPECT_TRUE(b.presence(318, peers, &err));

    EXPECT_TRUE(peers.size() >= 2);
    bool saw_alice = false, saw_bob = false;
    for (const auto &p : peers) {
        if (p.peer == "peer-alice") {
            saw_alice = true;
            EXPECT_EQ(p.current_frame, (int64_t)1042);
            EXPECT_STR_EQ(p.display_name, "Alice");
        }
        if (p.peer == "peer-bob") {
            saw_bob = true;
            EXPECT_EQ(p.current_frame, (int64_t)318);
        }
    }
    EXPECT_TRUE(saw_alice);
    EXPECT_TRUE(saw_bob);
    a.bye();
    b.bye();
}

// ── project sharing ──

static void write_file(const fs::path &p, const std::string &content) {
    fs::create_directories(p.parent_path());
    write_file_atomic(p, content, nullptr);
}

static void test_manifest_and_blobs(RelayFixture &fx) {
    std::string err;

    // A little project on "machine one".
    const fs::path src_root = fx.dir / "proj_src";
    const std::string video = std::string(300000, 'V') + "tail-marker";
    write_file(src_root / "media" / "cam1.mp4", video);
    write_file(src_root / "calibration" / "cam1.yaml", "fx: 1234\n");
    write_file(src_root / "labeled_data" / "2026_08_27_10_00_00" /
                   "keypoints3d.csv",
               "#red_csv v2\n#skeleton TestSkel\n");

    std::vector<FileRef> files;
    scan_dir(src_root / "media", src_root, category::kMedia, files);
    scan_dir(src_root / "calibration", src_root, category::kCalibration, files);
    scan_dir(src_root / "labeled_data", src_root, category::kLabels, files);
    EXPECT_EQ(files.size(), (size_t)3);

    Manifest m;
    m.project_name = "testproj";
    m.binding = standard_binding();
    m.created_by = "peer-alice";
    m.created_ms = now_ms();
    m.project_json = nlohmann::json{{"project_name", "testproj"},
                                    {"media_folder", "/machine/one/media"}};
    EXPECT_TRUE(build_manifest(files, m, &err));
    EXPECT_EQ(m.entries.size(), (size_t)3);
    EXPECT_TRUE(m.media_bytes() > 300000);

    // Publish it.
    RelaySession a;
    EXPECT_TRUE(a.connect(fx.config(), "peer-alice", "Alice",
                          standard_binding(), &err));
    EXPECT_TRUE(a.put_manifest(m, &err));

    std::vector<std::string> hashes;
    for (const auto &e : m.entries) hashes.push_back(e.sha256);

    std::vector<std::string> needed;
    EXPECT_TRUE(a.blobs_needed(hashes, needed, &err));
    EXPECT_EQ(needed.size(), (size_t)3);  // relay has none of them yet

    for (const auto &e : m.entries)
        EXPECT_TRUE(a.put_blob(e.sha256, (src_root / e.rel_path).string(),
                               nullptr, &err));

    // Now the relay has them all.
    std::vector<std::string> needed2;
    EXPECT_TRUE(a.blobs_needed(hashes, needed2, &err));
    EXPECT_EQ(needed2.size(), (size_t)0);
    a.bye();

    // "Machine two" clones.
    const fs::path dst_root = fx.dir / "proj_dst";
    fs::create_directories(dst_root);

    RelaySession b;
    EXPECT_TRUE(b.connect(fx.config(), "peer-bob", "Bob", standard_binding(),
                          &err));

    Manifest got;
    bool present = false;
    EXPECT_TRUE(b.get_manifest(got, present, &err));
    EXPECT_TRUE(present);
    EXPECT_STR_EQ(got.project_name, "testproj");
    EXPECT_EQ(got.entries.size(), (size_t)3);

    TransferPlan plan = plan_transfer(got, dst_root, /*include_media=*/true);
    EXPECT_EQ(plan.needed.size(), (size_t)3);
    EXPECT_EQ(plan.already_present.size(), (size_t)0);

    for (const auto &e : plan.needed)
        EXPECT_TRUE(b.get_blob(e.sha256, dst_root, dst_root / e.rel_path,
                               nullptr, &err));

    // Content must be byte-identical after the round trip.
    std::string landed;
    EXPECT_TRUE(read_file(dst_root / "media" / "cam1.mp4", landed, &err));
    EXPECT_TRUE(landed == video);

    // Re-planning now finds everything present -- a second clone moves zero
    // bytes, which is the property that makes USB-seeding the media work.
    TransferPlan plan2 = plan_transfer(got, dst_root, true);
    EXPECT_EQ(plan2.needed.size(), (size_t)0);
    EXPECT_EQ(plan2.already_present.size(), (size_t)3);
    EXPECT_EQ(plan2.bytes_needed, (uint64_t)0);

    // "Metadata and labels only" must skip the media.
    TransferPlan plan3 = plan_transfer(got, fx.dir / "proj_dst3", false);
    EXPECT_EQ(plan3.needed.size(), (size_t)2);
    EXPECT_TRUE(plan3.bytes_skipped > 300000);

    b.bye();
}

// An interrupted download must resume from its .part rather than restart, and
// must still verify on completion.
static void test_blob_resume(RelayFixture &fx) {
    std::string err;
    const fs::path src_root = fx.dir / "resume_src";
    const fs::path dst_root = fx.dir / "resume_dst";

    // Bigger than one 4 MiB chunk so the transfer is genuinely multi-step.
    const std::string big = std::string(5u * 1024 * 1024 + 777, 'R');
    write_file(src_root / "media" / "big.mp4", big);

    std::vector<FileRef> files;
    scan_dir(src_root / "media", src_root, category::kMedia, files);
    Manifest m;
    m.binding = standard_binding();
    EXPECT_TRUE(build_manifest(files, m, &err));
    EXPECT_EQ(m.entries.size(), (size_t)1);
    const std::string hash = m.entries[0].sha256;

    RelaySession a;
    EXPECT_TRUE(a.connect(fx.config(), "peer-alice", "Alice",
                          standard_binding(), &err));
    EXPECT_TRUE(a.put_blob(hash, (src_root / m.entries[0].rel_path).string(),
                           nullptr, &err));
    a.bye();

    // Download, cancelling partway through.
    RelaySession b;
    EXPECT_TRUE(b.connect(fx.config(), "peer-bob", "Bob", standard_binding(),
                          &err));
    fs::create_directories(dst_root);

    bool cancelled_once = false;
    const bool ok = b.get_blob(
        hash, dst_root, dst_root / "media" / "big.mp4",
        [&](uint64_t done, uint64_t total) {
            (void)total;
            if (done > 0 && !cancelled_once) {
                cancelled_once = true;
                return false;  // simulate the link dropping
            }
            return true;
        },
        &err);
    EXPECT_FALSE(ok);
    EXPECT_TRUE(cancelled_once);

    // Partial data must be retained for the retry.
    const uint64_t staged = resume_offset(dst_root, hash);
    EXPECT_TRUE(staged > 0);
    EXPECT_TRUE(staged < big.size());
    EXPECT_FALSE(file_exists(dst_root / "media" / "big.mp4"));

    // Retry resumes and completes.
    EXPECT_TRUE(b.get_blob(hash, dst_root, dst_root / "media" / "big.mp4",
                           nullptr, &err));
    std::string landed;
    EXPECT_TRUE(read_file(dst_root / "media" / "big.mp4", landed, &err));
    EXPECT_TRUE(landed == big);
    EXPECT_EQ(landed.size(), big.size());
    b.bye();
}

// A blob whose bytes do not hash to its claimed id must be refused, not landed.
static void test_corrupt_blob_refused(RelayFixture &fx) {
    std::string err;
    const fs::path root = fx.dir / "corrupt";
    fs::create_directories(incoming_dir(root));

    const std::string claimed = sha256_hex(std::string("the real content"));
    const std::string wrong = "totally different bytes";
    EXPECT_TRUE(append_chunk(root, claimed, 0, wrong.data(), wrong.size(),
                             &err));
    EXPECT_FALSE(finalize_blob(root, claimed, root / "out.bin", &err));
    EXPECT_TRUE(err.find("hash mismatch") != std::string::npos);
    EXPECT_FALSE(file_exists(root / "out.bin"));
    // The bad staging file is discarded so a retry starts clean.
    EXPECT_EQ(resume_offset(root, claimed), (uint64_t)0);
}

// ── path rewriting ──

static void test_project_path_rewrite() {
    nlohmann::json orig;
    orig["project_name"] = "rig";
    orig["project_path"] = "/home/alice/projects/rig";
    orig["project_root_path"] = "/home/alice/projects";
    orig["media_folder"] = "/home/alice/projects/rig/media";
    orig["calibration_folder"] = "/home/alice/projects/rig/calibration";
    orig["keypoints_root_folder"] = "/home/alice/projects/rig/labeled_data";
    orig["annotation_2d"] = false;
    orig["pump_offset_ms"] = 42;

    const nlohmann::json out =
        rewrite_project_paths(orig, "/data/bob/rig-clone", "rig");

    EXPECT_STR_EQ(out.value("project_path", std::string{}), "/data/bob/rig-clone");
    EXPECT_STR_EQ(out.value("project_root_path", std::string{}), "/data/bob");
    EXPECT_STR_EQ(out.value("media_folder", std::string{}),
                  "/data/bob/rig-clone/media");
    EXPECT_STR_EQ(out.value("calibration_folder", std::string{}),
                  "/data/bob/rig-clone/calibration");
    EXPECT_STR_EQ(out.value("keypoints_root_folder", std::string{}),
                  "/data/bob/rig-clone/labeled_data");

    // Settings unrelated to paths must survive the trip untouched.
    EXPECT_EQ(out.value("pump_offset_ms", 0), 42);
    EXPECT_FALSE(out.value("annotation_2d", true));
}

int main() {
    RelayFixture fx;
    if (!fx.start()) {
        std::fprintf(stderr, "could not start the test relay\n");
        return 1;
    }

    test_connect_and_auth(fx);
    test_wrong_psk_rejected(fx);
    test_unknown_room_rejected(fx);
    test_binding_mismatch_rejected(fx);
    test_ops_reach_other_peer(fx);
    test_incremental_pull(fx);
    test_forged_authorship_rejected(fx);
    test_duplicate_push_is_deduped(fx);
    test_two_peer_conflict_converges(fx);
    test_presence(fx);
    test_manifest_and_blobs(fx);
    test_blob_resume(fx);
    test_corrupt_blob_refused(fx);
    test_project_path_rewrite();

    fx.shutdown();

    std::printf("test_collab_loopback: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
