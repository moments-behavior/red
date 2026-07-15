// test_sync_plan.cpp — unit tests for src/sync_plan.h (no video/GPU needed).
//
// Ports both tests from cluster_pose/harness/test_sync.py:
//   1. frame-rate-agnostic drop detection (7 cams @ 200 fps, interior drop)
//   2. the slot<->position mapping invariant (present slots map exactly onto
//      mp4 positions 0..decoded_len-1, monotonic and gapless)
// plus red-specific coverage: slot_of_pos inversion, trim/clean statuses,
// decoded-count validation, and sync_plan.json import parity.
//
// Optionally pass a real cluster_pose sync_plan.json as argv[1] to run the
// mapping invariant against production data.

#include "test_framework.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "../src/sync_plan.h"

namespace ct = camera_timestamps;

// Synthesize a rig: `n_cams` cameras, `n` frames at `delta` ns, with one
// interior drop of `lost` frames at row `drop_at` in camera `drop_cam`
// (mirrors test_sync.py::test_framerate_agnostic's fixture).
static ct::CameraTimestamps make_rig(int n_cams, int n, int64_t delta,
                                     int drop_cam, int drop_at, int lost,
                                     int64_t base = 1000000000000LL) {
    ct::CameraTimestamps ts;
    ts.format = ct::Format::OrangePTP;
    for (int ci = 0; ci < n_cams; ++ci) {
        std::vector<int64_t> ns;
        for (int r = 0; r < n; ++r) {
            int64_t t = base + (int64_t)r * delta;
            if (ci == drop_cam && r >= drop_at) t += (int64_t)lost * delta;
            ns.push_back(t);
        }
        ts.frame_ns["900000" + std::to_string(ci)] = std::move(ns);
    }
    return ts;
}

static std::vector<std::string> rig_names(int n_cams) {
    std::vector<std::string> v;
    for (int i = 0; i < n_cams; ++i) v.push_back("Cam900000" + std::to_string(i));
    return v;
}

static void test_framerate_agnostic() {
    const int64_t delta = 5000000;  // 200 fps
    const int N = 12000, drop_at = 5000, lost = 3;
    auto ts = make_rig(7, N, delta, /*drop_cam=*/3, drop_at, lost);
    auto plan = sync_plan::build(ts, rig_names(7));

    printf("[framerate] delta_ns=%lld status=%s first_drop_slot=%lld\n",
           (long long)plan.delta_ns, sync_plan::status_name(plan.status),
           (long long)plan.first_drop_slot);
    EXPECT_TRUE(plan.valid);
    EXPECT_TRUE(std::llabs(plan.delta_ns - delta) <= 2);
    EXPECT_TRUE(plan.status == sync_plan::Status::Reindex);
    EXPECT_EQ(plan.first_drop_slot, drop_at);
    EXPECT_EQ(plan.canonical_len, N + lost);

    const sync_plan::SyncCam *cam3 = plan.cam("Cam9000003");
    EXPECT_TRUE(cam3 != nullptr);
    if (cam3) {
        int64_t total_lost = 0;
        for (const auto &g : cam3->gaps) total_lost += g.lost;
        EXPECT_EQ(total_lost, lost);
        EXPECT_EQ(cam3->gaps.size(), (size_t)1);
        EXPECT_EQ(cam3->gaps[0].slot, drop_at);
        // Frames before the gap are identity-mapped; after it they shift.
        EXPECT_TRUE(cam3->has(drop_at - 1));
        EXPECT_FALSE(cam3->has(drop_at));
        EXPECT_FALSE(cam3->has(drop_at + lost - 1));
        EXPECT_TRUE(cam3->has(drop_at + lost));
        EXPECT_EQ(cam3->pos(drop_at - 1), drop_at - 1);
        EXPECT_EQ(cam3->pos(drop_at + lost), drop_at);   // first frame after gap
        EXPECT_EQ(cam3->pos(drop_at), drop_at);          // inside gap -> next frame
        EXPECT_EQ(cam3->slot_of_pos(drop_at), drop_at + lost);
        EXPECT_EQ(cam3->slot_of_pos(drop_at - 1), drop_at - 1);
    }
    // Undropped camera is identity over the shared span but absent at the tail.
    const sync_plan::SyncCam *cam0 = plan.cam("Cam9000000");
    EXPECT_TRUE(cam0 != nullptr);
    if (cam0) {
        EXPECT_EQ(cam0->end_slot, N);
        EXPECT_FALSE(cam0->has(N));  // cam3 extends to N+lost, cam0 does not
        EXPECT_EQ(cam0->pos(1234), 1234);
    }

    std::string err;
    bool ok = sync_plan::self_check(plan, &err);
    if (!ok) fprintf(stderr, "self_check: %s\n", err.c_str());
    EXPECT_TRUE(ok);
}

static void test_trim_and_clean() {
    // Clean: identical spans, no drops.
    {
        auto ts = make_rig(3, 1000, 1250000, /*drop_cam=*/-1, 0, 0);
        auto plan = sync_plan::build(ts, rig_names(3));
        EXPECT_TRUE(plan.valid);
        EXPECT_TRUE(plan.status == sync_plan::Status::Clean);
        EXPECT_EQ(plan.canonical_len, 1000);
        EXPECT_TRUE(sync_plan::self_check(plan));
    }
    // Trim: camera 1 starts 4 triggers late (no interior drops).
    {
        auto ts = make_rig(3, 1000, 1250000, -1, 0, 0);
        auto &late = ts.frame_ns["9000001"];
        late.erase(late.begin(), late.begin() + 4);
        auto plan = sync_plan::build(ts, rig_names(3));
        EXPECT_TRUE(plan.valid);
        EXPECT_TRUE(plan.status == sync_plan::Status::Trim);
        EXPECT_EQ(plan.predict_start, 4);
        const sync_plan::SyncCam *c1 = plan.cam("Cam9000001");
        EXPECT_TRUE(c1 && c1->start_slot == 4);
        if (c1) {
            EXPECT_FALSE(c1->has(3));
            EXPECT_TRUE(c1->has(4));
            EXPECT_EQ(c1->pos(4), 0);
            EXPECT_EQ(c1->slot_of_pos(0), 4);
        }
        EXPECT_TRUE(sync_plan::self_check(plan));
    }
}

static void test_decoded_counts() {
    auto ts = make_rig(3, 1000, 1250000, 1, 500, 2);
    auto plan = sync_plan::build(ts, rig_names(3));
    EXPECT_TRUE(plan.valid);
    std::map<std::string, int> counts = {
        {"Cam9000000", 1000}, {"Cam9000001", 1000}, {"Cam9000002", 1000}};
    EXPECT_TRUE(sync_plan::check_decoded_counts(plan, counts));
    EXPECT_TRUE(plan.valid);
    counts["Cam9000001"] = 700;  // truncated mp4
    EXPECT_FALSE(sync_plan::check_decoded_counts(plan, counts));
    EXPECT_FALSE(plan.valid);
    printf("[counts] mismatch rejected: %s\n", plan.error.c_str());
}

static void test_missing_camera_invalidates() {
    auto ts = make_rig(3, 1000, 1250000, -1, 0, 0);
    auto names = rig_names(3);
    names.push_back("Cam9999999");  // loaded in viewer, no timestamps
    auto plan = sync_plan::build(ts, names);
    EXPECT_FALSE(plan.valid);
}

static void test_json_roundtrip() {
    // A hand-written v1 plan mirroring cluster_pose's schema.
    const char *json_text =
        "{\"version\":1,\"recording\":\"t\",\"delta_ns\":1249995,"
        "\"anchor_ts_ns\":100,\"canonical_len\":2000,\"predict_start\":0,"
        "\"predict_len\":1990,\"status\":\"reindex\",\"first_drop_slot\":800,"
        "\"cameras\":{"
        "\"CamA\":{\"start_slot\":0,\"decoded_len\":2000,\"true_span\":2000,"
        "\"frame_id_mode\":\"reindex\",\"gaps\":[]},"
        "\"CamB\":{\"start_slot\":0,\"decoded_len\":1990,\"true_span\":2000,"
        "\"frame_id_mode\":\"hole\",\"gaps\":[{\"slot\":800,\"lost\":8},"
        "{\"slot\":1200,\"lost\":2}]}}}";
    std::string path = "/tmp/test_sync_plan_roundtrip.json";
    {
        std::ofstream f(path);
        f << json_text;
    }
    auto plan = sync_plan::load_json(path);
    EXPECT_TRUE(plan.valid);
    EXPECT_TRUE(plan.status == sync_plan::Status::Reindex);
    EXPECT_EQ(plan.delta_ns, 1249995);
    EXPECT_EQ(plan.canonical_len, 2000);
    EXPECT_EQ(plan.first_drop_slot, 800);
    const sync_plan::SyncCam *b = plan.cam("CamB");
    EXPECT_TRUE(b != nullptr);
    if (b) {
        EXPECT_EQ(b->end_slot, 2000);
        EXPECT_FALSE(b->has(805));
        EXPECT_TRUE(b->has(808));
        EXPECT_EQ(b->pos(808), 800);
        EXPECT_EQ(b->slot_of_pos(800), 808);
        EXPECT_EQ(b->pos(1210), 1200);  // past both gaps: 1210 - 8 - 2
    }
    EXPECT_TRUE(sync_plan::self_check(plan));
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Simulation of the decoder emission state machine (decoder.cpp sync mode).
// The ring buffer must receive the exact canonical slot sequence; slots the
// camera dropped hold a duplicate of the nearest decoded frame, flagged.
// ---------------------------------------------------------------------------
struct SimSlot {
    int64_t label;
    bool dropped;
    int64_t content;  // mp4 index of the pixels shown in this slot
};

// Replicates decoder.cpp's per-frame logic: seek epoch state (next_slot,
// have_content), gap back-fill before converting the new frame, and the EOF
// trailing fill. seek_target < 0 means "initial load from mp4 frame 0".
static std::vector<SimSlot> simulate_decoder(const sync_plan::SyncCam &cam,
                                             int64_t canonical_len,
                                             int64_t seek_target = -1) {
    std::vector<SimSlot> out;
    int64_t next_slot = 0, mp4_pos = 0, content = -1;
    bool have_content = false;
    if (seek_target >= 0) {
        mp4_pos = cam.seek_pos(seek_target);
        next_slot = seek_target;
    }
    for (; mp4_pos < cam.decoded_len; ++mp4_pos) {
        int64_t c = cam.slot_of_pos(mp4_pos);
        if (c < next_slot) {
            if (!have_content) {
                content = mp4_pos;
                have_content = true;
            }
            continue;
        }
        if (!have_content) {
            content = mp4_pos;  // head fill duplicates the first frame
            have_content = true;
            while (next_slot < c) out.push_back({next_slot++, true, content});
        } else {
            while (next_slot < c) out.push_back({next_slot++, true, content});
            content = mp4_pos;  // convert the new frame after the fills
        }
        if (next_slot == c) out.push_back({next_slot++, false, content});
    }
    if (have_content)  // EOF trailing fill
        while (next_slot < canonical_len) out.push_back({next_slot++, true, content});
    return out;
}

static void check_stream(const sync_plan::SyncCam &cam, int64_t canonical_len,
                         const std::vector<SimSlot> &out, int64_t start) {
    EXPECT_EQ(out.size(), (size_t)(canonical_len - start));
    bool ok = true;
    for (size_t i = 0; i < out.size(); ++i) {
        const SimSlot &s = out[i];
        if (s.label != start + (int64_t)i) ok = false;  // exact slot sequence
        if (cam.has(s.label)) {
            if (s.dropped || s.content != cam.pos(s.label)) ok = false;
        } else {
            // A fill must freeze a frame adjacent to the absence: before the
            // start -> first frame; past the end -> last frame; inside an
            // interior gap -> the frame just before or just after the gap.
            int64_t lo = 0, hi = 0;
            if (s.label < cam.start_slot) {
                lo = hi = 0;
            } else if (s.label >= cam.end_slot) {
                lo = hi = cam.decoded_len - 1;
            } else {
                for (const auto &g : cam.gaps) {
                    if (g.slot <= s.label && s.label < g.slot + g.lost) {
                        lo = std::max<int64_t>(0, g.pos_before - 1);
                        hi = std::min<int64_t>(cam.decoded_len - 1,
                                               g.pos_before);
                        break;
                    }
                }
            }
            if (!s.dropped || s.content < lo || s.content > hi) ok = false;
        }
        if (!ok) {
            fprintf(stderr, "stream broken at slot %lld (label %lld)\n",
                    (long long)(start + i), (long long)s.label);
            break;
        }
    }
    EXPECT_TRUE(ok);
}

static void test_decoder_emission() {
    // Rig with interior drop (cam B), leading skew (cam C), short tail (cam A
    // relative to B's extended span).
    auto ts = make_rig(3, 2000, 1250000, /*drop_cam=*/1, 700, 5);
    auto &late = ts.frame_ns["9000002"];
    late.erase(late.begin(), late.begin() + 3);  // cam C starts 3 slots late
    auto plan = sync_plan::build(ts, rig_names(3));
    EXPECT_TRUE(plan.valid);
    int64_t T = plan.canonical_len;

    for (const auto &name : rig_names(3)) {
        const sync_plan::SyncCam &cam = *plan.cam(name);
        // Initial load: full canonical sequence 0..T-1.
        check_stream(cam, T, simulate_decoder(cam, T), 0);
        // Seeks: into the gap, right before/after it, into the leading skew,
        // near the end, and past this camera's end_slot.
        for (int64_t t : {(int64_t)0, (int64_t)1, (int64_t)699, (int64_t)700,
                          (int64_t)702, (int64_t)705, (int64_t)2, T - 2,
                          cam.end_slot - 1, cam.end_slot, T - 1}) {
            if (t < 0 || t >= T) continue;
            check_stream(cam, T, simulate_decoder(cam, T, t), t);
        }
    }
    printf("[emission] decoder state machine emits exact canonical streams\n");
}

static void test_real_plan(const char *path) {
    auto plan = sync_plan::load_json(path);
    printf("[mapping] %s status=%s cams=%zu canonical_len=%lld\n", path,
           sync_plan::status_name(plan.status), plan.cams.size(),
           (long long)plan.canonical_len);
    EXPECT_TRUE(plan.valid);
    std::string err;
    bool ok = sync_plan::self_check(plan, &err);
    if (!ok) fprintf(stderr, "self_check: %s\n", err.c_str());
    EXPECT_TRUE(ok);
    // Head before the first drop is fully present (test_sync.py:44-50).
    if (plan.first_drop_slot > 0) {
        bool head_ok = true, drop_confirmed = false;
        for (const auto &kv : plan.cams) {
            if (!kv.second.has(0) || !kv.second.has(plan.first_drop_slot - 1))
                head_ok = false;
            if (!kv.second.has(plan.first_drop_slot)) drop_confirmed = true;
        }
        EXPECT_TRUE(head_ok);
        EXPECT_TRUE(drop_confirmed);
    }
}

int main(int argc, char **argv) {
    test_framerate_agnostic();
    test_trim_and_clean();
    test_decoded_counts();
    test_missing_camera_invalidates();
    test_json_roundtrip();
    test_decoder_emission();
    if (argc > 1) test_real_plan(argv[1]);

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
