// test_pump_events.cpp — unit tests for src/pump_events_core.h (no video/GPU).
//
// Covers the pumpctl dispense-log contract as it actually is on disk:
//   - the real minified record shape (keys sorted, no schema version)
//   - "ptp_ns": null, which pumpctl emits whenever it cannot read /dev/ptpN
//   - microstep mode, where requested_uL is absent rather than zero
//   - a torn final line, expected because the log is appended live
//   - `seq` restarting at 1 in every file
//   - resolution against real frame timestamps, including refusing to place a
//     dispense that falls outside the recording
//
// Optionally pass a real recording folder as argv[1] and a pump log folder as
// argv[2] to resolve production data end to end.

#include "test_framework.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "../src/pump_events_core.h"

namespace pe = pump_events;
namespace ct = camera_timestamps;

// A verbatim line from /home/user/orange_data/config/pump/dispense_logs/
// pump_dispense_20260811_222503.jsonl (keys sorted, minified, as nlohmann dumps).
static const char *kRealLine =
    R"({"config_file":"/home/user/orange_data/config/pump/config.json","delay_us":8196,)"
    R"("direction":"push","estimated_actual_ms":1983.432,"experiment":"y 2s then z 4s",)"
    R"("loop":1,"mode":"uL","monotonic_ns":373100335093486,"ptp_device":"/dev/ptp1",)"
    R"("ptp_iface":"enp65s0f1np1","ptp_minus_realtime_ns":37000002513,)"
    R"("ptp_ns":1786504374353794473,"pump":"y","pump_config":{"control_mode":0,)"
    R"("cycles":500,"delay":20,"dispense_time_ms":100,"lead_mm":0.8,"microsteps":16,)"
    R"("push_direction":0,"repeat":false,"repeat_delay":50,"steps_per_rev":200,)"
    R"("syringe_ID_mm":9.144,"target_uL":2.0},"realtime_ns":1786504337353791960,)"
    R"("requested_ms":2000,"requested_uL":2.0,"seq":1,"source":"experiment","step":1,)"
    // Custom delimiter: the value "(paced)" would otherwise close R"( early.
    R"J("steps":121,"utc":"2026-08-12T03:12:17.353791Z","wire":"(paced)"})J";

static std::filesystem::path temp_dir() {
    static int counter = 0;
    std::filesystem::path d = std::filesystem::temp_directory_path() /
                              ("red_pump_test_" + std::to_string(counter++));
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

static std::string write_log(const std::filesystem::path &dir,
                             const std::string &name,
                             const std::vector<std::string> &lines) {
    std::filesystem::path p = dir / name;
    std::ofstream f(p);
    for (const auto &l : lines) f << l << "\n";
    return p.string();
}

// ---------------------------------------------------------------------------

static void test_parse_real_record() {
    std::vector<pe::PumpEvent> ev;
    pe::LoadReport rep;
    std::filesystem::path d = temp_dir();
    std::string p = write_log(d, "pump_dispense_20260811_222503.jsonl", {kRealLine});

    EXPECT_TRUE(pe::parse_jsonl(p, ev, rep));
    EXPECT_EQ((int)ev.size(), 1);
    EXPECT_EQ(rep.records, 1);
    EXPECT_EQ(rep.skipped_lines, 0);
    if (ev.empty()) return;

    const pe::PumpEvent &e = ev[0];
    EXPECT_TRUE(e.has_ptp);
    EXPECT_EQ(e.ptp_ns, (int64_t)1786504374353794473LL);
    EXPECT_EQ(e.monotonic_ns, (int64_t)373100335093486LL);
    EXPECT_EQ(e.realtime_ns, (int64_t)1786504337353791960LL);
    EXPECT_TRUE(e.pump == "y");
    EXPECT_TRUE(e.direction == "push");
    EXPECT_TRUE(e.is_dispense());
    EXPECT_TRUE(e.mode == "uL");
    EXPECT_TRUE(e.source == "experiment");
    EXPECT_TRUE(e.experiment == "y 2s then z 4s");
    EXPECT_TRUE(e.has_volume);
    EXPECT_NEAR(e.requested_uL, 2.0, 1e-9);
    EXPECT_EQ(e.requested_ms, 2000);
    EXPECT_EQ(e.steps, 121);
    EXPECT_EQ(e.delay_us, 8196);
    EXPECT_NEAR(e.estimated_actual_ms, 1983.432, 1e-6);
    EXPECT_TRUE(e.wire == "(paced)");
    EXPECT_EQ(e.seq, 1);
    EXPECT_EQ(e.step, 1);
    EXPECT_EQ(e.loop, 1);
    EXPECT_FALSE(e.dry);
    EXPECT_TRUE(e.src_file == "pump_dispense_20260811_222503.jsonl");
    // Display helpers.
    EXPECT_TRUE(pe::format_clock(e) == "03:12:17.353");
    EXPECT_TRUE(pe::format_volume(e) == "2.00 uL");

    std::filesystem::remove_all(d);
}

// "ptp_ns" is always emitted but is null when pumpctl lacks access to
// /dev/ptpN; that must not read as timestamp 0.
static void test_null_ptp() {
    std::vector<pe::PumpEvent> ev;
    pe::LoadReport rep;
    std::filesystem::path d = temp_dir();
    std::string p = write_log(
        d, "pump_dispense_20260101_000000.jsonl",
        {R"({"seq":1,"ptp_ns":null,"ptp_unavailable_reason":"permission denied",)"
         R"("monotonic_ns":500,"realtime_ns":9000,"pump":"x","direction":"push",)"
         R"("mode":"uL","requested_uL":1.5,"steps":60,"delay_us":100,)"
         R"("estimated_actual_ms":12.0,"wire":"hx 60 100","utc":"2026-01-01T00:00:00.000000Z"})"});

    EXPECT_TRUE(pe::parse_jsonl(p, ev, rep));
    EXPECT_EQ((int)ev.size(), 1);
    if (ev.empty()) return;
    EXPECT_FALSE(ev[0].has_ptp);
    EXPECT_EQ(ev[0].ptp_ns, (int64_t)0);
    EXPECT_EQ(ev[0].monotonic_ns, (int64_t)500);
    EXPECT_EQ(pe::event_time_on(ev[0], pe::ClockAxis::Ptp), (int64_t)0);
    EXPECT_EQ(pe::event_time_on(ev[0], pe::ClockAxis::Monotonic), (int64_t)500);

    std::filesystem::remove_all(d);
}

// A microstep-mode dispense genuinely has no requested volume.
static void test_microstep_mode_has_no_volume() {
    std::vector<pe::PumpEvent> ev;
    pe::LoadReport rep;
    std::filesystem::path d = temp_dir();
    std::string p = write_log(
        d, "pump_dispense_20260101_000001.jsonl",
        {R"({"seq":1,"ptp_ns":1000,"monotonic_ns":1,"realtime_ns":1,"pump":"z",)"
         R"("direction":"push","mode":"microsteps","steps":400,"delay_us":50,)"
         R"("estimated_actual_ms":40.0,"wire":"hz 400 50"})"});

    EXPECT_TRUE(pe::parse_jsonl(p, ev, rep));
    EXPECT_EQ((int)ev.size(), 1);
    if (ev.empty()) return;
    EXPECT_FALSE(ev[0].has_volume);
    EXPECT_NEAR(ev[0].requested_uL, 0.0, 1e-12);
    EXPECT_TRUE(pe::format_volume(ev[0]) == "400 steps");

    std::filesystem::remove_all(d);
}

// The log is appended live and flushed per record, so a torn final line is
// normal. Earlier records must survive it.
static void test_torn_line_is_skipped() {
    std::vector<pe::PumpEvent> ev;
    pe::LoadReport rep;
    std::filesystem::path d = temp_dir();
    std::string p = write_log(
        d, "pump_dispense_20260101_000002.jsonl",
        {R"({"seq":1,"ptp_ns":1000,"realtime_ns":1,"pump":"x","direction":"push"})",
         R"({"seq":2,"ptp_ns":2000,"realtime_ns":2,"pump":"x","direction":"push"})",
         R"({"seq":3,"ptp_ns":3000,"realt)"});

    EXPECT_TRUE(pe::parse_jsonl(p, ev, rep));
    EXPECT_EQ((int)ev.size(), 2);
    EXPECT_EQ(rep.records, 2);
    EXPECT_EQ(rep.skipped_lines, 1);

    std::filesystem::remove_all(d);
}

// `seq` restarts at 1 in every file, so identity is (file, seq) and a merge
// must not deduplicate on seq alone. Discovery must also find both files, and
// skip an empty one without complaint (pumpctl opens a log at startup whether
// or not anything dispenses).
static void test_multi_file_merge() {
    std::filesystem::path d = temp_dir();
    write_log(d, "pump_dispense_20260101_000100.jsonl",
              {R"({"seq":1,"ptp_ns":1000,"realtime_ns":100,"pump":"x","direction":"push"})",
               R"({"seq":2,"ptp_ns":2000,"realtime_ns":200,"pump":"x","direction":"push"})"});
    write_log(d, "pump_dispense_20260101_000200.jsonl",
              {R"({"seq":1,"ptp_ns":3000,"realtime_ns":300,"pump":"y","direction":"pull"})"});
    write_log(d, "pump_dispense_20260101_000300.jsonl", {});   // empty is normal
    write_log(d, "not_a_pump_log.jsonl",
              {R"({"seq":9,"ptp_ns":9000,"realtime_ns":900,"pump":"q"})"});

    std::vector<std::string> logs = pe::discover_logs(d.string());
    EXPECT_EQ((int)logs.size(), 3);

    pe::LoadReport rep;
    std::vector<pe::PumpEvent> ev = pe::load_files(logs, rep);
    EXPECT_EQ((int)ev.size(), 3);
    EXPECT_EQ(rep.files, 3);
    EXPECT_TRUE(rep.error.empty());
    // Sorted by time, and the two seq==1 records both survive.
    if (ev.size() == 3) {
        EXPECT_EQ(ev[0].realtime_ns, (int64_t)100);
        EXPECT_EQ(ev[2].realtime_ns, (int64_t)300);
        EXPECT_EQ(ev[0].seq, 1);
        EXPECT_EQ(ev[2].seq, 1);
        EXPECT_TRUE(ev[0].src_file != ev[2].src_file);
        EXPECT_FALSE(ev[2].is_dispense());   // "pull" is a draw
    }

    std::filesystem::remove_all(d);
}

// ---------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------

static void test_resolve_against_timestamps() {
    // 300 fps, 1000 frames, no drops.
    const int64_t base = 1786504351977233897LL;
    const int64_t delta = 3333330;
    std::vector<int64_t> ref;
    for (int i = 0; i < 1000; ++i) ref.push_back(base + (int64_t)i * delta);

    std::vector<pe::PumpEvent> ev(4);
    ev[0].has_ptp = true; ev[0].ptp_ns = base + 500 * delta;          // exact hit
    ev[1].has_ptp = true; ev[1].ptp_ns = base + 500 * delta + 100;    // just after
    ev[2].has_ptp = true; ev[2].ptp_ns = base - 60LL * 1000000000LL;  // a minute early
    ev[3].has_ptp = true; ev[3].ptp_ns = base + 999 * delta;
    ev[3].estimated_actual_ms = 2000.0;   // runs past the end of the recording

    pe::ResolveInputs in;
    in.ref_ns = &ref;
    in.axis = pe::ClockAxis::Ptp;
    in.tol_ns = delta;

    int placed = pe::resolve(ev, in);
    EXPECT_EQ(placed, 3);
    EXPECT_EQ(ev[0].frame, 500);
    EXPECT_EQ(ev[1].frame, 500);      // nearest, not next
    EXPECT_EQ(ev[2].frame, -1);       // outside: refused, not clamped to 0
    EXPECT_EQ(ev[3].frame, 999);
    EXPECT_EQ(ev[3].end_frame, 999);  // clipped to the last frame, not -1

    // A positive offset moves events later in time == later frames.
    in.offset_ns = 10 * delta;
    pe::resolve(ev, in);
    EXPECT_EQ(ev[0].frame, 510);

    // No usable axis => nothing is placed, and nothing is invented.
    in.offset_ns = 0;
    in.axis = pe::ClockAxis::None;
    EXPECT_EQ(pe::resolve(ev, in), 0);
    EXPECT_EQ(ev[0].frame, -1);
}

// On an orange recording, a record with no PTP time still resolves through
// monotonic_ns against the meta.csv timestamp_sys column.
static void test_monotonic_fallback() {
    const int64_t base = 1786504351977233897LL;
    const int64_t sys_base = 373071617463487LL;
    const int64_t delta = 3333330;
    std::vector<int64_t> ref, ref_sys;
    for (int i = 0; i < 1000; ++i) {
        ref.push_back(base + (int64_t)i * delta);
        ref_sys.push_back(sys_base + (int64_t)i * delta);
    }

    std::vector<pe::PumpEvent> ev(2);
    ev[0].has_ptp = true; ev[0].ptp_ns = base + 100 * delta;
    ev[1].has_ptp = false; ev[1].monotonic_ns = sys_base + 200 * delta;

    pe::ResolveInputs in;
    in.ref_ns = &ref;
    in.ref_sys_ns = &ref_sys;
    in.axis = pe::ClockAxis::Ptp;
    in.tol_ns = delta;

    EXPECT_EQ(pe::resolve(ev, in), 2);
    EXPECT_EQ(ev[0].frame, 100);
    EXPECT_EQ(ev[1].frame, 200);

    // Without the sys column there is no fallback, and we must not silently
    // compare monotonic_ns against PTP timestamps.
    in.ref_sys_ns = nullptr;
    EXPECT_EQ(pe::resolve(ev, in), 1);
    EXPECT_EQ(ev[1].frame, -1);
}

static void test_axis_selection() {
    EXPECT_TRUE(pe::pick_axis(ct::Format::OrangePTP) == pe::ClockAxis::Ptp);
    EXPECT_TRUE(pe::pick_axis(ct::Format::LabIsoCsv) == pe::ClockAxis::Realtime);
    EXPECT_TRUE(pe::pick_axis(ct::Format::None) == pe::ClockAxis::None);
}

static void test_navigation() {
    std::vector<pe::PumpEvent> ev(4);
    ev[0].frame = 10;  ev[0].direction = "push"; ev[0].pump = "x";
    ev[1].frame = -1;  ev[1].direction = "push"; ev[1].pump = "x";  // unplaced
    ev[2].frame = 50;  ev[2].direction = "pull"; ev[2].pump = "y";  // a draw
    ev[3].frame = 90;  ev[3].direction = "push"; ev[3].pump = "y";

    auto all = [](const pe::PumpEvent &) { return true; };
    auto pushes_only = [](const pe::PumpEvent &e) { return e.is_dispense(); };

    EXPECT_EQ(pe::next_after(ev, 0, all), 0);
    EXPECT_EQ(pe::next_after(ev, 10, all), 2);
    EXPECT_EQ(pe::next_after(ev, 10, pushes_only), 3);   // skips the pull
    EXPECT_EQ(pe::next_after(ev, 90, all), -1);
    EXPECT_EQ(pe::prev_before(ev, 90, all), 2);
    EXPECT_EQ(pe::prev_before(ev, 90, pushes_only), 0);
    EXPECT_EQ(pe::prev_before(ev, 10, all), -1);
    // An unplaced event is never a jump target in either direction.
    EXPECT_EQ(pe::next_after(ev, -1, all), 0);
}

// ---------------------------------------------------------------------------
// Optional: real data
// ---------------------------------------------------------------------------

static void test_real(const char *recording, const char *log_dir) {
    std::vector<std::string> logs = pe::discover_logs(log_dir);
    printf("  [real] %zu log file(s) in %s\n", logs.size(), log_dir);
    pe::LoadReport rep;
    std::vector<pe::PumpEvent> ev = pe::load_files(logs, rep);
    EXPECT_TRUE(rep.skipped_lines == 0);
    printf("  [real] %d record(s)\n", rep.records);

    // Discover camera tokens from the meta.csv sidecars present.
    std::vector<std::string> tokens;
    for (const auto &entry : std::filesystem::directory_iterator(recording)) {
        std::string n = entry.path().filename().string();
        if (n.rfind("Cam", 0) == 0 && n.find("_meta.csv") != std::string::npos)
            tokens.push_back(n.substr(3, n.find("_meta.csv") - 3));
    }
    std::sort(tokens.begin(), tokens.end());
    ct::CameraTimestamps ts = ct::load(recording, tokens, "");
    EXPECT_TRUE(ts.format == ct::Format::OrangePTP);
    if (ts.frame_ns.empty()) return;

    const std::vector<int64_t> &ref = ts.frame_ns.begin()->second;
    const std::vector<int64_t> &sys = ts.frame_sys_ns.begin()->second;
    EXPECT_EQ(ref.size(), sys.size());
    EXPECT_TRUE(sys.front() > 0);   // timestamp_sys column really was retained

    pe::ResolveInputs in;
    in.ref_ns = &ref;
    in.ref_sys_ns = &sys;
    in.axis = pe::pick_axis(ts.format);
    in.tol_ns = ct::frame_period_ns(ref);
    int placed = pe::resolve(ev, in);
    printf("  [real] %d of %zu dispense(s) inside the recording\n", placed,
           ev.size());

    // Every placed event must land on a frame whose timestamp is within half a
    // frame period of the dispense — the property the whole feature rests on.
    for (const auto &e : ev) {
        if (e.frame < 0) continue;
        EXPECT_TRUE(e.frame < (int)ref.size());
        int64_t err = ref[e.frame] - pe::event_time_on(e, pe::ClockAxis::Ptp);
        if (err < 0) err = -err;
        EXPECT_TRUE(err <= in.tol_ns);
    }
}

int main(int argc, char **argv) {
    test_parse_real_record();
    test_null_ptp();
    test_microstep_mode_has_no_volume();
    test_torn_line_is_skipped();
    test_multi_file_merge();
    test_resolve_against_timestamps();
    test_monotonic_fallback();
    test_axis_selection();
    test_navigation();
    if (argc > 2) test_real(argv[1], argv[2]);

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
