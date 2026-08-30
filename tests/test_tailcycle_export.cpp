// test_tailcycle_export.cpp — Test the tailcycle-dataset exporter.
//
// Self-contained: builds an AnnotationMap in memory, exports it, and reads the
// Parquet back with Arrow. No project on disk and no fixture data, so it runs
// anywhere red builds with Arrow.
//
// Build: cmake target "test_tailcycle_export"
// Run:   ./test_tailcycle_export [output_dir]

#include "annotation.h"
#include "camera.h"
#include "tailcycle_export.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>

#if defined(RED_HAVE_PARQUET)
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>
#endif

namespace fs = std::filesystem;

static int g_failures = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cout << "  FAIL: " << (msg) << "  [" << #cond << " at line "  \
                      << __LINE__ << "]\n";                                    \
            g_failures++;                                                      \
        }                                                                      \
    } while (0)

#if !defined(RED_HAVE_PARQUET)
int main() {
    std::cout << "test_tailcycle_export: built without Arrow -- nothing to test.\n";
    return 0;
}
#else

// ── reading helpers ──────────────────────────────────────────────────────────

static std::shared_ptr<arrow::Table> read_pq(const fs::path &p) {
    auto file = arrow::io::ReadableFile::Open(p.string());
    if (!file.ok()) return nullptr;
    auto reader = parquet::arrow::OpenFile(*file, arrow::default_memory_pool());
    if (!reader.ok()) return nullptr;
    auto t = (*reader)->ReadTable();
    if (!t.ok()) return nullptr;
    return *t;
}

static bool has_column(const std::shared_ptr<arrow::Table> &t, const char *name) {
    return t && t->schema()->GetFieldByName(name) != nullptr;
}

// Distinct values of a dictionary<int32,str> column. Dictionary32Builder only
// interns a value when it is appended, so the dictionary is exactly the set of
// values actually used.
static std::set<std::string> dict_values(const std::shared_ptr<arrow::Table> &t,
                                         const char *name) {
    std::set<std::string> out;
    auto col = t->GetColumnByName(name);
    if (!col) return out;
    for (int c = 0; c < col->num_chunks(); c++) {
        auto d = std::static_pointer_cast<arrow::DictionaryArray>(col->chunk(c));
        auto vals = std::static_pointer_cast<arrow::StringArray>(d->dictionary());
        for (int64_t i = 0; i < vals->length(); i++) out.insert(vals->GetString(i));
    }
    return out;
}

// Decoded value of a dictionary column at one row (chunk 0 only; these tables
// are small enough to arrive in a single chunk).
static std::string dict_at(const std::shared_ptr<arrow::Table> &t, const char *name,
                           int64_t row) {
    auto col = t->GetColumnByName(name);
    auto d = std::static_pointer_cast<arrow::DictionaryArray>(col->chunk(0));
    auto vals = std::static_pointer_cast<arrow::StringArray>(d->dictionary());
    auto idx = std::static_pointer_cast<arrow::Int32Array>(d->indices());
    return vals->GetString(idx->Value(row));
}

static int32_t int_at(const std::shared_ptr<arrow::Table> &t, const char *name,
                      int64_t row) {
    auto col = t->GetColumnByName(name);
    return std::static_pointer_cast<arrow::Int32Array>(col->chunk(0))->Value(row);
}

static std::string slurp(const fs::path &p) {
    std::ifstream f(p);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// ── fixture ──────────────────────────────────────────────────────────────────

static const int NC = 2, NN = 3, NF = 5;

static TailcycleExport::ExportConfig make_config(const std::string &out) {
    TailcycleExport::ExportConfig cfg;
    cfg.output_folder = out;
    cfg.split = "train";
    cfg.session_id = "sess1";
    cfg.camera_names = {"camA", "camB"};
    cfg.node_names = {"Snout", "EarL", "TailBase"};
    cfg.edges = {{0, 1}, {0, 2}};
    cfg.n_frames = NF;
    cfg.fps = 180.0f;
    cfg.source_video = "rec.mp4";
    cfg.provenance_source = "test_tailcycle_export";
    for (int i = 0; i < NC; i++) {
        CameraParams c;
        c.k = Eigen::Matrix3d::Identity();
        c.k(0, 0) = c.k(1, 1) = 1000;
        c.k(0, 2) = 640;
        c.k(1, 2) = 480;
        c.r = Eigen::Matrix3d::Identity();
        c.rvec = Eigen::Vector3d(0.1 * i, 0, 0);
        c.tvec = Eigen::Vector3d(i * 10, 0, 300);
        c.image_width = 1280;
        c.image_height = 960;
        cfg.calibration.push_back(c);
    }
    return cfg;
}

// frames 0..3 hand-labelled, frame 4 predicted; frame 3 / TailBase left
// unlabelled; 3D is triangulated on Snout and imported on EarL.
static AnnotationMap make_annotations(u32 first_frame = 0) {
    AnnotationMap amap;
    for (u32 i = 0; i < (u32)NF; i++) {
        const u32 f = first_frame + i;
        FrameAnnotation fa = make_frame(NN, NC, f);
        for (int c = 0; c < NC; c++)
            for (int n = 0; n < NN; n++) {
                if (i == 3 && n == 2) continue;   // deliberately unlabelled
                Keypoint2D &kp = fa.cameras[c].keypoints[n];
                kp.x = 100.0 + i * 10 + n;
                kp.y = 200.0 + c;
                kp.labeled = true;
                kp.source = (i == 4) ? LabelSource::Predicted : LabelSource::Manual;
                if (i == 4) kp.confidence = 0.75f;
            }
        fa.kp3d[0].x = 1; fa.kp3d[0].y = 2; fa.kp3d[0].z = 3;
        fa.kp3d[0].set_triangulated();
        fa.kp3d[1].x = 4; fa.kp3d[1].y = 5; fa.kp3d[1].z = 6;
        fa.kp3d[1].set_imported(0.9f);
        amap[f] = std::move(fa);
    }
    return amap;
}

// ── tests ────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    // temp_directory_path() rather than /tmp: on Windows that would land in
    // C:\tmp, which works but is not where anyone looks for scratch files.
    std::string root = argc > 1
                           ? argv[1]
                           : (fs::temp_directory_path() / "red_tailcycle_test").string();
    fs::remove_all(root);
    fs::create_directories(root);

    // ── 1. default export splits annotated from tracked ──
    {
        const std::string out = root + "/t1";
        auto cfg = make_config(out);
        TailcycleExport::ExportStats st;
        std::string status;
        CHECK(TailcycleExport::export_session(cfg, make_annotations(), &st, &status),
              "default export should succeed: " + status);
        CHECK(st.sessions_written == 2, "a project with manual and predicted points is two sessions");

        const fs::path A = fs::path(out) / "train" / "sess1_annotated";
        const fs::path T = fs::path(out) / "train" / "sess1_tracked";
        CHECK(fs::exists(A) && fs::exists(T), "both sessions exist");

        auto ka = read_pq(A / "keypoints.pq");
        CHECK(ka != nullptr, "annotated keypoints.pq is readable");
        // 4 frames x 2 cameras x 3 nodes, less the 2 unlabelled
        CHECK(ka && ka->num_rows() == 22, "annotated 2D row count");
        CHECK(dict_values(ka, "status") == std::set<std::string>{"projected"},
              "red has no occlusion channel, so every point is `projected`");
        CHECK(!has_column(ka, "score"),
              "score column omitted entirely when every label is hand-placed");

        auto kt = read_pq(T / "keypoints.pq");
        CHECK(kt && kt->num_rows() == 6, "tracked 2D row count");
        CHECK(has_column(kt, "score"), "predicted points carry a score");

        // Triangulated 3D is excluded by default, and triangulation is the only
        // producer of 3D in red -- so the annotated session has no 3D layer.
        CHECK(!fs::exists(A / "points3d.pq"),
              "no points3d.pq when the only 3D is triangulated");
        auto p3 = read_pq(T / "points3d.pq");
        CHECK(p3 && p3->num_rows() == NF, "imported 3D lands in the tracked session");
        CHECK(p3 && dict_values(p3, "bodypart") == std::set<std::string>{"EarL"},
              "only the imported bodypart, not the triangulated one");

        auto g = read_pq(A / "groups.pq");
        CHECK(g && g->num_rows() == 1, "one group row");
        CHECK(g && int_at(g, "n_frames", 0) == NF, "n_frames comes from the media");

        const std::string toml = slurp(A / "session.toml");
        CHECK(toml.find("mode = \"3d\"") != std::string::npos, "two cameras means 3d mode");
        CHECK(toml.find("labels = \"annotated\"") != std::string::npos, "annotated session labels");
        CHECK(toml.find("\"TailBase\"") != std::string::npos, "keypoint names written");
        CHECK(slurp(T / "session.toml").find("labels = \"tracked\"") != std::string::npos,
              "tracked session labels");

        const std::string calib = slurp(A / "calibration.toml");
        CHECK(calib.find("offset = [ 0.0, 0.0,]") != std::string::npos,
              "red has no crop model, so offset is the image origin");
        CHECK(calib.find("moving = false") != std::string::npos,
              "static rig, so extrinsics.pq is omitted");
        CHECK(!fs::exists(A / "extrinsics.pq") && !fs::exists(A / "regions.pq"),
              "optional tables red cannot fill are absent, not empty");

        // export_session creates the group folder but never fills it -- the
        // caller owns the pixels, because a group must hold exactly its own
        // frames (see the header).
        CHECK(fs::is_directory(A / "groups" / "sess1"), "group folder created");
        CHECK(fs::is_empty(A / "groups" / "sess1"), "group folder left for the caller to fill");
    }

    // ── 2. an unlabelled point writes no row at all ──
    {
        const std::string out = root + "/t2";
        auto cfg = make_config(out);
        TailcycleExport::ExportStats st;
        std::string status;
        TailcycleExport::export_session(cfg, make_annotations(), &st, &status);
        auto k = read_pq(fs::path(out) / "train" / "sess1_annotated" / "keypoints.pq");
        int found = 0;
        for (int64_t r = 0; k && r < k->num_rows(); r++)
            if (int_at(k, "frame", r) == 3 && dict_at(k, "bodypart", r) == "TailBase") found++;
        CHECK(found == 0, "unlabelled writes no row, rather than an `unlabeled` row");
    }

    // ── 3. include_triangulated_3d brings the derived solve back ──
    {
        const std::string out = root + "/t3";
        auto cfg = make_config(out);
        cfg.include_triangulated_3d = true;
        TailcycleExport::ExportStats st;
        std::string status;
        TailcycleExport::export_session(cfg, make_annotations(), &st, &status);
        auto p3 = read_pq(fs::path(out) / "train" / "sess1_annotated" / "points3d.pq");
        CHECK(p3 && p3->num_rows() == NF, "triangulated 3D written when asked for");
        CHECK(p3 && dict_values(p3, "bodypart") == std::set<std::string>{"Snout"},
              "the triangulated bodypart");
        CHECK(p3 && !has_column(p3, "score"),
              "a triangulated point's confidence describes the solve, not the point");
    }

    // ── 4. frame numbers are rebased into the group ──
    {
        const std::string out = root + "/t4";
        auto cfg = make_config(out);
        cfg.source_frame_start = 100;
        TailcycleExport::ExportStats st;
        std::string status;
        CHECK(TailcycleExport::export_session(cfg, make_annotations(100), &st, &status),
              "rebased export succeeds: " + status);
        auto k = read_pq(fs::path(out) / "train" / "sess1_annotated" / "keypoints.pq");
        int32_t lo = 1 << 30, hi = -1;
        for (int64_t r = 0; k && r < k->num_rows(); r++) {
            const int32_t f = int_at(k, "frame", r);
            lo = std::min(lo, f);
            hi = std::max(hi, f);
        }
        CHECK(k && lo == 0 && hi == 3,
              "red's absolute frame_number becomes a 0-based index into the group");
    }

    // ── 5. refusals: a file that loads cleanly and is wrong is worse than none ──
    {
        auto cfg = make_config(root + "/t5");
        cfg.calibration[0].telecentric = true;
        TailcycleExport::ExportStats st;
        std::string status;
        CHECK(!TailcycleExport::export_session(cfg, make_annotations(), &st, &status),
              "telecentric calibration is refused");
        CHECK(status.find("telecentric") != std::string::npos, "refusal names the reason");
    }
    {
        auto cfg = make_config(root + "/t6");
        cfg.calibration[1].r(2, 2) = -1.0;   // det = -1, no Rodrigues vector exists
        TailcycleExport::ExportStats st;
        std::string status;
        CHECK(!TailcycleExport::export_session(cfg, make_annotations(), &st, &status),
              "improper rotation is refused");
    }
    {
        auto cfg = make_config(root + "/t7");
        cfg.calibration[0].image_width = 0;
        TailcycleExport::ExportStats st;
        std::string status;
        CHECK(!TailcycleExport::export_session(cfg, make_annotations(), &st, &status),
              "a camera with no size is refused (validation rule 8)");
    }

    if (g_failures == 0) {
        std::cout << "ALL CHECKS PASSED\n";
        return 0;
    }
    std::cout << g_failures << " CHECK(S) FAILED\n";
    return 1;
}
#endif // RED_HAVE_PARQUET
