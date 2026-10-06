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
#include "tailcycle_import.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <string>

#if defined(RED_HAVE_PARQUET)
#include "tailcycle_schema.h"
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

static bool replace_once(std::string &text, const std::string &from,
                         const std::string &to) {
    const size_t p = text.find(from);
    if (p == std::string::npos) return false;
    text.replace(p, from.size(), to);
    return true;
}


// ── hand-written tables, for rows red never writes itself ───────────────────

static const float NA = std::numeric_limits<float>::quiet_NaN();

struct KRow { std::string g; int f; std::string a, c, b, s; float x, y; };
struct PRow { std::string g; int f; std::string a, b, s; float x, y, z; };
struct IRow { std::string g; int f; std::string a, c, s; float x0, y0, x1, y1; };

static void put(arrow::FloatBuilder &b, float v) {
    if (std::isnan(v)) (void)b.AppendNull(); else (void)b.Append(v);
}
static void write_keypoints(const fs::path &p, const std::vector<KRow> &rows) {
    arrow::StringDictionary32Builder g, a, c, b, s;
    arrow::Int32Builder f;
    arrow::FloatBuilder x, y;
    for (const auto &r : rows) {
        (void)g.Append(r.g); (void)f.Append(r.f); (void)a.Append(r.a); (void)c.Append(r.c);
        (void)b.Append(r.b); (void)s.Append(r.s); put(x, r.x); put(y, r.y);
    }
    std::vector<std::shared_ptr<arrow::Array>> v(8);
    (void)g.Finish(&v[0]); (void)f.Finish(&v[1]); (void)a.Finish(&v[2]); (void)c.Finish(&v[3]);
    (void)b.Finish(&v[4]); (void)s.Finish(&v[5]); (void)x.Finish(&v[6]); (void)y.Finish(&v[7]);
    (void)Tailcycle::write_table(arrow::Table::Make(Tailcycle::keypoints_schema(false), v), p.string());
}
static void write_points3d(const fs::path &p, const std::vector<PRow> &rows) {
    arrow::StringDictionary32Builder g, a, b, s;
    arrow::Int32Builder f;
    arrow::FloatBuilder x, y, z;
    for (const auto &r : rows) {
        (void)g.Append(r.g); (void)f.Append(r.f); (void)a.Append(r.a); (void)b.Append(r.b);
        (void)s.Append(r.s); put(x, r.x); put(y, r.y); put(z, r.z);
    }
    std::vector<std::shared_ptr<arrow::Array>> v(8);
    (void)g.Finish(&v[0]); (void)f.Finish(&v[1]); (void)a.Finish(&v[2]); (void)b.Finish(&v[3]);
    (void)s.Finish(&v[4]); (void)x.Finish(&v[5]); (void)y.Finish(&v[6]); (void)z.Finish(&v[7]);
    (void)Tailcycle::write_table(arrow::Table::Make(Tailcycle::points3d_schema(false), v), p.string());
}
static void write_instances(const fs::path &p, const std::vector<IRow> &rows) {
    arrow::StringDictionary32Builder g, a, c, s;
    arrow::Int32Builder f;
    arrow::FloatBuilder x0, y0, x1, y1;
    arrow::StringBuilder n;
    for (const auto &r : rows) {
        (void)g.Append(r.g); (void)f.Append(r.f); (void)a.Append(r.a); (void)c.Append(r.c);
        put(x0, r.x0); put(y0, r.y0); put(x1, r.x1); put(y1, r.y1);
        (void)s.Append(r.s); (void)n.AppendNull();
    }
    std::vector<std::shared_ptr<arrow::Array>> v(10);
    (void)g.Finish(&v[0]); (void)f.Finish(&v[1]); (void)a.Finish(&v[2]); (void)c.Finish(&v[3]);
    (void)x0.Finish(&v[4]); (void)y0.Finish(&v[5]); (void)x1.Finish(&v[6]); (void)y1.Finish(&v[7]);
    (void)s.Finish(&v[8]); (void)n.Finish(&v[9]);
    (void)Tailcycle::write_table(arrow::Table::Make(Tailcycle::instances_schema(), v), p.string());
}

// Every row of a table as a string, columns in name order, so two tables can
// be compared as sets whatever their row order or dictionary encoding.
static std::set<std::string> row_set(const std::shared_ptr<arrow::Table> &t) {
    std::set<std::string> out;
    if (!t) return out;
    auto flat = t->CombineChunks().ValueOrDie();
    std::vector<std::string> names = flat->ColumnNames();
    std::sort(names.begin(), names.end());
    for (int64_t r = 0; r < flat->num_rows(); r++) {
        std::string row;
        for (const auto &name : names) {
            auto arr = flat->GetColumnByName(name)->chunk(0);
            row += name + "=";
            if (arr->IsNull(r)) row += "null";
            else if (auto d = std::dynamic_pointer_cast<arrow::DictionaryArray>(arr))
                row += std::static_pointer_cast<arrow::StringArray>(d->dictionary())
                           ->GetString(std::static_pointer_cast<arrow::Int32Array>(d->indices())->Value(r));
            else if (auto fa = std::dynamic_pointer_cast<arrow::FloatArray>(arr))
                row += std::to_string(fa->Value(r));
            else if (auto ia = std::dynamic_pointer_cast<arrow::Int32Array>(arr))
                row += std::to_string(ia->Value(r));
            else if (auto sa = std::dynamic_pointer_cast<arrow::StringArray>(arr))
                row += sa->GetString(r);
            row += ";";
        }
        out.insert(row);
    }
    return out;
}
static std::set<std::string> row_set(const fs::path &p) { return row_set(read_pq(p)); }

// The rows of `t` whose columns contain every `name=value` in `want`.
static std::set<std::string> rows_where(const std::set<std::string> &rows,
                                        std::initializer_list<std::string> want) {
    std::set<std::string> out;
    for (const auto &r : rows) {
        bool all = true;
        for (const auto &w : want)
            if (r.find(w + ";") == std::string::npos) all = false;
        if (all) out.insert(r);
    }
    return out;
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
                // i==4 is a model's output; the rest are hand-placed. One of
                // the hand-placed ones is also marked reprojected, because a
                // refreshed manual observation must still export as visible.
                if (i == 4) kp.set_predicted();
                else        kp.set_manual();
                if (i == 0 && c == 1 && n == 0) kp.reprojected = true;
                if (i == 4) kp.confidence = 0.75f;
            }
        fa.kp3d[0].x = 1; fa.kp3d[0].y = 2; fa.kp3d[0].z = 3;
        fa.kp3d[0].set_triangulated();
        fa.kp3d[1].x = 4; fa.kp3d[1].y = 5; fa.kp3d[1].z = 6;
        fa.kp3d[1].set_predicted(0.9f);
        amap[f] = FrameInstances{std::move(fa)};
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
        CHECK(dict_values(ka, "status") == std::set<std::string>{"visible"},
              "user-annotated points are exported as `visible`");
        CHECK(!has_column(ka, "score"),
              "score column omitted entirely when every label is hand-placed");

        // red stores y with the origin at the BOTTOM of the image (ImPlot
        // coords); the format, the calibration and the JPEGs all use top-left.
        // The fixture puts every point at y = 200 + camera index, so an
        // unflipped export writes 200/201 and a correct one writes 760/759.
        {
            auto ycol = ka->GetColumnByName("y");
            auto yarr = std::static_pointer_cast<arrow::FloatArray>(ycol->chunk(0));
            bool flipped = true, unflipped = false;
            for (int64_t r = 0; r < ka->num_rows(); r++) {
                const float want = 960.0f - (200.0f + (dict_at(ka, "camera", r) == "camB" ? 1.0f : 0.0f));
                if (std::abs(yarr->Value(r) - want) > 0.01f) flipped = false;
                if (std::abs(yarr->Value(r) - (want - 960.0f + 2.0f * (960.0f - want))) < 0.01f) unflipped = true;
            }
            CHECK(flipped, "y is flipped from ImPlot (bottom-origin) to image coords");
            (void)unflipped;
        }

        auto kt = read_pq(T / "keypoints.pq");
        CHECK(kt && kt->num_rows() == 6, "tracked 2D row count");
        CHECK(dict_values(kt, "status") == std::set<std::string>{"projected"},
              "triangulated/projected points are exported as `projected`");
        CHECK(has_column(kt, "score"), "predicted points carry a score");

        // The default writes the 2D layer only, so neither session carries 3D.
        CHECK(!fs::exists(A / "points3d.pq") && !fs::exists(T / "points3d.pq"),
              "the default (2D keypoints) writes no points3d.pq");

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

        // Tailcycle stores crop-local labels but sensor-coordinate intrinsics.
        // Simulate a cropped source by adding an offset and restoring the
        // sensor-space principal point in the exported calibration; import
        // must normalize it back to red's stored-image calibration.
        std::string cropped_calib = slurp(A / "calibration.toml");
        CHECK(replace_once(cropped_calib,
                           "matrix = [ [ 1000,0,640,], [ 0,1000,480,], [ 0,0,1,],]",
                           "matrix = [ [ 1000,0,760,], [ 0,1000,620,], [ 0,0,1,],]"),
              "crop fixture restores sensor-space principal point");
        CHECK(replace_once(cropped_calib, "offset = [ 0.0, 0.0,]",
                           "offset = [ 120, 140,]"),
              "crop fixture writes a non-zero offset");
        {
            std::ofstream f(A / "calibration.toml");
            f << cropped_calib;
        }
        // Tailcycle also permits each skeleton entry to be a polyline. The
        // importer should connect consecutive names without crossing paths,
        // and collapse the repeated-anchor reverse edge.
        {
            std::string session = slurp(A / "session.toml");
            CHECK(replace_once(
                      session,
                      "skeleton = [ [ \"Snout\", \"EarL\",], [ \"Snout\", \"TailBase\",],]",
                      "skeleton = [\n"
                      " [ \"Snout\", \"EarL\", \"Snout\",],\n"
                      " [ \"Snout\", \"TailBase\",],\n"
                      "]"),
                  "polyline skeleton fixture written");
            std::ofstream f(A / "session.toml");
            f << session;
        }
        TailcycleImport::Session imported;
        TailcycleImport::ImportStats ist;
        std::string import_status;
        CHECK(TailcycleImport::read_session(A.string(), "sess1", &imported, &ist,
                                            &import_status),
              "cropped calibration imports: " + import_status);
        const std::vector<std::pair<int, int>> expected_edges{{0, 1}, {0, 2}};
        CHECK(imported.edges == expected_edges,
              "polyline skeleton imports as unique consecutive edges");
        CHECK(imported.calibration.size() == NC, "all cropped cameras import");
        CHECK(imported.calibration.size() == NC &&
              std::abs(imported.calibration[0].k(0, 2) - 640.0) < 1e-9 &&
              std::abs(imported.calibration[0].k(1, 2) - 480.0) < 1e-9,
              "crop offset is folded into the principal point");
        CHECK(imported.calibration.size() == NC &&
              (imported.calibration[0].projection_mat -
               red_math::projectionFromKRt(imported.calibration[0].k,
                                           imported.calibration[0].r,
                                           imported.calibration[0].tvec)).norm() < 1e-9,
              "projection matrix uses the crop-normalized calibration");
        CHECK(ist.keypoint_rows == 22, "cropped import keeps crop-local keypoint rows");
        const auto frame0 = imported.annotations.find(0);
        CHECK(frame0 != imported.annotations.end() && !frame0->second.empty() &&
              frame0->second.front().cameras.size() == NC &&
              std::abs(frame0->second.front().cameras[0].keypoints[0].x - 100.0) < 1e-6 &&
              std::abs(frame0->second.front().cameras[0].keypoints[0].y - 200.0) < 1e-6,
              "crop offset does not shift crop-local keypoint labels");
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

    // ── 2b. all keypoint visibility statuses round-trip ──
    {
        const std::string out = root + "/t2b";
        auto cfg = make_config(out);
        cfg.force_labels = "annotated";  // keep both visibility statuses together
        AnnotationMap amap;
        FrameAnnotation fa = make_frame(NN, NC, 0);

        auto &visible = fa.cameras[0].keypoints[0];
        visible.x = 11.0; visible.y = 22.0; visible.set_manual();
        visible.set_manual();

        auto &projected = fa.cameras[0].keypoints[1];
        projected.x = 33.0; projected.y = 44.0; projected.set_manual();
        projected.set_predicted();
        projected.reprojected = true;

        // Author only: an occluded point here has no coordinates, so it
        // must not claim a position.
        fa.cameras[0].keypoints[2].author = Keypoint2D::Author::Manual;
        fa.cameras[0].keypoints[2].set_occluded();
        amap[0] = FrameInstances{std::move(fa)};

        TailcycleExport::ExportStats st;
        std::string status;
        CHECK(TailcycleExport::export_session(cfg, amap, &st, &status),
              "all-status export succeeds: " + status);
        const fs::path d = fs::path(out) / "train" / "sess1";
        auto k = read_pq(d / "keypoints.pq");
        CHECK(k && k->num_rows() == 3, "visible/projected/missing write rows");
        CHECK((k && dict_values(k, "status") ==
                       std::set<std::string>{"visible", "projected", "missing"}),
              "visible/projected/missing are exported");

        TailcycleImport::Session imported;
        TailcycleImport::ImportStats ist;
        CHECK(TailcycleImport::read_session(d.string(), "sess1", &imported, &ist,
                                            &status),
              "all-status import succeeds: " + status);
        const auto fit = imported.annotations.find(0);
        CHECK(fit != imported.annotations.end() && !fit->second.empty(),
              "all-status import retains the frame");
        if (fit != imported.annotations.end() && !fit->second.empty()) {
            const auto &if0 = fit->second.front();
            CHECK(if0.cameras[0].keypoints[0].is_observed() &&
                  std::abs(if0.cameras[0].keypoints[0].x - 11.0) < 1e-6,
                  "visible imports as a labeled point");
            CHECK(if0.cameras[0].keypoints[1].has_pos &&
                  if0.cameras[0].keypoints[1].reprojected,
                  "projected imports as a projected point");
            CHECK(!if0.cameras[0].keypoints[2].has_pos &&
                  if0.cameras[0].keypoints[2].is_occluded(),
                  "missing imports as an occluded point");
            CHECK(!if0.cameras[1].keypoints[0].has_pos,
                  "unlabeled remains the default empty point");
        }
    }

    // ── 2c. a tracked session can still contain visible observations ──
    // The session-level labels field must not turn visible rows into
    // projected rows when the imported session is saved back in place.
    {
        const std::string out = root + "/t2c";
        auto cfg = make_config(out);
        cfg.force_labels = "tracked";
        AnnotationMap amap;
        FrameAnnotation fa = make_frame(NN, NC, 0);
        auto &kp = fa.cameras[0].keypoints[0];
        kp.x = 12.0; kp.y = 23.0; kp.set_manual();
        kp.set_manual();
        amap[0] = FrameInstances{std::move(fa)};
        TailcycleExport::ExportStats st;
        std::string status;
        CHECK(TailcycleExport::export_session(cfg, amap, &st, &status),
              "tracked visible export succeeds: " + status);

        const fs::path d = fs::path(out) / "train" / "sess1";
        TailcycleImport::Session imported;
        TailcycleImport::ImportStats ist;
        CHECK(TailcycleImport::read_session(d.string(), "sess1", &imported, &ist,
                                            &status),
              "tracked visible import succeeds: " + status);
        // What this has always been about is the visible STATUS surviving,
        // not authorship: it was written when one enum carried both. A tracked
        // session's points are the model's, so `predicted` -- and `visible`
        // shows up as a position that is present and not derived.
        {
            const Keypoint2D &k0 =
                imported.annotations.at(0).front().cameras[0].keypoints[0];
            CHECK(k0.usable() && !k0.is_occluded(),
                  "tracked session preserves visible status");
            CHECK(k0.is_predicted() && !k0.is_manual(),
                  "tracked session's points are the model's");
        }

        cfg.in_place = true;
        cfg.edited_views = {{0, 0}};   // rewrite the view rather than keep its old rows
        CHECK(TailcycleExport::export_session(cfg, imported.annotations, &st, &status),
              "tracked visible in-place round trip succeeds: " + status);
        auto k = read_pq(d / "keypoints.pq");
        CHECK((k && dict_values(k, "status") == std::set<std::string>{"visible"}),
              "tracked visible row remains visible after round trip");
    }

    // ── 3. asking for the 3D layer brings the derived solve back ──
    {
        const std::string out = root + "/t3";
        auto cfg = make_config(out);
        cfg.layers = TailcycleExport::ExportConfig::Layers::TwoDAndThreeD;
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

    // ── 3b. a 3D-only session writes no keypoints.pq at all ──
    {
        const std::string out = root + "/t3b";
        auto cfg = make_config(out);
        cfg.layers = TailcycleExport::ExportConfig::Layers::ThreeD;
        TailcycleExport::ExportStats st;
        std::string status;
        CHECK(TailcycleExport::export_session(cfg, make_annotations(), &st, &status),
              "3D-only export succeeds: " + status);
        // The imported 3D is the tracked bucket, the triangulated is annotated.
        bool any = false;
        for (const char *sfx : {"", "_annotated", "_tracked"}) {
            const fs::path d = fs::path(out) / "train" / (std::string("sess1") + sfx);
            if (!fs::exists(d)) continue;
            any = true;
            CHECK(!fs::exists(d / "keypoints.pq"),
                  "3D-only writes no keypoints.pq");
            CHECK(fs::exists(d / "points3d.pq"), "3D-only writes points3d.pq");
        }
        CHECK(any, "3D-only produced at least one session");
    }

    // ── 3c. force_labels puts every row in one session ──
    // What saving corrections back over an existing session needs: editing an
    // imported session mixes sources, and the usual split would replace it
    // with <name>_annotated and <name>_tracked beside frames belonging to
    // neither.
    {
        const std::string out = root + "/t3c";
        auto cfg = make_config(out);
        cfg.force_labels = "tracked";
        TailcycleExport::ExportStats st;
        std::string status;
        CHECK(TailcycleExport::export_session(cfg, make_annotations(), &st, &status),
              "forced-label export succeeds: " + status);
        CHECK(st.sessions_written == 1, "forced labels means exactly one session");
        const fs::path d = fs::path(out) / "train" / "sess1";
        CHECK(fs::exists(d), "written under the unsuffixed name");
        CHECK(!fs::exists(fs::path(out) / "train" / "sess1_annotated") &&
              !fs::exists(fs::path(out) / "train" / "sess1_tracked"),
              "no _annotated / _tracked split");
        auto k = read_pq(d / "keypoints.pq");
        // every 2D row, manual and predicted alike
        CHECK(k && k->num_rows() == 28, "all rows land in the one session");
        CHECK(slurp(d / "session.toml").find("labels = \"tracked\"") != std::string::npos,
              "the forced label is written");
    }

    // ── 3e. in-place saves replace label tables without creating a sibling ──
    {
        const std::string out = root + "/t3e";
        auto cfg = make_config(out);
        cfg.source_frame_start = 100;
        cfg.force_labels = "annotated";
        cfg.layers = TailcycleExport::ExportConfig::Layers::TwoDAndThreeD;
        TailcycleExport::ExportStats st;
        std::string status;
        CHECK(TailcycleExport::export_session(cfg, make_annotations(100), &st, &status),
              "initial in-place fixture export succeeds: " + status);
        const fs::path d = fs::path(out) / "train" / "sess1";
        CHECK(fs::exists(d / "keypoints.pq"), "in-place fixture has keypoints.pq");
        TailcycleImport::Session imported;
        TailcycleImport::ImportStats ist;
        CHECK(TailcycleImport::read_session(d.string(), "sess1", &imported, &ist,
                                            &status),
              "in-place fixture imports: " + status);
        AnnotationMap amap = imported.annotations;
        amap.at(0).front().cameras[0].keypoints[0].x = 777.0;
        // Imported annotations are rebased to group-local frames, as they are
        // in the GUI. The save path must not apply source_frame_start twice.
        cfg.source_frame_start = 0;
        cfg.in_place = true;
        cfg.edited_views = {{0, 0}};
        st = {};
        CHECK(TailcycleExport::export_session(cfg, amap, &st, &status),
              "in-place overwrite succeeds: " + status);
        auto k = read_pq(d / "keypoints.pq");
        bool found = false;
        for (int64_t r = 0; k && r < k->num_rows(); r++) {
            if (int_at(k, "frame", r) == 0 && dict_at(k, "camera", r) == "camA" &&
                dict_at(k, "bodypart", r) == "Snout") {
                auto col = k->GetColumnByName("x");
                auto arr = std::static_pointer_cast<arrow::FloatArray>(col->chunk(0));
                found = std::abs(arr->Value(r) - 777.0f) < 0.01f;
            }
        }
        CHECK(found, "in-place overwrite replaces the existing label table");

        // Clearing all corrections is valid: in-place export removes all three
        // optional annotation tables, and importing the session still works.
        AnnotationMap empty;
        cfg.edited_views.clear();
        for (int f = 0; f < NF; ++f)
            for (int c = 0; c < NC; ++c) cfg.edited_views.insert({f, c});
        for (int f = 0; f < NF; ++f) cfg.edited_frames_3d.insert(f);
        st = {};
        CHECK(TailcycleExport::export_session(cfg, empty, &st, &status),
              "in-place save can clear every annotation: " + status);
        CHECK(!fs::exists(d / "keypoints.pq") && !fs::exists(d / "points3d.pq") &&
              !fs::exists(d / "instances.pq"), "empty corrections remove all label tables");
        TailcycleImport::Session cleared;
        CHECK(TailcycleImport::read_session(d.string(), "sess1", &cleared, &ist, &status),
              "empty annotation session imports: " + status);
        CHECK(cleared.annotations.empty(), "empty session imports with no annotations");
    }

    // ── 3d. every animal gets its own animal_id ──
    // The row key is (group, frame, animal, camera, bodypart). A shared id
    // makes rows collide, and a reader keeping the last per key silently keeps
    // one animal out of N -- which is how a save destroyed four fish.
    {
        const std::string out = root + "/t3d";
        auto cfg = make_config(out);
        AnnotationMap amap = make_annotations();
        // give every frame a second animal
        for (auto &[f, fis] : amap) {
            FrameAnnotation second = fis.front();
            second.instance_id = 1;
            for (auto &cam : second.cameras)
                for (auto &kp : cam.keypoints) kp.x += 500.0;
            fis.push_back(std::move(second));
        }
        TailcycleExport::ExportStats st;
        std::string status;
        CHECK(TailcycleExport::export_session(cfg, amap, &st, &status),
              "two-animal export succeeds: " + status);
        auto k = read_pq(fs::path(out) / "train" / "sess1_annotated" / "keypoints.pq");
        const std::set<std::string> want{"a00", "a01"};
        CHECK(k && dict_values(k, "animal_id") == want,
              "each instance writes its own animal_id");
        // and no key collides
        std::set<std::string> keys;
        bool dup = false;
        for (int64_t r = 0; k && r < k->num_rows(); r++) {
            std::string key = dict_at(k, "animal_id", r) + "|" +
                              std::to_string(int_at(k, "frame", r)) + "|" +
                              dict_at(k, "camera", r) + "|" +
                              dict_at(k, "bodypart", r);
            if (!keys.insert(key).second) dup = true;
        }
        CHECK(!dup, "rule 9: no duplicate key once animals are distinguished");
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

    // ── 4b. single-camera 2D uses only the required calibration metadata ──
    {
        const std::string out = root + "/t4b";
        auto cfg = make_config(out);
        cfg.camera_names = {"camA"};
        cfg.calibration.resize(1);
        cfg.layers = TailcycleExport::ExportConfig::Layers::TwoD;
        TailcycleExport::ExportStats st;
        std::string status;
        CHECK(TailcycleExport::export_session(cfg, make_annotations(), &st, &status),
              "minimal 2D export succeeds without measured calibration: " + status);
        const fs::path d = fs::path(out) / "train" / "sess1_annotated";
        const std::string calib = slurp(d / "calibration.toml");
        CHECK(calib.find("name = \"camA\"") != std::string::npos &&
              calib.find("size = [ 1280, 960,") != std::string::npos &&
              calib.find("offset = [ 0.0, 0.0,") != std::string::npos,
              "2D calibration retains the required name, size, and zero offset");
        CHECK(calib.find("matrix =") == std::string::npos &&
              calib.find("rotation =") == std::string::npos &&
              calib.find("translation =") == std::string::npos,
              "2D calibration does not invent camera geometry");
        TailcycleImport::Session imported;
        TailcycleImport::ImportStats ist;
        CHECK(TailcycleImport::read_session(d.string(), "sess1", &imported, &ist, &status),
              "minimal 2D session imports: " + status);
        CHECK(imported.mode == "2d" && imported.camera_names == std::vector<std::string>{"camA"},
              "2D camera identity survives round-trip");
        CHECK(imported.calibration.size() == 1 && imported.calibration[0].image_width == 1280 &&
              imported.calibration[0].image_height == 960 &&
              std::abs(imported.calibration[0].k(0, 0) - 1280.0) < 1e-9,
              "image size and nominal pinhole survive round-trip");
    }

    // ── 4c. boxes go to instances.pq, one labeled row each ──
    {
        const std::string out = root + "/t4c";
        auto cfg = make_config(out);
        AnnotationMap amap = make_annotations();
        {
            CameraExtras &e = amap.at(0).front().cameras[0].get_extras();
            e.bbox_x = 10; e.bbox_y = 20; e.bbox_w = 30; e.bbox_h = 40; e.has_bbox = true;
        }
        {
            // A second animal with a box and nothing else.
            FrameAnnotation boxed = make_frame(NN, NC, 1, /*instance_id=*/1);
            CameraExtras &e = boxed.cameras[1].get_extras();
            e.bbox_x = 100; e.bbox_y = 110; e.bbox_w = 50; e.bbox_h = 60; e.has_bbox = true;
            amap.at(1).push_back(std::move(boxed));
        }
        TailcycleExport::ExportStats st;
        std::string status;
        CHECK(TailcycleExport::export_session(cfg, amap, &st, &status),
              "export with boxes succeeds: " + status);
        const fs::path A = fs::path(out) / "train" / "sess1_annotated";
        const fs::path T = fs::path(out) / "train" / "sess1_tracked";
        auto it = read_pq(A / "instances.pq");
        CHECK(it != nullptr, "instances.pq written when boxes exist");
        // Only the two boxes: a keypoint view without a box has no row.
        CHECK(it && it->num_rows() == 2, "one row per box and none for unboxed views");
        CHECK(it && (dict_values(it, "status") == std::set<std::string>{"labeled"}),
              "boxes are labeled");
        bool box_ok = false, present_ok = false;
        for (int64_t r = 0; it && r < it->num_rows(); r++) {
            auto fcol = [&](const char *n) {
                auto a = std::static_pointer_cast<arrow::FloatArray>(it->GetColumnByName(n)->chunk(0));
                return a->IsNull(r) ? -1.0f : a->Value(r);
            };
            if (int_at(it, "frame", r) == 0 && dict_at(it, "camera", r) == "camA" &&
                dict_at(it, "animal_id", r) == "a00")
                box_ok = fcol("x0") == 10.0f && fcol("y0") == 20.0f &&
                         fcol("x1") == 40.0f && fcol("y1") == 60.0f &&
                         dict_at(it, "status", r) == "labeled";
            if (dict_at(it, "animal_id", r) == "a01")
                present_ok = int_at(it, "frame", r) == 1 && dict_at(it, "camera", r) == "camB" &&
                             dict_at(it, "status", r) == "labeled" && fcol("x1") == 150.0f;
        }
        CHECK(box_ok, "box is [x0,x1) x [y0,y1) in image coordinates");
        CHECK(present_ok, "box-only animal is written as labeled with its box");
        CHECK(!fs::exists(T / "instances.pq"), "a session with no boxes gets no instances.pq");

        TailcycleImport::Session imported;
        TailcycleImport::ImportStats ist;
        CHECK(TailcycleImport::read_session(A.string(), "sess1", &imported, &ist, &status),
              "session with instances.pq imports: " + status);
        bool rt = false;
        if (imported.annotations.count(0))
            for (const auto &fa : imported.annotations.at(0))
                if (fa.instance_id == 0 && fa.cameras[0].has_bbox()) {
                    const auto &e = fa.cameras[0].get_extras();
                    rt = e.bbox_x == 10 && e.bbox_y == 20 && e.bbox_w == 30 && e.bbox_h == 40;
                }
        CHECK(rt, "box round-trips through instances.pq");
    }

    // ── 4c2. an absent mark is an `absent` row with no box, and comes back ──
    {
        const std::string out = root + "/t4c2";
        auto cfg = make_config(out);
        AnnotationMap amap = make_annotations();
        {
            CameraExtras &e = amap.at(0).front().cameras[0].get_extras();
            e.bbox_x = 10; e.bbox_y = 20; e.bbox_w = 30; e.bbox_h = 40; e.has_bbox = true;
        }
        // Animal a00 is not in camB on frame 0.
        set_absent(amap.at(0).front().cameras[1], true);
        TailcycleExport::ExportStats st;
        std::string status;
        CHECK(TailcycleExport::export_session(cfg, amap, &st, &status),
              "export with an absent mark succeeds: " + status);
        const fs::path A = fs::path(out) / "train" / "sess1_annotated";
        auto it = read_pq(A / "instances.pq");
        CHECK(it && it->num_rows() == 2, "a box row and an absent row");
        bool absent_ok = false;
        for (int64_t r = 0; it && r < it->num_rows(); r++) {
            if (dict_at(it, "status", r) != "absent") continue;
            auto x0 = std::static_pointer_cast<arrow::FloatArray>(
                it->GetColumnByName("x0")->chunk(0));
            absent_ok = int_at(it, "frame", r) == 0 && dict_at(it, "camera", r) == "camB" &&
                        dict_at(it, "animal_id", r) == "a00" && x0->IsNull(r);
        }
        CHECK(absent_ok, "absent row: frame 0, camB, a00, no box");

        TailcycleImport::Session imported;
        TailcycleImport::ImportStats ist;
        CHECK(TailcycleImport::read_session(A.string(), "sess1", &imported, &ist, &status),
              "session with an absent row imports: " + status);
        bool rt = false;
        if (imported.annotations.count(0))
            for (const auto &fa : imported.annotations.at(0))
                if (fa.instance_id == 0)
                    rt = fa.cameras[1].is_absent() && !fa.cameras[1].has_bbox() &&
                         fa.cameras[0].has_bbox() && !fa.cameras[0].is_absent();
        CHECK(rt, "absent round-trips through instances.pq");
    }

    // ── 4d. a boxes-only project exports as a detection-only session ──
    {
        const std::string out = root + "/t4d";
        auto cfg = make_config(out);
        cfg.camera_names = {"camA"};
        cfg.calibration.resize(1);
        AnnotationMap amap;
        for (u32 f : {1u, 3u}) {
            FrameAnnotation fa = make_frame(NN, 1, f);
            CameraExtras &e = fa.cameras[0].get_extras();
            e.bbox_x = 5.0 * f; e.bbox_y = 6; e.bbox_w = 70; e.bbox_h = 80; e.has_bbox = true;
            amap[f] = FrameInstances{std::move(fa)};
        }
        TailcycleExport::ExportStats st;
        std::string status;
        CHECK(TailcycleExport::export_session(cfg, amap, &st, &status),
              "boxes-only export succeeds: " + status);
        CHECK(st.sessions_written == 1 && st.instance_rows == 2, "one session, two box rows");
        const fs::path d = fs::path(out) / "train" / "sess1";
        CHECK(fs::exists(d / "instances.pq") && !fs::exists(d / "keypoints.pq") &&
              !fs::exists(d / "points3d.pq"),
              "instances.pq is the only label table");
        CHECK(slurp(d / "session.toml").find("labels = \"annotated\"") != std::string::npos,
              "hand-drawn boxes make an annotated session");
        auto it = read_pq(d / "instances.pq");
        CHECK(it && (dict_values(it, "status") == std::set<std::string>{"labeled"}),
              "box-only rows are labeled");

        TailcycleImport::Session imported;
        TailcycleImport::ImportStats ist;
        CHECK(TailcycleImport::read_session(d.string(), "sess1", &imported, &ist, &status),
              "boxes-only session imports: " + status);
        CHECK(imported.has_boxes && !imported.has_2d && !imported.has_3d,
              "import reports a boxes-only session");
        CHECK(imported.annotations.count(3) &&
              imported.annotations.at(3).front().cameras[0].has_bbox() &&
              imported.annotations.at(3).front().cameras[0].get_extras().bbox_x == 15.0,
              "boxes round-trip without any keypoints");

    }

    // ── 4e. several clips of one recording become several groups ──
    {
        const std::string out = root + "/t4e";
        auto cfg = make_config(out);
        cfg.force_labels = "annotated";
        // red frames 100..104 are labelled; clips cover 100-101 and 103-104,
        // so frame 102 falls between groups and is not exported.
        cfg.groups = {{"rec_ix100", 2, 100}, {"rec_ix103", 2, 103}};
        TailcycleExport::ExportStats st;
        std::string status;
        CHECK(TailcycleExport::export_session(cfg, make_annotations(100), &st, &status),
              "multi-group export succeeds: " + status);
        const fs::path d = fs::path(out) / "train" / "sess1";
        CHECK(fs::is_directory(d / "groups" / "rec_ix100") &&
              fs::is_directory(d / "groups" / "rec_ix103"), "one folder per group");
        auto g = read_pq(d / "groups.pq");
        CHECK(g && g->num_rows() == 2, "one groups.pq row per group");
        CHECK(g && int_at(g, "source_frame_start", 0) == 100 && int_at(g, "n_frames", 0) == 2 &&
              int_at(g, "source_frame_start", 1) == 103, "one groups.pq row per group, in order");
        auto k = read_pq(d / "keypoints.pq");
        CHECK(k && dict_values(k, "group_id") == (std::set<std::string>{"rec_ix100", "rec_ix103"}),
              "rows are keyed by their own group");
        bool rebased = true;
        for (int64_t r = 0; k && r < k->num_rows(); r++)
            if (int_at(k, "frame", r) < 0 || int_at(k, "frame", r) > 1) rebased = false;
        CHECK(rebased, "frames are rebased into each group");
        // 2 cameras x 3 nodes on frames 100, 101, 103 (frame 103 lacks TailBase) + 104
        CHECK(k && k->num_rows() == 6 + 6 + 4 + 6, "frame 102, between groups, is dropped");

        TailcycleImport::Session imported;
        TailcycleImport::ImportStats ist;
        CHECK(TailcycleImport::read_session(d.string(), "rec_ix103", &imported, &ist, &status),
              "a group of a multi-group session imports: " + status);
        CHECK(imported.n_frames == 2 && imported.source_frame_start == 103 &&
              imported.annotations.size() == 2, "the chosen group's frames only");
    }

    // ── 4f. a box whose view's keypoints are tracked goes to _tracked only ──
    {
        const std::string out = root + "/t4f";
        auto cfg = make_config(out);
        cfg.layers = TailcycleExport::ExportConfig::Layers::TwoDAndThreeD;
        AnnotationMap amap = make_annotations();
        // frame 4: predicted 2D, triangulated (annotated) Snout 3D
        CameraExtras &e = amap.at(4).front().cameras[0].get_extras();
        e.bbox_x = 1; e.bbox_y = 2; e.bbox_w = 3; e.bbox_h = 4; e.has_bbox = true;
        TailcycleExport::ExportStats st;
        std::string status;
        CHECK(TailcycleExport::export_session(cfg, amap, &st, &status),
              "split export with a box succeeds: " + status);
        const fs::path A = fs::path(out) / "train" / "sess1_annotated";
        const fs::path T = fs::path(out) / "train" / "sess1_tracked";
        CHECK(!fs::exists(A / "instances.pq"), "the box is not duplicated into _annotated");
        CHECK(row_set(T / "instances.pq").size() == 1, "the box goes with its view's keypoints");
    }

    // ── 4g. label comparison, which decides what an in-place save rewrites ──
    {
        AnnotationMap amap = make_annotations();
        const FrameInstances was = amap.at(0);
        FrameInstances now = was;
        CHECK(view_labels_equal(&was, &now, 0) && frame_3d_equal(&was, &now), "a copy is equal");
        now.front().cameras[0].get_extras().has_bbox = true;
        now.front().cameras[0].get_extras().bbox_w = 5;
        now.front().cameras[0].get_extras().bbox_h = 5;
        CHECK(!view_labels_equal(&was, &now, 0), "an added box edits its view");
        CHECK(view_labels_equal(&was, &now, 1), "... and no other view");
        CHECK(frame_3d_equal(&was, &now), "... and not the 3D");
        now = was;
        now.front().cameras[1].keypoints[0].x += 1;
        CHECK(!view_labels_equal(&was, &now, 1) && view_labels_equal(&was, &now, 0),
              "a moved keypoint edits its own view");
        now.front().cameras[1].keypoints[0].x -= 1;
        CHECK(view_labels_equal(&was, &now, 1), "an edit that is undone is not an edit");
        now.front().kp3d[0].x += 1;
        CHECK(!frame_3d_equal(&was, &now) && view_labels_equal(&was, &now, 0),
              "a 3D change edits the 3D only");
        now = was;
        now.push_back(make_frame(NN, NC, 0, /*instance_id=*/7));
        CHECK(view_labels_equal(&was, &now, 0) && frame_3d_equal(&was, &now),
              "an animal with nothing labelled is no animal");
        CameraExtras &e = now.back().cameras[1].get_extras();
        e.bbox_w = e.bbox_h = 3; e.has_bbox = true;
        CHECK(view_labels_equal(&was, &now, 0) && !view_labels_equal(&was, &now, 1),
              "a new animal edits only the views it is labelled in");
        FrameInstances empty{make_frame(NN, NC, 0)};
        CHECK(view_labels_equal(nullptr, &empty, 0) && frame_3d_equal(&empty, nullptr),
              "an empty frame is the same as no frame");
    }

    // ── 4h. saving corrections: rewrite what was edited, keep everything else ──
    // A two-group session with the rows red never writes itself: `unlabeled`
    // keypoints, a `missing` 3D point, `present` and `absent` instances, and a
    // null-box `labeled` row as older red exports wrote.
    {
        const std::string out = root + "/t4h";
        auto base = make_config(out);
        base.force_labels = "annotated";
        base.layers = TailcycleExport::ExportConfig::Layers::TwoDAndThreeD;
        base.groups = {{"gA", 2, 0}, {"gB", 2, 2}};
        const fs::path d = fs::path(out) / "train" / "sess1";
        const std::vector<KRow> kp0 = {
            {"gA", 0, "a00", "camA", "Snout", "visible", 10, 20},
            {"gA", 0, "a00", "camA", "EarL", "unlabeled", NA, NA},
            {"gA", 0, "a00", "camB", "Snout", "visible", 30, 40},
            {"gA", 0, "a00", "camB", "EarL", "unlabeled", NA, NA},
            {"gA", 1, "a00", "camA", "Snout", "visible", 11, 21},
            {"gB", 0, "a00", "camA", "Snout", "visible", 50, 60},
        };
        const std::vector<PRow> p30 = {
            {"gA", 0, "a00", "Snout", "visible", 1, 2, 3},
            {"gA", 0, "a00", "EarL", "missing", NA, NA, NA},
            {"gB", 0, "a00", "Snout", "visible", 4, 5, 6},
        };
        const std::vector<IRow> in0 = {
            {"gA", 0, "a00", "camA", "present", 1, 2, 11, 12},
            {"gA", 0, "a01", "camA", "absent", NA, NA, NA, NA},
            {"gA", 0, "a02", "camA", "present", NA, NA, NA, NA},
            {"gA", 0, "a00", "camB", "present", 5, 5, 15, 15},
            {"gA", 0, "a01", "camB", "absent", NA, NA, NA, NA},
            {"gA", 1, "a00", "camA", "labeled", NA, NA, NA, NA},   // invalid (rule 11)
            {"gA", 1, "a00", "camB", "labeled", 7, 7, 17, 17},
            {"gB", 0, "a00", "camA", "labeled", 3, 3, 13, 13},
            {"gB", 0, "a00", "camB", "labeled", NA, NA, NA, NA},   // other group: untouched
        };
        // Build the session (its TOMLs and groups.pq) and put the fixture tables in.
        auto reset = [&]() {
            fs::remove_all(out);
            TailcycleExport::ExportStats st;
            std::string status;
            if (!TailcycleExport::export_session(base, make_annotations(), &st, &status))
                std::cout << "  fixture export failed: " << status << "\n";
            write_keypoints(d / "keypoints.pq", kp0);
            write_points3d(d / "points3d.pq", p30);
            write_instances(d / "instances.pq", in0);
        };
        reset();
        const auto K0 = row_set(d / "keypoints.pq");
        const auto P0 = row_set(d / "points3d.pq");
        const auto I0 = row_set(d / "instances.pq");

        // What tailcycle_save_session does, for group gA.
        auto open_gA = [&](TailcycleImport::Session *s) {
            TailcycleImport::ImportStats ist;
            std::string status;
            bool ok = TailcycleImport::read_session(d.string(), "gA", s, &ist, &status);
            CHECK(ok, "fixture session imports: " + status);
            return ok;
        };
        auto save_gA = [&](const TailcycleImport::Session &s, const AnnotationMap &amap,
                           std::set<std::pair<int, int>> views, std::set<int> frames3d) {
            TailcycleExport::ExportConfig cfg = base;
            cfg.groups.clear();
            cfg.group_id = s.group_id;
            cfg.n_frames = s.n_frames;
            cfg.source_frame_start = 0;
            cfg.animal_ids = s.animal_ids;
            cfg.in_place = true;
            cfg.edited_views = std::move(views);
            cfg.edited_frames_3d = std::move(frames3d);
            TailcycleExport::ExportStats st;
            std::string status;
            const bool ok = TailcycleExport::export_session(cfg, amap, &st, &status);
            CHECK(ok, "in-place save succeeds: " + status);
            return ok;
        };
        const std::string legacy = *rows_where(I0, {"frame=1", "camera=camA", "status=labeled"}).begin();
        auto without = [](std::set<std::string> s, const std::set<std::string> &drop) {
            for (const auto &r : drop) s.erase(r);
            return s;
        };

        // Import: `labeled` and `present` boxes load as boxes, `absent` rows as
        // absent marks; box-less `present` rows do not load.
        {
            TailcycleImport::Session s;
            if (open_gA(&s)) {
                int boxes = 0, absents = 0;
                for (const auto &[f, fis] : s.annotations)
                    for (const auto &fa : fis)
                        for (const auto &cam : fa.cameras) {
                            boxes += cam.has_bbox();
                            absents += cam.is_absent();
                        }
                CHECK(boxes == 3, "the two present boxes and the labeled one load");
                CHECK(absents == 2, "a01's two absent rows load as absent marks");
            }
        }
        // An unedited save keeps every row; only the invalid null-box labeled row goes.
        {
            TailcycleImport::Session s;
            if (open_gA(&s) && save_gA(s, s.annotations, {}, {})) {
                CHECK(row_set(d / "keypoints.pq") == K0, "unedited: keypoints unchanged");
                CHECK(row_set(d / "points3d.pq") == P0, "unedited: 3D unchanged");
                CHECK(row_set(d / "instances.pq") == without(I0, {legacy}),
                      "unedited: instances unchanged but for the null-box labeled row");
            }
        }
        // Moving camA's box at frame 0 rewrites that view alone.
        reset();
        {
            TailcycleImport::Session s;
            if (open_gA(&s)) {
                AnnotationMap amap = s.annotations;
                for (auto &fa : amap.at(0))
                    if (fa.cameras[0].has_bbox()) fa.cameras[0].extras->bbox_x += 100;
                if (save_gA(s, amap, {{0, 0}}, {})) {
                    const auto I = row_set(d / "instances.pq");
                    const auto view = rows_where(I, {"group_id=gA", "frame=0", "camera=camA"});
                    // Exactly red's: a00's box and a01's absent mark; a02's
                    // box-less `present` row, which red cannot hold, goes.
                    CHECK(view.size() == 2, "edited view: exactly red's box and absent mark");
                    const auto lab = rows_where(I, {"group_id=gA", "frame=0", "camera=camA",
                                                    "status=labeled"});
                    CHECK(lab.size() == 1 &&
                              lab.begin()->find("x0=101.000000;") != std::string::npos,
                          "edited view: the present box is labeled, at its new place");
                    CHECK(rows_where(I, {"group_id=gA", "frame=0", "camera=camA",
                                         "animal_id=a01", "status=absent"}).size() == 1,
                          "edited view: the absent mark is written back");
                    CHECK(rows_where(I, {"frame=0", "camera=camB"}) ==
                              rows_where(I0, {"frame=0", "camera=camB"}),
                          "the other camera at that frame is untouched");
                    CHECK(rows_where(I, {"group_id=gB"}) == rows_where(I0, {"group_id=gB"}),
                          "the other group is untouched");
                    CHECK(row_set(d / "keypoints.pq") == K0,
                          "edited view: keypoints regenerate the same, unlabeled rows kept");
                    CHECK(row_set(d / "points3d.pq") == P0, "no 3D edit: 3D unchanged");
                }
            }
        }
        // A 3D edit rewrites that frame's 3D, keeping the `missing` row.
        reset();
        {
            TailcycleImport::Session s;
            if (open_gA(&s)) {
                AnnotationMap amap = s.annotations;
                amap.at(0).front().kp3d[0].x = 9;
                if (save_gA(s, amap, {}, {0})) {
                    const auto P = row_set(d / "points3d.pq");
                    CHECK(P.size() == 3, "3D: one row per point, none lost");
                    CHECK(rows_where(P, {"group_id=gA", "frame=0", "bodypart=Snout", "x=9.000000"}).size() == 1,
                          "3D: the edit is written");
                    CHECK(rows_where(P, {"status=missing"}).size() == 1,
                          "3D: the missing row red never loads is kept");
                    // (The edited point now carries a score column -- an imported
                    // point reads back with confidence 1 -- so compare values.)
                    CHECK(rows_where(P, {"group_id=gB", "x=4.000000", "y=5.000000", "z=6.000000"}).size() == 1,
                          "3D: the other group is untouched");
                    CHECK(row_set(d / "instances.pq") == without(I0, {legacy}),
                          "3D: instances untouched");
                }
            }
        }
        // Deleting every box and keypoint of gA empties gA and nothing else.
        reset();
        {
            TailcycleImport::Session s;
            if (open_gA(&s)) {
                AnnotationMap amap = s.annotations;
                for (auto &[f, fis] : amap)
                    for (auto &fa : fis)
                        for (auto &cam : fa.cameras) {
                            if (cam.extras) {
                                cam.extras->has_bbox = false;
                                cam.extras->absent = false;
                            }
                            for (auto &kp : cam.keypoints) kp = Keypoint2D{};
                        }
                if (save_gA(s, amap, {{0, 0}, {0, 1}, {1, 0}, {1, 1}}, {})) {
                    const auto I = row_set(d / "instances.pq");
                    CHECK(rows_where(I, {"group_id=gA"}).empty(),
                          "every gA row goes once its boxes and absent marks are cleared");
                    CHECK(rows_where(I, {"group_id=gB"}) == rows_where(I0, {"group_id=gB"}),
                          "gB's instances are untouched");
                    const auto K = row_set(d / "keypoints.pq");
                    CHECK(rows_where(K, {"group_id=gA", "status=visible"}).empty(),
                          "deleted keypoints are gone");
                    CHECK(rows_where(K, {"status=unlabeled"}).size() == 2 &&
                              rows_where(K, {"group_id=gB"}).size() == 1,
                          "unlabeled rows and gB's keypoints are kept");
                }
            }
        }
        // A box on a new animal gets an id no animal in the session has.
        reset();
        {
            TailcycleImport::Session s;
            if (open_gA(&s)) {
                AnnotationMap amap = s.annotations;
                FrameAnnotation extra = make_frame(NN, NC, 1, (int)s.animal_ids.size());
                CameraExtras &e = extra.cameras[0].get_extras();
                e.bbox_x = 50; e.bbox_y = 50; e.bbox_w = 10; e.bbox_h = 10; e.has_bbox = true;
                amap.at(1).push_back(std::move(extra));
                if (save_gA(s, amap, {{1, 0}}, {})) {
                    const auto view = rows_where(row_set(d / "instances.pq"),
                                                 {"group_id=gA", "frame=1", "camera=camA"});
                    CHECK(view.size() == 1 && view.begin()->find("animal_id=a03;") != std::string::npos,
                          "a01 and a02 are taken on disk, so the new animal is a03");
                }
            }
        }
    }

    // ── 4i. an in-place save that empties a table deletes it ──
    {
        const std::string out = root + "/t4i";
        auto cfg = make_config(out);
        cfg.camera_names = {"camA"};
        cfg.calibration.resize(1);
        cfg.force_labels = "annotated";
        AnnotationMap amap;
        FrameAnnotation fa = make_frame(NN, 1, 2);
        CameraExtras &e = fa.cameras[0].get_extras();
        e.bbox_x = 1; e.bbox_y = 1; e.bbox_w = 5; e.bbox_h = 5; e.has_bbox = true;
        amap[2] = FrameInstances{std::move(fa)};
        TailcycleExport::ExportStats st;
        std::string status;
        CHECK(TailcycleExport::export_session(cfg, amap, &st, &status), "box export: " + status);
        const fs::path d = fs::path(out) / "train" / "sess1";
        amap.at(2).front().cameras[0].extras->has_bbox = false;
        cfg.in_place = true;
        cfg.edited_views = {{2, 0}};
        CHECK(TailcycleExport::export_session(cfg, amap, &st, &status),
              "deleting the last box saves: " + status);
        CHECK(!fs::exists(d / "instances.pq"), "the emptied instances.pq is deleted");
        CHECK(!fs::exists(d / "instances.pq.tmp"), "no temporary file is left behind");
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
