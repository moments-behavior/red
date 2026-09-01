// Built as its own C++20 target (see CMakeLists): Arrow 25 uses std::span in
// its public headers, and red is C++17 everywhere else.

#include "tailcycle_export.h"

#if defined(RED_HAVE_PARQUET)
#include "tailcycle_schema.h"
#include "red_math.h"
#include <arrow/api.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <system_error>
#endif

namespace TailcycleExport {

#if !defined(RED_HAVE_PARQUET)

bool available() { return false; }
bool export_session(const ExportConfig &, const AnnotationMap &, ExportStats *,
                    std::string *status) {
    if (status) *status = "This build has no Parquet support (Arrow not found at configure time).";
    return false;
}

#else

bool available() { return true; }

namespace fs = std::filesystem;

namespace {

// Which of the two sessions (§2.6) a point belongs to. The partition is per
// *point*, not per frame: red lets a frame hold hand-placed keypoints in one
// camera and predicted ones in another.
enum class Bucket { Annotated, Tracked };

// `Imported` is machine output, not a third category. Nothing in red imports
// hand-made labels into it: red's own CSV round-trips the source letter, so
// hand labels reload as Manual, and the only writers of Imported are the
// JARVIS importer (gui/jarvis_import_window.h) and red.cpp, which sets it
// with the comment "predicted, awaiting review". If a future import path
// brings in genuine human labels, it should set Manual rather than teach this
// function a new case.
Bucket bucket_2d(LabelSource s) {
    switch (s) {
    case LabelSource::Manual:    return Bucket::Annotated;
    case LabelSource::Predicted: return Bucket::Tracked;
    case LabelSource::Imported:  return Bucket::Tracked;
    }
    return Bucket::Annotated;
}

Bucket bucket_3d(Kp3DSource s) {
    switch (s) {
    case Kp3DSource::Triangulated: return Bucket::Annotated;  // derived from 2D labels
    case Kp3DSource::Imported:     return Bucket::Tracked;    // model predictions
    case Kp3DSource::None:         return Bucket::Annotated;
    }
    return Bucket::Annotated;
}

std::string toml_str(const std::string &v) { return "\"" + v + "\""; }

std::string toml_name_list(const std::vector<std::string> &v) {
    std::string s = "[ ";
    for (size_t i = 0; i < v.size(); i++) { s += toml_str(v[i]); s += i + 1 < v.size() ? ", " : ",";}
    return s + "]";
}

// A rotation with det = -1 (red allows these; see projectPointR in red_math.h)
// has no Rodrigues vector. Writing rvec anyway would produce a calibration
// that parses and triangulates wrongly, which is the one outcome worth
// refusing over.
bool rotation_is_proper(const Eigen::Matrix3d &r) {
    return std::abs(r.determinant() - 1.0) < 1e-6;
}

bool write_session_toml(const fs::path &dir, const ExportConfig &cfg,
                        const char *labels, std::string *err) {
    std::ofstream o(dir / "session.toml");
    if (!o) { *err = "cannot write session.toml"; return false; }
    o << "mode = " << toml_str(cfg.camera_names.size() >= 2 ? "3d" : "2d") << "\n";
    o << "units = " << toml_str(cfg.units) << "\n";
    o << "labels = " << toml_str(labels) << "\n";
    o << "names = " << toml_name_list(cfg.node_names) << "\n";

    o << "skeleton = [";
    for (size_t i = 0; i < cfg.edges.size(); i++) {
        const auto &e = cfg.edges[i];
        if (e.first < 0 || e.second < 0 ||
            e.first >= (int)cfg.node_names.size() || e.second >= (int)cfg.node_names.size())
            continue;   // an edge naming a node outside `names` is rule-2 invalid
        o << " [ " << toml_str(cfg.node_names[e.first]) << ", "
          << toml_str(cfg.node_names[e.second]) << ",],";
    }
    o << "]\n";
    // red has no bilateral pairing metadata; guessing it from name prefixes
    // would be inventing data, so the list stays empty (§4 allows that).
    o << "flip_pairs = []\n\n";

    o << "[provenance]\n";
    o << "source = " << toml_str(cfg.provenance_source) << "\n";
    o << "annotator = " << toml_str(cfg.annotator) << "\n";
    o << "annotator_tool = " << toml_str("red") << "\n";
    return true;
}

bool write_calibration_toml(const fs::path &dir, const ExportConfig &cfg, std::string *err) {
    std::ofstream o(dir / "calibration.toml");
    if (!o) { *err = "cannot write calibration.toml"; return false; }
    o.precision(17);
    for (size_t i = 0; i < cfg.camera_names.size(); i++) {
        const CameraParams &c = cfg.calibration[i];
        o << "[cam_" << i << "]\n";
        o << "name = " << toml_str(cfg.camera_names[i]) << "\n";
        o << "size = [ " << c.image_width << ", " << c.image_height << ",]\n";
        o << "matrix = [";
        for (int r = 0; r < 3; r++) {
            o << " [ ";
            for (int k = 0; k < 3; k++) o << c.k(r, k) << ",";
            o << "],";
        }
        o << "]\n";
        o << "distortions = [";
        for (int d = 0; d < 5; d++) o << " " << c.dist_coeffs(d) << ",";
        o << "]\n";
        // red's rvec is world -> cam (red_math.h projectPoints applies
        // R * pt + tvec), which is the convention the format asks for.
        o << "rotation = [ " << c.rvec(0) << ", " << c.rvec(1) << ", " << c.rvec(2) << ",]\n";
        o << "translation = [ " << c.tvec(0) << ", " << c.tvec(1) << ", " << c.tvec(2) << ",]\n";
        o << "fisheye = false\n";
        // red has no crop model: its calibration already describes the stored
        // image, so the origin is the image origin.
        o << "offset = [ 0.0, 0.0,]\n";
        o << "moving = false\n\n";
    }
    o << "[metadata]\n";
    return true;
}

} // namespace

bool export_session(const ExportConfig &cfg, const AnnotationMap &amap,
                    ExportStats *stats, std::string *status) {
    auto t0 = std::chrono::steady_clock::now();
    ExportStats local;
    ExportStats &st = stats ? *stats : local;
    auto fail = [&](const std::string &m) { if (status) *status = m; return false; };

    // ── validation that must happen before anything is written ──
    if (cfg.camera_names.empty()) return fail("No cameras.");
    if (cfg.calibration.size() != cfg.camera_names.size())
        return fail("Calibration count does not match camera count.");
    if (cfg.node_names.empty()) return fail("Skeleton has no keypoint names.");
    if (cfg.n_frames <= 0) return fail("n_frames must come from the media and be > 0.");

    for (size_t i = 0; i < cfg.calibration.size(); i++) {
        const CameraParams &c = cfg.calibration[i];
        const std::string &n = cfg.camera_names[i];
        if (c.telecentric)
            return fail("Camera " + n + " is telecentric. calibration.toml is an aniposelib "
                        "CameraGroup, which has no telecentric model -- the file would load "
                        "cleanly and triangulate wrongly.");
        if (!rotation_is_proper(c.r))
            return fail("Camera " + n + " has an improper rotation (det != 1), which has no "
                        "Rodrigues representation.");
        if (c.image_width <= 0 || c.image_height <= 0)
            return fail("Camera " + n + " has no image size; validation rule 8 requires it.");
    }

    const std::string gid = cfg.group_id.empty() ? cfg.session_id : cfg.group_id;
    const std::string animal_id = "a00";   // matches the convention in johnson-mouse-tracked

    // ── which buckets actually have data ──
    bool has[2] = {false, false};
    for (const auto &[fnum, fa] : amap) {
        for (const auto &cam : fa.cameras)
            for (const auto &kp : cam.keypoints)
                if (kp.labeled) has[(int)bucket_2d(kp.source)] = true;
        for (const auto &k3 : fa.kp3d) {
            if (k3.source == Kp3DSource::None) continue;
            if (k3.source == Kp3DSource::Triangulated && !cfg.include_triangulated_3d) continue;
            has[(int)bucket_3d(k3.source)] = true;
        }
    }
    if (!has[0] && !has[1]) return fail("Nothing to export: no labelled points.");

    struct Job { Bucket b; const char *labels; std::string suffix; };
    std::vector<Job> jobs;
    const bool both = has[0] && has[1] && cfg.export_annotated && cfg.export_tracked;
    if (has[0] && cfg.export_annotated)
        jobs.push_back({Bucket::Annotated, Tailcycle::labels::kAnnotated, both ? "_annotated" : ""});
    if (has[1] && cfg.export_tracked)
        jobs.push_back({Bucket::Tracked, Tailcycle::labels::kTracked, both ? "_tracked" : ""});
    if (jobs.empty()) return fail("Nothing selected to export.");

    for (const Job &job : jobs) {
        const std::string sid = cfg.session_id + job.suffix;
        const fs::path dir = fs::path(cfg.output_folder) / cfg.split / sid;
        std::error_code ec;
        fs::create_directories(dir / "groups" / gid, ec);
        if (ec) return fail("Cannot create " + dir.string() + ": " + ec.message());

        std::string err;
        if (!write_session_toml(dir, cfg, job.labels, &err)) return fail(err);
        if (!write_calibration_toml(dir, cfg, &err)) return fail(err);

        // ── groups.pq ──
        {
            arrow::StringBuilder gid_b, src_b, notes_b;
            arrow::Int32Builder nf_b, start_b, step_b;
            arrow::FloatBuilder fps_b;
            auto ok = gid_b.Append(gid).ok() && nf_b.Append(cfg.n_frames).ok() &&
                      src_b.Append(cfg.source_video).ok() &&
                      start_b.Append(cfg.source_frame_start).ok() && step_b.Append(1).ok() &&
                      notes_b.Append("").ok() &&
                      (cfg.fps > 0 ? fps_b.Append(cfg.fps).ok() : fps_b.AppendNull().ok());
            if (!ok) return fail("groups.pq: builder append failed.");
            std::vector<std::shared_ptr<arrow::Array>> a(7);
            if (!gid_b.Finish(&a[0]).ok() || !nf_b.Finish(&a[1]).ok() || !fps_b.Finish(&a[2]).ok() ||
                !src_b.Finish(&a[3]).ok() || !start_b.Finish(&a[4]).ok() ||
                !step_b.Finish(&a[5]).ok() || !notes_b.Finish(&a[6]).ok())
                return fail("groups.pq: finish failed.");
            auto s = Tailcycle::write_table(
                arrow::Table::Make(Tailcycle::groups_schema(), a), (dir / "groups.pq").string());
            if (!s.ok()) return fail("groups.pq: " + s.ToString());
        }

        // ── keypoints.pq ──
        // Every point red exports is `projected`: red records placement, not
        // visibility, and §7 is explicit that labels which assert nothing about
        // occlusion must not be written as `visible`.
        {
            arrow::StringDictionary32Builder g_b, a_b, c_b, p_b, s_b;
            arrow::Int32Builder f_b;
            arrow::FloatBuilder x_b, y_b, sc_b;
            bool any_score = false;
            int rows = 0;

            for (const auto &[fnum, fa] : amap) {
                const int frame = (int)fnum - cfg.source_frame_start;
                if (frame < 0 || frame >= cfg.n_frames) continue;   // rule 6
                for (size_t ci = 0; ci < fa.cameras.size() && ci < cfg.camera_names.size(); ci++) {
                    const auto &cam = fa.cameras[ci];
                    // red stores 2D keypoints in ImPlot coordinates, whose origin
                    // is the BOTTOM-left of the image. Every other exporter flips
                    // (see jarvis_export.h, "ImPlot -> image coords"), and the
                    // format, the calibration and the extracted JPEGs all use a
                    // top-left origin. Without this the labels look plausible --
                    // they sit inside the frame and move smoothly -- but nothing
                    // triangulates: reprojection residuals run to hundreds of px.
                    const double img_h = (double)cfg.calibration[ci].image_height;
                    for (size_t ni = 0; ni < cam.keypoints.size() && ni < cfg.node_names.size(); ni++) {
                        const Keypoint2D &kp = cam.keypoints[ni];
                        if (!kp.labeled) continue;   // no row, not `unlabeled` (§7)
                        if (bucket_2d(kp.source) != job.b) continue;
                        if (!g_b.Append(gid).ok() || !f_b.Append(frame).ok() ||
                            !a_b.Append(animal_id).ok() ||
                            !c_b.Append(cfg.camera_names[ci]).ok() ||
                            !p_b.Append(cfg.node_names[ni]).ok() ||
                            !s_b.Append(Tailcycle::status::kProjected).ok() ||
                            !x_b.Append((float)kp.x).ok() ||
                            !y_b.Append((float)(img_h - kp.y)).ok())
                            return fail("keypoints.pq: builder append failed.");
                        // A human label carries no confidence -- red stores 0.0f,
                        // and passing that through would ship every hand-placed
                        // point with a score of zero (§7 says null).
                        const bool scored = kp.source != LabelSource::Manual && kp.confidence > 0.0f;
                        if (scored) any_score = true;
                        if (!(scored ? sc_b.Append(kp.confidence) : sc_b.AppendNull()).ok())
                            return fail("keypoints.pq: score append failed.");
                        rows++;
                    }
                }
            }

            if (rows > 0) {
                std::vector<std::shared_ptr<arrow::Array>> a(any_score ? 9 : 8);
                if (!g_b.Finish(&a[0]).ok() || !f_b.Finish(&a[1]).ok() || !a_b.Finish(&a[2]).ok() ||
                    !c_b.Finish(&a[3]).ok() || !p_b.Finish(&a[4]).ok() || !s_b.Finish(&a[5]).ok() ||
                    !x_b.Finish(&a[6]).ok() || !y_b.Finish(&a[7]).ok())
                    return fail("keypoints.pq: finish failed.");
                if (any_score && !sc_b.Finish(&a[8]).ok())
                    return fail("keypoints.pq: score finish failed.");
                auto s = Tailcycle::write_table(
                    arrow::Table::Make(Tailcycle::keypoints_schema(any_score), a),
                    (dir / "keypoints.pq").string());
                if (!s.ok()) return fail("keypoints.pq: " + s.ToString());
                st.keypoint_rows += rows;
            }
        }

        // ── points3d.pq ──
        {
            arrow::StringDictionary32Builder g_b, a_b, p_b, s_b;
            arrow::Int32Builder f_b;
            arrow::FloatBuilder x_b, y_b, z_b, sc_b;
            bool any_score = false;
            int rows = 0;

            for (const auto &[fnum, fa] : amap) {
                const int frame = (int)fnum - cfg.source_frame_start;
                if (frame < 0 || frame >= cfg.n_frames) continue;
                for (size_t ni = 0; ni < fa.kp3d.size() && ni < cfg.node_names.size(); ni++) {
                    const Keypoint3D &k3 = fa.kp3d[ni];
                    if (k3.source == Kp3DSource::None) continue;
                    if (k3.source == Kp3DSource::Triangulated && !cfg.include_triangulated_3d)
                        continue;
                    if (bucket_3d(k3.source) != job.b) continue;
                    if (!g_b.Append(gid).ok() || !f_b.Append(frame).ok() ||
                        !a_b.Append(animal_id).ok() || !p_b.Append(cfg.node_names[ni]).ok() ||
                        !s_b.Append(Tailcycle::status::kVisible).ok() ||
                        !x_b.Append((float)k3.x).ok() || !y_b.Append((float)k3.y).ok() ||
                        !z_b.Append((float)k3.z).ok())
                        return fail("points3d.pq: builder append failed.");
                    const bool scored = k3.source == Kp3DSource::Imported && k3.confidence > 0.0f;
                    if (scored) any_score = true;
                    if (!(scored ? sc_b.Append(k3.confidence) : sc_b.AppendNull()).ok())
                        return fail("points3d.pq: score append failed.");
                    rows++;
                }
            }

            if (rows > 0) {
                std::vector<std::shared_ptr<arrow::Array>> a(any_score ? 9 : 8);
                if (!g_b.Finish(&a[0]).ok() || !f_b.Finish(&a[1]).ok() || !a_b.Finish(&a[2]).ok() ||
                    !p_b.Finish(&a[3]).ok() || !s_b.Finish(&a[4]).ok() || !x_b.Finish(&a[5]).ok() ||
                    !y_b.Finish(&a[6]).ok() || !z_b.Finish(&a[7]).ok())
                    return fail("points3d.pq: finish failed.");
                if (any_score && !sc_b.Finish(&a[8]).ok())
                    return fail("points3d.pq: score finish failed.");
                auto s = Tailcycle::write_table(
                    arrow::Table::Make(Tailcycle::points3d_schema(any_score), a),
                    (dir / "points3d.pq").string());
                if (!s.ok()) return fail("points3d.pq: " + s.ToString());
                st.points3d_rows += rows;
            }
        }

        // At least one of keypoints.pq / points3d.pq must exist and be
        // non-empty (§3). Reaching here with neither means the bucket scan and
        // the row loops disagreed, which is a bug rather than bad input.
        if (!fs::exists(dir / "keypoints.pq") && !fs::exists(dir / "points3d.pq"))
            return fail("Session " + sid + " would have no label table.");

        st.sessions.push_back(dir.string());
        st.sessions_written++;
    }

    st.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (status) {
        char buf[256];
        snprintf(buf, sizeof(buf), "Wrote %d session%s: %d 2D rows, %d 3D rows (%.1fs)",
                 st.sessions_written, st.sessions_written == 1 ? "" : "s",
                 st.keypoint_rows, st.points3d_rows, st.elapsed_seconds);
        *status = buf;
    }
    return true;
}

#endif // RED_HAVE_PARQUET

} // namespace TailcycleExport
