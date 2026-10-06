// Built as its own C++20 target (see CMakeLists): Arrow 25 uses std::span in
// its public headers, and red is C++17 everywhere else.

#include "tailcycle_export.h"

#if defined(RED_HAVE_PARQUET)
#include "tailcycle_read.h"
#include "tailcycle_schema.h"
#include "red_math.h"
#include <arrow/api.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <system_error>
#include <tuple>
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
// camera and projected/predicted ones in another.
enum class Bucket { Annotated, Tracked };

// `Imported` is machine output, not a third category. Nothing in red imports
// hand-made labels into it: red's own CSV round-trips the source letter, so
// hand labels reload as Manual, and the only writers of Imported are the
// JARVIS importer (gui/jarvis_import_window.h) and red.cpp, which sets it
// with the comment "predicted, awaiting review". If a future import path
// brings in genuine human labels, it should set Manual rather than teach this
// function a new case.
Bucket bucket_2d(const Keypoint2D &kp) {
    // An OCCLUSION is a judgement, and only a model makes a predicted one.
    // So an occluded point is annotated unless something claimed to predict
    // it -- including one with no author at all, which is what marking a
    // never-placed node hidden produces. Requiring `manual` here instead made
    // every such row bucket as tracked, and a session containing both split
    // in two: half your afternoon filed as machine output.
    if (kp.is_occluded()) return kp.is_predicted() ? Bucket::Tracked : Bucket::Annotated;

    // A POSITION is annotated only if a person put it there; anything derived
    // or predicted is tracked. A reprojection buckets as tracked to match
    // what this did when reprojection and prediction shared one enum value --
    // whether a reprojection of YOUR labels ought to count as annotated is a
    // separate question, and it needs kp3d's origin to answer.
    return kp.is_manual() ? Bucket::Annotated : Bucket::Tracked;
}

Bucket bucket_3d(const Keypoint3D &k3) {
    // Solved from 2D labels -- by red, or by whoever made the dataset. An
    // imported session that declares itself annotated lands here, which is
    // what stops red re-exporting human work as machine output.
    return k3.is_predicted() ? Bucket::Tracked : Bucket::Annotated;
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

bool minimal_2d(const ExportConfig &cfg) {
    return cfg.camera_names.size() == 1 && cfg.layers == ExportConfig::Layers::TwoD;
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
        if (minimal_2d(cfg)) {   // §5: a 2D camera needs only name, size, offset
            o << "offset = [ 0.0, 0.0,]\n\n";
            continue;
        }
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
        // Red keeps calibration in stored-image coordinates (the importer
        // folds any source crop offset into the principal point), so exports
        // use the image origin as their crop offset.
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
    // Label tables are written beside their final name and renamed over it
    // once a session is complete, so a failure leaves the old tables intact.
    std::vector<std::pair<fs::path, fs::path>> staged;   // .tmp -> final
    auto fail = [&](const std::string &m) {
        std::error_code ec;
        for (const auto &p : staged) fs::remove(p.first, ec);
        if (status) *status = m;
        return false;
    };

    // ── validation that must happen before anything is written ──
    if (cfg.camera_names.empty()) return fail("No cameras.");
    if (cfg.calibration.size() != cfg.camera_names.size())
        return fail("Calibration count does not match camera count.");
    if (cfg.node_names.empty()) return fail("Skeleton has no keypoint names.");
    if (cfg.groups.empty() && cfg.n_frames <= 0)
        return fail("n_frames must come from the media and be > 0.");

    for (size_t i = 0; i < cfg.calibration.size(); i++) {
        const CameraParams &c = cfg.calibration[i];
        const std::string &n = cfg.camera_names[i];
        if (c.telecentric && !minimal_2d(cfg))
            return fail("Camera " + n + " is telecentric. calibration.toml is an aniposelib "
                        "CameraGroup, which has no telecentric model -- the file would load "
                        "cleanly and triangulate wrongly.");
        if (!rotation_is_proper(c.r) && !minimal_2d(cfg))
            return fail("Camera " + n + " has an improper rotation (det != 1), which has no "
                        "Rodrigues representation.");
        if (c.image_width <= 0 || c.image_height <= 0)
            return fail("Camera " + n + " has no image size; validation rule 8 requires it.");
    }

    const std::string gid = cfg.group_id.empty() ? cfg.session_id : cfg.group_id;
    std::vector<ExportConfig::Group> groups = cfg.groups;
    if (groups.empty()) groups.push_back({gid, cfg.n_frames, cfg.source_frame_start});
    // The group holding red frame `fnum`, with `*frame` set to its index there.
    auto locate = [&](u32 fnum, int *frame) -> const ExportConfig::Group * {
        for (const auto &g : groups)
            if ((int)fnum >= g.source_frame_start && (int)fnum < g.source_frame_start + g.n_frames) {
                *frame = (int)fnum - g.source_frame_start;
                return &g;
            }
        return nullptr;
    };
    // The format keys every row by (group, frame, animal, camera, bodypart),
    // so an id shared between animals is not a cosmetic problem: the rows
    // collide, and a reader keeping the last one per key silently keeps one
    // animal out of five.
    // A generated id must also miss every id already in use -- the session's
    // own, and in place every one on disk -- or a new animal is written as an
    // existing one.
    std::set<std::string> used_ids;
    for (const auto &id : cfg.animal_ids)
        if (!id.empty()) used_ids.insert(id);
    std::map<int, std::string> made_ids;
    auto animal_id_of = [&](int instance_id) -> std::string {
        if (instance_id >= 0 && instance_id < (int)cfg.animal_ids.size() &&
            !cfg.animal_ids[(size_t)instance_id].empty())
            return cfg.animal_ids[(size_t)instance_id];
        auto it = made_ids.find(instance_id);
        if (it != made_ids.end()) return it->second;
        char buf[16];
        snprintf(buf, sizeof(buf), "a%02d", instance_id < 0 ? 0 : instance_id);
        std::string id = buf;
        for (int n = 0; used_ids.count(id); n++) {
            snprintf(buf, sizeof(buf), "a%02d", n);
            id = buf;
        }
        used_ids.insert(id);
        made_ids.emplace(instance_id, id);
        return id;
    };

    auto has_box = [](const CameraAnnotation &cam) {
        return cam.has_bbox() && cam.extras->bbox_w > 0 && cam.extras->bbox_h > 0;
    };
    // A box on a view with no keypoint or 3D row to write is a box-only
    // (detection) label; it goes in the annotated session.
    auto box_only = [&](const FrameAnnotation &fa, size_t ci) {
        if (!has_box(fa.cameras[ci])) return false;
        if (cfg.layers != ExportConfig::Layers::ThreeD)
            for (const auto &kp : fa.cameras[ci].keypoints)
                if (keypoint2d_assessed(kp)) return false;
        if (cfg.layers != ExportConfig::Layers::TwoD)
            for (const auto &k3 : fa.kp3d)
                if (k3.exist) return false;
        return true;
    };

    // ── which buckets actually have data ──
    bool has[2] = {false, false};
    for (const auto &[fnum, fis] : amap)
      for (const FrameAnnotation &fa : fis) {
        for (size_t ci = 0; ci < fa.cameras.size(); ci++)
            if (box_only(fa, ci)) has[(int)Bucket::Annotated] = true;
        for (const auto &cam : fa.cameras)
            for (const auto &kp : cam.keypoints)
                if (keypoint2d_assessed(kp))
                    has[(int)bucket_2d(kp)] = true;
        for (const auto &k3 : fa.kp3d) {
            if (!k3.exist) continue;
            if (cfg.layers == ExportConfig::Layers::TwoD) continue;
            has[(int)bucket_3d(k3)] = true;
        }
    }
    // In place, deleting every label is a correction like any other.
    if (!has[0] && !has[1] && !cfg.in_place)
        return fail("Nothing to export: no labelled points or boxes.");

    struct Job { Bucket b; const char *labels; std::string suffix; };
    std::vector<Job> jobs;
    const bool forced = !cfg.force_labels.empty();
    if (forced) {
        // One session, every row, the caller's label. Bucket::Any is expressed
        // by running both buckets into the same job below.
        jobs.push_back({Bucket::Annotated, cfg.force_labels.c_str(), ""});
    }
    const bool both = !forced && has[0] && has[1] &&
                      cfg.export_annotated && cfg.export_tracked;
    if (!forced && has[0] && cfg.export_annotated)
        jobs.push_back({Bucket::Annotated, Tailcycle::labels::kAnnotated, both ? "_annotated" : ""});
    if (!forced && has[1] && cfg.export_tracked)
        jobs.push_back({Bucket::Tracked, Tailcycle::labels::kTracked, both ? "_tracked" : ""});
    if (jobs.empty()) return fail("Nothing selected to export.");

    // Every box is written once: with that view's keypoints (annotated if
    // both sessions have some), else with the animal's 3D, else -- a box-only
    // label -- with the annotated session, or the first one written.
    const Job *job_for[2] = {nullptr, nullptr};
    for (const Job &j : jobs)
        if (!job_for[(int)j.b]) job_for[(int)j.b] = &j;
    auto box_job = [&](const FrameAnnotation &fa, size_t ci) -> const Job * {
        if (forced) return &jobs.front();
        bool kp[2] = {false, false}, p3[2] = {false, false};
        if (cfg.layers != ExportConfig::Layers::ThreeD)
            for (const auto &k : fa.cameras[ci].keypoints)
                if (keypoint2d_assessed(k)) kp[(int)bucket_2d(k)] = true;
        if (cfg.layers != ExportConfig::Layers::TwoD)
            for (const auto &k3 : fa.kp3d)
                if (k3.exist) p3[(int)bucket_3d(k3)] = true;
        for (const bool *b : {kp, p3})
            for (int k = 0; k < 2; k++)
                if (b[k] && job_for[k]) return job_for[k];
        return job_for[(int)Bucket::Annotated] ? job_for[(int)Bucket::Annotated] : &jobs.front();
    };

    // ── in place: the tables already on disk, merged below ──
    std::shared_ptr<arrow::Table> old_kp, old_p3, old_in;
    if (cfg.in_place) {
        if (!forced) return fail("An in-place save rewrites one session; it needs force_labels.");
        const fs::path dir = fs::path(cfg.output_folder) / cfg.split / cfg.session_id;
        for (auto [t, name] : {std::pair{&old_kp, "keypoints.pq"}, std::pair{&old_p3, "points3d.pq"},
                               std::pair{&old_in, "instances.pq"}}) {
            if (!fs::exists(dir / name)) continue;
            *t = Tailcycle::read_pq(dir / name);
            if (!*t) return fail(std::string("Cannot read the existing ") + name + ".");
            Tailcycle::DictCol aid(*t, "animal_id");
            for (const auto &v : aid.vals) used_ids.insert(v);
        }
    }
    auto written_group = [&](const std::string &id) -> const ExportConfig::Group * {
        for (const auto &g : groups)
            if (g.id == id) return &g;
        return nullptr;
    };
    auto camera_index = [&](const std::string &name) {
        for (size_t i = 0; i < cfg.camera_names.size(); i++)
            if (cfg.camera_names[i] == name) return (int)i;
        return -1;
    };
    // Whether an old row survives an in-place save. Other groups' rows always
    // do; a row outside its group's frames never does (rule 6). Otherwise it
    // does when red did not rewrite its unit (`edited(red_frame)`), or when it
    // did but the row is one red never loads and red wrote nothing in its key.
    auto keep_old_row = [&](const std::string &g, double frame, auto &&edited,
                            bool keep_if_edited) {
        const ExportConfig::Group *grp = written_group(g);
        if (!grp) return true;
        if (frame < 0 || frame >= grp->n_frames) return false;
        return !edited(grp->source_frame_start + (int)frame) || keep_if_edited;
    };
    auto row_key = [](std::initializer_list<std::string> parts) {
        std::string k;
        for (const auto &p : parts) { k += p; k += '\x1f'; }
        return k;
    };

    for (const Job &job : jobs) {
        const std::string sid = cfg.session_id + job.suffix;
        const fs::path dir = fs::path(cfg.output_folder) / cfg.split / sid;
        std::error_code ec;
        if (cfg.in_place) {
            // The session is already there, frames and all. Creating
            // groups/<gid> would leave an empty directory beside the real one
            // -- or an empty one in a session whose media lives elsewhere,
            // which is exactly the kind of litter red has no business leaving
            // in someone's dataset.
            if (!fs::is_directory(dir))
                return fail("Session folder is gone: " + dir.string());
        } else {
            for (const auto &g : groups) fs::create_directories(dir / "groups" / g.id, ec);
            if (ec) return fail("Cannot create " + dir.string() + ": " + ec.message());
        }

        std::string err;
        // In place: these describe the session, not its labels, and what is
        // already on disk says more than this config can reproduce.
        if (!cfg.in_place) {
            if (!write_session_toml(dir, cfg, job.labels, &err)) return fail(err);
            if (!write_calibration_toml(dir, cfg, &err)) return fail(err);
        }

        // ── groups.pq ──
        // Skipped in place for the same reason as the TOMLs, and one of its
        // own: source_frame_step and notes are hardcoded here (1 and empty),
        // so rewriting it would overwrite whatever the session recorded.
        if (!cfg.in_place) {
            {
                arrow::StringBuilder gid_b, src_b, notes_b;
                arrow::Int32Builder nf_b, start_b, step_b;
                arrow::FloatBuilder fps_b;
                for (const auto &g : groups) {
                    auto ok = gid_b.Append(g.id).ok() && nf_b.Append(g.n_frames).ok() &&
                              src_b.Append(cfg.source_video).ok() &&
                              start_b.Append(g.source_frame_start).ok() && step_b.Append(1).ok() &&
                              notes_b.Append("").ok() &&
                              (cfg.fps > 0 ? fps_b.Append(cfg.fps).ok() : fps_b.AppendNull().ok());
                    if (!ok) return fail("groups.pq: builder append failed.");
                }
                std::vector<std::shared_ptr<arrow::Array>> a(7);
                if (!gid_b.Finish(&a[0]).ok() || !nf_b.Finish(&a[1]).ok() || !fps_b.Finish(&a[2]).ok() ||
                    !src_b.Finish(&a[3]).ok() || !start_b.Finish(&a[4]).ok() ||
                    !step_b.Finish(&a[5]).ok() || !notes_b.Finish(&a[6]).ok())
                    return fail("groups.pq: finish failed.");
                auto s = Tailcycle::write_table(
                    arrow::Table::Make(Tailcycle::groups_schema(), a), (dir / "groups.pq").string());
                if (!s.ok()) return fail("groups.pq: " + s.ToString());
            }
        }

        // Keys (rule 9) of the rows red writes, which an old row in an edited
        // unit must not duplicate. And the files an in-place save empties.
        std::set<std::string> kp_keys, p3_keys;
        std::vector<fs::path> emptied;
        auto stage = [&](const std::shared_ptr<arrow::Table> &t, const char *name) {
            staged.push_back({dir / (std::string(name) + ".tmp"), dir / name});
            return Tailcycle::write_table(t, staged.back().first.string());
        };

        // ── keypoints.pq ──
        // Red keeps the user provenance (`Manual`) as `visible`, even when a
        // triangulation refreshes that point's coordinates. Non-manual 2D
        // values filled from 3D are `projected`; explicit occlusions are
        // `missing`. An unlabelled point has no row (§7).
        {
            arrow::StringDictionary32Builder g_b, a_b, c_b, p_b, s_b;
            arrow::Int32Builder f_b;
            arrow::FloatBuilder x_b, y_b, sc_b;
            bool any_score = false;
            int rows = 0;

            for (const auto &[fnum, fis] : amap)
      for (const FrameAnnotation &fa : fis) {
                int frame = 0;
                const ExportConfig::Group *grp = locate(fnum, &frame);
                if (!grp) continue;   // rule 6
                if (cfg.layers == ExportConfig::Layers::ThreeD) break;
                for (size_t ci = 0; ci < fa.cameras.size() && ci < cfg.camera_names.size(); ci++) {
                    if (cfg.in_place && !cfg.edited_views.count({(int)fnum, (int)ci})) continue;
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
                        if (!keypoint2d_assessed(kp))
                            continue;   // no row, not `unlabeled` (§7)
                        if (cfg.force_labels.empty() &&
                            bucket_2d(kp) != job.b) continue;
                        if (!g_b.Append(grp->id).ok() || !f_b.Append(frame).ok() ||
                            !a_b.Append(animal_id_of(fa.instance_id)).ok() ||
                            !c_b.Append(cfg.camera_names[ci]).ok() ||
                            !p_b.Append(cfg.node_names[ni]).ok() ||
                            // The visibility CLAIM, which is its own field --
                            // not authorship. Reading `manual` here is why the
                            // importer had to mark a tracked session's visible
                            // rows hand-placed: it was the only way to get
                            // `visible` back out, and it cost the session its
                            // provenance.
                            // One field, one question, three answers --
                            // and the three statuses are exactly those
                            // answers, so this is a straight mapping.
                            !s_b.Append(
                                 kp.vis == Keypoint2D::Vis::Occluded
                                     ? Tailcycle::status::kMissing
                                 : kp.vis == Keypoint2D::Vis::Observed
                                     ? Tailcycle::status::kVisible
                                     : Tailcycle::status::kProjected).ok())
                            return fail("keypoints.pq: builder append failed.");
                        if (kp.is_occluded()) {
                            if (!x_b.AppendNull().ok() || !y_b.AppendNull().ok())
                                return fail("keypoints.pq: coordinate append failed.");
                        } else if (!x_b.Append((float)kp.x).ok() ||
                                   !y_b.Append((float)(img_h - kp.y)).ok()) {
                            return fail("keypoints.pq: coordinate append failed.");
                        }
                        // A human label carries no confidence -- red stores 0.0f,
                        // and passing that through would ship every hand-placed
                        // point with a score of zero (§7 says null).
                        const bool scored = !kp.is_occluded() &&
                                             !kp.is_manual() &&
                                             kp.confidence > 0.0f;
                        if (scored) any_score = true;
                        if (!(scored ? sc_b.Append(kp.confidence) : sc_b.AppendNull()).ok())
                            return fail("keypoints.pq: score append failed.");
                        kp_keys.insert(row_key({grp->id, std::to_string(frame),
                                                animal_id_of(fa.instance_id),
                                                cfg.camera_names[ci], cfg.node_names[ni]}));
                        rows++;
                    }
                }
            }

            const int red_rows = rows;
            const bool in_scope = cfg.layers != ExportConfig::Layers::ThreeD;
            if (cfg.in_place && in_scope && old_kp) {
                Tailcycle::DictCol g(old_kp, "group_id"), a(old_kp, "animal_id"),
                    c(old_kp, "camera"), p(old_kp, "bodypart"), s(old_kp, "status");
                Tailcycle::NumCol f(old_kp, "frame"), x(old_kp, "x"), y(old_kp, "y"),
                    sc(old_kp, "score");
                if (!c.ok || !p.ok || !s.ok || !f.ok || !x.ok || !y.ok)
                    return fail("keypoints.pq: unexpected column types; nothing saved.");
                for (size_t i = 0; i < f.vals.size(); i++) {
                    const std::string gi = g.ok ? g.vals[i] : groups.front().id;
                    const std::string ai = a.ok ? a.vals[i] : std::string("a00");
                    const int ci = camera_index(c.vals[i]);
                    const bool has_xy = !x.null[i] && !y.null[i];
                    auto edited = [&](int rf) { return ci >= 0 && cfg.edited_views.count({rf, ci}); };
                    if (!keep_old_row(gi, f.vals[i], edited,
                                      !Tailcycle::keypoint_row_loaded(s.vals[i], has_xy) &&
                                          !kp_keys.count(row_key({gi, std::to_string((int)f.vals[i]),
                                                                  ai, c.vals[i], p.vals[i]}))))
                        continue;
                    const bool scored = sc.ok && !sc.null[i];
                    if (scored) any_score = true;
                    if (!g_b.Append(gi).ok() || !f_b.Append((int)f.vals[i]).ok() ||
                        !a_b.Append(ai).ok() || !c_b.Append(c.vals[i]).ok() ||
                        !p_b.Append(p.vals[i]).ok() || !s_b.Append(s.vals[i]).ok() ||
                        !(x.null[i] ? x_b.AppendNull() : x_b.Append((float)x.vals[i])).ok() ||
                        !(y.null[i] ? y_b.AppendNull() : y_b.Append((float)y.vals[i])).ok() ||
                        !(scored ? sc_b.Append((float)sc.vals[i]) : sc_b.AppendNull()).ok())
                        return fail("keypoints.pq: builder append failed.");
                    rows++;
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
                auto s = stage(arrow::Table::Make(Tailcycle::keypoints_schema(any_score), a),
                               "keypoints.pq");
                if (!s.ok()) return fail("keypoints.pq: " + s.ToString());
                st.keypoint_rows += red_rows;
            } else if (cfg.in_place && in_scope) {
                emptied.push_back(dir / "keypoints.pq");
            }
        }

        // ── points3d.pq ──
        {
            arrow::StringDictionary32Builder g_b, a_b, p_b, s_b;
            arrow::Int32Builder f_b;
            arrow::FloatBuilder x_b, y_b, z_b, sc_b;
            bool any_score = false;
            int rows = 0;

            for (const auto &[fnum, fis] : amap)
      for (const FrameAnnotation &fa : fis) {
                int frame = 0;
                const ExportConfig::Group *grp = locate(fnum, &frame);
                if (!grp) continue;
                if (cfg.in_place && !cfg.edited_frames_3d.count((int)fnum)) continue;
                for (size_t ni = 0; ni < fa.kp3d.size() && ni < cfg.node_names.size(); ni++) {
                    const Keypoint3D &k3 = fa.kp3d[ni];
                    if (!k3.exist) continue;
                    if (cfg.layers == ExportConfig::Layers::TwoD) continue;
                    if (cfg.force_labels.empty() &&
                        bucket_3d(k3) != job.b) continue;
                    if (!g_b.Append(grp->id).ok() || !f_b.Append(frame).ok() ||
                        !a_b.Append(animal_id_of(fa.instance_id)).ok() ||
                        !p_b.Append(cfg.node_names[ni]).ok() ||
                        !s_b.Append(Tailcycle::status::kVisible).ok() ||
                        !x_b.Append((float)k3.x).ok() || !y_b.Append((float)k3.y).ok() ||
                        !z_b.Append((float)k3.z).ok())
                        return fail("points3d.pq: builder append failed.");
                    const bool scored = k3.is_predicted() && k3.confidence > 0.0f;
                    if (scored) any_score = true;
                    if (!(scored ? sc_b.Append(k3.confidence) : sc_b.AppendNull()).ok())
                        return fail("points3d.pq: score append failed.");
                    p3_keys.insert(row_key({grp->id, std::to_string(frame),
                                            animal_id_of(fa.instance_id), cfg.node_names[ni]}));
                    rows++;
                }
            }

            const int red_rows = rows;
            const bool in_scope = cfg.layers != ExportConfig::Layers::TwoD;
            if (cfg.in_place && in_scope && old_p3) {
                Tailcycle::DictCol g(old_p3, "group_id"), a(old_p3, "animal_id"),
                    p(old_p3, "bodypart"), s(old_p3, "status");
                Tailcycle::NumCol f(old_p3, "frame"), x(old_p3, "x"), y(old_p3, "y"),
                    z(old_p3, "z"), sc(old_p3, "score");
                if (!p.ok || !s.ok || !f.ok || !x.ok || !y.ok || !z.ok)
                    return fail("points3d.pq: unexpected column types; nothing saved.");
                for (size_t i = 0; i < f.vals.size(); i++) {
                    const std::string gi = g.ok ? g.vals[i] : groups.front().id;
                    const std::string ai = a.ok ? a.vals[i] : std::string("a00");
                    const bool has_xyz = !x.null[i] && !y.null[i] && !z.null[i];
                    auto edited = [&](int rf) { return cfg.edited_frames_3d.count(rf) > 0; };
                    if (!keep_old_row(gi, f.vals[i], edited,
                                      !Tailcycle::point3d_row_loaded(s.vals[i], has_xyz) &&
                                          !p3_keys.count(row_key({gi, std::to_string((int)f.vals[i]),
                                                                  ai, p.vals[i]}))))
                        continue;
                    const bool scored = sc.ok && !sc.null[i];
                    if (scored) any_score = true;
                    if (!g_b.Append(gi).ok() || !f_b.Append((int)f.vals[i]).ok() ||
                        !a_b.Append(ai).ok() || !p_b.Append(p.vals[i]).ok() ||
                        !s_b.Append(s.vals[i]).ok() ||
                        !(x.null[i] ? x_b.AppendNull() : x_b.Append((float)x.vals[i])).ok() ||
                        !(y.null[i] ? y_b.AppendNull() : y_b.Append((float)y.vals[i])).ok() ||
                        !(z.null[i] ? z_b.AppendNull() : z_b.Append((float)z.vals[i])).ok() ||
                        !(scored ? sc_b.Append((float)sc.vals[i]) : sc_b.AppendNull()).ok())
                        return fail("points3d.pq: builder append failed.");
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
                auto s = stage(arrow::Table::Make(Tailcycle::points3d_schema(any_score), a),
                               "points3d.pq");
                if (!s.ok()) return fail("points3d.pq: " + s.ToString());
                st.points3d_rows += red_rows;
            } else if (cfg.in_place && in_scope) {
                emptied.push_back(dir / "points3d.pq");
            }
        }

        // ── instances.pq (§9) ──
        // One `labeled` row per box, and nothing else: a view without a box
        // has no row. A box-only annotation is also `labeled`: it is a
        // human-placed box, and `present` would make it an ignore region a
        // detector never learns from. red's bboxes are already top-left image
        // coordinates, so only origin+extent -> [x0,x1) is converted.
        //
        // In place, an edited view's rows become exactly red's boxes (a
        // `present` box becomes `labeled`, `absent` and box-less rows go); an
        // unedited view keeps its rows as they are, except `labeled` rows with
        // no box, which rule 11 forbids.
        {
            arrow::StringDictionary32Builder g_b, a_b, c_b, s_b;
            arrow::Int32Builder f_b;
            arrow::FloatBuilder x0_b, y0_b, x1_b, y1_b;
            arrow::StringBuilder n_b;
            int rows = 0;

            for (const auto &[fnum, fis] : amap)
            for (const FrameAnnotation &fa : fis) {
                int frame = 0;
                const ExportConfig::Group *grp = locate(fnum, &frame);
                if (!grp) continue;
                for (size_t ci = 0; ci < fa.cameras.size() && ci < cfg.camera_names.size(); ci++) {
                    const CameraAnnotation &cam = fa.cameras[ci];
                    if (!has_box(cam)) continue;
                    if (cfg.in_place && !cfg.edited_views.count({(int)fnum, (int)ci})) continue;
                    if (box_job(fa, ci) != &job) continue;
                    const CameraExtras &e = *cam.extras;
                    if (!g_b.Append(grp->id).ok() || !f_b.Append(frame).ok() ||
                        !a_b.Append(animal_id_of(fa.instance_id)).ok() ||
                        !c_b.Append(cfg.camera_names[ci]).ok() ||
                        !s_b.Append(Tailcycle::status::kLabeled).ok() ||
                        !n_b.AppendNull().ok() ||
                        !x0_b.Append((float)e.bbox_x).ok() ||
                        !y0_b.Append((float)e.bbox_y).ok() ||
                        !x1_b.Append((float)(e.bbox_x + e.bbox_w)).ok() ||
                        !y1_b.Append((float)(e.bbox_y + e.bbox_h)).ok())
                        return fail("instances.pq: builder append failed.");
                    rows++;
                }
            }

            const int red_rows = rows;
            if (cfg.in_place && old_in) {
                Tailcycle::DictCol g(old_in, "group_id"), a(old_in, "animal_id"),
                    c(old_in, "camera"), s(old_in, "status");
                Tailcycle::NumCol f(old_in, "frame"), x0(old_in, "x0"), y0(old_in, "y0"),
                    x1(old_in, "x1"), y1(old_in, "y1");
                Tailcycle::StrCol n(old_in, "notes");
                if (!c.ok || !f.ok || !x0.ok || !y0.ok || !x1.ok || !y1.ok)
                    return fail("instances.pq: unexpected column types; nothing saved.");
                for (size_t i = 0; i < f.vals.size(); i++) {
                    const std::string gi = g.ok ? g.vals[i] : groups.front().id;
                    const std::string ai = a.ok ? a.vals[i] : std::string("a00");
                    const std::string si = s.ok ? s.vals[i] : std::string(Tailcycle::status::kLabeled);
                    const int ci = camera_index(c.vals[i]);
                    const bool box = !x0.null[i] && !y0.null[i] && !x1.null[i] && !y1.null[i] &&
                                     Tailcycle::box_nonempty(x0.vals[i], y0.vals[i],
                                                             x1.vals[i], y1.vals[i]);
                    auto edited = [&](int rf) { return ci >= 0 && cfg.edited_views.count({rf, ci}); };
                    if (!keep_old_row(gi, f.vals[i], edited, false)) continue;
                    if (written_group(gi) && si == Tailcycle::status::kLabeled && !box) continue;
                    const bool noted = n.ok && !n.null[i];
                    if (!g_b.Append(gi).ok() || !f_b.Append((int)f.vals[i]).ok() ||
                        !a_b.Append(ai).ok() || !c_b.Append(c.vals[i]).ok() ||
                        !s_b.Append(si).ok() ||
                        !(noted ? n_b.Append(n.vals[i]) : n_b.AppendNull()).ok() ||
                        !(x0.null[i] ? x0_b.AppendNull() : x0_b.Append((float)x0.vals[i])).ok() ||
                        !(y0.null[i] ? y0_b.AppendNull() : y0_b.Append((float)y0.vals[i])).ok() ||
                        !(x1.null[i] ? x1_b.AppendNull() : x1_b.Append((float)x1.vals[i])).ok() ||
                        !(y1.null[i] ? y1_b.AppendNull() : y1_b.Append((float)y1.vals[i])).ok())
                        return fail("instances.pq: builder append failed.");
                    rows++;
                }
            }

            if (rows > 0) {
                std::vector<std::shared_ptr<arrow::Array>> a(10);
                if (!g_b.Finish(&a[0]).ok() || !f_b.Finish(&a[1]).ok() || !a_b.Finish(&a[2]).ok() ||
                    !c_b.Finish(&a[3]).ok() || !x0_b.Finish(&a[4]).ok() || !y0_b.Finish(&a[5]).ok() ||
                    !x1_b.Finish(&a[6]).ok() || !y1_b.Finish(&a[7]).ok() || !s_b.Finish(&a[8]).ok() ||
                    !n_b.Finish(&a[9]).ok())
                    return fail("instances.pq: finish failed.");
                auto s = stage(arrow::Table::Make(Tailcycle::instances_schema(), a),
                               "instances.pq");
                if (!s.ok()) return fail("instances.pq: " + s.ToString());
                st.instance_rows += red_rows;
            } else if (cfg.in_place) {
                emptied.push_back(dir / "instances.pq");
            }
        }

        // Every table of this session is written; put them in place.
        {
            std::error_code ec;
            for (const auto &[tmp, dst] : staged) {
                fs::rename(tmp, dst, ec);
                if (ec) return fail("Cannot replace " + dst.string() + ": " + ec.message());
            }
            staged.clear();
            for (const auto &p : emptied) fs::remove(p, ec);
        }

        // §3 asks for keypoints.pq or points3d.pq. red also accepts a session
        // whose only labels are boxes in instances.pq -- a detection-only
        // dataset -- which is a deliberate extension of the format. Reaching
        // here with none of the three means the bucket scan and the row loops
        // disagreed, which is a bug rather than bad input. In place, emptying
        // a session is a legitimate correction (rule 6: no table is required).
        if (!cfg.in_place && !fs::exists(dir / "keypoints.pq") && !fs::exists(dir / "points3d.pq") &&
            !fs::exists(dir / "instances.pq"))
            return fail("Session " + sid + " would have no label table.");

        st.sessions.push_back(dir.string());
        st.sessions_written++;
    }

    st.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (status) {
        char buf[256];
        snprintf(buf, sizeof(buf), "Wrote %d session%s: %d 2D rows, %d 3D rows, %d boxes (%.1fs)",
                 st.sessions_written, st.sessions_written == 1 ? "" : "s",
                 st.keypoint_rows, st.points3d_rows, st.instance_rows, st.elapsed_seconds);
        *status = buf;
    }
    return true;
}

#endif // RED_HAVE_PARQUET

} // namespace TailcycleExport
