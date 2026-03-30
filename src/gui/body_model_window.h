#pragma once
// body_model_window.h — MuJoCo body model viewer + IK controls
//
// Renders a posed body model via Metal, with controls for loading models,
// running inverse kinematics, and adjusting solver parameters.

#include "imgui.h"
#include "annotation.h"
#include "app_context.h"
#include "gui/panel.h"
#include "mujoco_context.h"
#include "mujoco_ik.h"
#include "mujoco_stac.h"
#include "arena_alignment.h"
#include <ImGuiFileDialog.h>
#include <fstream>
#include <iomanip>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>

#ifdef RED_HAS_MUJOCO
#include "mujoco_metal_renderer.h"
#endif

enum SitePlacement { SITES_DEFAULT = 0, SITES_STAC = 1 };

// A saved calibration result (default, STAC, STAC symmetric, etc.)
struct CalibrationEntry {
    std::string name;
    bool   symmetric      = false;
    int    n_iters        = 0;
    int    n_samples      = 0;
    int    m_max_iters    = 0;
    double m_lr           = 0;
    double m_reg_coef     = 0;
    double pre_residual   = 0;   // mm
    double post_residual  = 0;   // mm
    double time_s         = 0;
    std::vector<double> site_offsets;       // 3*nsite (delta from original)
    std::vector<std::string> site_names;    // nsite
    std::vector<double> displacement_mm;    // per-site |offset| in mm
};

// Per-frame result for parallel export
struct QposResult {
    int frame_num;
    std::vector<double> qpos;
    double residual_mm;
    int iterations;
    bool converged;
};

// Export state for background thread progress
struct MujocoExportState {
    std::atomic<bool> running{false};
    std::atomic<int>  frames_done{0};
    int               frames_total = 0;
    int               num_threads = 0;
    std::string       output_path;
    std::string       result_msg;
    bool              result_ok = false;
};

struct BodyModelState {
    bool show = false;

    // IK method selection
    int site_placement = SITES_DEFAULT;

    // IK solver state (owns warm-start)
    MujocoIKState ik_state;

    // STAC calibration state
    StacState stac_state;
    bool stac_calibrating = false;

    // Calibration history: entry 0 is always "Default" (zero offsets)
    std::vector<CalibrationEntry> calib_history;
    int active_calibration = 0; // index into calib_history

    // Export state
    MujocoExportState export_state;

    // Arena alignment (calibration → MuJoCo transform)
    ArenaAlignment arena_align;
    bool alignment_mode = false;
    // Saved state during alignment mode
    SkeletonContext saved_skeleton;
    AnnotationMap saved_annotations;

    // Model scale (saved for session persistence)
    float saved_model_scale = 1.0f;

    // Arena dimensions (model units: meters for rodent, cm for fly)
    float arena_width = 1.828f;   // X extent (rodent default: 1828mm = 1.828m)
    float arena_depth = 1.828f;   // Y extent (rodent default: same, square)
    float arena_offset[3] = {0, 0, 0}; // arena center offset from origin

    // Controls section height (user can drag splitter)
    float controls_height = 300.0f;

    // Display options
    bool auto_solve = false;
    bool show_skin = true;
    bool show_bodies = true;
    bool show_site_markers = true;

    // Deferred unload (processed at start of next frame)
    bool unload_requested = false;

    // Camera for 3D view (uses MuJoCo's mjvCamera for native controls)
    bool  cam_initialized = false;
    bool  show_arena      = true;

    // Track which frame we last solved for
    int last_solved_frame = -1;

    // Calibration camera view: -1 = free camera, 0..N-1 = calibration camera index
    int selected_camera = -1;
    ViewOverride calib_view_override;
    float calib_aspect = 0.0f;
    bool show_video_bg = false;
    float scene_opacity = 1.0f;
    bool calib_cam_user_override = false;
    float cam_zoom = 1.0f;       // zoom on calibration camera view
    float cam_pan[2] = {0, 0};   // pan offset (normalized image coords)

    // Renderer handle + native MuJoCo camera
#ifdef RED_HAS_MUJOCO
    MujocoRenderer *renderer = nullptr;
    mjvCamera mjcam;  // initialized in draw function on first use
#endif

    // Model path for file dialog
    std::string model_path;

    // Set arena defaults based on loaded model type
    void set_arena_defaults_for_model(const MujocoContext &mj) {
#ifdef RED_HAS_MUJOCO
        if (!mj.model) return;
        bool is_fly = (mj_name2id(mj.model, mjOBJ_BODY, "thorax") >= 0 &&
                       mj_name2id(mj.model, mjOBJ_BODY, "wing_left") >= 0);
        if (is_fly) {
            // Fly arena: 24mm x 5.6mm in cm (model units), origin at corner
            arena_width = 2.4f;   // 24mm in cm
            arena_depth = 0.56f;  // 5.6mm in cm
            arena_offset[0] = arena_width * 0.5f;  // center X at 12mm
            arena_offset[1] = arena_depth * 0.5f;  // center Y at 2.8mm
            arena_offset[2] = 0.0f;
        } else {
            // Rodent arena: 1828mm x 1828mm in meters, centered at origin
            arena_width = 1.828f;
            arena_depth = 1.828f;
            arena_offset[0] = 0.0f;
            arena_offset[1] = 0.0f;
            arena_offset[2] = 0.0f;
        }
#endif
    }

    // Apply loaded calibration offsets to the model after session load.
    // Handles double-offset protection (resets to original first).
    void apply_loaded_calibration(MujocoContext &mj) {
        // Apply saved model scale
        if (std::abs(saved_model_scale - 1.0f) > 1e-6f &&
            std::abs(mj.model_scale - saved_model_scale) > 1e-6f) {
            float rel = saved_model_scale / mj.model_scale;
            mj.apply_model_scale(rel);
        }
        // Reset any existing offsets first to avoid double-application
        if (stac_state.calibrated)
            stac_reset(mj, stac_state);
        if (active_calibration > 0 &&
            active_calibration < (int)calib_history.size()) {
            auto &c = calib_history[active_calibration];
            if (!c.site_offsets.empty() &&
                (int)c.site_offsets.size() == 3 * (int)mj.model->nsite) {
                int ns = (int)mj.model->nsite;
                stac_state.original_site_pos.assign(
                    mj.model->site_pos, mj.model->site_pos + 3 * ns);
                stac_state.site_offsets = c.site_offsets;
                stac_state.calibrated = true;
                stac_apply_offsets(mj, stac_state);
            }
        }
        mujoco_ik_reset(ik_state);
        last_solved_frame = -1;
    }

    // Save session to JSON
    bool save_session(const std::string &path) const {
        try {
            nlohmann::json j;
            j["version"] = 1;
            j["model_path"] = model_path;
            j["site_placement"] = site_placement;

            // IK settings
            j["ik"] = {{"max_iterations", ik_state.max_iterations},
                       {"lr", ik_state.lr}, {"beta", ik_state.beta},
                       {"reg_strength", ik_state.reg_strength}};

            // Arena alignment
            if (arena_align.valid) {
                std::vector<double> R_flat(9), t_vec(3), corners_flat(12);
                for (int i = 0; i < 9; i++) R_flat[i] = arena_align.R.data()[i];
                for (int i = 0; i < 3; i++) t_vec[i] = arena_align.t[i];
                for (int i = 0; i < 4; i++)
                    for (int c = 0; c < 3; c++)
                        corners_flat[3*i+c] = arena_align.calib_corners[i][c];
                j["arena"] = {{"valid", true}, {"scale", arena_align.scale},
                              {"R", R_flat}, {"t", t_vec},
                              {"corners", corners_flat},
                              {"residual_mm", arena_align.residual_mm}};
            }

            // STAC settings
            j["stac"] = {{"symmetric", stac_state.symmetric},
                         {"n_iters", stac_state.n_iters},
                         {"n_sample_frames", stac_state.n_sample_frames},
                         {"m_max_iters", stac_state.m_max_iters},
                         {"m_lr", stac_state.m_lr},
                         {"m_momentum", stac_state.m_momentum},
                         {"m_reg_coef", stac_state.m_reg_coef}};

            // Calibration history
            nlohmann::json hist = nlohmann::json::array();
            for (const auto &c : calib_history) {
                nlohmann::json entry;
                entry["name"] = c.name;
                entry["symmetric"] = c.symmetric;
                entry["n_iters"] = c.n_iters;
                entry["n_samples"] = c.n_samples;
                entry["m_max_iters"] = c.m_max_iters;
                entry["m_lr"] = c.m_lr;
                entry["m_reg_coef"] = c.m_reg_coef;
                entry["pre_residual"] = c.pre_residual;
                entry["post_residual"] = c.post_residual;
                entry["time_s"] = c.time_s;
                entry["site_offsets"] = c.site_offsets;
                entry["displacement_mm"] = c.displacement_mm;
                entry["site_names"] = c.site_names;
                hist.push_back(entry);
            }
            j["calibration_history"] = hist;
            j["active_calibration"] = active_calibration;

            // Display
            j["display"] = {{"show_skin", show_skin}, {"show_bodies", show_bodies},
                            {"show_sites", show_site_markers}, {"show_arena", show_arena},
                            {"arena_width", arena_width}, {"arena_depth", arena_depth},
                            {"arena_offset", std::vector<float>{arena_offset[0], arena_offset[1], arena_offset[2]}},
                            {"model_scale", saved_model_scale}};

            std::ofstream f(path);
            if (!f) return false;
            f << j.dump(2);
            return true;
        } catch (...) { return false; }
    }

    // Load session from JSON. Returns true on success.
    // Call AFTER model is loaded (needs nsite for offset application).
    bool load_session(const std::string &path) {
        try {
            std::ifstream f(path);
            if (!f) return false;
            nlohmann::json j;
            f >> j;
            if (!j.contains("version")) return false;

            model_path = j.value("model_path", model_path);
            site_placement = j.value("site_placement", site_placement);

            if (j.contains("ik")) {
                auto &ik = j["ik"];
                ik_state.max_iterations = ik.value("max_iterations", ik_state.max_iterations);
                ik_state.lr = ik.value("lr", ik_state.lr);
                ik_state.beta = ik.value("beta", ik_state.beta);
                ik_state.reg_strength = ik.value("reg_strength", ik_state.reg_strength);
            }

            if (j.contains("arena")) {
                auto &a = j["arena"];
                arena_align.valid = a.value("valid", false);
                arena_align.scale = a.value("scale", 0.001);
                arena_align.residual_mm = a.value("residual_mm", 0.0);
                if (a.contains("R")) {
                    auto R = a["R"].get<std::vector<double>>();
                    if (R.size() == 9)
                        arena_align.R = Eigen::Map<Eigen::Matrix3d>(R.data());
                }
                if (a.contains("t")) {
                    auto t = a["t"].get<std::vector<double>>();
                    if (t.size() == 3)
                        arena_align.t = Eigen::Map<Eigen::Vector3d>(t.data());
                }
                if (a.contains("corners")) {
                    auto c = a["corners"].get<std::vector<double>>();
                    if (c.size() == 12) {
                        for (int i = 0; i < 4; i++)
                            arena_align.calib_corners[i] = Eigen::Vector3d(c[3*i], c[3*i+1], c[3*i+2]);
                        arena_align.corners_set = true;
                    }
                }
            }

            if (j.contains("stac")) {
                auto &s = j["stac"];
                stac_state.symmetric = s.value("symmetric", true);
                stac_state.n_iters = s.value("n_iters", 3);
                stac_state.n_sample_frames = s.value("n_sample_frames", 100);
                stac_state.m_max_iters = s.value("m_max_iters", 500);
                stac_state.m_lr = s.value("m_lr", 5e-4);
                stac_state.m_momentum = s.value("m_momentum", 0.9);
                stac_state.m_reg_coef = s.value("m_reg_coef", 0.1);
            }

            if (j.contains("calibration_history")) {
                calib_history.clear();
                for (auto &entry : j["calibration_history"]) {
                    CalibrationEntry c;
                    c.name = entry.value("name", "");
                    c.symmetric = entry.value("symmetric", false);
                    c.n_iters = entry.value("n_iters", 0);
                    c.n_samples = entry.value("n_samples", 0);
                    c.m_max_iters = entry.value("m_max_iters", 0);
                    c.m_lr = entry.value("m_lr", 0.0);
                    c.m_reg_coef = entry.value("m_reg_coef", 0.0);
                    c.pre_residual = entry.value("pre_residual", 0.0);
                    c.post_residual = entry.value("post_residual", 0.0);
                    c.time_s = entry.value("time_s", 0.0);
                    c.site_offsets = entry.value("site_offsets", std::vector<double>{});
                    c.displacement_mm = entry.value("displacement_mm", std::vector<double>{});
                    c.site_names = entry.value("site_names", std::vector<std::string>{});
                    calib_history.push_back(std::move(c));
                }
                active_calibration = j.value("active_calibration", 0);
                if (!calib_history.empty())
                    active_calibration = std::clamp(active_calibration, 0, (int)calib_history.size() - 1);
            }

            if (j.contains("display")) {
                auto &d = j["display"];
                show_skin = d.value("show_skin", true);
                show_bodies = d.value("show_bodies", true);
                show_site_markers = d.value("show_sites", true);
                show_arena = d.value("show_arena", true);
                arena_width = d.value("arena_width", 1.828f);
                arena_depth = d.value("arena_depth", 1.828f);
                saved_model_scale = d.value("model_scale", 1.0f);
                if (d.contains("arena_offset")) {
                    auto ao = d["arena_offset"].get<std::vector<float>>();
                    if (ao.size() >= 3) {
                        arena_offset[0] = ao[0]; arena_offset[1] = ao[1]; arena_offset[2] = ao[2];
                    }
                }
            }

            return true;
        } catch (...) { return false; }
    }
};

#ifdef RED_HAS_MUJOCO

// Solve a chunk of frames on a dedicated model/data copy.
// Each thread gets its own mjModel + mjData + IKState for full isolation.
inline void SolveChunk(mjModel *model, const std::vector<std::pair<int, std::vector<Keypoint3D>>> &frames,
                       const MujocoIKState &ik_template,
                       const std::vector<int> &skeleton_to_site,
                       int num_nodes, float scale_factor,
                       std::vector<QposResult> &results,
                       std::atomic<int> &progress) {
    mjData *data = mj_makeData(model);
    int nq = model->nq;

    // Fresh IK state per thread
    MujocoIKState ik;
    ik.max_iterations = ik_template.max_iterations;
    ik.lr = ik_template.lr;
    ik.beta = ik_template.beta;
    ik.reg_strength = ik_template.reg_strength;

    // Temporary MujocoContext wrapper for mujoco_ik_solve
    MujocoContext thread_mj;
    thread_mj.model = model;
    thread_mj.data = data;
    thread_mj.loaded = true;
    thread_mj.scale_factor = scale_factor;
    thread_mj.skeleton_to_site = skeleton_to_site;
    thread_mj.mapped_count = (int)skeleton_to_site.size();

    results.resize(frames.size());
    for (int i = 0; i < (int)frames.size(); i++) {
        const auto &[fnum, kp3d] = frames[i];
        mujoco_ik_solve(thread_mj, ik, kp3d.data(), num_nodes, fnum);

        results[i].frame_num = fnum;
        results[i].qpos.assign(data->qpos, data->qpos + nq);
        results[i].residual_mm = ik.final_residual * 1000.0;
        results[i].iterations = ik.iterations_used;
        results[i].converged = ik.converged;

        progress.fetch_add(1, std::memory_order_relaxed);
    }

    // Don't free model (shared across threads via mj_copyModel, caller frees)
    // But DO free data (per-thread)
    thread_mj.model = nullptr; // prevent MujocoContext destructor from freeing
    thread_mj.data = nullptr;
    thread_mj.loaded = false;
    mj_deleteData(data);
}

// Launch parallel batch export in a background thread.
inline void LaunchExport(MujocoContext &mj, MujocoIKState &ik_template,
                         const AnnotationMap &annotations, int num_nodes,
                         int site_placement, const StacState &stac,
                         const std::vector<std::string> &node_names,
                         const ArenaAlignment &arena_align,
                         MujocoExportState &es) {
    int nq = (int)mj.model->nq;

    // Collect frames with 3D data, sorted by frame number
    struct FrameRef { int frame_num; const Keypoint3D *kp3d; int kp3d_size; };
    std::vector<FrameRef> frame_refs;
    for (const auto &[fnum, fa] : annotations) {
        if (!fa.kp3d.empty()) {
            int active = 0;
            for (int n = 0; n < num_nodes && n < (int)fa.kp3d.size(); n++)
                if (fa.kp3d[n].triangulated) active++;
            if (active >= 4)
                frame_refs.push_back({(int)fnum, fa.kp3d.data(), (int)fa.kp3d.size()});
        }
    }
    std::sort(frame_refs.begin(), frame_refs.end(),
              [](const FrameRef &a, const FrameRef &b) { return a.frame_num < b.frame_num; });

    if (frame_refs.empty()) {
        es.result_msg = "No frames with 3D keypoints";
        es.result_ok = false;
        es.running = false;
        return;
    }

    // Deep-copy keypoint data and apply arena alignment if active
    std::vector<std::pair<int, std::vector<Keypoint3D>>> all_frames;
    all_frames.reserve(frame_refs.size());
    for (const auto &fr : frame_refs) {
        std::vector<Keypoint3D> kp(fr.kp3d, fr.kp3d + fr.kp3d_size);
        if (arena_align.valid) {
            for (auto &k : kp) {
                if (!k.triangulated) continue;
                Eigen::Vector3d p = arena_align.transform(Eigen::Vector3d(k.x, k.y, k.z));
                k.x = p.x(); k.y = p.y(); k.z = p.z();
            }
        }
        all_frames.push_back({fr.frame_num, std::move(kp)});
    }

    int N = (int)all_frames.size();
    int n_threads = std::max(1, (int)std::thread::hardware_concurrency());
    n_threads = std::min(n_threads, std::max(1, N / 20));
    n_threads = std::min(n_threads, 16);

    es.frames_total = N;
    es.frames_done = 0;
    es.num_threads = n_threads;

    // Capture all metadata for the background thread (snapshot at launch time)
    std::string path = es.output_path;
    MujocoIKState ik_settings = ik_template;
    std::vector<int> skel_to_site = mj.skeleton_to_site;
    float sf = mj.scale_factor;
    std::string model_path = mj.model_path;
    int method = site_placement;

    // Snapshot STAC state for metadata
    bool stac_calibrated = stac.calibrated;
    int stac_n_iters = stac.n_iters;
    int stac_n_samples = stac.n_sample_frames;
    int stac_m_iters = stac.m_max_iters;
    double stac_m_lr = stac.m_lr;
    double stac_m_momentum = stac.m_momentum;
    double stac_m_reg = stac.m_reg_coef;
    double stac_pre_res = stac.pre_residual;
    double stac_post_res = stac.post_residual;
    std::vector<double> stac_offsets = stac.site_offsets;
    std::vector<double> stac_orig_sites = stac.original_site_pos;

    // Snapshot site positions (includes STAC offsets if calibrated)
    int nsite = (int)mj.model->nsite;
    std::vector<double> site_pos(mj.model->site_pos, mj.model->site_pos + 3 * nsite);
    std::vector<int> site_bodyid(mj.model->site_bodyid, mj.model->site_bodyid + nsite);

    // Site names and node names for header
    std::vector<std::string> site_names(nsite);
    for (int i = 0; i < nsite; i++) {
        const char *name = mj_id2name(mj.model, mjOBJ_SITE, i);
        site_names[i] = name ? name : "";
    }
    std::vector<std::string> kp_names = node_names;

    // Copy the model N_threads times (each thread needs its own)
    // Do this on the main thread before launching
    std::vector<mjModel *> thread_models(n_threads);
    for (int t = 0; t < n_threads; t++)
        thread_models[t] = mj_copyModel(nullptr, mj.model);

    // Launch background thread that coordinates the workers
    std::thread([=, &es, thread_models = std::move(thread_models),
                 all_frames = std::move(all_frames)]() mutable {
        // Split frames into contiguous chunks (not round-robin) for warm-start benefit
        int total = (int)all_frames.size();
        std::vector<std::vector<std::pair<int, std::vector<Keypoint3D>>>> chunks(n_threads);
        for (int t = 0; t < n_threads; t++) {
            int start = (int64_t)t * total / n_threads;
            int end   = (int64_t)(t + 1) * total / n_threads;
            for (int i = start; i < end; i++)
                chunks[t].push_back(std::move(all_frames[i]));
        }

        // Launch worker threads
        std::vector<std::vector<QposResult>> thread_results(n_threads);
        std::vector<std::thread> workers;
        for (int t = 0; t < n_threads; t++) {
            workers.emplace_back(SolveChunk,
                thread_models[t], std::cref(chunks[t]),
                std::cref(ik_settings), std::cref(skel_to_site),
                num_nodes, sf,
                std::ref(thread_results[t]),
                std::ref(es.frames_done));
        }

        // Wait for all workers
        for (auto &w : workers) w.join();

        // Free model copies
        for (auto *m : thread_models) mj_deleteModel(m);

        // Merge and sort results
        std::vector<QposResult> all_results;
        for (auto &tr : thread_results)
            all_results.insert(all_results.end(),
                std::make_move_iterator(tr.begin()),
                std::make_move_iterator(tr.end()));
        std::sort(all_results.begin(), all_results.end(),
                  [](const QposResult &a, const QposResult &b) {
                      return a.frame_num < b.frame_num;
                  });

        // Write CSV
        int nq = all_results.empty() ? 0 : (int)all_results[0].qpos.size();
        std::ofstream out(path);
        if (!out.is_open()) {
            es.result_msg = "Failed to write " + path;
            es.result_ok = false;
            es.running = false;
            return;
        }

        // --- Metadata header (comment lines for reproducibility) ---
        out << "# RED qpos export\n";
        out << "# model: " << model_path << "\n";
        out << "# nq: " << nq << "\n";
        out << "# nv: " << (all_results.empty() ? 0 : nq) << "\n"; // nv not easily available, omit or compute
        out << "# threads: " << n_threads << "\n";
        out << "# site_placement: " << (method == SITES_STAC ? "STAC_calibrated" : "default_xml") << "\n";
        out << "# ik_algorithm: IK_dm_control\n";
        out << "# scale_factor: " << std::setprecision(6) << sf
            << (sf <= 0.0f ? " (auto)" : "") << "\n";
        out << "# max_iterations: " << ik_settings.max_iterations << "\n";
        out << "# lr: " << std::setprecision(6) << ik_settings.lr << "\n";
        out << "# beta: " << ik_settings.beta << "\n";
        out << "# reg_strength: " << std::setprecision(8) << ik_settings.reg_strength << "\n";
        out << "# progress_thresh: " << ik_settings.progress_thresh << "\n";
        out << "# check_every: " << ik_settings.check_every << "\n";

        // Joint names
        // (not easily available from thread — use model copy if needed)
        // We captured site_names and kp_names instead
        out << "# keypoint_names:";
        for (const auto &n : kp_names) out << " " << n;
        out << "\n";

        // STAC metadata
        if (method == SITES_STAC) {
            out << "# stac_calibrated: " << (stac_calibrated ? "true" : "false") << "\n";
            if (stac_calibrated) {
                out << "# stac_n_iters: " << stac_n_iters << "\n";
                out << "# stac_n_sample_frames: " << stac_n_samples << "\n";
                out << "# stac_m_max_iters: " << stac_m_iters << "\n";
                out << "# stac_m_lr: " << std::setprecision(6) << stac_m_lr << "\n";
                out << "# stac_m_momentum: " << stac_m_momentum << "\n";
                out << "# stac_m_reg_coef: " << stac_m_reg << "\n";
                out << "# stac_pre_residual_mm: " << std::setprecision(4) << stac_pre_res << "\n";
                out << "# stac_post_residual_mm: " << stac_post_res << "\n";

                // Site offsets (the calibrated delta from original positions)
                out << "# stac_site_offsets (site_name, body_id, dx, dy, dz):\n";
                int n_off = (int)stac_offsets.size() / 3;
                for (int i = 0; i < n_off && i < (int)site_names.size(); i++) {
                    double ox = stac_offsets[3*i], oy = stac_offsets[3*i+1], oz = stac_offsets[3*i+2];
                    if (std::abs(ox) > 1e-10 || std::abs(oy) > 1e-10 || std::abs(oz) > 1e-10) {
                        out << "#   " << site_names[i] << ", body=" << site_bodyid[i]
                            << ", " << std::setprecision(8)
                            << ox << ", " << oy << ", " << oz << "\n";
                    }
                }

                // Final site positions (original + offset)
                out << "# final_site_positions (site_name, x, y, z in model coords):\n";
                for (int i = 0; i < (int)site_pos.size() / 3 && i < (int)site_names.size(); i++) {
                    if (!site_names[i].empty()) {
                        out << "#   " << site_names[i] << ", "
                            << std::setprecision(8)
                            << site_pos[3*i] << ", " << site_pos[3*i+1]
                            << ", " << site_pos[3*i+2] << "\n";
                    }
                }
            }
        }

        out << "#\n"; // blank comment line before data

        // Column header
        out << "frame";
        for (int j = 0; j < nq; j++) out << ",qpos_" << j;
        out << ",residual_mm,iterations,converged\n";

        double total_res = 0.0;
        for (const auto &r : all_results) {
            out << r.frame_num;
            for (int j = 0; j < nq; j++)
                out << "," << std::setprecision(8) << r.qpos[j];
            out << "," << std::setprecision(4) << r.residual_mm
                << "," << r.iterations
                << "," << (r.converged ? 1 : 0) << "\n";
            total_res += r.residual_mm;
        }
        out.close();

        double mean_res = total_res / all_results.size();
        es.result_msg = std::to_string((int)all_results.size()) + " frames exported ("
            + std::to_string(n_threads) + " threads, mean "
            + std::to_string((int)mean_res) + " mm)";
        es.result_ok = true;
        es.running = false;
    }).detach();
}

#endif

inline void DrawBodyModelWindow(BodyModelState &state, MujocoContext &mj,
                                AppContext &ctx) {
    DrawPanel("Body Model", state.show,
        [&]() {
#ifndef RED_HAS_MUJOCO
            ImGui::TextColored(ImVec4(1, 0.5, 0, 1),
                "MuJoCo not available. Install mujoco.framework to lib/ and rebuild.");
            return;
#else
            // --- Deferred unload (safe: no ImGui widgets active) ---
            if (state.unload_requested) {
                state.unload_requested = false;
                if (state.renderer) {
                    mujoco_renderer_destroy(state.renderer);
                    state.renderer = nullptr;
                }
                mujoco_ik_reset(state.ik_state);
                mj.unload();
                state.last_solved_frame = -1;
            }

            // --- Model loading ---
            if (!mj.loaded) {
                ImGui::TextWrapped("Body Model — IK_dm_control inverse kinematics "
                    "with MuJoCo body models, STAC site calibration, and arena alignment.");
                ImGui::Spacing();

                // One-click restore: load previous model + session
                bool has_session = !ctx.pm.project_path.empty() &&
                    std::filesystem::exists(ctx.pm.project_path + "/mujoco_session.json");
                bool has_prev_model = !ctx.user_settings.last_mujoco_model.empty() &&
                    std::filesystem::exists(ctx.user_settings.last_mujoco_model);

                if (has_prev_model && has_session) {
                    if (ImGui::Button("Restore Previous Session")) {
                        state.model_path = ctx.user_settings.last_mujoco_model;
                        if (mj.load(state.model_path, ctx.skeleton)) {
                            state.set_arena_defaults_for_model(mj);
                            std::string sp = ctx.pm.project_path + "/mujoco_session.json";
                            if (state.load_session(sp))
                                state.apply_loaded_calibration(mj);
                            ctx.toasts.push("MuJoCo model + session restored");
                        } else {
                            ctx.toasts.push("Failed: " + mj.load_error, Toast::Error);
                        }
                    }
                    ImGui::SameLine();
                    ImGui::TextDisabled("(%s)",
                        std::filesystem::path(ctx.user_settings.last_mujoco_model)
                            .filename().string().c_str());
                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::Spacing();
                }

                if (state.model_path.empty()) {
                    if (has_prev_model)
                        state.model_path = ctx.user_settings.last_mujoco_model;
                    else {
                        std::string default_path = "models/rodent/rodent_no_collision.xml";
                        if (std::filesystem::exists(default_path))
                            state.model_path = default_path;
                    }
                }

                ImGui::InputText("Model XML", &state.model_path);
                ImGui::SameLine();
                if (ImGui::Button("Browse...")) {
                    IGFD::FileDialogConfig cfg;
                    cfg.countSelectionMax = 1;
                    cfg.flags = ImGuiFileDialogFlags_Modal;
                    if (!state.model_path.empty()) {
                        std::filesystem::path p(state.model_path);
                        if (p.has_parent_path())
                            cfg.path = p.parent_path().string();
                    }
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "ChooseMujocoModel", "Select MuJoCo Model", ".xml,.mjb", cfg);
                }

                if (ImGui::Button("Load Model") && !state.model_path.empty()) {
                    if (mj.load(state.model_path, ctx.skeleton)) {
                        ctx.user_settings.last_mujoco_model = state.model_path;
                        save_user_settings(ctx.user_settings);
                        state.set_arena_defaults_for_model(mj);
                        // Auto-load session if it exists (overrides defaults)
                        std::string sp = ctx.pm.project_path + "/mujoco_session.json";
                        if (!ctx.pm.project_path.empty() &&
                            std::filesystem::exists(sp) && state.load_session(sp)) {
                            state.apply_loaded_calibration(mj);
                            ctx.toasts.push("MuJoCo model + session loaded");
                        } else {
                            ctx.toasts.push("MuJoCo model loaded: " +
                                std::to_string(mj.mapped_count) + "/" +
                                std::to_string(ctx.skeleton.num_nodes) +
                                " sites matched");
                        }
                    } else {
                        ctx.toasts.push("Failed: " + mj.load_error, Toast::Error);
                    }
                }

                // Quick-load the last used model
                if (!ctx.user_settings.last_mujoco_model.empty() &&
                    std::filesystem::exists(ctx.user_settings.last_mujoco_model)) {
                    ImGui::SameLine();
                    if (ImGui::Button("Load Previous")) {
                        state.model_path = ctx.user_settings.last_mujoco_model;
                        if (mj.load(state.model_path, ctx.skeleton)) {
                            state.set_arena_defaults_for_model(mj);
                            std::string sp = ctx.pm.project_path + "/mujoco_session.json";
                            if (!ctx.pm.project_path.empty() &&
                                std::filesystem::exists(sp) && state.load_session(sp)) {
                                state.apply_loaded_calibration(mj);
                                ctx.toasts.push("MuJoCo model + session loaded");
                            } else {
                                ctx.toasts.push("MuJoCo model loaded: " +
                                    std::to_string(mj.mapped_count) + "/" +
                                    std::to_string(ctx.skeleton.num_nodes) +
                                    " sites matched");
                            }
                        } else {
                            ctx.toasts.push("Failed: " + mj.load_error, Toast::Error);
                        }
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("%s",
                            ctx.user_settings.last_mujoco_model.c_str());
                    }
                }

                if (!mj.load_error.empty())
                    ImGui::TextColored(ImVec4(1, 0.3, 0.3, 1), "%s", mj.load_error.c_str());
                return;
            }

            if (!mj.model) return; // Defensive: loaded but model null

            // Helper: get kp3d pointer for IK, applying arena transform if active.
            // When not aligned, returns the original data directly (zero overhead).
            std::vector<Keypoint3D> transformed_kp3d_buf; // reusable buffer
            auto get_kp3d = [&](const std::vector<Keypoint3D> &src) -> const Keypoint3D * {
                if (!state.arena_align.valid) return src.data();
                transformed_kp3d_buf = src;
                for (auto &kp : transformed_kp3d_buf) {
                    if (!kp.triangulated) continue;
                    Eigen::Vector3d p = state.arena_align.transform(
                        Eigen::Vector3d(kp.x, kp.y, kp.z));
                    kp.x = p.x(); kp.y = p.y(); kp.z = p.z();
                }
                return transformed_kp3d_buf.data();
            };
            // When arena alignment is active, the transform handles mm→m scaling,
            // so disable IK's internal scaling by setting scale_factor=1.0.
            // Save/restore the user's original value so the slider still works.
            static float saved_scale = 0.0f;
            if (state.arena_align.valid && mj.scale_factor != 1.0f) {
                saved_scale = mj.scale_factor;
                mj.scale_factor = 1.0f;
            } else if (!state.arena_align.valid && mj.scale_factor == 1.0f && saved_scale != 1.0f) {
                mj.scale_factor = saved_scale;
            }

            // --- Scrollable controls region ---
            float min_ctrl_h = 60.0f, max_ctrl_h = ImGui::GetContentRegionAvail().y - 150.0f;
            state.controls_height = std::clamp(state.controls_height, min_ctrl_h, max_ctrl_h);
            ImGui::BeginChild("##BodyModelControls", ImVec2(0, state.controls_height), true);

            if (ImGui::CollapsingHeader("Controls", ImGuiTreeNodeFlags_DefaultOpen)) {

            ImGui::Text("Model: %s", mj.model_path.c_str());
            ImGui::Text("Sites: %d/%d matched  |  Bodies: %d  |  Joints: %d",
                        mj.mapped_count, ctx.skeleton.num_nodes,
                        (int)mj.model->nbody, (int)mj.model->njnt);

            // Session save/load
            {
                std::string session_path = ctx.pm.project_path + "/mujoco_session.json";
                if (ImGui::SmallButton("Save Session")) {
                    if (state.save_session(session_path))
                        ctx.toasts.push("Session saved");
                    else
                        ctx.toasts.push("Failed to save session", Toast::Error);
                }
                ImGui::SameLine();
                if (std::filesystem::exists(session_path)) {
                    if (ImGui::SmallButton("Load Session")) {
                        if (state.load_session(session_path)) {
                            state.apply_loaded_calibration(mj);
                            ctx.toasts.push("Session loaded");
                        } else {
                            ctx.toasts.push("Failed to load session", Toast::Error);
                        }
                    }
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("Unload Model"))
                    state.unload_requested = true;
            }

            // Model scale (uniform, affects physics).
            // The input shows the TARGET scale (absolute, not relative).
            // Applied as a relative change when the user presses Enter/Tab.
            {
                ImGui::SetNextItemWidth(200);
                ImGui::InputFloat("Model Scale", &state.saved_model_scale, 0, 0, "%.4f");
                if (ImGui::IsItemDeactivatedAfterEdit()) {
                    float target = state.saved_model_scale;
                    if (target > 0.01f && target < 100.0f &&
                        std::abs(target - mj.model_scale) > 1e-6f) {
                        float rel = target / mj.model_scale;
                        mj.apply_model_scale(rel);
                        mujoco_ik_reset(state.ik_state);
                        state.last_solved_frame = -1;
                    }
                }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Uniformly scale the body model.\n"
                        "1.0 = original size. Press Enter or Tab to apply.\n"
                        "Affects mass and inertia for physics.");
            }

            ImGui::Separator();

            if (ImGui::TreeNode("Solver Settings")) {
                if (!state.arena_align.valid) {
                    ImGui::SetNextItemWidth(200);
                    if (ImGui::SliderFloat("Scale factor", &mj.scale_factor, 0.0f, 3.0f,
                                           mj.scale_factor == 0.0f ? "auto" : "%.4f"))
                        mujoco_ik_reset(state.ik_state);
                    if (mj.scale_factor == 0.0f)
                        ImGui::SameLine(), ImGui::TextDisabled("(auto)");
                }
                ImGui::SetNextItemWidth(200);
                ImGui::SliderInt("Max iterations", &state.ik_state.max_iterations, 100, 20000,
                                "%d", ImGuiSliderFlags_Logarithmic);
                ImGui::SetNextItemWidth(200);
                float reg_log = (state.ik_state.reg_strength > 0)
                                    ? log10f((float)state.ik_state.reg_strength) : -6.0f;
                if (ImGui::SliderFloat("Regularization (log10)", &reg_log, -6.0f, 0.0f, "%.1f"))
                    state.ik_state.reg_strength = pow(10.0, reg_log);
                ImGui::Checkbox("Auto-solve on frame change", &state.auto_solve);
                if (state.auto_solve) {
                    ImGui::SameLine();
                    float budget = (float)state.ik_state.time_budget_ms;
                    ImGui::SetNextItemWidth(120);
                    if (ImGui::SliderFloat("Budget (ms)", &budget, 0.0f, 100.0f,
                                           budget == 0.0f ? "unlimited" : "%.0f ms"))
                        state.ik_state.time_budget_ms = (double)budget;
                }
                ImGui::TreePop();
            }

            if (ImGui::Button("Solve Frame")) {
                auto it = ctx.annotations.find(ctx.current_frame_num);
                if (it != ctx.annotations.end() && !it->second.kp3d.empty()) {
                    const Keypoint3D *kp = get_kp3d(it->second.kp3d);
                    double saved_budget = state.ik_state.time_budget_ms;
                    state.ik_state.time_budget_ms = 0.0;
                    mujoco_ik_solve(mj, state.ik_state,
                                    kp,
                                    ctx.skeleton.num_nodes,
                                    ctx.current_frame_num);
                    state.ik_state.time_budget_ms = saved_budget;
                    state.last_solved_frame = ctx.current_frame_num;
                    // Auto-center camera on the torso body after solve
                    int torso_id = mj_name2id(mj.model, mjOBJ_BODY, "torso");
                    if (torso_id >= 0) {
                        state.mjcam.lookat[0] = mj.data->xpos[3*torso_id+0];
                        state.mjcam.lookat[1] = mj.data->xpos[3*torso_id+1];
                        state.mjcam.lookat[2] = mj.data->xpos[3*torso_id+2];
                    }
                }
            }
            // Continue button: run more iterations from current pose
            if (state.ik_state.active_sites > 0 && !state.ik_state.converged) {
                ImGui::SameLine();
                if (ImGui::Button("Continue")) {
                    auto it = ctx.annotations.find(ctx.current_frame_num);
                    if (it != ctx.annotations.end() && !it->second.kp3d.empty()) {
                        const Keypoint3D *kp = get_kp3d(it->second.kp3d);
                        mujoco_ik_continue(mj, state.ik_state,
                                           kp,
                                           ctx.skeleton.num_nodes,
                                           ctx.current_frame_num,
                                           state.ik_state.max_iterations);
                    }
                }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Run %d more iterations from current pose",
                                      state.ik_state.max_iterations);
            }

            // Export button (same line as Solve Frame)
            ImGui::SameLine();
            if (state.export_state.running) {
                int done = state.export_state.frames_done.load(std::memory_order_relaxed);
                int total = state.export_state.frames_total;
                float frac = total > 0 ? (float)done / total : 0.0f;
                char buf[64];
                snprintf(buf, sizeof(buf), "%d / %d frames (%d threads)",
                         done, total, state.export_state.num_threads);
                ImGui::ProgressBar(frac, ImVec2(-1, 0), buf);
            } else {
                // Check if export just finished
                if (!state.export_state.result_msg.empty()) {
                    if (state.export_state.result_ok)
                        ctx.toasts.push(state.export_state.result_msg);
                    else
                        ctx.toasts.push(state.export_state.result_msg, Toast::Error);
                    state.export_state.result_msg.clear();
                }
                if (ImGui::Button("Solve All & Export")) {
                    state.export_state.output_path =
                        ctx.pm.project_path + "/qpos_export.csv";
                    state.export_state.running = true;
                    LaunchExport(mj, state.ik_state, ctx.annotations,
                                 ctx.skeleton.num_nodes, state.site_placement,
                                 state.stac_state, ctx.skeleton.node_names,
                                 state.arena_align, state.export_state);
                }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Solve IK on all labeled frames in parallel\n"
                                      "and save qpos to %s/qpos_export.csv",
                                      ctx.pm.project_path.c_str());
            }

            // Solver status
            if (state.ik_state.active_sites > 0) {
                const char *status = state.ik_state.converged    ? "Converged"
                                   : state.ik_state.time_limited ? "Time limit"
                                   :                               "Not converged";
                ImVec4 color = state.ik_state.converged
                                   ? ImVec4(0.2f, 0.9f, 0.2f, 1.0f)
                                   : ImVec4(1.0f, 0.7f, 0.2f, 1.0f);
                ImGui::TextColored(color, "%s in %d iters (%.1f ms)  |  "
                                   "Residual: %.1f mm  |  Sites: %d",
                                   status,
                                   state.ik_state.iterations_used,
                                   state.ik_state.solve_time_ms,
                                   state.ik_state.final_residual * 1000.0,
                                   state.ik_state.active_sites);
            }

            // --- Site Placement ---
            ImGui::Separator();
            const char *placement_labels[] = {"Default (XML)", "STAC Calibrated"};
            ImGui::SetNextItemWidth(200);
            int prev_placement = state.site_placement;
            ImGui::Combo("Site Placement", &state.site_placement, placement_labels, 2);

            // Apply/remove STAC offsets when user switches
            if (state.site_placement != prev_placement) {
                if (state.site_placement == SITES_DEFAULT) {
                    // Switch to default: remove offsets
                    if (state.stac_state.calibrated)
                        stac_reset(mj, state.stac_state);
                } else if (state.site_placement == SITES_STAC) {
                    // Switch to STAC: apply active calibration offsets
                    if (state.active_calibration > 0 &&
                        state.active_calibration < (int)state.calib_history.size()) {
                        auto &c = state.calib_history[state.active_calibration];
                        if (!c.site_offsets.empty() &&
                            (int)c.site_offsets.size() == 3 * (int)mj.model->nsite) {
                            if (state.stac_state.original_site_pos.empty()) {
                                int ns = (int)mj.model->nsite;
                                state.stac_state.original_site_pos.assign(
                                    mj.model->site_pos, mj.model->site_pos + 3 * ns);
                            }
                            state.stac_state.site_offsets = c.site_offsets;
                            state.stac_state.calibrated = true;
                            stac_apply_offsets(mj, state.stac_state);
                        }
                    }
                }
                mujoco_ik_reset(state.ik_state);
                state.last_solved_frame = -1;
            }

            if (state.site_placement == SITES_STAC) {
                ImGui::Separator();

                // Initialize default entry if history is empty
                if (state.calib_history.empty()) {
                    CalibrationEntry def;
                    def.name = "Default (XML)";
                    int ns = (int)mj.model->nsite;
                    def.site_offsets.assign(3 * ns, 0.0);
                    def.displacement_mm.assign(ns, 0.0);
                    def.site_names.resize(ns);
                    for (int i = 0; i < ns; i++) {
                        const char *n = mj_id2name(mj.model, mjOBJ_SITE, i);
                        def.site_names[i] = n ? n : "";
                    }
                    state.calib_history.push_back(std::move(def));
                    state.active_calibration = 0;
                }

                // --- Progress display during calibration ---
                if (state.stac_calibrating) {
                    int round = state.stac_state.current_round.load(std::memory_order_relaxed);
                    int total_rounds = state.stac_state.total_rounds.load(std::memory_order_relaxed);
                    int phase = state.stac_state.phase.load(std::memory_order_relaxed);
                    int frames_done = state.stac_state.round_frames_done.load(std::memory_order_relaxed);
                    int frames_total = state.stac_state.round_frames_total.load(std::memory_order_relaxed);
                    int m_done = state.stac_state.m_iter_done.load(std::memory_order_relaxed);

                    const char *phase_name = (phase == 1) ? "Q-phase (IK solve)"
                                           : (phase == 2) ? "M-phase (offset SGD)"
                                           : (phase == 3) ? "Final Q-phase"
                                           : "Starting...";
                    ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.3f, 1.0f),
                        "STAC Calibrating — Round %d/%d", round, total_rounds);
                    ImGui::Text("%s", phase_name);
                    if (phase == 1 || phase == 3) {
                        float frac = frames_total > 0 ? (float)frames_done / frames_total : 0.0f;
                        char buf[64]; snprintf(buf, sizeof(buf), "%d / %d frames", frames_done, frames_total);
                        ImGui::ProgressBar(frac, ImVec2(-1, 0), buf);
                    } else if (phase == 2) {
                        float frac = state.stac_state.m_max_iters > 0
                            ? (float)m_done / state.stac_state.m_max_iters : 0.0f;
                        char buf[64]; snprintf(buf, sizeof(buf), "%d / %d SGD iters", m_done, state.stac_state.m_max_iters);
                        ImGui::ProgressBar(frac, ImVec2(-1, 0), buf);
                    }

                    // Check if calibration finished
                    if (state.stac_state.calibrated) {
                        state.stac_calibrating = false;
                        mujoco_ik_reset(state.ik_state);
                        state.last_solved_frame = -1;

                        // Build calibration entry from completed STAC
                        CalibrationEntry entry;
                        entry.name = state.stac_state.symmetric ? "STAC Symmetric" : "STAC";
                        entry.symmetric = state.stac_state.symmetric;
                        entry.n_iters = state.stac_state.n_iters;
                        entry.n_samples = state.stac_state.frames_used;
                        entry.m_max_iters = state.stac_state.m_max_iters;
                        entry.m_lr = state.stac_state.m_lr;
                        entry.m_reg_coef = state.stac_state.m_reg_coef;
                        entry.pre_residual = state.stac_state.pre_residual;
                        entry.post_residual = state.stac_state.post_residual;
                        entry.time_s = state.stac_state.calibration_time_s;
                        entry.site_offsets = state.stac_state.site_offsets;
                        int ns = (int)entry.site_offsets.size() / 3;
                        entry.site_names.resize(ns);
                        entry.displacement_mm.resize(ns);
                        for (int i = 0; i < ns; i++) {
                            const char *n = mj_id2name(mj.model, mjOBJ_SITE, i);
                            entry.site_names[i] = n ? n : "";
                            double ox = entry.site_offsets[3*i], oy = entry.site_offsets[3*i+1], oz = entry.site_offsets[3*i+2];
                            entry.displacement_mm[i] = std::sqrt(ox*ox + oy*oy + oz*oz) * 1000.0;
                        }
                        state.calib_history.push_back(std::move(entry));
                        state.active_calibration = (int)state.calib_history.size() - 1;
                        ctx.toasts.push("STAC: " +
                            std::to_string((int)state.stac_state.pre_residual) + " mm -> " +
                            std::to_string((int)state.stac_state.post_residual) + " mm");
                    }
                }

                // --- New calibration controls ---
                if (!state.stac_calibrating) {
                    ImGui::SliderInt("Alternating rounds", &state.stac_state.n_iters, 1, 10);
                    ImGui::SliderInt("Sample frames", &state.stac_state.n_sample_frames, 50, 500);
                    ImGui::Checkbox("Symmetric KP Sites", &state.stac_state.symmetric);
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("Force bilateral symmetry: midline sites stay on "
                            "midline,\nL/R pairs get mirrored offsets (Y-axis reflection)");
                    if (ImGui::Button("Calibrate Offsets")) {
                        state.stac_calibrating = true;
                        // Reset to default offsets before calibrating
                        stac_reset(mj, state.stac_state);
                        std::thread([&mj, &state, &ctx]() {
                            stac_calibrate(mj, state.stac_state, state.ik_state,
                                           ctx.annotations, ctx.skeleton.num_nodes,
                                           &ctx.skeleton.node_names,
                                           state.arena_align.valid ? &state.arena_align : nullptr);
                        }).detach();
                    }
                }

                // --- Calibration history table ---
                if (state.calib_history.size() > 1) {
                    ImGui::Separator();
                    ImGui::Text("Site Placements");

                    if (ImGui::BeginTable("##CalibHistory", 5,
                            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_SizingFixedFit)) {
                        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 20);
                        ImGui::TableSetupColumn("Name");
                        ImGui::TableSetupColumn("Residual");
                        ImGui::TableSetupColumn("Settings");
                        ImGui::TableSetupColumn("Time");
                        ImGui::TableHeadersRow();

                        for (int ci = 0; ci < (int)state.calib_history.size(); ci++) {
                            auto &c = state.calib_history[ci];
                            ImGui::TableNextRow();
                            bool is_active = (ci == state.active_calibration);

                            // Radio button
                            ImGui::TableSetColumnIndex(0);
                            char radio_id[32]; snprintf(radio_id, sizeof(radio_id), "##cr%d", ci);
                            if (ImGui::RadioButton(radio_id, is_active) && !is_active) {
                                // Switch to this calibration
                                state.active_calibration = ci;
                                if (ci == 0) {
                                    stac_reset(mj, state.stac_state);
                                    state.site_placement = SITES_DEFAULT;
                                } else {
                                    // Apply these offsets
                                    if (state.stac_state.original_site_pos.empty()) {
                                        int ns = (int)mj.model->nsite;
                                        state.stac_state.original_site_pos.assign(
                                            mj.model->site_pos, mj.model->site_pos + 3*ns);
                                    }
                                    state.stac_state.site_offsets = c.site_offsets;
                                    stac_apply_offsets(mj, state.stac_state);
                                    state.stac_state.calibrated = true;
                                }
                                mujoco_ik_reset(state.ik_state);
                                state.last_solved_frame = -1;
                            }

                            // Name
                            ImGui::TableSetColumnIndex(1);
                            if (is_active)
                                ImGui::TextColored(ImVec4(0.3f, 0.9f, 0.3f, 1.0f), "%s", c.name.c_str());
                            else
                                ImGui::Text("%s", c.name.c_str());

                            // Residual
                            ImGui::TableSetColumnIndex(2);
                            if (ci == 0)
                                ImGui::TextDisabled("—");
                            else
                                ImGui::Text("%.1f -> %.1f mm", c.pre_residual, c.post_residual);

                            // Settings
                            ImGui::TableSetColumnIndex(3);
                            if (ci == 0)
                                ImGui::TextDisabled("—");
                            else
                                ImGui::Text("%dr %ds %s", c.n_iters, c.n_samples,
                                    c.symmetric ? "sym" : "");

                            // Time
                            ImGui::TableSetColumnIndex(4);
                            if (ci == 0)
                                ImGui::TextDisabled("—");
                            else
                                ImGui::Text("%.1fs", c.time_s);
                        }
                        ImGui::EndTable();
                    }

                    // Per-site displacement details for active calibration
                    auto &active = state.calib_history[state.active_calibration];
                    if (state.active_calibration > 0 && !active.displacement_mm.empty()) {
                        if (ImGui::TreeNode("Per-site displacements")) {
                            if (ImGui::BeginTable("##SiteDisp", 5,
                                    ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                    ImGuiTableFlags_Sortable | ImGuiTableFlags_SizingFixedFit)) {
                                ImGui::TableSetupColumn("Site");
                                ImGui::TableSetupColumn("dx (mm)", ImGuiTableColumnFlags_DefaultSort);
                                ImGui::TableSetupColumn("dy (mm)");
                                ImGui::TableSetupColumn("dz (mm)");
                                ImGui::TableSetupColumn("|d| (mm)");
                                ImGui::TableHeadersRow();

                                int ns = (int)active.displacement_mm.size();
                                for (int i = 0; i < ns; i++) {
                                    if (active.displacement_mm[i] < 0.01) continue; // skip unchanged
                                    ImGui::TableNextRow();
                                    ImGui::TableSetColumnIndex(0);
                                    ImGui::Text("%s", active.site_names[i].c_str());
                                    ImGui::TableSetColumnIndex(1);
                                    ImGui::Text("%.1f", active.site_offsets[3*i+0] * 1000.0);
                                    ImGui::TableSetColumnIndex(2);
                                    ImGui::Text("%.1f", active.site_offsets[3*i+1] * 1000.0);
                                    ImGui::TableSetColumnIndex(3);
                                    ImGui::Text("%.1f", active.site_offsets[3*i+2] * 1000.0);
                                    ImGui::TableSetColumnIndex(4);
                                    float mag = (float)active.displacement_mm[i];
                                    ImVec4 color = (mag > 20.0f) ? ImVec4(1,0.3f,0.3f,1)
                                                 : (mag > 10.0f) ? ImVec4(1,0.7f,0.2f,1)
                                                 : ImVec4(0.8f,0.8f,0.8f,1);
                                    ImGui::TextColored(color, "%.1f", mag);
                                }
                                ImGui::EndTable();
                            }
                            ImGui::TreePop();
                        }
                    }
                }
            }

            ImGui::Separator();

            // --- Display options ---
            ImGui::Checkbox("Skin", &state.show_skin);
            ImGui::SameLine();
            ImGui::Checkbox("Bodies", &state.show_bodies);
            ImGui::SameLine();
            ImGui::Checkbox("Sites", &state.show_site_markers);
            ImGui::SameLine();
            ImGui::Checkbox("Arena", &state.show_arena);

            if (state.show_arena) {
                ImGui::SetNextItemWidth(100);
                ImGui::InputFloat("Arena W", &state.arena_width, 0, 0, "%.3f");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(100);
                ImGui::InputFloat("Arena D", &state.arena_depth, 0, 0, "%.3f");
                ImGui::SetNextItemWidth(200);
                ImGui::InputFloat3("Arena offset", state.arena_offset, "%.4f");
            }

            if (state.show_skin && mj.model->nskin > 0) {
                ImGui::SetNextItemWidth(200);
                ImGui::SliderFloat("Skin inflate", &mj.model->skin_inflate[0],
                                   0.0f, 0.01f, "%.4f m");
            }

            // --- Initialize MuJoCo camera on first use ---
            if (!state.cam_initialized) {
                mjv_defaultCamera(&state.mjcam);
                state.mjcam.lookat[0] = 0.0;
                state.mjcam.lookat[1] = 0.0;
                state.mjcam.lookat[2] = 0.04;
                state.mjcam.distance  = 0.4;
                state.mjcam.azimuth   = 160.0;
                state.mjcam.elevation = -30.0;
                state.cam_initialized = true;
            }

            // --- Camera selector: free camera or calibration cameras ---
            if (!ctx.pm.camera_params.empty()) {
                ImGui::Separator();
                ImGui::Text("Camera View");
                const char *preview = (state.selected_camera < 0)
                    ? "Free Camera"
                    : ctx.pm.camera_names[state.selected_camera].c_str();
                ImGui::SetNextItemWidth(-1);
                if (ImGui::BeginCombo("##CameraSelect", preview)) {
                    if (ImGui::Selectable("Free Camera", state.selected_camera < 0)) {
                        state.selected_camera = -1;
                        state.calib_cam_user_override = false;
                        state.cam_zoom = 1.0f; state.cam_pan[0] = state.cam_pan[1] = 0;
                    }
                    for (int i = 0; i < (int)ctx.pm.camera_names.size(); i++) {
                        if (i >= (int)ctx.pm.camera_params.size()) break;
                        bool selected = (state.selected_camera == i);
                        if (ImGui::Selectable(ctx.pm.camera_names[i].c_str(), selected)) {
                            state.selected_camera = i;
                            state.calib_cam_user_override = false;
                            state.cam_zoom = 1.0f; state.cam_pan[0] = state.cam_pan[1] = 0;
                        }
                    }
                    ImGui::EndCombo();
                }

                if (state.selected_camera >= 0) {
                    ImGui::Checkbox("Video Background", &state.show_video_bg);
                    if (state.show_video_bg) {
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(120);
                        ImGui::SliderFloat("Opacity", &state.scene_opacity, 0.0f, 1.0f, "%.2f");
                    }
                    if (state.calib_cam_user_override) {
                        ImGui::SameLine();
                        if (ImGui::SmallButton("Reset View")) {
                            state.calib_cam_user_override = false;
                            state.cam_zoom = 1.0f;
                            state.cam_pan[0] = state.cam_pan[1] = 0;
                        }
                    }
                }

                // Build direct view/projection override from calibration extrinsics.
                // Skip if user has manually overridden (dragged/scrolled).
                if (state.selected_camera >= 0 &&
                    state.selected_camera < (int)ctx.pm.camera_params.size() &&
                    !state.calib_cam_user_override) {
                    // Stored in BodyModelState for the render call below
                    state.calib_view_override.active = true;

                    const CameraParams &cp = ctx.pm.camera_params[state.selected_camera];
                    double img_w = cp.image_width  > 0 ? cp.image_width  : 1024.0;
                    double img_h = cp.image_height > 0 ? cp.image_height : 1024.0;

                    Eigen::Matrix4f V = Eigen::Matrix4f::Zero();
                    Eigen::Matrix4f P = Eigen::Matrix4f::Zero();
                    Eigen::Vector3d eye_w;

                    if (cp.telecentric) {
                        // Telecentric/orthographic: build combined view+proj from DLT
                        // The projection_mat is a 3x4 affine matrix [A t; 0 0 0 1]
                        // that maps 3D world points (mm) to 2D pixel coords directly.
                        // We need to convert this to NDC for Metal.
                        double sf = (state.arena_align.valid) ? 1.0 : 0.1; // mm→cm or aligned
                        Eigen::Matrix<double,3,4> PM = cp.projection_mat;

                        // Build the 4x4 Metal projection:
                        // NDC_x = (2*u/w - 1), NDC_y = (1 - 2*v/h), NDC_z in [0,1]
                        // u = PM.row(0) * [X Y Z 1], v = PM.row(1) * [X Y Z 1]
                        // But we need to account for arena alignment: X_mj = align(X_calib)
                        // For simplicity, build the combined matrix P*V that maps
                        // MuJoCo world coords to NDC.
                        Eigen::Matrix4d M = Eigen::Matrix4d::Zero();
                        // Row 0: NDC_x = 2*(PM.row(0)*p)/w - 1
                        for (int c = 0; c < 4; c++)
                            M(0,c) = 2.0 * PM(0,c) / img_w;
                        M(0,3) -= 1.0;
                        // Row 1: NDC_y = -(2*(PM.row(1)*p)/h - 1) = 1 - 2*(PM.row(1)*p)/h
                        for (int c = 0; c < 4; c++)
                            M(1,c) = -2.0 * PM(1,c) / img_h;
                        M(1,3) += 1.0;
                        // Row 2: depth — map a reasonable Z range to [0,1]
                        // Use the camera's forward direction (R row 2) for depth
                        double near_z = -10.0, far_z = 10.0; // generous range in world units
                        for (int c = 0; c < 3; c++)
                            M(2,c) = cp.r(2,c) / (far_z - near_z);
                        M(2,3) = 0.5; // center of depth range
                        // Row 3: w = 1 (orthographic, no perspective divide)
                        M(3,3) = 1.0;

                        // If arena alignment is active, compose: M maps calib mm coords.
                        // We need to map MuJoCo coords. P_final = M * align_inverse.
                        // But it's simpler: pass V=identity, P=M, and let the IK transform
                        // handle the calib→MuJoCo mapping. However the renderer operates
                        // in MuJoCo space. So we need M to map MuJoCo → NDC.
                        // M currently maps calib_mm → NDC. We need: MuJoCo → calib_mm → NDC.
                        // calib_mm = (p_mj - t) / (scale * R) ... inverse of arena transform.
                        if (state.arena_align.valid) {
                            // Build 4x4 inverse arena transform
                            Eigen::Matrix4d A_inv = Eigen::Matrix4d::Identity();
                            double inv_s = 1.0 / state.arena_align.scale;
                            Eigen::Matrix3d Rt = state.arena_align.R.transpose();
                            A_inv.block<3,3>(0,0) = Rt * inv_s;
                            A_inv.block<3,1>(0,3) = -Rt * state.arena_align.t * inv_s;
                            M = M * A_inv;
                        } else {
                            // Scale: MuJoCo is in model units, DLT expects mm
                            Eigen::Matrix4d Sc = Eigen::Matrix4d::Identity();
                            Sc(0,0) = 1.0/sf; Sc(1,1) = 1.0/sf; Sc(2,2) = 1.0/sf;
                            M = M * Sc;
                        }

                        // Use M as combined view-projection (V=identity)
                        V = Eigen::Matrix4f::Identity();
                        P = M.cast<float>();
                        // Eye position: place along camera's optical axis for lighting
                        Eigen::Vector3d cam_dir(cp.r(2,0), cp.r(2,1), cp.r(2,2));
                        if (state.arena_align.valid)
                            cam_dir = state.arena_align.R * cam_dir;
                        eye_w = cam_dir.normalized() * 5.0; // arbitrary distance along axis

                    } else {
                        // Perspective camera (existing code)
                        Eigen::Vector3d eye_calib = -cp.r.transpose() * cp.tvec;
                        Eigen::Vector3d cam_fwd_calib = cp.r.transpose() * Eigen::Vector3d(0, 0, 1);
                        Eigen::Vector3d cam_up_calib  = cp.r.transpose() * Eigen::Vector3d(0, -1, 0);

                        Eigen::Vector3d cam_fwd, cam_up;
                        if (state.arena_align.valid) {
                            eye_w   = state.arena_align.transform(eye_calib);
                            cam_fwd = state.arena_align.scale * state.arena_align.R * cam_fwd_calib;
                            cam_up  = state.arena_align.scale * state.arena_align.R * cam_up_calib;
                        } else {
                            double sf = (double)mj.scale_factor;
                            if (sf <= 0.0) sf = 0.001;
                            eye_w   = eye_calib * sf;
                            cam_fwd = cam_fwd_calib;
                            cam_up  = cam_up_calib;
                        }
                        cam_fwd.normalize();
                        cam_up.normalize();
                        Eigen::Vector3d lookat_pt = eye_w + cam_fwd;

                        Eigen::Vector3d f = (lookat_pt - eye_w).normalized();
                        Eigen::Vector3d s = f.cross(cam_up).normalized();
                        Eigen::Vector3d u = s.cross(f);
                        V(0,0) = s.x();  V(0,1) = s.y();  V(0,2) = s.z();  V(0,3) = -s.dot(eye_w);
                        V(1,0) = u.x();  V(1,1) = u.y();  V(1,2) = u.z();  V(1,3) = -u.dot(eye_w);
                        V(2,0) = -f.x(); V(2,1) = -f.y(); V(2,2) = -f.z(); V(2,3) = f.dot(eye_w);
                        V(3,3) = 1.0f;

                        double fx = cp.k(0,0), fy = cp.k(1,1);
                        double cx = cp.k(0,2), cy = cp.k(1,2);
                        double near_z = 0.001, far_z = 100.0;
                        P(0,0) = 2.0f * fx / img_w;
                        P(0,2) = 1.0f - 2.0f * cx / img_w;
                        P(1,1) = 2.0f * fy / img_h;
                        P(1,2) = 2.0f * cy / img_h - 1.0f;
                        P(2,2) = far_z / (near_z - far_z);
                        P(2,3) = far_z * near_z / (near_z - far_z);
                        P(3,2) = -1.0f;
                    }

                    // Apply zoom/pan as a post-projection NDC transform.
                    // This matches the blit UV transform exactly at all zoom levels:
                    //   NDC_new = (NDC_old - 2*pan) / zoom
                    float iz = 1.0f / state.cam_zoom;
                    Eigen::Matrix4f S = Eigen::Matrix4f::Identity();
                    S(0,0) = iz;
                    S(1,1) = iz;
                    S(0,3) = -2.0f * state.cam_pan[0] * iz;
                    S(1,3) =  2.0f * state.cam_pan[1] * iz; // opposite sign: blit flips Y
                    P = S * P;

                    memcpy(state.calib_view_override.view, V.data(), 16 * sizeof(float));
                    memcpy(state.calib_view_override.proj, P.data(), 16 * sizeof(float));
                    state.calib_view_override.eye[0] = (float)eye_w.x();
                    state.calib_view_override.eye[1] = (float)eye_w.y();
                    state.calib_view_override.eye[2] = (float)eye_w.z();

                    // Set mjvCamera for smooth transition if user starts dragging
                    {
                        Eigen::Vector3d dir(cp.r(2,0), cp.r(2,1), cp.r(2,2)); // camera Z axis
                        if (state.arena_align.valid)
                            dir = state.arena_align.R * dir;
                        dir.normalize();
                        Eigen::Vector3d lk = eye_w + dir * std::max(eye_w.norm() * 0.8, 0.01);
                        state.mjcam.lookat[0] = lk.x(); state.mjcam.lookat[1] = lk.y();
                        state.mjcam.lookat[2] = lk.z();
                        state.mjcam.distance = (eye_w - lk).norm();
                        state.mjcam.azimuth = atan2(dir.y(), dir.x()) * 180.0 / M_PI;
                        state.mjcam.elevation = asin(std::clamp(dir.z(), -1.0, 1.0)) * 180.0 / M_PI;
                        state.mjcam.orthographic = cp.telecentric ? 1 : 0;
                    }

                    // Store aspect ratio for viewport sizing
                    state.calib_aspect = (float)(img_w / img_h);
                } else {
                    state.calib_view_override.active = false;
                }
            }

            // --- Arena Alignment ---
            if (ImGui::CollapsingHeader("Arena Alignment")) {
                if (state.arena_align.valid) {
                    // Human-readable rotation description
                    const auto &R = state.arena_align.R;
                    const char *rot_desc = "custom rotation";
                    if (R.isApprox(Eigen::Matrix3d::Identity(), 0.01)) rot_desc = "identity (no rotation)";
                    else if (R(0,0) < -0.9 && R(1,1) < -0.9 && R(2,2) > 0.9) rot_desc = "180° around Z (flip X/Y)";
                    else if (R(0,0) < -0.9 && R(2,2) < -0.9 && R(1,1) > 0.9) rot_desc = "180° around Y (flip X/Z)";
                    else if (R(1,1) < -0.9 && R(2,2) < -0.9 && R(0,0) > 0.9) rot_desc = "180° around X (flip Y/Z)";
                    ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.2f, 1.0f),
                        "Aligned: %s, residual=%.2f mm",
                        rot_desc, state.arena_align.residual_mm);
                    // Z offset: fine-tune vertical position (compensates for
                    // labeling corners slightly above/below the arena surface)
                    float z_off_mm = (float)(state.arena_align.t.z() * 1000.0);
                    ImGui::SetNextItemWidth(200);
                    if (ImGui::SliderFloat("Z offset (mm)", &z_off_mm, -50.0f, 50.0f, "%.1f")) {
                        state.arena_align.t.z() = z_off_mm / 1000.0;
                        mujoco_ik_reset(state.ik_state);
                        state.last_solved_frame = -1;
                    }
                    if (ImGui::Button("Clear Alignment")) {
                        state.arena_align = ArenaAlignment{};
                        mujoco_ik_reset(state.ik_state);
                        state.last_solved_frame = -1;
                    }
                }

                if (!state.alignment_mode) {
                    // Quick axis flip: 180° rotation around Z (flips X and Y)
                    // This is the most common mismatch between calibration and MuJoCo frames.
                    {
                        bool flip = false;
                        if (!state.arena_align.valid) {
                            // Check if R is approximately diag(-1,-1,1)
                            flip = false;
                        } else {
                            flip = (state.arena_align.R(0,0) < -0.9 && state.arena_align.R(1,1) < -0.9);
                        }
                        if (ImGui::Checkbox("Flip X/Y (180° around Z)", &flip)) {
                            if (flip) {
                                state.arena_align.valid = true;
                                state.arena_align.R = Eigen::Matrix3d::Identity();
                                state.arena_align.R(0,0) = -1.0;
                                state.arena_align.R(1,1) = -1.0;
                                state.arena_align.t = Eigen::Vector3d::Zero();
                                state.arena_align.scale = 0.001; // mm → m
                                state.arena_align.residual_mm = 0.0;
                                state.arena_align.corners_set = false;
                            } else {
                                state.arena_align = ArenaAlignment{};
                            }
                            mujoco_ik_reset(state.ik_state);
                            state.last_solved_frame = -1;
                        }
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Common fix: calibration X/Y axes point opposite\n"
                                "to MuJoCo. Applies R=diag(-1,-1,1) with mm->m scaling.");
                    }

                    ImGui::TextWrapped("Or label the 4 arena corners for full alignment:");

                    // Load Previous: reuse saved corners without re-labeling
                    if (ctx.user_settings.arena_corners.size() == 12) {
                        if (ImGui::Button("Load Previous Corners")) {
                            for (int i = 0; i < 4; i++) {
                                state.arena_align.calib_corners[i] = Eigen::Vector3d(
                                    ctx.user_settings.arena_corners[3*i+0],
                                    ctx.user_settings.arena_corners[3*i+1],
                                    ctx.user_settings.arena_corners[3*i+2]);
                            }
                            state.arena_align.corners_set = true;
                            compute_arena_alignment(state.arena_align, state.arena_width, state.arena_depth, state.arena_offset[0], state.arena_offset[1], state.arena_offset[2]);
                            if (state.arena_align.valid) {
                                mujoco_ik_reset(state.ik_state);
                                state.last_solved_frame = -1;
                                ctx.toasts.push("Arena aligned from saved corners: " +
                                    std::to_string(state.arena_align.residual_mm).substr(0, 5) +
                                    " mm residual");
                            }
                        }
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("(%.0f, %.0f, %.0f), (%.0f, %.0f, %.0f),\n"
                                "(%.0f, %.0f, %.0f), (%.0f, %.0f, %.0f)",
                                ctx.user_settings.arena_corners[0], ctx.user_settings.arena_corners[1],
                                ctx.user_settings.arena_corners[2], ctx.user_settings.arena_corners[3],
                                ctx.user_settings.arena_corners[4], ctx.user_settings.arena_corners[5],
                                ctx.user_settings.arena_corners[6], ctx.user_settings.arena_corners[7],
                                ctx.user_settings.arena_corners[8], ctx.user_settings.arena_corners[9],
                                ctx.user_settings.arena_corners[10], ctx.user_settings.arena_corners[11]);
                        }
                    }

                    if (ctx.user_settings.arena_corners.size() == 12)
                        ImGui::SameLine();
                    if (ImGui::Button("Enter Alignment Mode")) {
                        // Save current skeleton and annotations
                        state.saved_skeleton = ctx.skeleton;
                        state.saved_annotations = ctx.annotations;
                        // Switch to ArenaCorners4 skeleton
                        ctx.skeleton.node_colors.clear();
                        ctx.skeleton.edges.clear();
                        ctx.skeleton.node_names.clear();
                        skeleton_initialize("ArenaCorners4", &ctx.skeleton, ArenaCorners4);
                        ctx.skeleton.has_skeleton = true;
                        // Clear annotations for alignment labeling
                        ctx.annotations.clear();
                        state.alignment_mode = true;
                        state.last_solved_frame = -1;
                        ctx.toasts.push("Alignment mode: label 4 arena corners, then triangulate");
                    }
                } else {
                    ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.3f, 1.0f),
                        "ALIGNMENT MODE ACTIVE");
                    ImGui::TextWrapped("Label the 4 arena corners (A=red, B=green, "
                        "C=blue, D=yellow) in at least 2 camera views each, "
                        "then triangulate. The corners form a square (A-B-C-D).");

                    // Check if we have triangulated 3D for all 4 corners
                    int n_triangulated = 0;
                    for (auto &[fnum, fa] : ctx.annotations) {
                        if (fa.kp3d.empty()) continue;
                        for (int i = 0; i < 4 && i < (int)fa.kp3d.size(); i++)
                            if (fa.kp3d[i].triangulated) n_triangulated++;
                        break; // only need one frame
                    }

                    ImGui::Text("Triangulated corners: %d / 4", n_triangulated);

                    if (n_triangulated >= 4 && ImGui::Button("Compute Alignment")) {
                        // Extract the 4 corner 3D positions
                        for (auto &[fnum, fa] : ctx.annotations) {
                            if (fa.kp3d.empty()) continue;
                            for (int i = 0; i < 4 && i < (int)fa.kp3d.size(); i++) {
                                if (fa.kp3d[i].triangulated) {
                                    state.arena_align.calib_corners[i] = Eigen::Vector3d(
                                        fa.kp3d[i].x, fa.kp3d[i].y, fa.kp3d[i].z);
                                }
                            }
                            break;
                        }
                        state.arena_align.corners_set = true;
                        compute_arena_alignment(state.arena_align, state.arena_width, state.arena_depth, state.arena_offset[0], state.arena_offset[1], state.arena_offset[2]);
                        if (state.arena_align.valid) {
                            // Save corners to settings for "Load Previous"
                            ctx.user_settings.arena_corners.resize(12);
                            for (int i = 0; i < 4; i++) {
                                ctx.user_settings.arena_corners[3*i+0] = state.arena_align.calib_corners[i].x();
                                ctx.user_settings.arena_corners[3*i+1] = state.arena_align.calib_corners[i].y();
                                ctx.user_settings.arena_corners[3*i+2] = state.arena_align.calib_corners[i].z();
                            }
                            save_user_settings(ctx.user_settings);
                            ctx.toasts.push("Arena aligned: " +
                                std::to_string(state.arena_align.residual_mm).substr(0, 5) +
                                " mm residual");
                        }
                    }

                    if (ImGui::Button("Exit Alignment Mode")) {
                        // Restore original skeleton and annotations
                        ctx.skeleton = state.saved_skeleton;
                        ctx.annotations = std::move(state.saved_annotations);
                        state.alignment_mode = false;
                        mujoco_ik_reset(state.ik_state);
                        state.last_solved_frame = -1;
                    }
                }
            }

            } // end CollapsingHeader("Controls")

            // Auto-solve runs every frame regardless of header state
            if (state.auto_solve && ctx.current_frame_num != state.last_solved_frame) {
                state.last_solved_frame = ctx.current_frame_num;
                auto it = ctx.annotations.find(ctx.current_frame_num);
                if (it != ctx.annotations.end() && !it->second.kp3d.empty()) {
                    const Keypoint3D *kp = get_kp3d(it->second.kp3d);
                    mujoco_ik_solve(mj, state.ik_state,
                                    kp,
                                    ctx.skeleton.num_nodes,
                                    ctx.current_frame_num);
                    int torso_id = mj_name2id(mj.model, mjOBJ_BODY, "torso");
                    if (torso_id >= 0) {
                        state.mjcam.lookat[0] = mj.data->xpos[3*torso_id+0];
                        state.mjcam.lookat[1] = mj.data->xpos[3*torso_id+1];
                        state.mjcam.lookat[2] = mj.data->xpos[3*torso_id+2];
                    }
                }
            }

            ImGui::EndChild(); // ##BodyModelControls

            // --- Draggable splitter ---
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.3f, 0.3f, 0.3f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.6f, 0.6f, 0.6f, 1.0f));
            ImGui::Button("##splitter", ImVec2(-1, 4));
            ImGui::PopStyleColor(3);
            if (ImGui::IsItemActive()) {
                state.controls_height += ImGui::GetIO().MouseDelta.y;
            }
            if (ImGui::IsItemHovered())
                ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);

            // --- 3D Viewport (fills remaining space) ---
            ImVec2 avail = ImGui::GetContentRegionAvail();
            float vp_w = avail.x > 100 ? avail.x : 400;
            float vp_h = avail.y > 100 ? avail.y : 400;

            // When a calibration camera is selected, match its aspect ratio
            if (state.calib_view_override.active && state.calib_aspect > 0.0f) {
                float fit_h = vp_w / state.calib_aspect;
                float fit_w = vp_h * state.calib_aspect;
                if (fit_h <= vp_h) vp_h = fit_h; // width-limited
                else               vp_w = fit_w; // height-limited
            }

            uint32_t w = (uint32_t)vp_w, h = (uint32_t)vp_h;

            // Create or resize renderer
            if (!state.renderer) {
                state.renderer = mujoco_renderer_create(w, h);
            } else {
                uint32_t cw, ch;
                mujoco_renderer_get_size(state.renderer, &cw, &ch);
                if (cw != w || ch != h)
                    mujoco_renderer_resize(state.renderer, w, h);
            }

            if (state.renderer) {
                const ViewOverride *vo = state.calib_view_override.active
                    ? &state.calib_view_override : nullptr;
                // Get video background texture if enabled
                void *bg_tex = nullptr;
                if (state.show_video_bg && state.selected_camera >= 0 &&
                    state.selected_camera < (int)ctx.pm.camera_names.size() &&
                    ctx.scene) {
                    // selected_camera indexes ctx.pm.camera_names which matches
                    // the scene camera order
                    int cam_idx = state.selected_camera;
                    if (cam_idx < (int)ctx.scene->num_cams &&
                        ctx.scene->image_descriptor[cam_idx])
                        bg_tex = (void *)ctx.scene->image_descriptor[cam_idx];
                }
                float opacity = (state.show_video_bg && bg_tex) ? state.scene_opacity : 1.0f;
                mujoco_renderer_render(state.renderer, &mj, &state.mjcam,
                                       state.show_skin, state.show_bodies,
                                       state.show_site_markers, state.show_arena, vo,
                                       state.alignment_mode, bg_tex, opacity,
                                       state.cam_zoom, state.cam_pan,
                                       state.arena_width, state.arena_depth,
                                       state.arena_offset);
                ImTextureID tex = mujoco_renderer_get_texture(state.renderer);
                if (tex) {
                    ImVec2 img_pos = ImGui::GetCursorScreenPos();
                    // Draw the image as background
                    ImGui::GetWindowDrawList()->AddImage(
                        tex, img_pos, ImVec2(img_pos.x + vp_w, img_pos.y + vp_h));

                    // InvisibleButton captures mouse focus properly (unlike Image
                    // which is passive and can lose hover during screen recording)
                    ImGui::InvisibleButton("##mj_viewport", ImVec2(vp_w, vp_h));
                    bool vp_hovered = ImGui::IsItemHovered();
                    bool vp_active  = ImGui::IsItemActive();


                    // Mouse controls on the viewport
                    if (vp_hovered || vp_active) {
                        ImGuiIO &io = ImGui::GetIO();
                        double dx =  (double)io.MouseDelta.x / (double)vp_h;
                        double dy = -(double)io.MouseDelta.y / (double)vp_h;

                        bool any_drag = ImGui::IsMouseDragging(ImGuiMouseButton_Left) ||
                                        ImGui::IsMouseDragging(ImGuiMouseButton_Right) ||
                                        ImGui::IsMouseDragging(ImGuiMouseButton_Middle);
                        bool any_scroll = (vp_hovered && io.MouseWheel != 0.0f);

                        // Double-click to reset to calibration camera view
                        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) &&
                            state.selected_camera >= 0) {
                            state.calib_cam_user_override = false;
                            state.cam_zoom = 1.0f;
                            state.cam_pan[0] = state.cam_pan[1] = 0;
                        }

                        if (state.calib_view_override.active && state.selected_camera >= 0) {
                            // Calibration camera: zoom/pan in image space (video stays aligned)
                            if (ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
                                state.cam_pan[0] += (float)dx * state.cam_zoom;
                                state.cam_pan[1] -= (float)dy * state.cam_zoom; // match dy sign
                            }
                            if (vp_hovered && io.MouseWheel != 0.0f) {
                                float zf = 1.0f + io.MouseWheel * 0.1f;
                                state.cam_zoom = std::clamp(state.cam_zoom / zf, 0.1f, 20.0f);
                            }
                        } else {
                            // Free camera: orbit/pan/zoom via MuJoCo
                            if (ImGui::IsMouseDragging(ImGuiMouseButton_Left))
                                mjv_moveCamera(mj.model, mjMOUSE_ROTATE_V, dx, -dy,
                                               &mj.scene, &state.mjcam);
                            if (ImGui::IsMouseDragging(ImGuiMouseButton_Right))
                                mjv_moveCamera(mj.model, mjMOUSE_MOVE_V, dx, -dy,
                                               &mj.scene, &state.mjcam);
                            if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle))
                                mjv_moveCamera(mj.model, mjMOUSE_ZOOM, dx, dy,
                                               &mj.scene, &state.mjcam);
                            if (vp_hovered && io.MouseWheel != 0.0f)
                                mjv_moveCamera(mj.model, mjMOUSE_ZOOM, 0,
                                               -0.05 * io.MouseWheel,
                                               &mj.scene, &state.mjcam);
                        }
                    }
                }
            }

#endif // RED_HAS_MUJOCO
        },
        [&]() {
            // File dialog (runs every frame regardless of panel visibility)
            if (ImGuiFileDialog::Instance()->Display("ChooseMujocoModel")) {
                if (ImGuiFileDialog::Instance()->IsOk())
                    state.model_path = ImGuiFileDialog::Instance()->GetFilePathName();
                ImGuiFileDialog::Instance()->Close();
            }
        },
        ImVec2(600, 700),
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
}
