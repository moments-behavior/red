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

enum IKMethod { IK_DM_CONTROL = 0, IK_STAC = 1 };

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
    int ik_method = IK_DM_CONTROL;

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

    // Renderer handle + native MuJoCo camera
#ifdef RED_HAS_MUJOCO
    MujocoRenderer *renderer = nullptr;
    mjvCamera mjcam;  // initialized in draw function on first use
#endif

    // Model path for file dialog
    std::string model_path;
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
                         int ik_method, const StacState &stac,
                         const std::vector<std::string> &node_names,
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

    // Deep-copy keypoint data (annotations may be modified on main thread)
    std::vector<std::pair<int, std::vector<Keypoint3D>>> all_frames;
    all_frames.reserve(frame_refs.size());
    for (const auto &fr : frame_refs)
        all_frames.push_back({fr.frame_num, std::vector<Keypoint3D>(fr.kp3d, fr.kp3d + fr.kp3d_size)});

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
    int method = ik_method;

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
        out << "# ik_method: " << (method == IK_STAC ? "IK_STAC" : "IK_dm_control") << "\n";
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
        if (method == IK_STAC) {
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
                ImGui::TextWrapped("Load a MuJoCo XML body model to enable "
                                   "inverse kinematics and body model visualization.");
                ImGui::Spacing();

                if (state.model_path.empty()) {
                    // Try last-used path from settings, then default
                    if (!ctx.user_settings.last_mujoco_model.empty() &&
                        std::filesystem::exists(ctx.user_settings.last_mujoco_model))
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
                        "ChooseMujocoModel", "Select MuJoCo XML", ".xml", cfg);
                }

                if (ImGui::Button("Load Model") && !state.model_path.empty()) {
                    if (mj.load(state.model_path, ctx.skeleton)) {
                        ctx.user_settings.last_mujoco_model = state.model_path;
                        save_user_settings(ctx.user_settings);
                        ctx.toasts.push("MuJoCo model loaded: " +
                                        std::to_string(mj.mapped_count) + "/" +
                                        std::to_string(ctx.skeleton.num_nodes) +
                                        " sites matched");
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
                            ctx.toasts.push("MuJoCo model loaded: " +
                                            std::to_string(mj.mapped_count) + "/" +
                                            std::to_string(ctx.skeleton.num_nodes) +
                                            " sites matched");
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

            // --- Scrollable controls region ---
            float min_ctrl_h = 60.0f, max_ctrl_h = ImGui::GetContentRegionAvail().y - 150.0f;
            state.controls_height = std::clamp(state.controls_height, min_ctrl_h, max_ctrl_h);
            ImGui::BeginChild("##BodyModelControls", ImVec2(0, state.controls_height), true);

            if (ImGui::CollapsingHeader("Controls", ImGuiTreeNodeFlags_DefaultOpen)) {

            ImGui::Text("Model: %s", mj.model_path.c_str());
            ImGui::Text("Sites: %d/%d matched  |  Bodies: %d  |  Joints: %d",
                        mj.mapped_count, ctx.skeleton.num_nodes,
                        (int)mj.model->nbody, (int)mj.model->njnt);
            ImGui::Separator();

            // IK method selector
            const char *ik_methods[] = {"IK_dm_control", "IK_STAC"};
            ImGui::SetNextItemWidth(200);
            ImGui::Combo("IK Method", &state.ik_method, ik_methods, 2);

            if (ImGui::SliderFloat("Scale factor", &mj.scale_factor, 0.0f, 3.0f,
                                   mj.scale_factor == 0.0f ? "auto" : "%.4f"))
                mujoco_ik_reset(state.ik_state); // reset warm-start on scale change
            if (mj.scale_factor == 0.0f)
                ImGui::SameLine(), ImGui::TextDisabled("(auto-detect from data)");

            ImGui::SliderInt("Max iterations", &state.ik_state.max_iterations, 100, 20000,
                            "%d", ImGuiSliderFlags_Logarithmic);
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

            if (ImGui::Button("Solve Frame")) {
                auto it = ctx.annotations.find(ctx.current_frame_num);
                if (it != ctx.annotations.end() && !it->second.kp3d.empty()) {
                    double saved_budget = state.ik_state.time_budget_ms;
                    state.ik_state.time_budget_ms = 0.0; // no time limit for manual solve
                    mujoco_ik_solve(mj, state.ik_state,
                                    it->second.kp3d.data(),
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
                        mujoco_ik_continue(mj, state.ik_state,
                                           it->second.kp3d.data(),
                                           ctx.skeleton.num_nodes,
                                           ctx.current_frame_num,
                                           state.ik_state.max_iterations);
                    }
                }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Run %d more iterations from current pose",
                                      state.ik_state.max_iterations);
            }

            // Export button / progress
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
                                 ctx.skeleton.num_nodes, state.ik_method,
                                 state.stac_state, ctx.skeleton.node_names,
                                 state.export_state);
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

            // --- STAC calibration controls ---
            if (state.ik_method == IK_STAC) {
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
                                           &ctx.skeleton.node_names);
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
                    if (ImGui::Selectable("Free Camera", state.selected_camera < 0))
                        state.selected_camera = -1;
                    for (int i = 0; i < (int)ctx.pm.camera_names.size(); i++) {
                        if (i >= (int)ctx.pm.camera_params.size()) break;
                        if (ctx.pm.camera_params[i].telecentric) continue;
                        bool selected = (state.selected_camera == i);
                        if (ImGui::Selectable(ctx.pm.camera_names[i].c_str(), selected))
                            state.selected_camera = i;
                    }
                    ImGui::EndCombo();
                }

                // Apply calibration camera pose to mjvCamera
                if (state.selected_camera >= 0 &&
                    state.selected_camera < (int)ctx.pm.camera_params.size()) {
                    const CameraParams &cp = ctx.pm.camera_params[state.selected_camera];

                    // Camera position in world: eye = -R^T * t
                    Eigen::Vector3d eye_world = -cp.r.transpose() * cp.tvec;

                    // Scale from calibration units (mm) to model units (m)
                    double sf = (double)mj.scale_factor;
                    if (sf <= 0.0) sf = 0.001; // auto: assume mm→m
                    eye_world *= sf;

                    // Camera forward direction in world: R^T * [0,0,1]
                    // (OpenCV camera looks along +Z in camera frame)
                    Eigen::Vector3d fwd = cp.r.transpose() * Eigen::Vector3d(0, 0, 1);
                    fwd.normalize();

                    // Set lookat to a point along the forward direction,
                    // at the distance from camera to world origin
                    double dist_to_origin = eye_world.norm();
                    double lookat_dist = std::max(dist_to_origin * 0.8, 0.1);

                    Eigen::Vector3d lookat = eye_world + lookat_dist * fwd;

                    state.mjcam.lookat[0] = lookat.x();
                    state.mjcam.lookat[1] = lookat.y();
                    state.mjcam.lookat[2] = lookat.z();
                    state.mjcam.distance = lookat_dist;

                    // MuJoCo: forward = { ce*ca, ce*sa, se }
                    state.mjcam.azimuth = atan2(fwd.y(), fwd.x()) * 180.0 / M_PI;
                    state.mjcam.elevation = asin(std::clamp(fwd.z(), -1.0, 1.0))
                                            * 180.0 / M_PI;
                }
            }

            } // end CollapsingHeader("Controls")

            // Auto-solve runs every frame regardless of header state
            if (state.auto_solve && ctx.current_frame_num != state.last_solved_frame) {
                state.last_solved_frame = ctx.current_frame_num;
                auto it = ctx.annotations.find(ctx.current_frame_num);
                if (it != ctx.annotations.end() && !it->second.kp3d.empty()) {
                    mujoco_ik_solve(mj, state.ik_state,
                                    it->second.kp3d.data(),
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
                mujoco_renderer_render(state.renderer, &mj, &state.mjcam,
                                       state.show_skin, state.show_bodies,
                                       state.show_site_markers, state.show_arena);
                ImTextureID tex = mujoco_renderer_get_texture(state.renderer);
                if (tex) {
                    ImVec2 img_pos = ImGui::GetCursorScreenPos();
                    ImGui::Image(tex, ImVec2(vp_w, vp_h));

                    // Overlay "Unload" button in top-left corner of viewport
                    ImVec2 saved_cursor = ImGui::GetCursorScreenPos();
                    ImGui::SetCursorScreenPos(ImVec2(img_pos.x + 4, img_pos.y + 4));
                    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.7f);
                    if (ImGui::SmallButton("Unload"))
                        state.unload_requested = true;
                    ImGui::PopStyleVar();
                    ImGui::SetCursorScreenPos(saved_cursor);

                    // Mouse controls (matches MuJoCo simulate.cc):
                    //   Left-drag:   orbit (rotate)
                    //   Right-drag:  pan (translate lookat)
                    //   Scroll:      zoom (distance)
                    if (ImGui::IsItemHovered()) {
                        ImGuiIO &io = ImGui::GetIO();
                        double dx =  (double)io.MouseDelta.x / (double)vp_h;
                        double dy = -(double)io.MouseDelta.y / (double)vp_h;

                        // Left-drag to orbit (negate dy for Blender-style:
                        // drag down = camera orbits above, looking down)
                        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left))
                            mjv_moveCamera(mj.model, mjMOUSE_ROTATE_V, dx, -dy,
                                           &mj.scene, &state.mjcam);

                        // Right-drag to pan (negate dy so drag-up pans up)
                        if (ImGui::IsMouseDragging(ImGuiMouseButton_Right))
                            mjv_moveCamera(mj.model, mjMOUSE_MOVE_V, dx, -dy,
                                           &mj.scene, &state.mjcam);

                        // Middle-drag to zoom
                        if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle))
                            mjv_moveCamera(mj.model, mjMOUSE_ZOOM, dx, dy,
                                           &mj.scene, &state.mjcam);

                        // Scroll to zoom
                        if (io.MouseWheel != 0.0f)
                            mjv_moveCamera(mj.model, mjMOUSE_ZOOM, 0,
                                           -0.05 * io.MouseWheel,
                                           &mj.scene, &state.mjcam);
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
