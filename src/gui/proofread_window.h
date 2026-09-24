#pragma once
// proofread_window.h — bad-frames navigator panel that appears after a
// Proofread project is loaded. Reads bad-frame indices from the
// dashboard's /api/bad_frames_all endpoint, scoped to (pm.proofread_animal,
// pm.proofread_session), and lets the user click any frame to seek.
//
// Cameras section: the user decides which cameras have bad calibration by
// looking at the prediction overlay on every camera. Unticking "Use" adds
// the camera to pm.excluded_cameras: triangulation, save and export then
// ignore it. red's own leave-one-out check (camera_check.h) is shown as a
// hint only.
//
// Action signal:
//   - open_requested + requested_frame
//     The main loop reads these and issues an accurate seek_all_cameras
//     to the requested video frame.

#include "imgui.h"
#include "proofread_client.h"
#include "app_context.h"
#include "gui/gui_keypoints.h"

#include <misc/cpp/imgui_stdlib.h>

#include <algorithm>
#include <thread>
#include <set>
#include <memory>
#include <atomic>
#include <cstdio>
#include <string>


// One background prediction download. The worker owns a shared_ptr, so a
// job abandoned by the UI (refresh, project close) finishes harmlessly.
struct ProofreadPredJob {
    std::string session;
    std::vector<int> frames;
    ProofreadPrediction result;
    bool ok = false;
    std::atomic<bool> done{false};
};

struct ProofreadWindowState {
    bool show = false;
    ProofreadState server;   // url, threshold, sessions (fetched lazily)

    // Action signal: the user clicked a frame and wants to seek there.
    bool        open_requested = false;
    int         requested_frame = 0;
    std::string requested_animal;
    std::string requested_session;
    std::string requested_recording_path;

    // Seek-only path (no video reload needed); cleared once the main loop
    // performs the seek.
    int pending_seek_frame = -1;

    // True once we've auto-fetched on first show.
    bool initial_fetch_done = false;

    // Cached client-side analysis of pm.camera_check (recomputed only when
    // the samples or the exclusion set change).
    CameraCheckResult analysis;
    int analysis_version = -1;
    std::vector<bool> analysis_mask;

    // Tailcycle prediction for the queue's frames. With auto_pred on, a Seek
    // to an unlabeled frame overlays it on every camera (proofread instead
    // of relabel).
    ProofreadPrediction pred;
    bool auto_pred = true;
    std::string pred_note;   // last apply result, for the panel
    int last_overlay_frame = -1;  // frame the auto-overlay last looked at
    // Downloads run off the UI thread (a first request makes the server
    // parse the session CSV, ~1 s): one job at a time, frames queued.
    std::shared_ptr<ProofreadPredJob> pred_job;
    std::vector<int> pred_queue;
    std::set<int> pred_asked;     // ever requested since the last refresh
    int force_frame = -1;         // "Load on this frame" waiting on data

    // Camera set changed (excluded / re-included): the main loop reloads the
    // project so excluded cameras disappear (see setup_project).
    bool cameras_dirty = false;
    bool reload_requested = false;
};


inline void proofread_set_camera_used(ProofreadWindowState &w,
                                      AppContext &ctx, int cam_idx, bool use);

namespace proofread_window_detail {

inline const ProofreadSession *find_session(const ProofreadState &s,
                                             const std::string &animal,
                                             const std::string &session) {
    for (const auto &ps : s.sessions) {
        if (ps.animal == animal && ps.session == session) return &ps;
    }
    return nullptr;
}

inline void save_redproj(const ProjectManager &pm) {
    if (pm.project_path.empty() || pm.project_name.empty()) return;
    save_project_manager_json(pm, std::filesystem::path(pm.project_path) /
                                      (pm.project_name + ".redproj"));
}

inline void draw_cameras_section(ProofreadWindowState &w, AppContext &ctx) {
    auto &pm = ctx.pm;
    const int nc = (int)pm.camera_names.size();
    if (nc == 0 || pm.camera_params.empty() || !ctx.scene) {
        ImGui::TextDisabled("(no cameras / calibration loaded)");
        return;
    }

    const auto excluded = excluded_camera_mask(pm);
    int n_used = 0;
    for (bool ex : excluded) n_used += !ex;

    ImGui::TextDisabled("Untick Use on a camera whose predicted points are "
                        "off: its keypoints are\ncleared and it is left out "
                        "of Triangulate (T), save and export.");
    if (ImGui::SmallButton("Scan annotated frames")) {
        for (const auto &[frame, fa] : ctx.annotations)
            pm.camera_check.record(frame, collect_camera_check_obs(
                                              fa, &ctx.skeleton, ctx.scene));
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Sample every frame with un-triangulated 2D "
                          "(e.g. after batch predict).\n"
                          "Frames are also sampled automatically on each "
                          "Triangulate.");
    ImGui::SameLine();
    if (ImGui::SmallButton("Reset samples")) pm.camera_check.clear();
    if (w.analysis_version != pm.camera_check.version ||
        w.analysis_mask != excluded) {
        w.analysis = analyze_camera_check(pm.camera_check, pm.camera_params,
                                          nc, excluded);
        w.analysis_version = pm.camera_check.version;
        w.analysis_mask = excluded;
    }
    const auto &res = w.analysis;
    ImGui::TextDisabled("Error = median reprojection error vs. the "
                        "consistent cameras, %d frame(s) sampled",
                        res.frames);

    const ImVec4 red(1.0f, 0.45f, 0.45f, 1.0f);
    if (ImGui::BeginTable("##cams", 4,
                          ImGuiTableFlags_RowBg |
                              ImGuiTableFlags_BordersInnerH |
                              ImGuiTableFlags_SizingFixedFit)) {
        ImGui::TableSetupColumn("Use");
        ImGui::TableSetupColumn("Camera");
        ImGui::TableSetupColumn("Error px");
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        for (int c = 0; c < nc; ++c) {
            const std::string &cam = pm.camera_names[c];
            ImGui::PushID(c);
            ImGui::TableNextRow();

            ImGui::TableNextColumn();
            bool use = !excluded[c];
            // Keep >= 2 cameras: triangulation needs two views.
            ImGui::BeginDisabled(use && n_used <= 2);
            if (ImGui::Checkbox("##use", &use))
                proofread_set_camera_used(w, ctx, c, use);
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(use ? "Untick: this camera's calibration is "
                                        "bad - clear its keypoints and leave it "
                                        "out of 3D, save and export"
                                      : "Tick: use this camera again");

            ImGui::TableNextColumn();
            if (excluded[c]) ImGui::TextDisabled("%s", cam.c_str());
            else ImGui::TextUnformatted(cam.c_str());

            ImGui::TableNextColumn();
            const auto &rc = res.cams[c];
            if (rc.samples == 0)
                ImGui::TextDisabled("-");
            else if (rc.suggested)
                ImGui::TextColored(red, "%.1f", rc.loo_px);
            else
                ImGui::Text("%.1f", rc.loo_px);
            if (rc.samples > 0 && ImGui::IsItemHovered()) {
                if (std::isfinite(rc.drop_px))
                    ImGui::SetTooltip(
                        "%d keypoint samples\nWith this camera left out, the "
                        "others agree to %.1f px\n(typical: %.1f px)",
                        rc.samples, rc.drop_px, res.typical_drop_px);
                else
                    ImGui::SetTooltip("%d keypoint samples", rc.samples);
            }

            ImGui::TableNextColumn();
            if (!excluded[c] && rc.suggested)
                ImGui::TextColored(red, "suggest exclude");
            else if (excluded[c])
                ImGui::TextDisabled("excluded");
            ImGui::PopID();
        }

        // Excluded cameras that were never loaded (bad calibration): listed
        // so they can be brought back.
        for (const auto &cam : std::vector<std::string>(pm.excluded_cameras)) {
            if (std::find(pm.camera_names.begin(), pm.camera_names.end(),
                          cam) != pm.camera_names.end())
                continue;
            ImGui::PushID(cam.c_str());
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            bool use = false;
            if (ImGui::Checkbox("##use", &use) && use) {
                auto &ex = pm.excluded_cameras;
                ex.erase(std::remove(ex.begin(), ex.end(), cam), ex.end());
                pm.camera_names.push_back(cam);
                std::sort(pm.camera_names.begin(), pm.camera_names.end());
                save_redproj(pm);
                w.cameras_dirty = true;
            }
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%s", cam.c_str());
            ImGui::TableNextColumn();
            ImGui::TextDisabled("-");
            ImGui::TableNextColumn();
            ImGui::TextDisabled("not loaded");
            ImGui::PopID();
        }
        ImGui::EndTable();
    }

    if (w.cameras_dirty) {
        ImGui::TextColored(red, "Camera set changed.");
        ImGui::SameLine();
        if (ImGui::SmallButton("Apply (reload project)"))
            w.reload_requested = true;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Saves labels, then reloads so excluded "
                              "cameras are not loaded.\nUntil then they are "
                              "already ignored by Triangulate, save and export.");
    }
}

// Prefer the prediction whose keypoint set matches the skeleton.
inline std::string preferred_pred_source(const AppContext &ctx) {
    return ctx.skeleton.num_nodes >= 47 ? "tailcycle47" : "tailcycle";
}

inline bool pred_pending(const ProofreadWindowState &w, int f) {
    if (std::find(w.pred_queue.begin(), w.pred_queue.end(), f) !=
        w.pred_queue.end())
        return true;
    return w.pred_job && std::find(w.pred_job->frames.begin(),
                                   w.pred_job->frames.end(), f) !=
                             w.pred_job->frames.end();
}

// Requested and answered, but the server had nothing for it.
inline bool pred_missing(const ProofreadWindowState &w, int f) {
    return w.pred_asked.count(f) && !w.pred.frames.count(f) &&
           !pred_pending(w, f);
}

inline void pred_request(ProofreadWindowState &w, const std::vector<int> &frames) {
    for (int f : frames)
        if (f >= 0 && !w.pred.frames.count(f) && w.pred_asked.insert(f).second)
            w.pred_queue.push_back(f);
}

// Merge a finished job; start the next one if frames are queued.
inline void pred_poll(ProofreadWindowState &w, AppContext &ctx) {
    const auto &pm = ctx.pm;
    if (w.pred_job && w.pred_job->done.load(std::memory_order_acquire)) {
        auto job = std::move(w.pred_job);
        if (job->session == pm.proofread_session) {
            if (job->ok) {
                for (auto &[f, v] : job->result.frames)
                    w.pred.frames[f] = std::move(v);
                w.pred.keypoints = job->result.keypoints;
                w.pred.source = job->result.source;
                w.pred.frame = job->result.frame;
                w.pred.fetched = true;
                w.pred.status = "Prediction: " + w.pred.source + " (" +
                                w.pred.frame + "), " +
                                std::to_string(w.pred.frames.size()) +
                                " frames cached";
            } else {
                w.pred.status = job->result.status;
            }
        }
    }
    if (!w.pred_job && !w.pred_queue.empty()) {
        const size_t n = std::min<size_t>(w.pred_queue.size(), 400);
        auto job = std::make_shared<ProofreadPredJob>();
        job->session = pm.proofread_session;
        job->frames.assign(w.pred_queue.begin(), w.pred_queue.begin() + n);
        w.pred_queue.erase(w.pred_queue.begin(), w.pred_queue.begin() + n);
        const std::string url = !pm.proofread_server_url.empty()
                                    ? pm.proofread_server_url
                                    : w.server.url;
        std::thread([job, url, animal = pm.proofread_animal,
                     source = preferred_pred_source(ctx)]() {
            job->ok = proofread_fetch_prediction(url, animal, job->session,
                                                 job->frames, source,
                                                 job->result);
            job->done.store(true, std::memory_order_release);
        }).detach();
        w.pred_job = std::move(job);
    }
}

// Prediction for one frame as per-skeleton-node 3D (by keypoint name;
// NaN where the prediction has no such keypoint).
inline std::vector<Eigen::Vector3d>
pred_points(const ProofreadWindowState &w, const AppContext &ctx,
            const std::vector<std::array<float, 4>> &kps) {
    auto lower = [](std::string v) {
        for (auto &ch : v) ch = (char)std::tolower((unsigned char)ch);
        return v;
    };
    const int nn = ctx.skeleton.num_nodes;
    std::vector<Eigen::Vector3d> pts(nn, Eigen::Vector3d::Constant(NAN));
    for (int n = 0; n < nn && n < (int)ctx.skeleton.node_names.size(); ++n) {
        const std::string want = lower(ctx.skeleton.node_names[n]);
        for (size_t k = 0; k < w.pred.keypoints.size() && k < kps.size(); ++k) {
            if (lower(w.pred.keypoints[k]) != want) continue;
            pts[n] = Eigen::Vector3d(kps[k][0], kps[k][1], kps[k][2]);
            break;
        }
    }
    return pts;
}

// (Re)request the prediction for every frame in the loaded session's queue.
inline void fetch_queue_predictions(ProofreadWindowState &w, AppContext &ctx) {
    const auto &pm = ctx.pm;
    w.pred = ProofreadPrediction{};
    w.pred_asked.clear();
    w.pred_queue.clear();
    w.pred_job.reset();          // abandon; the worker still finishes safely
    w.last_overlay_frame = -1;
    const ProofreadSession *ps =
        find_session(w.server, pm.proofread_animal, pm.proofread_session);
    if (ps) pred_request(w, ps->frames);
    pred_poll(w, ctx);
}

}  // namespace proofread_window_detail

// Overlay the tailcycle prediction for `frame` on every non-excluded camera.
// Without `force`, a frame that already has labels is left alone (never
// clobber the user's corrections). A frame not downloaded yet is requested
// in the background (never blocks the UI); true only if a pose was placed.
inline bool proofread_apply_prediction(ProofreadWindowState &w,
                                       AppContext &ctx, int frame,
                                       bool force) {
    auto &pm = ctx.pm;
    if (pm.camera_params.empty() || !ctx.scene || !ctx.skeleton.has_skeleton)
        return false;
    auto it = ctx.annotations.find((u32)frame);
    if (!force && it != ctx.annotations.end() &&
        frame_has_any_labels(it->second)) {
        w.pred_note = "Frame " + std::to_string(frame) +
                      " already labeled - prediction not loaded";
        return false;
    }
    auto pf = w.pred.frames.find(frame);
    if (pf == w.pred.frames.end()) {
        if (proofread_window_detail::pred_missing(w, frame)) {
            w.pred_note = "No prediction for frame " + std::to_string(frame);
            return false;
        }
        // Not downloaded yet: request a window around it (so stepping
        // through neighbours is instant) and apply when it arrives.
        std::vector<int> win;
        for (int f = std::max(0, frame - 30); f <= frame + 30; ++f)
            win.push_back(f);
        proofread_window_detail::pred_request(w, win);
        if (force) w.force_frame = frame;
        w.pred_note = "Loading prediction for frame " + std::to_string(frame) +
                      "...";
        return false;
    }

    const int nn = ctx.skeleton.num_nodes;
    auto &fa = get_or_create_frame(ctx.annotations, (u32)frame, nn,
                                   (int)ctx.scene->num_cams);
    // Excluded cameras stay empty.
    int placed = apply_predicted_pose(
        fa, &ctx.skeleton, pm.camera_params, ctx.scene,
        proofread_window_detail::pred_points(w, ctx, pf->second),
        excluded_camera_mask(pm));
    if (placed > 0 && !force)   // unreviewed until dragged / Triangulated
        pm.overlay_snapshots[(u32)frame] = snapshot_2d(fa);
    else
        pm.overlay_snapshots.erase((u32)frame);
    w.pred_note = "Frame " + std::to_string(frame) + ": " + w.pred.source +
                  " prediction on " + std::to_string(placed) + "/" +
                  std::to_string(nn) + " keypoints";
    return placed > 0;
}


// Exclude (use=false) or re-include a loaded camera. Excluding clears its
// keypoints on every frame; re-including puts the prediction back on frames
// the user hasn't touched (reviewed frames get it on the next T). Untouched
// prediction frames stay untouched, so they still aren't saved.
inline void proofread_set_camera_used(ProofreadWindowState &w,
                                      AppContext &ctx, int cam_idx, bool use) {
    auto &pm = ctx.pm;
    if (cam_idx < 0 || cam_idx >= (int)pm.camera_names.size()) return;
    const std::string cam = pm.camera_names[cam_idx];
    const auto untouched = untouched_overlay_frames(pm, ctx.annotations);

    auto &ex = pm.excluded_cameras;
    ex.erase(std::remove(ex.begin(), ex.end(), cam), ex.end());
    if (!use) ex.push_back(cam);
    proofread_window_detail::save_redproj(pm);
    w.cameras_dirty = true;

    const auto mask = excluded_camera_mask(pm);
    for (auto &[f, fa] : ctx.annotations) {
        if (cam_idx >= (int)fa.cameras.size()) continue;
        if (!use) {
            for (auto &kp : fa.cameras[cam_idx].keypoints) kp = Keypoint2D{};
        } else if (untouched.count(f)) {
            auto pf = w.pred.frames.find((int)f);
            if (pf != w.pred.frames.end())
                apply_predicted_pose(
                    fa, &ctx.skeleton, pm.camera_params, ctx.scene,
                    proofread_window_detail::pred_points(w, ctx, pf->second),
                    mask);
        }
        if (untouched.count(f)) pm.overlay_snapshots[f] = snapshot_2d(fa);
    }
}


// Draw the proofread bad-frame panel. Scopes to (pm.proofread_animal,
// pm.proofread_session) — i.e. the session loaded by the current project.
inline void DrawProofreadWindow(ProofreadWindowState &w, AppContext &ctx) {
    const auto &pm = ctx.pm;

    // If the loaded project is not a proofread project, this panel is a no-op
    // (still drawable if the user opens it via the menu, but with a hint).
    const bool is_proofread = !pm.proofread_animal.empty() &&
                              !pm.proofread_session.empty();

    if (!w.show) return;

    ImGui::SetNextWindowSize(ImVec2(420, 540), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Proofread Queue", &w.show)) {
        ImGui::End();
        return;
    }

    if (!is_proofread) {
        ImGui::TextDisabled(
            "No proofread project loaded.\n"
            "Use Proofread → Create Proofread Project (or Load) first.");
        ImGui::End();
        return;
    }

    // Bind the server URL from the project on first draw, then auto-fetch.
    if (!w.initial_fetch_done) {
        if (!pm.proofread_server_url.empty())
            w.server.url = pm.proofread_server_url;
        proofread_fetch(w.server);
        proofread_window_detail::fetch_queue_predictions(w, ctx);
        w.initial_fetch_done = true;
    }

    // ── Top bar ───────────────────────────────────────────────────────
    ImGui::TextDisabled("server:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-180.0f);
    ImGui::InputText("##proof_panel_url", &w.server.url);
    ImGui::SameLine();
    if (ImGui::Button("Refresh##proof_panel")) {
        proofread_fetch(w.server);
        proofread_window_detail::fetch_queue_predictions(w, ctx);
    }

    // ── Source selector: IK residual vs Scorer ────────────────────────
    // Both are offered because scorer coverage is still partial — a session
    // with no scorer.parquet simply won't appear when Scorer is selected.
    {
        int src = (w.server.source == ProofreadState::Source::Scorer) ? 1 : 0;
        ImGui::TextDisabled("bad by:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(150.0f);
        if (ImGui::Combo("##proof_src", &src, "IK residual\0Scorer\0")) {
            w.server.source = src == 1 ? ProofreadState::Source::Scorer
                                        : ProofreadState::Source::Residual;
            proofread_fetch(w.server);   // re-pull from the other endpoint
            proofread_window_detail::fetch_queue_predictions(w, ctx);
        }
    }

    const bool scorer_src = (w.server.source == ProofreadState::Source::Scorer);
    if (scorer_src) {
        ImGui::SetNextItemWidth(130.0f);
        ImGui::InputFloat("score < ##proof_panel",
                           &w.server.scorer_threshold, 0.05f, 0.25f, "%.2f");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(90.0f);
        ImGui::InputInt("min bad kps##proof_panel", &w.server.min_bad_kps, 1, 1);
        if (w.server.min_bad_kps < 1) w.server.min_bad_kps = 1;
    } else {
        ImGui::SetNextItemWidth(130.0f);
        ImGui::InputFloat("residual ≥ mm##proof_panel",
                           &w.server.residual_threshold_mm, 1.0f, 5.0f, "%.1f");
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100.0f);
    ImGui::InputInt("min gap##proof_panel",
                     &w.server.min_gap, 1, 10);
    if (w.server.min_gap < 0) w.server.min_gap = 0;
    ImGui::SameLine();
    if (ImGui::SmallButton("Apply##proof_panel")) {
        proofread_fetch(w.server);
        proofread_window_detail::fetch_queue_predictions(w, ctx);
    }

    if (!w.server.status.empty()) {
        bool bad =
            w.server.status.find("Cannot") != std::string::npos ||
            w.server.status.find("failed") != std::string::npos ||
            w.server.status.find("error")  != std::string::npos;
        ImVec4 col = bad ? ImVec4(1.0f, 0.45f, 0.45f, 1.0f)
                          : ImVec4(0.6f, 0.95f, 0.6f, 1.0f);
        ImGui::TextColored(col, "%s", w.server.status.c_str());
    }
    ImGui::Separator();

    // ── Cameras: calibration check + exclusion ───────────────────────
    if (ImGui::CollapsingHeader("Cameras", ImGuiTreeNodeFlags_DefaultOpen)) {
        proofread_window_detail::draw_cameras_section(w, ctx);
        ImGui::Separator();
    }

    // ── Tailcycle prediction overlay ─────────────────────────────────
    if (ImGui::Checkbox("Overlay tailcycle prediction", &w.auto_pred))
        w.last_overlay_frame = -1;   // apply to the current frame now
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Every unlabeled frame you land on (while paused) "
                          "starts from the pipeline's 3D prediction on every "
                          "camera:\ndrag the wrong points and press T instead "
                          "of labelling from scratch.\nFrames with labels are "
                          "never overwritten. A prediction you haven't "
                          "touched\n(dragged or Triangulated) is not saved "
                          "or exported.");
    ImGui::SameLine();
    if (ImGui::SmallButton("Load on this frame"))
        proofread_apply_prediction(w, ctx, ctx.current_frame_num, true);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Replace this frame's keypoints with the prediction");
    if (!w.pred.status.empty())
        ImGui::TextDisabled("%s", w.pred.status.c_str());
    if (!w.pred_note.empty())
        ImGui::TextDisabled("%s", w.pred_note.c_str());
    ImGui::Separator();

    // ── Per-session header ────────────────────────────────────────────
    ImGui::Text("%s   %s",
                 pm.proofread_animal.c_str(), pm.proofread_session.c_str());
    const ProofreadSession *ps = proofread_window_detail::find_session(
        w.server, pm.proofread_animal, pm.proofread_session);
    if (!ps) {
        ImGui::TextDisabled("(session not in current server response — "
                             "press Refresh)");
        ImGui::End();
        return;
    }
    if (scorer_src) {
        ImGui::Text("bad frames: %d / %d  (core kp score < %.2f)",
                     ps->n_frames_bad, ps->n_frames_total,
                     w.server.scorer_threshold);
    } else {
        ImGui::Text("bad frames: %d / %d  (residual >= %.1f mm)",
                     ps->n_frames_bad, ps->n_frames_total,
                     w.server.residual_threshold_mm);
    }
    ImGui::Separator();

    // ── Frame table ───────────────────────────────────────────────────
    if (ImGui::BeginTable("##frames", 4,
                           ImGuiTableFlags_RowBg |
                           ImGuiTableFlags_BordersInnerH |
                           ImGuiTableFlags_ScrollY |
                           ImGuiTableFlags_SizingFixedFit,
                           ImVec2(0.0f, 0.0f))) {
        ImGui::TableSetupColumn("Frame", ImGuiTableColumnFlags_WidthFixed, 90.0f);
        ImGui::TableSetupColumn(scorer_src ? "Score" : "Residual (mm)",
                                 ImGuiTableColumnFlags_WidthFixed, 110.0f);
        ImGui::TableSetupColumn("Worst kp", ImGuiTableColumnFlags_WidthFixed, 90.0f);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableHeadersRow();

        for (int fi = 0; fi < (int)ps->frames.size(); ++fi) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%d", ps->frames[fi]);
            ImGui::TableNextColumn();
            if (scorer_src) {
                float sc = fi < (int)ps->scores.size() ? ps->scores[fi] : 0.0f;
                ImGui::Text("%.2f", sc);
            } else {
                float r = fi < (int)ps->residuals_mm.size()
                            ? ps->residuals_mm[fi] : 0.0f;
                ImGui::Text("%.1f", r);
            }
            ImGui::TableNextColumn();
            if (scorer_src && fi < (int)ps->worst_kps.size()) {
                ImGui::TextUnformatted(ps->worst_kps[fi].c_str());
            } else {
                ImGui::TextDisabled("-");
            }
            ImGui::TableNextColumn();
            ImGui::PushID(fi);
            if (ImGui::SmallButton("Seek")) {
                w.open_requested = true;
                w.requested_animal = pm.proofread_animal;
                w.requested_session = pm.proofread_session;
                w.requested_recording_path = ps->recording_path;
                w.requested_frame = ps->frames[fi];
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    ImGui::End();
}

// Called every UI frame: overlay the prediction on whatever frame is shown
// (including the first frame after the project opens), once per frame
// change, while paused. Frames that already have labels are left alone.
inline void proofread_auto_overlay(ProofreadWindowState &w, AppContext &ctx,
                                   bool playing) {
    const auto &pm = ctx.pm;
    if (pm.proofread_animal.empty() || pm.proofread_session.empty()) return;
    proofread_window_detail::pred_poll(w, ctx);

    if (w.force_frame >= 0 &&
        (w.pred.frames.count(w.force_frame) ||
         proofread_window_detail::pred_missing(w, w.force_frame))) {
        proofread_apply_prediction(w, ctx, w.force_frame, true);
        w.force_frame = -1;
    }

    if (!w.auto_pred || playing || !pm.plot_keypoints_flag) return;
    const int f = ctx.current_frame_num;
    if (f == w.last_overlay_frame) return;
    auto it = ctx.annotations.find((u32)f);
    if (it != ctx.annotations.end() && frame_has_any_labels(it->second)) {
        w.last_overlay_frame = f;
        return;
    }
    // Done with this frame once placed or known missing; otherwise the
    // download is in flight and we retry next UI frame.
    if (proofread_apply_prediction(w, ctx, f, false) ||
        proofread_window_detail::pred_missing(w, f))
        w.last_overlay_frame = f;
}
