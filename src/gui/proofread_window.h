#pragma once
// proofread_window.h — bad-frames navigator panel that appears after a
// Proofread project is loaded. Reads bad-frame indices from the
// dashboard's /api/bad_frames_all endpoint, scoped to (pm.proofread_animal,
// pm.proofread_session), and lets the user click any frame to seek.
//
// Action signal:
//   - open_requested + requested_frame
//     The main loop reads these and issues an accurate seek_all_cameras
//     to the requested video frame.

#include "imgui.h"
#include "proofread_client.h"
#include "app_context.h"

#include <misc/cpp/imgui_stdlib.h>

#include <algorithm>
#include <cstdio>
#include <string>


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
};


namespace proofread_window_detail {

inline const ProofreadSession *find_session(const ProofreadState &s,
                                             const std::string &animal,
                                             const std::string &session) {
    for (const auto &ps : s.sessions) {
        if (ps.animal == animal && ps.session == session) return &ps;
    }
    return nullptr;
}

}  // namespace proofread_window_detail


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
