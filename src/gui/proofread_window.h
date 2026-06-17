#pragma once
// proofread_window.h — ImGui panel for the JARVIS pose-proofreading workflow.
//
// Talks to the mouse_dashboard server via proofread_client.h:
//   GET /api/bad_frames_all?residual_threshold_mm=...&min_gap=...
//
// Wire it from red.cpp's main loop:
//   DrawProofreadWindow(win.proofread);
//
// Action signals (caller is responsible for handling them):
//   - open_requested + requested_recording_path + requested_frame
//     → the user picked a row and wants the viewer to jump there.
//   - refresh_requested
//     → the user pressed the Refresh button. The window triggers the
//       fetch itself; this flag is left for the caller to do extra
//       housekeeping (e.g. close a stale session if its frames vanished).

#include "imgui.h"
#include <misc/cpp/imgui_stdlib.h>   // ImGui::InputText with std::string
#include "proofread_client.h"

#include <algorithm>
#include <cstdio>
#include <string>


struct ProofreadWindowState {
    bool show = false;
    ProofreadState server;   // url, threshold, sessions, status, ...

    // Which session row is expanded (index into server.sessions, or -1).
    int expanded_session = -1;

    // The user clicked an "Open" button — caller handles it.
    bool        open_requested = false;
    std::string requested_animal;
    std::string requested_session;
    std::string requested_recording_path;  // /mnt/free/<animal>/<session>
    int         requested_frame = 0;

    // Deferred seek: set after a successful load_videos() so the next
    // frame can issue the seek once decoders are ready. -1 = no pending.
    int pending_seek_frame = -1;

    // The user asked for a fresh fetch (informational; the window already
    // ran the fetch by the time the caller sees this flag).
    bool refresh_requested = false;
};


namespace proofread_window_detail {

// Compact summary line shown under the URL bar.
inline void summary_line(const ProofreadState &s) {
    if (!s.reachable && s.sessions.empty()) {
        ImGui::TextDisabled("Not connected. Set URL and press Refresh.");
        return;
    }
    ImGui::Text("%d sessions · %d bad frames · last fetch %.0f ms",
                static_cast<int>(s.sessions.size()),
                s.n_bad_total, s.last_ms);
    if (!s.generated_at.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled(" · server time %s", s.generated_at.c_str());
    }
}

// Color a status string the same way the posetail panel does.
inline void status_line(const std::string &status) {
    if (status.empty()) return;
    bool bad =
        status.find("Cannot") != std::string::npos ||
        status.find("failed") != std::string::npos ||
        status.find("error")  != std::string::npos ||
        status.find("Bad")    != std::string::npos;
    ImVec4 col = bad ? ImVec4(1.0f, 0.45f, 0.45f, 1.0f)
                     : ImVec4(0.6f, 0.95f, 0.6f, 1.0f);
    ImGui::TextColored(col, "%s", status.c_str());
}

// Sum bad frames across consecutive sessions of one animal.
inline int animal_bad_total(const ProofreadState &s, const std::string &animal) {
    int n = 0;
    for (const auto &ps : s.sessions) if (ps.animal == animal) n += ps.n_frames_bad;
    return n;
}

// Distinct animals, in the order they appear in s.sessions.
inline std::vector<std::string> distinct_animals(const ProofreadState &s) {
    std::vector<std::string> out;
    for (const auto &ps : s.sessions) {
        if (std::find(out.begin(), out.end(), ps.animal) == out.end()) {
            out.push_back(ps.animal);
        }
    }
    return out;
}

}  // namespace proofread_window_detail


inline void DrawProofreadWindow(ProofreadWindowState &w) {
    if (!w.show) return;

    ImGui::SetNextWindowSize(ImVec2(620, 540), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Proofread Queue", &w.show)) {
        ImGui::End();
        return;
    }

    // ─── Server config ─────────────────────────────────────────────────
    ImGui::TextUnformatted("Dashboard server");
    ImGui::SameLine();
    ImGui::TextDisabled("(http://host:port)");
    ImGui::SetNextItemWidth(-1);
    ImGui::InputText("##proofread_url", &w.server.url);

    ImGui::SetNextItemWidth(140);
    ImGui::InputFloat("residual ≥ (mm)", &w.server.residual_threshold_mm,
                       1.0f, 5.0f, "%.1f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    ImGui::InputInt("min gap (fr)", &w.server.min_gap, 1, 10);
    if (w.server.min_gap < 0) w.server.min_gap = 0;
    ImGui::SameLine();
    if (ImGui::Button("Refresh")) {
        proofread_fetch(w.server);
        w.refresh_requested = true;
        w.expanded_session = -1;
    }

    proofread_window_detail::status_line(w.server.status);
    proofread_window_detail::summary_line(w.server);

    ImGui::Separator();

    // ─── Session list ──────────────────────────────────────────────────
    if (w.server.sessions.empty()) {
        ImGui::TextDisabled(
            "No sessions to proofread yet — press Refresh once the dashboard "
            "is reachable.");
        ImGui::End();
        return;
    }

    const auto animals = proofread_window_detail::distinct_animals(w.server);

    if (ImGui::BeginChild("##proofread_scroll", ImVec2(0, 0), false,
                           ImGuiWindowFlags_HorizontalScrollbar)) {
        for (const auto &animal : animals) {
            int animal_total = proofread_window_detail::animal_bad_total(
                w.server, animal);
            char header[160];
            std::snprintf(header, sizeof(header),
                          "%s   (%d bad frames)###proofread_%s",
                          animal.c_str(), animal_total, animal.c_str());

            if (!ImGui::CollapsingHeader(header,
                                          ImGuiTreeNodeFlags_DefaultOpen)) {
                continue;
            }

            // Per-session rows for this animal.
            for (int si = 0; si < static_cast<int>(w.server.sessions.size()); ++si) {
                const auto &ps = w.server.sessions[si];
                if (ps.animal != animal) continue;

                bool expanded = (w.expanded_session == si);
                ImGui::PushID(si);

                // Header row: session id, bad count, expand/open button.
                if (ImGui::ArrowButton("##expand",
                                        expanded ? ImGuiDir_Down : ImGuiDir_Right)) {
                    w.expanded_session = expanded ? -1 : si;
                }
                ImGui::SameLine();
                ImGui::Text("%s", ps.session.c_str());
                ImGui::SameLine();
                ImGui::TextDisabled(" · %d / %d",
                                     ps.n_frames_bad, ps.n_frames_total);
                ImGui::SameLine(ImGui::GetWindowWidth() - 180.0f);
                if (ImGui::SmallButton("Open first")) {
                    w.open_requested = true;
                    w.requested_animal = ps.animal;
                    w.requested_session = ps.session;
                    w.requested_recording_path = ps.recording_path;
                    w.requested_frame = ps.frames.empty() ? 0 : ps.frames.front();
                }

                if (expanded) {
                    if (ImGui::BeginTable("##frames", 3,
                                           ImGuiTableFlags_RowBg |
                                           ImGuiTableFlags_BordersInnerH |
                                           ImGuiTableFlags_ScrollY |
                                           ImGuiTableFlags_SizingFixedFit,
                                           ImVec2(0.0f, 220.0f))) {
                        ImGui::TableSetupColumn("Frame", ImGuiTableColumnFlags_WidthFixed, 90.0f);
                        ImGui::TableSetupColumn("Residual (mm)", ImGuiTableColumnFlags_WidthFixed, 130.0f);
                        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);
                        ImGui::TableHeadersRow();

                        for (int fi = 0; fi < static_cast<int>(ps.frames.size()); ++fi) {
                            ImGui::TableNextRow();
                            ImGui::TableNextColumn();
                            ImGui::Text("%d", ps.frames[fi]);
                            ImGui::TableNextColumn();
                            float r = fi < static_cast<int>(ps.residuals_mm.size())
                                        ? ps.residuals_mm[fi] : 0.0f;
                            ImGui::Text("%.1f", r);
                            ImGui::TableNextColumn();
                            ImGui::PushID(fi);
                            if (ImGui::SmallButton("Open")) {
                                w.open_requested = true;
                                w.requested_animal = ps.animal;
                                w.requested_session = ps.session;
                                w.requested_recording_path = ps.recording_path;
                                w.requested_frame = ps.frames[fi];
                            }
                            ImGui::PopID();
                        }
                        ImGui::EndTable();
                    }
                }

                ImGui::PopID();
                ImGui::Separator();
            }
        }
    }
    ImGui::EndChild();

    ImGui::End();
}
