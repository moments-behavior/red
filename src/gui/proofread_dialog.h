#pragma once
// proofread_dialog.h — "Create Proofread Project" form.
//
// Mirrors gui/annotation_dialog.h structurally so it feels familiar, but
// the inputs are server-driven instead of user-picked:
//   - Server URL  (the JARVIS dashboard, http://host:port)
//   - Animal / Session  (populated from /api/bad_frames_all)
//   - media_folder    auto-derived = /mnt/free/<animal>/<session>
//   - calibration_folder  auto-fetched from /api/session_calib_zip into
//                          ~/.cache/red/proofread/<date>/
// The user still fills in project name, root path, and a skeleton.

#include "imgui.h"
#include "app_context.h"
#include "proofread_client.h"
#include "gui/panel.h"

#include <ImGuiFileDialog.h>
#include <misc/cpp/imgui_stdlib.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <set>
#include <string>
#include <vector>

namespace proofread_dialog_detail {

inline std::vector<std::string> distinct_animals(const ProofreadState &s) {
    std::vector<std::string> out;
    for (const auto &ps : s.sessions) {
        if (std::find(out.begin(), out.end(), ps.animal) == out.end()) {
            out.push_back(ps.animal);
        }
    }
    return out;
}

inline std::vector<std::string> sessions_for_animal(const ProofreadState &s,
                                                     const std::string &animal) {
    std::vector<std::string> out;
    for (const auto &ps : s.sessions) {
        if (ps.animal == animal) out.push_back(ps.session);
    }
    return out;
}

// "<session>_proofread" — keyed on the session (a unique timestamp) with a
// _proofread suffix so an exported/uploaded proofread dataset never collides
// with the original session's data. Animal is unused (kept for call sites).
inline std::string default_project_name(const std::string &animal,
                                         const std::string &session) {
    (void)animal;
    if (session.empty()) return {};
    return session + "_proofread";
}

}  // namespace proofread_dialog_detail


struct ProofreadDialogState {
    bool show = false;
    ProofreadState server;  // URL, threshold, fetched sessions
    std::string selected_animal;
    std::string selected_session;
    std::string status;
    // Cache for the calib dir we mirror to (filled at Create time so the
    // caller can verify it).
    std::string calib_cache_dir;
    // Set by the menu / welcome button when the dialog is opened so the
    // first DrawProofreadDialog call silently pulls the animal+session
    // list from the default URL. Cleared after the fetch fires.
    bool pending_fetch = false;
};


using ProofreadCreateCallback =
    std::function<bool(ProjectManager &pm, std::string &error_message)>;


inline void DrawProofreadDialog(ProofreadDialogState &state,
                                 AppContext &ctx,
                                 const ProofreadCreateCallback &on_create) {
    auto &pm = ctx.pm;
    const auto &skeleton_map = ctx.skeleton_map;
    const auto &skeleton_dir = ctx.skeleton_dir;

    // Auto-grab this session's calibration from the server into a per-date
    // cache dir and point pm.calibration_folder at it. Idempotent: if the
    // yamls are already cached we skip the network round-trip. Called both
    // when the user picks a session (so calib is ready before Create) and
    // again at Create time as a safety net.
    auto ensure_calib = [&](const std::string &animal,
                            const std::string &session) -> bool {
        if (animal.empty() || session.empty()) return false;
        const std::string date =
            session.size() >= 10 ? session.substr(0, 10) : session;
        const char *home = std::getenv("HOME");
        std::filesystem::path calib_dir =
            std::filesystem::path(home ? home : "/tmp") /
            ".cache" / "red" / "proofread" / date;
        bool have_yaml = false;
        if (std::filesystem::is_directory(calib_dir)) {
            for (const auto &e :
                 std::filesystem::directory_iterator(calib_dir)) {
                if (e.is_regular_file() && e.path().extension() == ".yaml") {
                    have_yaml = true;
                    break;
                }
            }
        }
        if (!have_yaml) {
            std::string err;
            if (!proofread_fetch_calib(state.server, animal, session,
                                        calib_dir, &err)) {
                state.status = "Could not fetch calibration: " + err;
                return false;
            }
        }
        state.calib_cache_dir = calib_dir.string();
        pm.calibration_folder = calib_dir.string();
        return true;
    };

    // ── File dialogs (must run every frame) ───────────────────────────
    if (ImGuiFileDialog::Instance()->Display(
            "ChooseProofRootDir", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            pm.project_root_path =
                ImGuiFileDialog::Instance()->GetCurrentPath();
        }
        ImGuiFileDialog::Instance()->Close();
    }
    if (ImGuiFileDialog::Instance()->Display(
            "ChooseProofSkeleton", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            pm.skeleton_file = ImGuiFileDialog::Instance()->GetFilePathName();
        }
        ImGuiFileDialog::Instance()->Close();
    }

    if (!state.show) return;

    // Auto-pull animal/session list from the default URL on first show.
    // The menu / welcome buttons set pending_fetch = true so we re-fetch
    // each time the dialog is opened (in case new sessions appeared).
    if (state.pending_fetch) {
        state.pending_fetch = false;
        proofread_fetch(state.server);
        state.status = state.server.status;
    }

    ImGui::SetNextWindowSize(ImVec2(720, 480), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Create Proofread Project", &state.show,
                      ImGuiWindowFlags_NoCollapse)) {
        if (!state.status.empty()) {
            bool bad =
                state.status.find("Cannot") != std::string::npos ||
                state.status.find("failed") != std::string::npos ||
                state.status.find("error")  != std::string::npos ||
                state.status.find("Bad")    != std::string::npos;
            ImVec4 col = bad ? ImVec4(1.0f, 0.45f, 0.45f, 1.0f)
                              : ImVec4(0.6f, 0.95f, 0.6f, 1.0f);
            ImGui::PushStyleColor(ImGuiCol_Text, col);
            ImGui::TextUnformatted(state.status.c_str());
            ImGui::PopStyleColor();
            ImGui::Separator();
        }

        if (ImGui::SmallButton("< Back")) {
            state.show = false;
            ImGui::End();
            return;
        }
        ImGui::Spacing();

        std::vector<const char *> skel_labels;
        skel_labels.reserve(skeleton_map.size());
        for (auto &kv : skeleton_map) skel_labels.push_back(kv.first.c_str());
        static int skel_idx = 0;
        if (skel_idx >= (int)skel_labels.size()) skel_idx = 0;

        if (ImGui::BeginTable(
                "proofForm", 3,
                ImGuiTableFlags_SizingStretchProp |
                ImGuiTableFlags_PadOuterX |
                ImGuiTableFlags_RowBg |
                ImGuiTableFlags_BordersInnerV)) {
            ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 160.0f);
            ImGui::TableSetupColumn("Field", ImGuiTableColumnFlags_WidthStretch, 1.0f);
            ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed, 110.0f);

            auto LabelCell = [](const char *t) {
                ImGui::TableSetColumnIndex(0);
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted(t);
            };

            // ── Server URL ──
            ImGui::TableNextRow();
            LabelCell("Server URL");
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::InputText("##proof_url", &state.server.url);
            ImGui::TableSetColumnIndex(2);
            if (ImGui::Button("Refresh##proof", ImVec2(-FLT_MIN, 0))) {
                proofread_fetch(state.server);
                state.status = state.server.status;
                // Reset the picked session if it disappeared.
                if (!state.selected_session.empty()) {
                    bool still =
                        std::any_of(state.server.sessions.begin(),
                                    state.server.sessions.end(),
                                    [&](const ProofreadSession &p){
                            return p.animal == state.selected_animal &&
                                   p.session == state.selected_session;
                        });
                    if (!still) state.selected_session.clear();
                }
            }

            // ── Login (HTTP Basic auth) ──
            ImGui::TableNextRow();
            LabelCell("Login (user / pass)");
            ImGui::TableSetColumnIndex(1);
            {
                float gap = ImGui::GetStyle().ItemInnerSpacing.x;
                float half = (ImGui::GetContentRegionAvail().x - gap) * 0.5f;
                ImGui::SetNextItemWidth(half);
                ImGui::InputText("##proof_user", &state.server.username);
                ImGui::SameLine(0, gap);
                ImGui::SetNextItemWidth(half);
                ImGui::InputText("##proof_pass", &state.server.password,
                                 ImGuiInputTextFlags_Password);
            }
            ImGui::TableSetColumnIndex(2);
            ImGui::Dummy(ImVec2(1, 1));

            // ── Threshold + min-gap (compact row) ──
            ImGui::TableNextRow();
            LabelCell("Bad-frame filter");
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(140);
            ImGui::InputFloat("residual ≥ (mm)##proof",
                              &state.server.residual_threshold_mm, 1.0f, 5.0f, "%.1f");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(100);
            ImGui::InputInt("min gap (fr)##proof", &state.server.min_gap, 1, 10);
            if (state.server.min_gap < 0) state.server.min_gap = 0;
            ImGui::TableSetColumnIndex(2);
            ImGui::Dummy(ImVec2(1, 1));

            // ── Animal combo ──
            const auto animals = proofread_dialog_detail::distinct_animals(
                state.server);
            ImGui::TableNextRow();
            LabelCell("Animal");
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-FLT_MIN);
            {
                const char *cur = state.selected_animal.empty()
                                    ? "<pick one>"
                                    : state.selected_animal.c_str();
                if (ImGui::BeginCombo("##proof_animal", cur)) {
                    for (const auto &a : animals) {
                        bool sel = (a == state.selected_animal);
                        if (ImGui::Selectable(a.c_str(), sel)) {
                            state.selected_animal = a;
                            state.selected_session.clear();
                            pm.project_name =
                                proofread_dialog_detail::default_project_name(a, {});
                        }
                        if (sel) ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
            }
            ImGui::TableSetColumnIndex(2);
            ImGui::TextDisabled("%d total", (int)animals.size());

            // ── Session combo ──
            const auto sessions = proofread_dialog_detail::sessions_for_animal(
                state.server, state.selected_animal);
            ImGui::TableNextRow();
            LabelCell("Session");
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-FLT_MIN);
            {
                const char *cur = state.selected_session.empty()
                                    ? "<pick one>"
                                    : state.selected_session.c_str();
                ImGui::BeginDisabled(sessions.empty());
                if (ImGui::BeginCombo("##proof_session", cur)) {
                    for (const auto &s : sessions) {
                        // Pull bad-frame count for the row hint.
                        int nbad = 0;
                        for (const auto &ps : state.server.sessions) {
                            if (ps.animal == state.selected_animal &&
                                ps.session == s) {
                                nbad = ps.n_frames_bad; break;
                            }
                        }
                        char label[256];
                        std::snprintf(label, sizeof(label),
                                       "%s   (%d bad)",
                                       s.c_str(), nbad);
                        bool sel = (s == state.selected_session);
                        if (ImGui::Selectable(label, sel)) {
                            state.selected_session = s;
                            pm.project_name =
                                proofread_dialog_detail::default_project_name(
                                    state.selected_animal, s);
                            // Auto-grab calibration from the server the moment
                            // a session is picked — these yamls are already in
                            // red's format (camera_matrix/rc_ext/tc_ext), so
                            // triangulation works with no conversion.
                            if (ensure_calib(state.selected_animal, s)) {
                                state.status = "Calibration ready → " +
                                               state.calib_cache_dir;
                            }
                        }
                        if (sel) ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
                ImGui::EndDisabled();
            }
            ImGui::TableSetColumnIndex(2);
            ImGui::TextDisabled("%d sess", (int)sessions.size());

            // ── Recording path (derived, read-only) ──
            std::string recording_path;
            if (!state.selected_animal.empty() && !state.selected_session.empty()) {
                recording_path = "/mnt/free/" + state.selected_animal +
                                 "/" + state.selected_session;
            }
            ImGui::TableNextRow();
            LabelCell("Recording");
            ImGui::TableSetColumnIndex(1);
            ImGui::BeginDisabled();
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::InputText("##proof_rec", &recording_path,
                              ImGuiInputTextFlags_ReadOnly);
            ImGui::EndDisabled();
            ImGui::TableSetColumnIndex(2);
            ImGui::Dummy(ImVec2(1, 1));

            // ── Project Name ──
            ImGui::TableNextRow();
            LabelCell("Project Name");
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::InputText("##proof_projname", &pm.project_name);
            ImGui::TableSetColumnIndex(2);
            ImGui::Dummy(ImVec2(1, 1));

            // ── Project Root Path ──
            ImGui::TableNextRow();
            LabelCell("Project Root Path");
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::InputText("##proof_rootpath", &pm.project_root_path);
            ImGui::TableSetColumnIndex(2);
            if (ImGui::Button("Browse##proof_root")) {
                IGFD::FileDialogConfig cfg;
                cfg.countSelectionMax = 1;
                cfg.path = pm.project_root_path;
                cfg.flags = ImGuiFileDialogFlags_Modal;
                ImGuiFileDialog::Instance()->OpenDialog(
                    "ChooseProofRootDir", "Choose Project Root", nullptr, cfg);
            }

            // ── Full Path (computed) ──
            {
                std::filesystem::path p =
                    std::filesystem::path(pm.project_root_path) / pm.project_name;
                pm.project_path = p.string();
            }
            ImGui::TableNextRow();
            LabelCell("Full Path");
            ImGui::TableSetColumnIndex(1);
            ImGui::BeginDisabled();
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::InputText("##proof_fullpath", &pm.project_path);
            ImGui::EndDisabled();
            ImGui::TableSetColumnIndex(2);
            ImGui::Dummy(ImVec2(1, 1));

            // ── Skeleton ──
            int skel_mode = pm.load_skeleton_from_json ? 0 : 1;
            ImGui::TableNextRow();
            LabelCell("Skeleton");
            ImGui::TableSetColumnIndex(1);
            {
                if (pm.load_skeleton_from_json) {
                    float avail = ImGui::GetContentRegionAvail().x;
                    const char *btxt = "Browse##proof_skel";
                    float browse_w = ImGui::CalcTextSize(btxt).x +
                                     ImGui::GetStyle().FramePadding.x * 2.0f;
                    float gap = ImGui::GetStyle().ItemInnerSpacing.x;
                    ImGui::PushID("proof_skelfile");
                    ImGui::SetNextItemWidth(ImMax(50.0f, avail - browse_w - gap));
                    ImGui::InputText("##path", &pm.skeleton_file);
                    ImGui::SameLine(0.0f, gap);
                    if (ImGui::Button(btxt)) {
                        IGFD::FileDialogConfig config;
                        config.countSelectionMax = 1;
                        config.path = skeleton_dir;
                        config.flags = ImGuiFileDialogFlags_Modal;
                        ImGuiFileDialog::Instance()->OpenDialog(
                            "ChooseProofSkeleton", "Choose Skeleton",
                            ".json", config);
                    }
                    ImGui::PopID();
                } else {
                    ImGui::BeginDisabled(skel_labels.empty());
                    ImGui::SetNextItemWidth(-FLT_MIN);
                    ImGui::Combo("##proof_skel_preset", &skel_idx,
                                 skel_labels.data(), (int)skel_labels.size());
                    ImGui::EndDisabled();
                }
            }
            ImGui::TableSetColumnIndex(2);
            ImGui::SetNextItemWidth(90.0f);
            if (ImGui::Combo("##proof_skel_mode", &skel_mode, "File\0Preset\0")) {
                pm.load_skeleton_from_json = (skel_mode == 0);
                if (pm.load_skeleton_from_json) pm.skeleton_name.clear();
            }
            pm.skeleton_name =
                pm.load_skeleton_from_json
                    ? std::string()
                    : (skel_labels.empty()
                           ? std::string()
                           : std::string(skel_labels[skel_idx]));

            // ── Calibration (auto-fetched note) ──
            ImGui::TableNextRow();
            LabelCell("Calibration");
            ImGui::TableSetColumnIndex(1);
            ImGui::TextDisabled(
                "auto-fetched from server on Create  → ~/.cache/red/proofread/<date>/");
            ImGui::TableSetColumnIndex(2);
            ImGui::Dummy(ImVec2(1, 1));

            ImGui::EndTable();
        }

        ImGui::Separator();

        const bool ok = !state.selected_animal.empty() &&
                        !state.selected_session.empty() &&
                        !pm.project_name.empty() &&
                        !pm.project_root_path.empty() &&
                        (!pm.load_skeleton_from_json || !pm.skeleton_file.empty());

        float avail = ImGui::GetContentRegionAvail().x;
        const char *create_label = "Create Project##proof_action";
        float w = ImGui::CalcTextSize(create_label).x +
                  ImGui::GetStyle().FramePadding.x * 2.0f;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (avail - w));

        ImGui::BeginDisabled(!ok);
        if (ImGui::Button(create_label)) {
            state.status.clear();

            // 1) Ensure calib is present (usually already fetched on select).
            //    ensure_calib sets pm.calibration_folder + state.calib_cache_dir.
            if (!ensure_calib(state.selected_animal,
                              state.selected_session)) {
                // ensure_calib already set state.status with the reason.
            } else {
                std::filesystem::path calib_dir = state.calib_cache_dir;

                // 2) Build the camera set from the calibration directory —
                // *only* cameras that have a Cam{ID}.yaml are part of the
                // calibrated set. Untracked cams on disk (e.g. Cam710040)
                // are intentionally skipped: they have no calibration and
                // red would otherwise try to load + fail on them.
                std::set<std::string> calibrated_cams;
                if (std::filesystem::is_directory(calib_dir)) {
                    for (const auto &e :
                         std::filesystem::directory_iterator(calib_dir)) {
                        if (!e.is_regular_file()) continue;
                        if (e.path().extension() != ".yaml") continue;
                        const std::string stem = e.path().stem().string();
                        if (stem.rfind("Cam", 0) == 0) {
                            calibrated_cams.insert(stem);
                        }
                    }
                }

                pm.media_folder = "/mnt/free/" + state.selected_animal + "/" +
                                  state.selected_session;
                pm.camera_names.clear();
                if (std::filesystem::is_directory(pm.media_folder)) {
                    for (const auto &e :
                         std::filesystem::directory_iterator(pm.media_folder)) {
                        if (!e.is_regular_file()) continue;
                        auto ext = e.path().extension().string();
                        if (ext != ".mp4" && ext != ".MP4") continue;
                        const std::string stem = e.path().stem().string();
                        if (calibrated_cams.count(stem) == 0) continue;
                        pm.camera_names.push_back(stem);
                    }
                    std::sort(pm.camera_names.begin(), pm.camera_names.end());
                }
                if (pm.camera_names.empty()) {
                    state.status = "No Cam*.mp4 found under " + pm.media_folder;
                } else {
                    // 3) Stamp proofread metadata so Load can refetch.
                    pm.proofread_server_url = state.server.url;
                    pm.proofread_username   = state.server.username;
                    pm.proofread_animal     = state.selected_animal;
                    pm.proofread_session    = state.selected_session;

                    // 4) Hand off to the host's project setup callback.
                    std::string err;
                    if (!on_create(pm, err)) {
                        state.status = err;
                    } else {
                        state.show = false;
                    }
                }
            }
        }
        ImGui::EndDisabled();
    }
    ImGui::End();
}
