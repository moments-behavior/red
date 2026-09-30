#pragma once
// tracktail_window.h — tracktail forward temporal tracker panel.
//
// Seeds from the CURRENT frame's annotations of the active animal (its
// triangulated 3D keypoints; 2D labels are triangulated first if needed),
// predicts the next N frames with tracktail, and writes the predicted 3D +
// reprojected 2D into the annotation buffer for frames [current+1 .. current+N].
//
// Inference runs on the tracktail HTTP server (server/server.py in
// github.com/AI-HHMI/tracktail): one chunk of the model's n_frames per click, no GPU needed
// locally.
//
// This header is UI only. The request flags below are consumed by
// tracktail_handle_requests() in tracktail_actions.h from the main loop, which
// is where the HTTP state lives. Keep it that way so this file stays
// free of CUDA / httplib includes (it is pulled into test_gui via
// window_states.h).

#include "app_context.h"
#include "gui/panel.h"
#include "imgui.h"
#include <misc/cpp/imgui_stdlib.h>
#include <cstdio>
#include <string>

struct TracktailWindowState {
    bool show = false;

    // ── Seed ──
    // If the current frame has 2D labels but no triangulated 3D, run the
    // standard Triangulate (T) first so the seed is available. Off = require
    // the user to press T themselves.
    bool auto_triangulate = true;

    // ── Server (HTTP) ──
    std::string server_url = "http://10.102.10.88:8000";
    bool server_probe_requested = false;
    std::string server_status;
    // How many future frames to write back. One chunk of n_frames holds
    // n_frames-1 of them; a larger value is clamped to that at Forward.
    int server_n_keep = 4;
    // Cached /info reply for display.
    int server_n_frames = 0;
    int server_image_size = 0;
    std::string server_device;
    std::string server_mode_3d;
    // Timings of the last /predict call.
    float server_last_total_ms = 0.0f;
    float server_last_request_ms = 0.0f;
    float server_last_encode_ms = 0.0f;
    float server_last_decode_ms = 0.0f;

    // ── Run ──
    bool forward_requested = false;
    std::string last_result;  // one-line summary of the last Forward
    bool last_result_ok = true;
};

namespace tracktail_ui_detail {

inline bool looks_like_error(const std::string &s) {
    return s.find("WARNING") != std::string::npos ||
           s.find("Cannot") != std::string::npos ||
           s.find("failed") != std::string::npos ||
           s.find("FAILED") != std::string::npos ||
           s.find("error") != std::string::npos;
}

}  // namespace tracktail_ui_detail

inline void DrawTracktailWindow(TracktailWindowState &st, AppContext &ctx) {
    DrawPanel("tracktail", st.show, [&]() {
        auto &pm = ctx.pm;
        auto *scene = ctx.scene;
        const bool is_2d = project_is_2d(pm);
        const bool videos_loaded = scene && scene->num_cams > 0 &&
                                   ctx.ps.video_loaded;

        // ── Seed status: what the current frame gives us ──
        ImGui::SeparatorText("Seed (current frame, active animal)");
        int n_3d = 0, n_2d = 0, n_inst = 0, inst_id = -1;
        auto it = ctx.annotations.find((u32)ctx.current_frame_num);
        if (it != ctx.annotations.end() && !it->second.empty()) {
            n_inst = (int)it->second.size();
            const FrameAnnotation &fa =
                instance_or_first(it->second, ctx.active_instance);
            inst_id = fa.instance_id;
            for (int k = 0; k < (int)fa.kp3d.size() &&
                            k < ctx.skeleton.num_nodes; ++k)
                if (fa.kp3d[k].triangulated) n_3d++;
            for (const auto &cam : fa.cameras)
                for (int k = 0; k < (int)cam.keypoints.size() &&
                                k < ctx.skeleton.num_nodes; ++k)
                    if (cam.keypoints[k].labeled) { n_2d++; break; }
        }
        ImGui::Text("Frame %d   animal %d/%d (id %d)", ctx.current_frame_num,
                    n_inst ? ctx.active_instance + 1 : 0, n_inst, inst_id);
        ImGui::Text("3D keypoints: %d / %d   cameras with 2D labels: %d",
                    n_3d, ctx.skeleton.num_nodes, n_2d);
        if (is_2d) {
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f),
                               "No calibration loaded: tracktail needs a 3D "
                               "project.");
        } else if (!videos_loaded) {
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f),
                               "No videos loaded.");
        } else if (n_3d == 0 && n_2d == 0) {
            ImGui::TextDisabled("Label the current frame first (2D in >= 2 "
                                "cameras, or press T after labeling).");
        } else if (n_3d == 0) {
            ImGui::TextDisabled("No triangulated 3D yet%s.",
                                st.auto_triangulate
                                    ? " -- Forward will triangulate first"
                                    : " -- press T first");
        }
        ImGui::Checkbox("Triangulate 2D labels first if there is no 3D",
                        &st.auto_triangulate);

        // ── Server ──
        ImGui::SeparatorText("Server");
        ImGui::Text("URL");
        ImGui::SetNextItemWidth(-160);
        ImGui::InputText("##tracktail_server_url", &st.server_url);
        ImGui::SameLine();
        if (ImGui::Button("Probe##tracktail_server"))
            st.server_probe_requested = true;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "GET /info on the configured URL to check the server is\n"
                "reachable and read the model's frames per chunk and\n"
                "image size. Forward probes by itself if needed.");

        if (st.server_n_frames > 0) {
            ImGui::TextDisabled(
                "Model: n_frames=%d image_size=%d  device=%s  mode=%s",
                st.server_n_frames, st.server_image_size,
                st.server_device.c_str(), st.server_mode_3d.c_str());
        }
        if (!st.server_status.empty()) {
            ImVec4 col = tracktail_ui_detail::looks_like_error(
                             st.server_status)
                             ? ImVec4(1.0f, 0.6f, 0.3f, 1.0f)
                             : ImVec4(0.5f, 1.0f, 0.5f, 1.0f);
            ImGui::TextColored(col, "%s", st.server_status.c_str());
        }
        if (st.server_last_total_ms > 0.0f) {
            ImGui::TextDisabled(
                "Last: total=%.0f ms  (encode=%.0f, request=%.0f, "
                "decode=%.0f)",
                st.server_last_total_ms, st.server_last_encode_ms,
                st.server_last_request_ms, st.server_last_decode_ms);
        }
        ImGui::SetNextItemWidth(160);
        ImGui::SliderInt("N future frames to keep", &st.server_n_keep, 1,
                         24);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "The server returns one chunk of n_frames per click: t=0 is\n"
                "the seed (current frame), the rest are future predictions.\n"
                "This slider picks how many to write into annotations for\n"
                "frames [current+1 .. current+N], up to n_frames-1.");

        // ── Run ──
        ImGui::SeparatorText("Run");
        char fwd_label[64];
        std::snprintf(fwd_label, sizeof(fwd_label), "tracktail Forward +%d",
                      st.server_n_keep);
        const bool can_run = !is_2d && videos_loaded && (n_3d > 0 ||
                             (st.auto_triangulate && n_2d >= 2));
        ImGui::BeginDisabled(!can_run);
        if (ImGui::Button(fwd_label)) st.forward_requested = true;
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip(
                "Seed from the current frame's 3D keypoints of the active\n"
                "animal, send all cameras x n_frames to the server, and\n"
                "write the first N future-frame predictions (3D +\n"
                "reprojected 2D) into annotations for that animal.\n"
                "Requires all cameras to have frames in the display buffer.");
        if (!st.last_result.empty()) {
            ImVec4 col = st.last_result_ok ? ImVec4(0.5f, 1.0f, 0.5f, 1.0f)
                                           : ImVec4(1.0f, 0.6f, 0.3f, 1.0f);
            ImGui::TextColored(col, "%s", st.last_result.c_str());
        }
        ImGui::TextDisabled("Predicted labels are marked Predicted; edit them "
                            "in the Labeling Tool like any other label.");
    }, nullptr, ImVec2(560, 420));
}
