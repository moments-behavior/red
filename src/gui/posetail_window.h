#pragma once
// posetail_window.h — PoseTail forward temporal tracker panel.
//
// Seeds from the CURRENT frame's annotations of the active animal (its
// triangulated 3D keypoints; 2D labels are triangulated first if needed),
// predicts the next N frames with PoseTail, and writes the predicted 3D +
// reprojected 2D into the annotation buffer for frames [current+1 .. current+N].
//
// Two backends:
//   Server (HTTP) — default. One 16-frame chunk per click sent to
//                   posetail/server/server.py; no GPU / ONNX needed locally.
//   Local ONNX    — chains 16-frame chunks to reach +N; needs the ONNX
//                   Runtime bundle at lib/onnxruntime (RED_HAS_ONNXRUNTIME).
//
// This header is UI only. The request flags below are consumed by
// posetail_handle_requests() in posetail_actions.h from the main loop, which
// is where the model / HTTP state lives. Keep it that way so this file stays
// free of ONNX / CUDA / httplib includes (it is pulled into test_gui via
// window_states.h).

#include "app_context.h"
#include "gui/panel.h"
#include "imgui.h"
#include <misc/cpp/imgui_stdlib.h>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

struct PosetailWindowState {
    bool show = false;

    // ── Seed ──
    // If the current frame has 2D labels but no triangulated 3D, run the
    // standard Triangulate (T) first so the seed is available. Off = require
    // the user to press T themselves.
    bool auto_triangulate = true;

    // ── Backend ──
    // false = local ONNX, true = HTTP server (default: the server host has
    // the GPU headroom, and it needs no local bundle).
    bool use_server = true;

    // ── Server (HTTP) ──
    std::string server_url = "http://10.102.10.88:8000";
    bool server_probe_requested = false;
    std::string server_status;
    // How many of the 15 future frames from a 16-frame chunk to write back.
    int server_n_keep = 15;
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

    // ── Local ONNX ──
    std::string onnx_path;
    bool load_requested = false;
    bool use_cpu = false;   // default GPU; toggle if VRAM-constrained
    int gpu_id = 1;         // default GPU 1 (assumes the idle/bigger one)
    int n_forward = 7;      // short horizon: long chains drift
    int max_queries = 24;   // keypoints per ONNX Run (per-alloc cap)
    std::string status;     // load status from posetail_init

    // ── Run ──
    bool forward_requested = false;
    std::string last_result;  // one-line summary of the last Forward
    bool last_result_ok = true;
};

namespace posetail_ui_detail {

// Look for a *tracker*.onnx under the usual checkout locations so the path
// box is pre-filled on first open. Depth-limited so a big $HOME is cheap.
inline std::string find_default_onnx() {
    namespace fs = std::filesystem;
    const char *home = std::getenv("HOME");
    if (!home) return {};
    std::vector<fs::path> candidates = {
        fs::path(home) / "src/posetail-onnx",
        fs::path(home) / "posetail-onnx",
    };
    std::function<std::string(const fs::path &, int)> find_onnx;
    find_onnx = [&](const fs::path &root, int depth) -> std::string {
        std::error_code ec;
        if (depth > 3 || !fs::is_directory(root, ec)) return {};
        for (auto &e : fs::directory_iterator(root, ec)) {
            if (e.is_regular_file(ec) && e.path().extension() == ".onnx" &&
                e.path().filename().string().find("tracker") !=
                    std::string::npos)
                return e.path().string();
        }
        for (auto &e : fs::directory_iterator(root, ec)) {
            if (!e.is_directory(ec)) continue;
            auto hit = find_onnx(e.path(), depth + 1);
            if (!hit.empty()) return hit;
        }
        return {};
    };
    for (auto &c : candidates) {
        auto hit = find_onnx(c, 0);
        if (!hit.empty()) return hit;
    }
    return {};
}

inline bool looks_like_error(const std::string &s) {
    return s.find("WARNING") != std::string::npos ||
           s.find("Cannot") != std::string::npos ||
           s.find("failed") != std::string::npos ||
           s.find("FAILED") != std::string::npos ||
           s.find("error") != std::string::npos;
}

}  // namespace posetail_ui_detail

inline void DrawPosetailWindow(PosetailWindowState &st, AppContext &ctx) {
    DrawPanel("PoseTail Tracker", st.show, [&]() {
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
                               "No calibration loaded: PoseTail needs a 3D "
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

        // ── Backend ──
        ImGui::SeparatorText("Backend");
        if (ImGui::RadioButton("Local ONNX", !st.use_server))
            st.use_server = false;
        ImGui::SameLine();
        if (ImGui::RadioButton("Server (HTTP)", st.use_server))
            st.use_server = true;

        if (st.use_server) {
            ImGui::Text("URL");
            ImGui::SetNextItemWidth(-160);
            ImGui::InputText("##posetail_server_url", &st.server_url);
            ImGui::SameLine();
            if (ImGui::Button("Probe##posetail_server"))
                st.server_probe_requested = true;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "GET /info on the configured URL to check the server is\n"
                    "reachable and the model matches red's expected\n"
                    "16 frames x 256x256 input. Status appears below.");

            if (st.server_n_frames > 0) {
                ImGui::TextDisabled(
                    "Model: n_frames=%d image_size=%d  device=%s  mode=%s",
                    st.server_n_frames, st.server_image_size,
                    st.server_device.c_str(), st.server_mode_3d.c_str());
            }
            if (!st.server_status.empty()) {
                ImVec4 col = posetail_ui_detail::looks_like_error(
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
                             15);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Server returns one 16-frame chunk per click: t=0 is the\n"
                    "seed (current frame), t=1..15 are future predictions.\n"
                    "This slider picks how many of those 15 to write into\n"
                    "annotations for frames [current+1 .. current+N].");
        } else {
            if (st.onnx_path.empty())
                st.onnx_path = posetail_ui_detail::find_default_onnx();
            ImGui::Text("ONNX");
            ImGui::SetNextItemWidth(-160);
            ImGui::InputText("##posetail_path", &st.onnx_path);
            ImGui::SameLine();
            if (ImGui::Button("Load##posetail")) st.load_requested = true;

            ImGui::Checkbox("Use CPU (slower, falls back if GPU OOMs)",
                            &st.use_cpu);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "GPU is much faster (~30 ms/frame) but uses ~6-8 GB VRAM\n"
                    "for 16 cams x 16 frames x 256x256. If VRAM is already\n"
                    "tight from the display buffer + NvDecoder, switch to\n"
                    "CPU (~3 sec/frame) which always fits.");
            if (!st.use_cpu) {
                ImGui::SameLine();
                ImGui::SetNextItemWidth(80);
                ImGui::InputInt("GPU id", &st.gpu_id, 0, 0);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(
                        "CUDA device id for PoseTail inference. Load picks\n"
                        "the card with the most free VRAM automatically;\n"
                        "this is the starting point. NB: CUDA ids are not\n"
                        "nvidia-smi ids unless CUDA_DEVICE_ORDER=PCI_BUS_ID.");
            }
            if (!st.status.empty()) ImGui::TextDisabled("%s", st.status.c_str());

            ImGui::SetNextItemWidth(120);
            ImGui::SliderInt("N forward frames", &st.n_forward, 1, 200);
            ImGui::SetNextItemWidth(120);
            ImGui::SliderInt("Queries per pass", &st.max_queries, 1, 32);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Max keypoints fed to one ONNX Run. Some GPU drivers can't\n"
                    "grant a single >4 GB allocation, so feeding all 24\n"
                    "keypoints at once OOMs even on a 24 GB card. Splitting\n"
                    "into batches of 12 doubles inference calls but halves\n"
                    "peak per-call memory.");
        }

        // ── Run ──
        ImGui::SeparatorText("Run");
        char fwd_label[64];
        if (st.use_server)
            std::snprintf(fwd_label, sizeof(fwd_label),
                          "PoseTail Forward (server, +%d)", st.server_n_keep);
        else
            std::snprintf(fwd_label, sizeof(fwd_label),
                          "PoseTail Forward +%d", st.n_forward);
        const bool can_run = !is_2d && videos_loaded && (n_3d > 0 ||
                             (st.auto_triangulate && n_2d >= 2));
        ImGui::BeginDisabled(!can_run);
        if (ImGui::Button(fwd_label)) st.forward_requested = true;
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            if (st.use_server)
                ImGui::SetTooltip(
                    "Seed from the current frame's 3D keypoints of the active\n"
                    "animal, send all cameras x 16 frames to the server, and\n"
                    "write the first N future-frame predictions (3D +\n"
                    "reprojected 2D) into annotations for that animal.\n"
                    "Requires all cameras to have frames in the display buffer.");
            else
                ImGui::SetTooltip(
                    "Seed from the current frame's 3D keypoints of the active\n"
                    "animal and predict forward N frames with PoseTail across\n"
                    "ALL cameras. Writes 3D + reprojected 2D into annotations\n"
                    "for frames [current+1 .. current+N] of that animal.\n"
                    "Requires all cameras to have frames in the display buffer.");
        }
        if (!st.last_result.empty()) {
            ImVec4 col = st.last_result_ok ? ImVec4(0.5f, 1.0f, 0.5f, 1.0f)
                                           : ImVec4(1.0f, 0.6f, 0.3f, 1.0f);
            ImGui::TextColored(col, "%s", st.last_result.c_str());
        }
        ImGui::TextDisabled("Predicted labels are marked Predicted; edit them "
                            "in the Labeling Tool like any other label.");
    }, nullptr, ImVec2(560, 420));
}
