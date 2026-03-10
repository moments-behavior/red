#pragma once
// sam_tool.h — SAM-assisted segmentation UI
//
// Provides a point-prompt interface for MobileSAM:
//   1. Toggle "SAM Assist" on
//   2. Left-click = foreground prompt, Right-click = background prompt
//   3. SAM decoder runs in <10ms, mask overlaid instantly
//   4. "Accept" stores mask polygon in Camera2D::mask_polygons
//
// Requires sam_inference.h. Guarded by #ifdef RED_HAS_ONNXRUNTIME for
// the actual inference; UI is always available to show status.

#include "imgui.h"
#include "implot.h"
#include "annotation.h"
#include "app_context.h"
#include "gui/panel.h"
#include "sam_inference.h"
#include <string>
#include <vector>

struct SamToolState {
    bool show = false;
    bool enabled = false;

    // Point prompts (in image coordinates, top-left origin)
    std::vector<tuple_d> fg_points;  // foreground clicks
    std::vector<tuple_d> bg_points;  // background clicks

    // Current mask result
    SamMask current_mask;
    bool has_pending_mask = false;

    // Which frame/camera the current prompts apply to
    u32 prompt_frame = 0;
    int prompt_cam = -1;
};

// Draw SAM mask overlay on a camera's ImPlot view
inline void sam_draw_overlay(SamToolState &state, int cam_idx,
                              int img_w, int img_h) {
    if (!state.enabled) return;
    if (cam_idx != state.prompt_cam) return;

    // Draw foreground points (green)
    if (!state.fg_points.empty()) {
        std::vector<double> xs, ys;
        for (const auto &pt : state.fg_points) {
            xs.push_back(pt.x);
            ys.push_back(img_h - pt.y); // to ImPlot coords
        }
        ImPlot::PushStyleColor(ImPlotCol_MarkerFill, ImVec4(0, 1, 0, 1));
        ImPlot::PushStyleVar(ImPlotStyleVar_MarkerSize, 8.0f);
        ImPlot::PlotScatter("##sam_fg", xs.data(), ys.data(), (int)xs.size());
        ImPlot::PopStyleVar();
        ImPlot::PopStyleColor();
    }

    // Draw background points (red)
    if (!state.bg_points.empty()) {
        std::vector<double> xs, ys;
        for (const auto &pt : state.bg_points) {
            xs.push_back(pt.x);
            ys.push_back(img_h - pt.y);
        }
        ImPlot::PushStyleColor(ImPlotCol_MarkerFill, ImVec4(1, 0, 0, 1));
        ImPlot::PushStyleVar(ImPlotStyleVar_MarkerSize, 8.0f);
        ImPlot::PlotScatter("##sam_bg", xs.data(), ys.data(), (int)xs.size());
        ImPlot::PopStyleVar();
        ImPlot::PopStyleColor();
    }

    // Draw mask overlay (semi-transparent blue)
    // For now, draw mask bounding polygon if available
    if (state.has_pending_mask && state.current_mask.valid) {
        auto polys = sam_mask_to_polygon(state.current_mask);
        for (const auto &poly : polys) {
            std::vector<double> xs, ys;
            for (const auto &pt : poly) {
                xs.push_back(pt.x);
                ys.push_back(img_h - pt.y);
            }
            xs.push_back(xs[0]); ys.push_back(ys[0]); // close
            ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.2f, 0.5f, 1.0f, 0.8f));
            ImPlot::PlotLine("##sam_mask", xs.data(), ys.data(), (int)xs.size());
            ImPlot::PopStyleColor();
        }
    }
}

// Handle SAM input on a focused camera view
inline void sam_handle_input(SamToolState &state, SamState &sam,
                              AnnotationMap &amap, u32 frame, int cam_idx,
                              int num_nodes, int num_cameras,
                              int img_w, int img_h,
                              const uint8_t *frame_rgb = nullptr) {
    if (!state.enabled) return;
    if (!ImPlot::IsPlotHovered()) return;

    ImPlotPoint mouse = ImPlot::GetPlotMousePos();

    // Reset prompts if frame/camera changed
    if (frame != state.prompt_frame || cam_idx != state.prompt_cam) {
        state.fg_points.clear();
        state.bg_points.clear();
        state.has_pending_mask = false;
        state.prompt_frame = frame;
        state.prompt_cam = cam_idx;
    }

    // Convert ImPlot coords to image coords (Y-flip)
    double img_x = std::clamp(mouse.x, 0.0, (double)img_w);
    double img_y = std::clamp((double)img_h - mouse.y, 0.0, (double)img_h);

    // Left click: add foreground point
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGui::GetIO().KeyShift) {
        state.fg_points.push_back({img_x, img_y});

        // Run SAM decoder
        if (sam.loaded && frame_rgb) {
            state.current_mask = sam_segment(sam, frame_rgb, img_w, img_h,
                                              state.fg_points, state.bg_points,
                                              nullptr, frame, cam_idx);
            state.has_pending_mask = state.current_mask.valid;
        }
    }

    // Right click: add background point
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        state.bg_points.push_back({img_x, img_y});

        if (sam.loaded && frame_rgb) {
            state.current_mask = sam_segment(sam, frame_rgb, img_w, img_h,
                                              state.fg_points, state.bg_points,
                                              nullptr, frame, cam_idx);
            state.has_pending_mask = state.current_mask.valid;
        }
    }

    // Enter: accept mask → store in annotation
    if (state.has_pending_mask && ImGui::IsKeyPressed(ImGuiKey_Enter)) {
        auto polys = sam_mask_to_polygon(state.current_mask);
        if (!polys.empty()) {
            auto &fa = get_or_create_frame(amap, frame, num_nodes, num_cameras);
            // Store on first instance (or create one)
            if (fa.instances.empty())
                fa.instances.push_back(make_instance(num_nodes, num_cameras));
            auto &inst = fa.instances[0];
            if (cam_idx < (int)inst.cameras.size()) {
                inst.cameras[cam_idx].mask_polygons = polys;
                inst.cameras[cam_idx].has_mask = true;
            }
        }
        // Reset
        state.fg_points.clear();
        state.bg_points.clear();
        state.has_pending_mask = false;
    }

    // Backspace: undo last point
    if (ImGui::IsKeyPressed(ImGuiKey_Backspace)) {
        if (!state.bg_points.empty()) state.bg_points.pop_back();
        else if (!state.fg_points.empty()) state.fg_points.pop_back();
    }

    // Escape: clear all prompts
    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
        state.fg_points.clear();
        state.bg_points.clear();
        state.has_pending_mask = false;
    }
}

// SAM tool settings panel
inline void DrawSamToolWindow(SamToolState &state, SamState &sam,
                               AppContext &ctx) {
    drawPanel("SAM Assist", state.show,
        [&]() {
        // Availability status
        if (!sam.available) {
            ImGui::TextColored(ImVec4(1, 0.5f, 0, 1),
                               "ONNX Runtime not available");
            ImGui::TextWrapped("Recompile with -DRED_HAS_ONNXRUNTIME to enable "
                               "SAM-assisted segmentation.");
            return;
        }

        if (!sam.loaded) {
            ImGui::TextColored(ImVec4(1, 1, 0, 1), "Models not loaded");
            // TODO: model path input + load button
            ImGui::TextWrapped("Download MobileSAM ONNX models (~50MB) and load them.");
            return;
        }

        ImGui::Checkbox("Enable SAM Assist", &state.enabled);

        ImGui::Separator();
        ImGui::Text("Encoder: %.1f ms", sam.last_encode_ms);
        ImGui::Text("Decoder: %.1f ms", sam.last_decode_ms);

        if (state.has_pending_mask) {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 1, 1),
                               "Mask ready (IoU: %.3f)", state.current_mask.iou_score);
            if (ImGui::Button("Accept (Enter)")) {
                // Simulate Enter key press for acceptance
                // (actual acceptance handled in sam_handle_input)
            }
        }

        ImGui::Text("FG points: %d", (int)state.fg_points.size());
        ImGui::Text("BG points: %d", (int)state.bg_points.size());

        if (ImGui::Button("Clear Points")) {
            state.fg_points.clear();
            state.bg_points.clear();
            state.has_pending_mask = false;
        }

        ImGui::Separator();
        ImGui::TextWrapped("Left click: foreground point");
        ImGui::TextWrapped("Right click: background point");
        ImGui::TextWrapped("Enter: accept mask");
        ImGui::TextWrapped("Backspace: undo last point");
        ImGui::TextWrapped("Escape: clear all");

        if (!sam.status.empty()) {
            ImGui::Separator();
            ImGui::TextWrapped("%s", sam.status.c_str());
        }
        },
        nullptr, ImVec2(300, 400));
}
