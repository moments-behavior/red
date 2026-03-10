#pragma once
// sam_tool.h — SAM-assisted segmentation UI
//
// Provides a point-prompt interface for MobileSAM / SAM 2.1:
//   1. Select model + load ONNX files
//   2. Toggle "SAM Assist" on
//   3. Left-click = foreground prompt, Right-click = background prompt
//   4. SAM decoder runs in <20ms, mask overlaid instantly
//   5. "Accept" stores mask polygon in CameraAnnotation extras

#include "imgui.h"
#include "implot.h"
#include "annotation.h"
#include "app_context.h"
#include "gui/panel.h"
#include "sam_inference.h"
#include <ImGuiFileDialog.h>
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

    // Model selection UI state
    int model_idx = 0; // 0 = MobileSAM, 1 = SAM 2.1 Tiny
    std::string encoder_path;
    std::string decoder_path;
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

    // Draw mask overlay (semi-transparent blue polygon)
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

    // Enter: accept mask -> store in annotation
    if (state.has_pending_mask && ImGui::IsKeyPressed(ImGuiKey_Enter)) {
        auto polys = sam_mask_to_polygon(state.current_mask);
        if (!polys.empty()) {
            auto &fa = get_or_create_frame(amap, frame, num_nodes, num_cameras);
            if (cam_idx < (int)fa.cameras.size()) {
                auto &ext = fa.cameras[cam_idx].get_extras();
                ext.mask_polygons = polys;
                ext.has_mask = true;
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
            ImGui::TextWrapped("Compile with ONNX Runtime in lib/onnxruntime/ "
                               "to enable SAM-assisted segmentation.");
            return;
        }

        // --- Model selection + loading ---
        ImGui::Text("Model");
        const char *model_names[] = {"MobileSAM (~9 MB)", "SAM 2.1 Tiny (~117 MB)"};
        ImGui::Combo("##sam_model", &state.model_idx, model_names, 2);

        ImGui::Separator();
        ImGui::Text("Encoder ONNX");
        ImGui::SetNextItemWidth(-60);
        ImGui::InputText("##sam_enc", &state.encoder_path);
        ImGui::SameLine();
        if (ImGui::Button("...##enc")) {
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 1;
            cfg.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "SamBrowseEncoder", "Select Encoder ONNX", ".onnx", cfg);
        }

        ImGui::Text("Decoder ONNX");
        ImGui::SetNextItemWidth(-60);
        ImGui::InputText("##sam_dec", &state.decoder_path);
        ImGui::SameLine();
        if (ImGui::Button("...##dec")) {
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 1;
            cfg.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "SamBrowseDecoder", "Select Decoder ONNX", ".onnx", cfg);
        }

        // Load button
        bool can_load = !state.encoder_path.empty() && !state.decoder_path.empty();
        if (!can_load) ImGui::BeginDisabled();
        if (ImGui::Button("Load Model")) {
            SamModel model_type = (state.model_idx == 0) ? SamModel::MobileSAM
                                                          : SamModel::SAM2;
            sam_init(sam, model_type, state.encoder_path.c_str(),
                     state.decoder_path.c_str());
        }
        if (!can_load) ImGui::EndDisabled();

        ImGui::SameLine();
        if (sam.loaded) {
            ImGui::TextColored(ImVec4(0, 1, 0, 1), "Loaded");
        } else if (!sam.status.empty()) {
            ImGui::TextColored(ImVec4(1, 1, 0, 1), "%s", sam.status.c_str());
        }

        ImGui::Separator();

        // --- Enable / usage ---
        if (!sam.loaded) ImGui::BeginDisabled();
        ImGui::Checkbox("Enable SAM Assist", &state.enabled);
        if (!sam.loaded) ImGui::EndDisabled();

        if (sam.loaded) {
            ImGui::Text("Encoder: %.1f ms  Decoder: %.1f ms",
                        sam.last_encode_ms, sam.last_decode_ms);
        }

        if (state.has_pending_mask) {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 1, 1),
                               "Mask ready (IoU: %.3f)", state.current_mask.iou_score);
        }

        ImGui::Text("FG: %d  BG: %d",
                     (int)state.fg_points.size(), (int)state.bg_points.size());

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
        },
        // always_fn: handle file dialog results every frame
        [&]() {
            if (ImGuiFileDialog::Instance()->Display("SamBrowseEncoder")) {
                if (ImGuiFileDialog::Instance()->IsOk()) {
                    state.encoder_path =
                        ImGuiFileDialog::Instance()->GetFilePathName();
                }
                ImGuiFileDialog::Instance()->Close();
            }
            if (ImGuiFileDialog::Instance()->Display("SamBrowseDecoder")) {
                if (ImGuiFileDialog::Instance()->IsOk()) {
                    state.decoder_path =
                        ImGuiFileDialog::Instance()->GetFilePathName();
                }
                ImGuiFileDialog::Instance()->Close();
            }
        },
        ImVec2(320, 480));
}
