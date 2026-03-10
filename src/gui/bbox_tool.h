#pragma once
// bbox_tool.h — Axis-aligned bounding box labeling tool
//
// Shift+drag draws a new bbox. Bboxes are stored in the unified
// AnnotationMap (Camera2D::has_bbox + bbox fields). Class/ID selection,
// keyboard shortcuts, and ImPlot interaction follow the original patterns.

#include "imgui.h"
#include "implot.h"
#include "annotation.h"
#include "app_context.h"
#include "gui/panel.h"
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

struct BBoxToolState {
    bool show = false;
    bool enabled = false; // master toggle for bbox drawing mode

    // Class and instance tracking
    std::vector<std::string> class_names = {"animal"};
    std::vector<ImVec4> class_colors = {ImVec4(0.3f, 1.0f, 1.0f, 1.0f)};
    int current_class = 0;
    int current_instance = 0;
    bool show_ids = true;

    // Drawing state
    bool drawing = false;       // currently dragging out a new bbox
    double start_x = 0, start_y = 0;

    // Hover state
    int hovered_instance = -1;  // instance index under cursor
    int hovered_cam = -1;

    ImVec4 next_class_color() const {
        float hue = class_colors.size() * 0.618033f;
        hue -= std::floor(hue);
        return (ImVec4)ImColor::HSV(hue, 0.85f, 0.95f);
    }
};

// Draw bbox rectangles on a camera's ImPlot view
inline void bbox_draw_overlays(BBoxToolState &state, const AnnotationMap &amap,
                                u32 frame, int cam_idx, int img_w, int img_h) {
    auto it = amap.find(frame);
    if (it == amap.end()) return;

    for (int i = 0; i < (int)it->second.instances.size(); ++i) {
        const auto &inst = it->second.instances[i];
        if (cam_idx >= (int)inst.cameras.size()) continue;
        const auto &cam = inst.cameras[cam_idx];
        if (!cam.has_bbox) continue;

        int ci = inst.category_id;
        ImVec4 color = (ci < (int)state.class_colors.size())
                           ? state.class_colors[ci]
                           : ImVec4(1, 1, 1, 1);

        // Highlight hovered
        if (i != state.hovered_instance || cam_idx != state.hovered_cam)
            color.w *= 0.6f;

        // Draw filled rect
        double x1 = cam.bbox_x;
        double y1_img = cam.bbox_y; // top-left in image coords
        double x2 = x1 + cam.bbox_w;
        double y2_img = y1_img + cam.bbox_h;
        // Convert to ImPlot coords (Y is flipped: ImPlot y = img_h - img_y)
        double y1_plot = img_h - y2_img;
        double y2_plot = img_h - y1_img;

        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(color.x, color.y, color.z, 0.15f));
        ImPlot::PushStyleColor(ImPlotCol_Line, color);
        double xs[] = {x1, x2, x2, x1, x1};
        double ys[] = {y1_plot, y1_plot, y2_plot, y2_plot, y1_plot};
        ImPlot::PlotLine("##bbox", xs, ys, 5);
        ImPlot::PopStyleColor(2);

        // Label
        if (state.show_ids) {
            char label[64];
            snprintf(label, sizeof(label), "%s #%d",
                     (ci < (int)state.class_names.size()) ? state.class_names[ci].c_str() : "?",
                     inst.instance_id);
            ImPlot::PlotText(label, x1 + 4, y2_plot - 4);
        }
    }

    // Draw in-progress bbox (while shift-dragging)
    if (state.drawing) {
        ImPlotPoint mouse = ImPlot::GetPlotMousePos();
        double xs[] = {state.start_x, mouse.x, mouse.x, state.start_x, state.start_x};
        double ys[] = {state.start_y, state.start_y, mouse.y, mouse.y, state.start_y};
        ImVec4 c = state.class_colors[state.current_class];
        ImPlot::PushStyleColor(ImPlotCol_Line, c);
        ImPlot::PlotLine("##bbox_new", xs, ys, 5);
        ImPlot::PopStyleColor();
    }
}

// Handle bbox input on a focused camera view
inline void bbox_handle_input(BBoxToolState &state, AnnotationMap &amap,
                               u32 frame, int cam_idx, int num_nodes,
                               int num_cameras, int img_w, int img_h) {
    if (!state.enabled) return;
    if (!ImPlot::IsPlotHovered()) return;

    ImPlotPoint mouse = ImPlot::GetPlotMousePos();

    // Clamp to image bounds
    double mx = std::clamp(mouse.x, 0.0, (double)img_w);
    double my = std::clamp(mouse.y, 0.0, (double)img_h);

    bool shift = ImGui::GetIO().KeyShift;

    // Shift-down: start drawing
    if (shift && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        state.drawing = true;
        state.start_x = mx;
        state.start_y = my;
    }

    // Shift released while drawing: commit bbox
    if (state.drawing && !shift) {
        state.drawing = false;

        // Normalize coords (ImPlot → image space)
        double x1 = std::min(state.start_x, mx);
        double x2 = std::max(state.start_x, mx);
        double y1_plot = std::min(state.start_y, my);
        double y2_plot = std::max(state.start_y, my);

        // Skip tiny accidental drags
        if (x2 - x1 < 3 || y2_plot - y1_plot < 3) return;

        // Convert to image coords (Y-flip)
        double bbox_x = x1;
        double bbox_y = img_h - y2_plot; // top-left in image coords
        double bbox_w = x2 - x1;
        double bbox_h = y2_plot - y1_plot;

        // Get or create frame + instance
        auto &fa = get_or_create_frame(amap, frame, num_nodes, num_cameras);

        // Find or create instance with matching class/id
        InstanceAnnotation *target = nullptr;
        for (auto &inst : fa.instances) {
            if (inst.category_id == state.current_class &&
                inst.instance_id == state.current_instance) {
                target = &inst;
                break;
            }
        }
        if (!target) {
            fa.instances.push_back(make_instance(num_nodes, num_cameras,
                                                  state.current_instance,
                                                  state.current_class));
            target = &fa.instances.back();
        }

        if (cam_idx < (int)target->cameras.size()) {
            auto &cam = target->cameras[cam_idx];
            cam.bbox_x = bbox_x;
            cam.bbox_y = bbox_y;
            cam.bbox_w = bbox_w;
            cam.bbox_h = bbox_h;
            cam.has_bbox = true;
        }
    }

    // Hover detection
    state.hovered_instance = -1;
    state.hovered_cam = -1;
    auto it = amap.find(frame);
    if (it != amap.end()) {
        for (int i = 0; i < (int)it->second.instances.size(); ++i) {
            const auto &inst = it->second.instances[i];
            if (cam_idx >= (int)inst.cameras.size()) continue;
            const auto &cam = inst.cameras[cam_idx];
            if (!cam.has_bbox) continue;

            double plot_y = img_h - cam.bbox_y - cam.bbox_h; // bottom in plot
            if (mx >= cam.bbox_x && mx <= cam.bbox_x + cam.bbox_w &&
                my >= plot_y && my <= plot_y + cam.bbox_h) {
                state.hovered_instance = i;
                state.hovered_cam = cam_idx;
                break;
            }
        }
    }

    // F key: delete hovered bbox from this camera
    if (state.hovered_instance >= 0 && ImGui::IsKeyPressed(ImGuiKey_F)) {
        auto &inst = amap[frame].instances[state.hovered_instance];
        if (cam_idx < (int)inst.cameras.size()) {
            inst.cameras[cam_idx].has_bbox = false;
        }
        state.hovered_instance = -1;
    }

    // Z/X: switch class
    if (ImGui::IsKeyPressed(ImGuiKey_Z)) {
        state.current_class = (state.current_class - 1 + (int)state.class_names.size())
                              % (int)state.class_names.size();
        state.current_instance = 0;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_X)) {
        state.current_class = (state.current_class + 1) % (int)state.class_names.size();
        state.current_instance = 0;
    }
    // C/V: switch instance ID
    if (ImGui::IsKeyPressed(ImGuiKey_C) && state.current_instance > 0)
        state.current_instance--;
    if (ImGui::IsKeyPressed(ImGuiKey_V))
        state.current_instance++;
}

// Settings panel for the bbox tool
inline void DrawBBoxToolWindow(BBoxToolState &state, AppContext &ctx) {
    drawPanel("Bbox Tool", state.show,
        [&]() {
        ImGui::Checkbox("Enable Bbox Drawing", &state.enabled);
        ImGui::Checkbox("Show IDs", &state.show_ids);

        ImGui::Separator();
        ImGui::Text("Class: %s (%d)",
                    state.class_names[state.current_class].c_str(),
                    state.current_class);
        ImGui::Text("Instance: %d", state.current_instance);

        ImGui::Separator();
        ImGui::TextWrapped("Shift+drag: draw bbox");
        ImGui::TextWrapped("F: delete hovered bbox");
        ImGui::TextWrapped("Z/X: prev/next class, C/V: prev/next instance");

        // Class list
        ImGui::SeparatorText("Classes");
        for (int i = 0; i < (int)state.class_names.size(); ++i) {
            ImGui::ColorButton(("##clr" + std::to_string(i)).c_str(),
                               state.class_colors[i], 0, ImVec2(14, 14));
            ImGui::SameLine();
            bool sel = (i == state.current_class);
            if (ImGui::Selectable(state.class_names[i].c_str(), sel))
                state.current_class = i;
        }
        if (ImGui::Button("+ Add Class")) {
            int n = (int)state.class_names.size();
            state.class_names.push_back("Class_" + std::to_string(n + 1));
            state.class_colors.push_back(state.next_class_color());
        }
        },
        nullptr, ImVec2(300, 350));
}
