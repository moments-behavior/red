#pragma once
// bbox_tool.h — Axis-aligned bounding box labeling tool
//
// Shift+drag draws a new bbox. Each camera view can hold any number of
// bboxes; each box carries its own class + instance ID (CameraExtras::bboxes).
// Class names live in the project (AnnotationConfig::class_names) so they
// persist with the .redproj and are shared with the OBB tool and exporters.

#include "imgui.h"
#include "implot.h"
#include <misc/cpp/imgui_stdlib.h>
#include "annotation.h"
#include "app_context.h"
#include "gui/panel.h"
#include "project.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

struct BBoxToolState {
    bool show = false;
    bool enabled = false; // master toggle for bbox drawing mode

    // Class and instance applied to newly drawn boxes (bbox + OBB tools)
    int current_class = 0;
    int current_instance = 0;
    bool show_ids = true;

    // Drawing state
    bool drawing = false;       // currently dragging out a new bbox
    double start_x = 0, start_y = 0;

    // Hovered box (index into this camera's bboxes), -1 = none
    int hovered_cam = -1;
    int hovered_idx = -1;

    // Class list edited (added/renamed) → re-save the .redproj
    bool classes_dirty = false;
};

// Deterministic per-class color: class 0 cyan, then golden-ratio hues.
inline ImVec4 bbox_class_color(int ci) {
    if (ci <= 0) return ImVec4(0.3f, 1.0f, 1.0f, 1.0f);
    float hue = ci * 0.618033f;
    hue -= std::floor(hue);
    return (ImVec4)ImColor::HSV(hue, 0.85f, 0.95f);
}

inline const char *bbox_class_name(const std::vector<std::string> &class_names, int ci) {
    return (ci >= 0 && ci < (int)class_names.size()) ? class_names[ci].c_str() : "?";
}

// Keep the class list non-empty and current_class in range.
inline void bbox_sanitize_classes(BBoxToolState &state,
                                  std::vector<std::string> &class_names) {
    if (class_names.empty()) class_names.push_back("animal");
    state.current_class = std::clamp(state.current_class, 0,
                                     (int)class_names.size() - 1);
}

inline void bbox_add_class(BBoxToolState &state,
                           std::vector<std::string> &class_names) {
    class_names.push_back("Class_" + std::to_string(class_names.size() + 1));
    state.current_class = (int)class_names.size() - 1;
    state.current_instance = 0;
    state.classes_dirty = true;
}

// Draw bbox rectangles on a camera's ImPlot view
inline void bbox_draw_overlays(BBoxToolState &state, const AnnotationMap &amap,
                                const std::vector<std::string> &class_names,
                                u32 frame, int cam_idx, int img_w, int img_h) {
    auto it = amap.find(frame);
    if (it != amap.end() && cam_idx < (int)it->second.cameras.size()) {
        const auto &cam = it->second.cameras[cam_idx];
        const auto &boxes = cam.get_extras().bboxes;
        for (int i = 0; i < (int)boxes.size(); ++i) {
            const auto &b = boxes[i];
            ImVec4 color = bbox_class_color(b.category_id);
            bool hovered = (cam_idx == state.hovered_cam && i == state.hovered_idx);
            if (!hovered) color.w *= 0.6f;

            // Image coords (y-down) → ImPlot coords (y-up)
            double x1 = b.x, x2 = b.x + b.w;
            double y1_plot = img_h - (b.y + b.h);
            double y2_plot = img_h - b.y;

            ImPlot::SetNextLineStyle(color, hovered ? 2.5f : 1.0f);
            double xs[] = {x1, x2, x2, x1, x1};
            double ys[] = {y1_plot, y1_plot, y2_plot, y2_plot, y1_plot};
            ImPlot::PlotLine("##bbox", xs, ys, 5);

            if (state.show_ids) {
                char label[96];
                snprintf(label, sizeof(label), "%s #%d",
                         bbox_class_name(class_names, b.category_id), b.instance_id);
                ImPlot::PushStyleColor(ImPlotCol_InlayText, color);
                // PlotText centers on the point; shift it inside the top-left corner
                ImVec2 ts = ImGui::CalcTextSize(label);
                ImPlot::PlotText(label, x1, y2_plot,
                                 ImVec2(ts.x * 0.5f + 3, ts.y * 0.5f + 2));
                ImPlot::PopStyleColor();
            }
        }
    }

    // Draw in-progress bbox (while shift-dragging)
    if (state.drawing && ImPlot::IsPlotHovered()) {
        ImPlotPoint mouse = ImPlot::GetPlotMousePos();
        double dxs[] = {state.start_x, mouse.x, mouse.x, state.start_x, state.start_x};
        double dys[] = {state.start_y, state.start_y, mouse.y, mouse.y, state.start_y};
        ImPlot::SetNextLineStyle(bbox_class_color(state.current_class));
        ImPlot::PlotLine("##bbox_new", dxs, dys, 5);
    }
}

// Handle bbox input on a focused camera view
inline void bbox_handle_input(BBoxToolState &state, AnnotationMap &amap,
                               std::vector<std::string> &class_names,
                               u32 frame, int cam_idx, int num_nodes,
                               int num_cameras, int img_w, int img_h) {
    // Clear this camera's hover first so it doesn't go stale when the
    // cursor leaves the view.
    if (state.hovered_cam == cam_idx) { state.hovered_cam = -1; state.hovered_idx = -1; }
    if (!state.enabled) return;
    if (!ImPlot::IsPlotHovered()) return;
    bbox_sanitize_classes(state, class_names);

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

    // Shift released while drawing: commit bbox (appended — never replaces)
    if (state.drawing && !shift) {
        state.drawing = false;

        double x1 = std::min(state.start_x, mx);
        double x2 = std::max(state.start_x, mx);
        double y1_plot = std::min(state.start_y, my);
        double y2_plot = std::max(state.start_y, my);

        // Skip tiny accidental drags
        if (x2 - x1 >= 3 && y2_plot - y1_plot >= 3) {
            auto &fa = get_or_create_frame(amap, frame, num_nodes, num_cameras);
            if (cam_idx < (int)fa.cameras.size()) {
                BBox b;
                b.x = x1;
                b.y = img_h - y2_plot; // top-left in image coords (Y-flip)
                b.w = x2 - x1;
                b.h = y2_plot - y1_plot;
                b.category_id = state.current_class;
                b.instance_id = state.current_instance;
                fa.cameras[cam_idx].get_extras().bboxes.push_back(b);
            }
        }
    }

    // Hover detection: smallest box under the cursor, so nested boxes stay
    // reachable.
    auto it = amap.find(frame);
    if (it != amap.end() && cam_idx < (int)it->second.cameras.size()) {
        const auto &boxes = it->second.cameras[cam_idx].get_extras().bboxes;
        double img_y = img_h - my;
        double best_area = 1e300;
        for (int i = 0; i < (int)boxes.size(); ++i) {
            const auto &b = boxes[i];
            if (mx >= b.x && mx <= b.x + b.w && img_y >= b.y && img_y <= b.y + b.h &&
                b.w * b.h < best_area) {
                best_area = b.w * b.h;
                state.hovered_cam = cam_idx;
                state.hovered_idx = i;
            }
        }
    }

    // Keyboard shortcuts below must not fire while typing in a text field.
    if (ImGui::GetIO().WantTextInput) return;

    bool have_hover = (state.hovered_cam == cam_idx && state.hovered_idx >= 0);

    // Delete: delete hovered bbox from this camera
    // (not F — F already deletes the active keypoint across all views)
    if (have_hover && ImGui::IsKeyPressed(ImGuiKey_Delete, false)) {
        auto &boxes = amap[frame].cameras[cam_idx].get_extras().bboxes;
        boxes.erase(boxes.begin() + state.hovered_idx);
        state.hovered_cam = state.hovered_idx = -1;
        have_hover = false;
    }

    // O: delete the hovered object (same class + instance) from ALL cameras
    if (have_hover && ImGui::IsKeyPressed(ImGuiKey_O, false)) {
        auto &fa = amap[frame];
        BBox target = fa.cameras[cam_idx].get_extras().bboxes[state.hovered_idx];
        for (auto &cam : fa.cameras) {
            if (!cam.has_bbox()) continue;
            auto &boxes = cam.get_extras().bboxes;
            boxes.erase(std::remove_if(boxes.begin(), boxes.end(),
                            [&](const BBox &b) {
                                return b.category_id == target.category_id &&
                                       b.instance_id == target.instance_id;
                            }),
                        boxes.end());
        }
        state.hovered_cam = state.hovered_idx = -1;
        have_hover = false;
    }

    // I: pick class + instance from the hovered box (to label the same
    // object in another camera)
    if (have_hover && ImGui::IsKeyPressed(ImGuiKey_I, false)) {
        const auto &b = amap[frame].cameras[cam_idx].get_extras().bboxes[state.hovered_idx];
        state.current_class = std::clamp(b.category_id, 0, (int)class_names.size() - 1);
        state.current_instance = b.instance_id;
    }

    int nclass = (int)class_names.size();
    // Z/X: switch class
    if (ImGui::IsKeyPressed(ImGuiKey_Z, false)) {
        state.current_class = (state.current_class - 1 + nclass) % nclass;
        state.current_instance = 0;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_X, false)) {
        state.current_class = (state.current_class + 1) % nclass;
        state.current_instance = 0;
    }
    // N key: create new class
    if (ImGui::IsKeyPressed(ImGuiKey_N, false))
        bbox_add_class(state, class_names);
    // C/V: switch instance ID
    if (ImGui::IsKeyPressed(ImGuiKey_C, false) && state.current_instance > 0)
        state.current_instance--;
    if (ImGui::IsKeyPressed(ImGuiKey_V, false))
        state.current_instance++;
}

// Settings panel for the bbox tool
inline void DrawBBoxToolWindow(BBoxToolState &state, AppContext &ctx) {
    auto &class_names = ctx.pm.annotation_config.class_names;
    DrawPanel("Bbox Tool", state.show,
        [&]() {
        bbox_sanitize_classes(state, class_names);
        ImGui::Checkbox("Enable Bbox Drawing", &state.enabled);
        ImGui::Checkbox("Show IDs", &state.show_ids);

        ImGui::Separator();
        ImGui::TextColored(bbox_class_color(state.current_class), "Class: %s (%d)",
                           class_names[state.current_class].c_str(),
                           state.current_class);
        ImGui::Text("Instance: %d", state.current_instance);

        // Boxes on the current frame
        auto it = ctx.annotations.find((u32)ctx.current_frame_num);
        int n_boxes = 0;
        if (it != ctx.annotations.end())
            for (const auto &cam : it->second.cameras)
                n_boxes += (int)cam.get_extras().bboxes.size();
        ImGui::Text("Boxes on this frame (all cameras): %d", n_boxes);

        ImGui::Separator();
        ImGui::TextWrapped("Shift+drag: draw bbox (adds; frames can hold many)");
        ImGui::TextWrapped("Delete: delete hovered bbox (this camera)");
        ImGui::TextWrapped("O: delete hovered object (same class+ID, all cameras)");
        ImGui::TextWrapped("I: pick class+ID from hovered bbox");
        ImGui::TextWrapped("Z/X: prev/next class, N: new class");
        ImGui::TextWrapped("C/V: prev/next instance ID");

        // Class list (click to select, edit to rename)
        ImGui::SeparatorText("Classes");
        for (int i = 0; i < (int)class_names.size(); ++i) {
            ImGui::PushID(i);
            ImGui::ColorButton("##clr", bbox_class_color(i), 0, ImVec2(14, 14));
            ImGui::SameLine();
            if (ImGui::RadioButton("##sel", i == state.current_class)) {
                state.current_class = i;
                state.current_instance = 0;
            }
            ImGui::SameLine();
            ImGui::SetNextItemWidth(-1);
            if (ImGui::InputText("##name", &class_names[i],
                                 ImGuiInputTextFlags_EnterReturnsTrue) ||
                ImGui::IsItemDeactivatedAfterEdit())
                state.classes_dirty = true;
            ImGui::PopID();
        }
        if (ImGui::Button("+ Add Class"))
            bbox_add_class(state, class_names);
        },
        // Always: persist class list edits to the .redproj
        [&]() {
            if (!state.classes_dirty) return;
            state.classes_dirty = false;
            const auto &pm = ctx.pm;
            if (pm.project_path.empty() || pm.project_name.empty()) return;
            save_project_manager_json(pm, std::filesystem::path(pm.project_path) /
                                              (pm.project_name + ".redproj"));
        },
        ImVec2(320, 420));
}
