#pragma once
// bbox_tool.h — Axis-aligned bounding box labeling tool
//
// Shift+drag draws a new bbox: press with Shift held, drag, let go. Bboxes are stored in the unified
// AnnotationMap (CameraAnnotation extras). Class/ID selection,
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
    // Empty until the first box (or + Add Class / N) makes Class_1.
    std::vector<std::string> class_names;
    std::vector<ImVec4> class_colors;
    int current_class = 0;
    int current_instance = 0;
    bool show_ids = true;

    // Drawing state
    bool drawing = false;       // currently dragging out a new bbox
    int drawing_cam = -1;       // the camera view it is being drawn in
    double start_x = 0, start_y = 0;

    // Hover state
    bool hovered = false;       // bbox under cursor on this frame
    int hovered_cam = -1;

    ImVec4 next_class_color() const {
        float hue = class_colors.size() * 0.618033f;
        hue -= std::floor(hue);
        return (ImVec4)ImColor::HSV(hue, 0.85f, 0.95f);
    }

    // Adds Class_<n+1> and makes it current.
    void add_class() {
        const int n = (int)class_names.size();
        class_names.push_back("Class_" + std::to_string(n + 1));
        class_colors.push_back(next_class_color());
        current_class = n;
        current_instance = 0;
    }
    ImVec4 current_color() const {
        return current_class >= 0 && current_class < (int)class_colors.size()
                   ? class_colors[current_class]
                   : ImVec4(0.3f, 1.0f, 1.0f, 1.0f);
    }
};

// Draw bbox rectangles on a camera's ImPlot view
inline void bbox_draw_overlays(BBoxToolState &state, const AnnotationMap &amap,
                                u32 frame, int cam_idx, int img_w, int img_h) {
    // Draw in-progress bbox (while shift-dragging), in its own view only
    if (state.drawing && cam_idx == state.drawing_cam) {
        ImPlotPoint mouse = ImPlot::GetPlotMousePos();
        double dxs[] = {state.start_x, mouse.x, mouse.x, state.start_x, state.start_x};
        double dys[] = {state.start_y, state.start_y, mouse.y, mouse.y, state.start_y};
        ImVec4 c = state.current_color();
        ImPlotSpec nspec;
        nspec.LineColor = c;
        ImPlot::PlotLine("##bbox_new", dxs, dys, 5, nspec);
    }

    auto it = amap.find(frame);
    if (it == amap.end() || it->second.empty()) return;
    // The bbox tools work on the animal being labelled; multi-animal boxes
    // wait on the active-instance UI.
    const auto &fa = it->second.front();

    if (cam_idx >= (int)fa.cameras.size()) return;
    const auto &cam = fa.cameras[cam_idx];
    if (!cam.has_bbox()) return;

    int ci = fa.category_id;
    ImVec4 color = (ci < (int)state.class_colors.size())
                       ? state.class_colors[ci]
                       : ImVec4(1, 1, 1, 1);

    // Highlight hovered
    if (!state.hovered || cam_idx != state.hovered_cam)
        color.w *= 0.6f;

    // Draw filled rect
    double x1 = cam.extras->bbox_x;
    double y1_img = cam.extras->bbox_y; // top-left in image coords
    double x2 = x1 + cam.extras->bbox_w;
    double y2_img = y1_img + cam.extras->bbox_h;
    // Convert to ImPlot coords (Y is flipped: ImPlot y = img_h - img_y)
    double y1_plot = img_h - y2_img;
    double y2_plot = img_h - y1_img;

    // ImPlot v1.0: item colors moved from PushStyleColor(ImPlotCol_Line/Fill)
    // to a per-call ImPlotSpec.
    double xs[] = {x1, x2, x2, x1, x1};
    double ys[] = {y1_plot, y1_plot, y2_plot, y2_plot, y1_plot};
    ImPlotSpec bspec;
    bspec.LineColor = color;
    bspec.FillColor = ImVec4(color.x, color.y, color.z, 0.15f);
    ImPlot::PlotLine("##bbox", xs, ys, 5, bspec);

    // Label
    if (state.show_ids) {
        char label[64];
        snprintf(label, sizeof(label), "%s #%d",
                 (ci < (int)state.class_names.size()) ? state.class_names[ci].c_str() : "?",
                 fa.instance_id);
        ImPlot::PlotText(label, x1 + 4, y2_plot - 4);
    }
}

// True while the camera views must not pan: a left drag with Shift held is
// drawing a box, not moving the image (ImPlot pans on left drag whatever the
// modifiers).
inline bool bbox_blocks_pan(const BBoxToolState &state) {
    return state.enabled && (state.drawing || ImGui::GetIO().KeyShift);
}

// The pointer while a box can be drawn (Shift over a view) or is being drawn:
// crosshairs with a small box beside them, in the class colour. Replaces
// ImPlot's "not allowed" cursor, which it shows for a drag on the views
// bbox_blocks_pan locks. Call inside the camera's plot.
inline void bbox_draw_cursor(const BBoxToolState &state, int cam_idx) {
    if (!state.enabled) return;
    const bool here = state.drawing ? cam_idx == state.drawing_cam
                                    : ImGui::GetIO().KeyShift && ImPlot::IsPlotHovered();
    if (!here) return;
    ImGui::SetMouseCursor(ImGuiMouseCursor_None);
    ImDrawList *dl = ImPlot::GetPlotDrawList();
    const ImVec2 m = ImGui::GetIO().MousePos;
    const ImVec2 lo = ImPlot::GetPlotPos();
    const ImVec2 hi(lo.x + ImPlot::GetPlotSize().x, lo.y + ImPlot::GetPlotSize().y);
    const ImU32 col = ImGui::GetColorU32(state.current_color());
    const ImU32 cross = IM_COL32(255, 255, 255, 150);
    dl->PushClipRect(lo, hi, true);
    dl->AddLine(ImVec2(lo.x, m.y), ImVec2(m.x - 5, m.y), cross);
    dl->AddLine(ImVec2(m.x + 5, m.y), ImVec2(hi.x, m.y), cross);
    dl->AddLine(ImVec2(m.x, lo.y), ImVec2(m.x, m.y - 5), cross);
    dl->AddLine(ImVec2(m.x, m.y + 5), ImVec2(m.x, hi.y), cross);
    // The box badge, below-right of the pointer.
    const ImVec2 b0(m.x + 9, m.y + 9), b1(m.x + 23, m.y + 19);
    dl->AddRectFilled(b0, b1, IM_COL32(0, 0, 0, 120));
    dl->AddRect(b0, b1, col, 0.0f, 0, 2.0f);
    dl->PopClipRect();
}

// Handle bbox input on a focused camera view
inline void bbox_handle_input(BBoxToolState &state, AnnotationMap &amap,
                               u32 frame, int cam_idx, int num_nodes,
                               int num_cameras, int img_w, int img_h) {
    if (!state.enabled) return;
    // A drag belongs to the view it started in; the others leave it alone.
    if (state.drawing && cam_idx != state.drawing_cam) return;
    if (!state.drawing && !ImPlot::IsPlotHovered()) return;

    ImPlotPoint mouse = ImPlot::GetPlotMousePos();

    // Clamp to image bounds
    double mx = std::clamp(mouse.x, 0.0, (double)img_w);
    double my = std::clamp(mouse.y, 0.0, (double)img_h);

    // Shift + press starts a box at the cursor ...
    if (!state.drawing && ImGui::GetIO().KeyShift &&
        ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        // The first box needs a class to belong to.
        if (state.class_names.empty()) state.add_class();
        state.drawing = true;
        state.drawing_cam = cam_idx;
        state.start_x = mx;
        state.start_y = my;
        return;
    }
    // ... Escape drops it ...
    if (state.drawing && ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
        state.drawing = false;
        state.drawing_cam = -1;
        return;
    }

    // ... and letting go of the mouse commits it, Shift still held or not.
    if (state.drawing && !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        state.drawing = false;
        state.drawing_cam = -1;

        // Normalize coords (ImPlot -> image space)
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

        // Get or create frame annotation
        auto &fa = get_or_create_frame(amap, frame, num_nodes, num_cameras);
        fa.category_id  = state.current_class;
        fa.instance_id  = state.current_instance;

        if (cam_idx < (int)fa.cameras.size()) {
            auto &ext = fa.cameras[cam_idx].get_extras();
            ext.bbox_x = bbox_x;
            ext.bbox_y = bbox_y;
            ext.bbox_w = bbox_w;
            ext.bbox_h = bbox_h;
            ext.has_bbox = true;
        }
    }

    // Hover detection
    state.hovered = false;
    state.hovered_cam = -1;
    auto it = amap.find(frame);
    if (it != amap.end() && !it->second.empty()) {
        const auto &fa = it->second.front();
        if (cam_idx < (int)fa.cameras.size()) {
            const auto &cam = fa.cameras[cam_idx];
            if (cam.has_bbox()) {
                double plot_y = img_h - cam.extras->bbox_y - cam.extras->bbox_h; // bottom in plot
                if (mx >= cam.extras->bbox_x && mx <= cam.extras->bbox_x + cam.extras->bbox_w &&
                    my >= plot_y && my <= plot_y + cam.extras->bbox_h) {
                    state.hovered = true;
                    state.hovered_cam = cam_idx;
                }
            }
        }
    }

    // Keyboard shortcuts below must not fire while typing in a text field.
    if (ImGui::GetIO().WantTextInput) return;

    // F key: delete hovered bbox from this camera
    if (state.hovered && ImGui::IsKeyPressed(ImGuiKey_F)) {
        auto &fis = amap[frame];
        if (!fis.empty() && cam_idx < (int)fis.front().cameras.size()) {
            auto &fa = fis.front();
            fa.cameras[cam_idx].get_extras().has_bbox = false;
        }
        state.hovered = false;
    }

    // O key: delete all bboxes from ALL cameras on this frame
    if (state.hovered && ImGui::IsKeyPressed(ImGuiKey_O)) {
        auto it2 = amap.find(frame);
        if (it2 != amap.end()) {
            // "All boxes on this frame" means every camera of every animal.
            for (auto &fa : it2->second)
              for (auto &cam : fa.cameras) {
                if (cam.has_bbox())
                    cam.get_extras().has_bbox = false;
            }
        }
        state.hovered = false;
    }

    // Z/X: switch class
    const bool have_classes = !state.class_names.empty();
    if (have_classes && ImGui::IsKeyPressed(ImGuiKey_Z)) {
        state.current_class = (state.current_class - 1 + (int)state.class_names.size())
                              % (int)state.class_names.size();
        state.current_instance = 0;
    }
    if (have_classes && ImGui::IsKeyPressed(ImGuiKey_X)) {
        state.current_class = (state.current_class + 1) % (int)state.class_names.size();
        state.current_instance = 0;
    }
    // N key: create new class
    if (ImGui::IsKeyPressed(ImGuiKey_N)) {
        state.add_class();
    }
    // C/V: switch instance ID
    if (ImGui::IsKeyPressed(ImGuiKey_C) && state.current_instance > 0)
        state.current_instance--;
    if (ImGui::IsKeyPressed(ImGuiKey_V))
        state.current_instance++;
}

// Settings panel for the bbox tool
inline void DrawBBoxToolWindow(BBoxToolState &state, AppContext &ctx) {
    DrawPanel("Bbox Tool", state.show,
        [&]() {
        ImGui::Checkbox("Enable Bbox Drawing", &state.enabled);
        ImGui::Checkbox("Show IDs", &state.show_ids);

        ImGui::Separator();
        if (state.class_names.empty())
            ImGui::TextDisabled("Class: none yet -- the first box adds Class_1");
        else
            ImGui::Text("Class: %s (%d)",
                        state.class_names[state.current_class].c_str(),
                        state.current_class);
        ImGui::Text("Instance: %d", state.current_instance);

        ImGui::Separator();
        ImGui::TextWrapped("Shift+drag: draw bbox (Esc cancels)");
        ImGui::TextWrapped("F: delete hovered bbox (this camera)");
        ImGui::TextWrapped("O: delete hovered class (all cameras)");
        ImGui::TextWrapped("Z/X: prev/next class, N: new class");
        ImGui::TextWrapped("C/V: prev/next instance ID");

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
        if (ImGui::Button("+ Add Class")) state.add_class();
        },
        nullptr, ImVec2(300, 350));
}
