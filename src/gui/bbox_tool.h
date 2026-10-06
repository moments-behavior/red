#pragma once
// bbox_tool.h — Axis-aligned bounding box labeling tool
//
// Shift+drag draws a new bbox: press with Shift held, drag, let go. Bboxes are
// stored in the unified AnnotationMap (CameraAnnotation extras), on the
// instance being edited. One class for now: tailcycle, which red exports to,
// has no class column, so a second class would not survive it. The labels
// keep a class per instance (0) so classes can come back. The class list is
// pm.annotation_config.class_names, saved with the labels (annotations.json).

#include "imgui.h"
#include "implot.h"
#include "annotation.h"
#include "app_context.h"
#include "gui/gui_keypoints.h"   // keypoint_hovered_now: F is shared
#include "gui/panel.h"
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

struct BBoxToolState {
    bool show = false;
    bool enabled = false; // master toggle for bbox drawing mode

    bool show_ids = true;

    // Drawing state
    bool drawing = false;       // currently dragging out a new bbox
    int drawing_cam = -1;       // the camera view it is being drawn in
    double start_x = 0, start_y = 0;

    // Hover state: the box under the pointer, by camera and animal (index
    // into the frame's instances).
    bool hovered = false;
    int hovered_cam = -1;
    int hovered_instance = -1;
};

// Boxes are drawn in one colour; the hovered one at full strength.
inline ImVec4 box_color() { return ImVec4(0.3f, 1.0f, 1.0f, 1.0f); }

// The instance a new box or OBB of class `cat` goes on, made the one being
// edited (active_instance, an index into the frame's instances):
//   - the instance being edited, if it is of that class;
//   - else that class's instance with the same id, if the frame has it;
//   - else a new instance of that class, with its next free id.
inline FrameAnnotation &box_target(AnnotationMap &amap, u32 frame, int cat,
                                   int &active_instance, int num_nodes,
                                   int num_cameras) {
    FrameInstances &fis = amap[frame];
    int id = 0;
    if (!fis.empty()) {
        FrameAnnotation &editing = instance_or_first(fis, active_instance);
        if (editing.category_id == cat) return editing;
        id = find_instance(fis, editing.instance_id, cat)
                 ? editing.instance_id
                 : next_free_instance_id(fis, cat);
    }
    FrameAnnotation &fa = get_or_create_frame(amap, frame, num_nodes,
                                              num_cameras, id, cat);
    active_instance = (int)(&fa - fis.data());
    return fa;
}

// Keys for the box tools: plain presses only, so Cmd/Ctrl shortcuts that
// share the letter do not also fire them.
inline bool box_key(ImGuiKey k) {
    const ImGuiIO &io = ImGui::GetIO();
    return !io.WantTextInput && !io.KeyCtrl && !io.KeySuper && !io.KeyAlt &&
           ImGui::IsKeyPressed(k, false);
}

// Draw bbox rectangles on a camera's ImPlot view: every animal's.
inline void bbox_draw_overlays(const BBoxToolState &state,
                               const AnnotationMap &amap, u32 frame,
                               int cam_idx, int img_w, int img_h) {
    (void)img_w;
    // Draw in-progress bbox (while shift-dragging), in its own view only
    if (state.drawing && cam_idx == state.drawing_cam) {
        ImPlotPoint mouse = ImPlot::GetPlotMousePos();
        double dxs[] = {state.start_x, mouse.x, mouse.x, state.start_x, state.start_x};
        double dys[] = {state.start_y, state.start_y, mouse.y, mouse.y, state.start_y};
        ImPlotSpec nspec;
        nspec.LineColor = box_color();
        ImPlot::PlotLine("##bbox_new", dxs, dys, 5, nspec);
    }

    auto it = amap.find(frame);
    if (it == amap.end()) return;
    const FrameInstances &fis = it->second;
    for (size_t inst = 0; inst < fis.size(); ++inst) {
        const FrameAnnotation &fa = fis[inst];
        if (cam_idx >= (int)fa.cameras.size()) continue;
        const auto &cam = fa.cameras[cam_idx];
        if (!cam.has_bbox()) continue;

        ImVec4 color = box_color();
        const bool hot = state.hovered && cam_idx == state.hovered_cam &&
                         (int)inst == state.hovered_instance;
        if (!hot) color.w *= 0.6f;

        double x1 = cam.extras->bbox_x;
        double y1_img = cam.extras->bbox_y; // top-left in image coords
        double x2 = x1 + cam.extras->bbox_w;
        double y2_img = y1_img + cam.extras->bbox_h;
        // Convert to ImPlot coords (Y is flipped: ImPlot y = img_h - img_y)
        double y1_plot = img_h - y2_img;
        double y2_plot = img_h - y1_img;

        double xs[] = {x1, x2, x2, x1, x1};
        double ys[] = {y1_plot, y1_plot, y2_plot, y2_plot, y1_plot};
        ImPlotSpec bspec;
        bspec.LineColor = color;
        bspec.FillColor = ImVec4(color.x, color.y, color.z, 0.15f);
        ImGui::PushID((int)inst);
        ImPlot::PlotLine("##bbox", xs, ys, 5, bspec);
        ImGui::PopID();

        if (state.show_ids) {
            char label[32];
            snprintf(label, sizeof(label), "#%d", fa.instance_id);
            ImPlot::PlotText(label, x1 + 4, y2_plot - 4);
        }
    }
}

// True while the camera views must not pan: a left drag with Shift held is
// drawing a box, not moving the image (ImPlot pans on left drag whatever the
// modifiers).
inline bool bbox_blocks_pan(const BBoxToolState &state) {
    return state.enabled && (state.drawing || ImGui::GetIO().KeyShift);
}

// The pointer while a box can be drawn (Shift over a view) or is being drawn:
// crosshairs with a small box beside them, in the box colour. Replaces
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
    const ImU32 col = ImGui::GetColorU32(box_color());
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

// Handle bbox input on a focused camera view. Where a box goes: box_target.
inline void bbox_handle_input(BBoxToolState &state, AnnotationMap &amap, u32 frame, int cam_idx,
                              int &active_instance, int num_nodes,
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

        auto &fa = box_target(amap, frame, /*class*/ 0, active_instance,
                              num_nodes, num_cameras);
        if (cam_idx < (int)fa.cameras.size()) {
            auto &ext = fa.cameras[cam_idx].get_extras();
            ext.bbox_x = x1;
            ext.bbox_y = img_h - y2_plot; // top-left in image coords (Y-flip)
            ext.bbox_w = x2 - x1;
            ext.bbox_h = y2_plot - y1_plot;
            ext.has_bbox = true;
        }
        return;
    }
    if (state.drawing) return;

    // Hover: the smallest box under the pointer, over every animal, so a box
    // inside another can still be reached.
    state.hovered = false;
    state.hovered_cam = -1;
    state.hovered_instance = -1;
    auto it = amap.find(frame);
    if (it != amap.end()) {
        double best_area = 0;
        for (size_t inst = 0; inst < it->second.size(); ++inst) {
            const auto &fa = it->second[inst];
            if (cam_idx >= (int)fa.cameras.size()) continue;
            const auto &cam = fa.cameras[cam_idx];
            if (!cam.has_bbox()) continue;
            const auto &e = *cam.extras;
            double plot_y = img_h - e.bbox_y - e.bbox_h; // bottom in plot
            if (mx < e.bbox_x || mx > e.bbox_x + e.bbox_w || my < plot_y ||
                my > plot_y + e.bbox_h)
                continue;
            const double area = e.bbox_w * e.bbox_h;
            if (!state.hovered || area < best_area) {
                state.hovered = true;
                state.hovered_cam = cam_idx;
                state.hovered_instance = (int)inst;
                best_area = area;
            }
        }
    }

    if (state.hovered) {
        auto &fa = it->second[(size_t)state.hovered_instance];
        // F: delete the hovered box on this camera. A keypoint under the
        // pointer keeps F for itself (delete it from all views).
        if (!keypoint_hovered_now() && box_key(ImGuiKey_F)) {
            fa.cameras[cam_idx].get_extras().has_bbox = false;
            state.hovered = false;
        }
        // O: delete that instance's box on every camera.
        else if (box_key(ImGuiKey_O)) {
            for (auto &cam : fa.cameras)
                if (cam.has_bbox()) cam.get_extras().has_bbox = false;
            state.hovered = false;
        }
    }
}

// Settings panel for the bbox tool
inline void DrawBBoxToolWindow(BBoxToolState &state, AppContext &ctx) {
    (void)ctx;
    DrawPanel("Bbox Tool", state.show,
        [&]() {
        ImGui::Checkbox("Enable Bbox Drawing", &state.enabled);
        ImGui::Checkbox("Show IDs", &state.show_ids);

        ImGui::Separator();
        ImGui::TextWrapped("Shift+drag: draw a box on the instance being "
                           "edited (N: next instance). Esc cancels.");
        ImGui::TextWrapped("F: delete the hovered box (this camera)");
        ImGui::TextWrapped("O: delete the hovered instance's box (all cameras)");
        },
        nullptr, ImVec2(300, 350));
}
