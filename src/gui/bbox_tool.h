#pragma once
// bbox_tool.h — Axis-aligned bounding box labeling tool
//
// Shift+drag draws a new bbox: press with Shift held, drag, let go. Bboxes are
// stored in the unified AnnotationMap (CameraAnnotation extras), on the
// instance being edited, which takes the box's class (YOLO needs one per box;
// ids stay unique across classes, so the CSVs need no class column). The class list is
// pm.annotation_config.box_classes, saved with the labels (annotations.json).

#include "imgui.h"
#include "implot.h"
#include "annotation.h"
#include "app_context.h"
#include "gui/gui_keypoints.h"   // keypoint_hovered_now: F is shared
#include "gui/panel.h"
#include <misc/cpp/imgui_stdlib.h>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

struct BBoxToolState {
    bool show = false;
    bool enabled = true;  // Shift+drag draws boxes; on by default

    int current_class = 0;      // index into the project's class list
    bool show_ids = true;
    int editing_class = -1;     // class whose name is being edited in the panel
    bool focus_edit = false;    // put the cursor in that name field next frame

    // Drawing state
    bool drawing = false;       // currently dragging out a new bbox
    int drawing_cam = -1;       // the camera view it is being drawn in
    double start_x = 0, start_y = 0;

    // Hover state: the box under the pointer, by camera and instance (index
    // into the frame's instances).
    bool hovered = false;
    int hovered_cam = -1;
    int hovered_instance = -1;
};

// A class's colour, the same every session: the first cyan, the rest spread
// round the hue circle by the golden ratio.
inline ImVec4 default_box_class_color(int i) {
    if (i <= 0) return ImVec4(0.3f, 1.0f, 1.0f, 1.0f);
    float hue = i * 0.618033f;
    hue -= std::floor(hue);
    return (ImVec4)ImColor::HSV(hue, 0.85f, 0.95f);
}

// A class's colour: the one picked for it, else its default.
inline ImVec4 box_class_color(const BoxClasses &classes, int i) {
    if (i >= 0 && classes.has_color((size_t)i)) {
        const auto &c = classes.colors[(size_t)i];
        return ImVec4(c[0], c[1], c[2], 1.0f);
    }
    return default_box_class_color(i);
}

inline const char *box_class_name(const BoxClasses &classes, int i) {
    return i >= 0 && i < (int)classes.names.size() ? classes.names[i].c_str() : "?";
}

// Adds Class_<n>, n its index, to the list and makes it current.
inline void add_box_class(BBoxToolState &state, BoxClasses &classes) {
    state.current_class = (int)classes.names.size();
    classes.names.push_back("Class_" + std::to_string(classes.names.size()));
}

// The class a new box gets: the current one, after making Class_0 if the list
// is empty (a new project's is).
inline int box_class_for_new(BBoxToolState &state, BoxClasses &classes) {
    if (classes.names.empty()) add_box_class(state, classes);
    state.current_class = std::clamp(state.current_class, 0, (int)classes.names.size() - 1);
    return state.current_class;
}

// The instance a new box goes on: the one being edited, or a first instance
// on a frame that has none.
inline FrameAnnotation &box_target(AnnotationMap &amap, u32 frame,
                                   int active_instance, int num_nodes,
                                   int num_cameras) {
    auto it = amap.find(frame);
    if (it != amap.end() && !it->second.empty())
        return instance_or_first(it->second, active_instance);
    return get_or_create_frame(amap, frame, num_nodes, num_cameras);
}

// Keys for the box tools: plain presses only, so Cmd/Ctrl shortcuts that
// share the letter do not also fire them.
inline bool box_key(ImGuiKey k) {
    const ImGuiIO &io = ImGui::GetIO();
    return !io.WantTextInput && !io.KeyCtrl && !io.KeySuper && !io.KeyAlt &&
           ImGui::IsKeyPressed(k, false);
}

// Draw bbox rectangles on a camera's ImPlot view: every instance's.
inline void bbox_draw_overlays(const BBoxToolState &state,
                               const BoxClasses &classes,
                               const AnnotationMap &amap, u32 frame,
                               int cam_idx, int img_w, int img_h) {
    (void)img_w;
    // Draw in-progress bbox (while shift-dragging), in its own view only
    if (state.drawing && cam_idx == state.drawing_cam) {
        ImPlotPoint mouse = ImPlot::GetPlotMousePos();
        double dxs[] = {state.start_x, mouse.x, mouse.x, state.start_x, state.start_x};
        double dys[] = {state.start_y, state.start_y, mouse.y, mouse.y, state.start_y};
        ImPlotSpec nspec;
        nspec.LineColor = box_class_color(classes, state.current_class);
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

        const int ci = fa.category_id;
        ImVec4 color = box_class_color(classes, ci);
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
            char label[96];
            // Class, then which one of that class: "Class_0 #1".
            snprintf(label, sizeof(label), "%s #%d",
                     box_class_name(classes, ci), fa.instance_id);
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
// crosshairs with a small box beside them, in the class colour. Replaces
// ImPlot's "not allowed" cursor, which it shows for a drag on the views
// bbox_blocks_pan locks. Call inside the camera's plot.
inline void bbox_draw_cursor(const BBoxToolState &state,
                             const BoxClasses &classes, int cam_idx) {
    if (!state.enabled) return;
    const bool here = state.drawing ? cam_idx == state.drawing_cam
                                    : ImGui::GetIO().KeyShift && ImPlot::IsPlotHovered();
    if (!here) return;
    ImGui::SetMouseCursor(ImGuiMouseCursor_None);
    ImDrawList *dl = ImPlot::GetPlotDrawList();
    const ImVec2 m = ImGui::GetIO().MousePos;
    const ImVec2 lo = ImPlot::GetPlotPos();
    const ImVec2 hi(lo.x + ImPlot::GetPlotSize().x, lo.y + ImPlot::GetPlotSize().y);
    const ImU32 col = ImGui::GetColorU32(box_class_color(classes, state.current_class));
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
inline void bbox_handle_input(BBoxToolState &state,
                              BoxClasses &classes,
                              AnnotationMap &amap, u32 frame, int cam_idx,
                              int active_instance, int num_nodes,
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

        const int cat = box_class_for_new(state, classes);
        auto &fa = box_target(amap, frame, active_instance, num_nodes,
                              num_cameras);
        fa.category_id = cat;
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

    // Hover: the smallest box under the pointer, over every instance, so a box
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

    // R / F, as for keypoints: the hovered box on this camera / that
    // instance's box on every camera. A keypoint under the pointer keeps
    // both keys for itself.
    if (state.hovered) {
        mark_box_hovered();
        auto &fa = it->second[(size_t)state.hovered_instance];
        if (!keypoint_hovered_now() && box_key(ImGuiKey_R)) {
            fa.cameras[cam_idx].get_extras().has_bbox = false;
            state.hovered = false;
        } else if (!keypoint_hovered_now() && box_key(ImGuiKey_F)) {
            for (auto &cam : fa.cameras)
                if (cam.has_bbox()) cam.get_extras().has_bbox = false;
            state.hovered = false;
        }
    }
}

// Settings panel for the bbox tool
inline void DrawBBoxToolWindow(BBoxToolState &state, AppContext &ctx) {
    DrawPanel("Bbox Tool", state.show,
        [&]() {
        auto &classes = ctx.pm.annotation_config.box_classes;
        ImGui::Checkbox("Enable Bbox Drawing", &state.enabled);
        ImGui::Checkbox("Show IDs", &state.show_ids);

        ImGui::Separator();
        ImGui::TextWrapped("Shift+drag: draw a box of the selected class on "
                           "the instance being edited (N: next instance). "
                           "Esc cancels.");
        ImGui::TextWrapped("R: delete the hovered box (this camera)");
        ImGui::TextWrapped("F: delete the hovered instance's box (all cameras)");

        // Class list: saved with the labels.
        ImGui::SeparatorText("Classes");
        if (classes.names.empty())
            ImGui::TextDisabled("None yet -- the first box adds Class_0.");
        // One row per class: click to pick it, double-click to rename it in
        // place (YOLO export writes these names).
        for (int i = 0; i < (int)classes.names.size(); ++i) {
            ImGui::PushID(i);
            // Double-click the swatch to pick the class's colour.
            ImVec4 col = box_class_color(classes, i);
            ImGui::ColorButton("##clr", col, ImGuiColorEditFlags_NoTooltip,
                               ImVec2(14, 14));
            if (ImGui::IsItemHovered() &&
                ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
                ImGui::OpenPopup("##pick_color");
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal))
                ImGui::SetTooltip("Double-click to change the colour");
            if (ImGui::BeginPopup("##pick_color")) {
                if (ImGui::ColorPicker3("##picker", &col.x,
                                        ImGuiColorEditFlags_NoSidePreview |
                                            ImGuiColorEditFlags_NoSmallPreview)) {
                    if (classes.colors.size() <= (size_t)i)
                        classes.colors.resize((size_t)i + 1, {-1.f, -1.f, -1.f});
                    classes.colors[(size_t)i] = {col.x, col.y, col.z};
                }
                if (ImGui::SmallButton("Default")) {
                    if ((size_t)i < classes.colors.size())
                        classes.colors[(size_t)i] = {-1.f, -1.f, -1.f};
                    ImGui::CloseCurrentPopup();
                }
                ImGui::EndPopup();
            }
            ImGui::SameLine();
            if (state.editing_class == i) {
                ImGui::SetNextItemWidth(-FLT_MIN);
                if (state.focus_edit) {
                    ImGui::SetKeyboardFocusHere();
                    state.focus_edit = false;
                }
                ImGui::InputText("##name", &classes.names[(size_t)i],
                                 ImGuiInputTextFlags_AutoSelectAll);
                if (ImGui::IsItemDeactivated()) {
                    if (classes.names[(size_t)i].empty())   // keep a name to export
                        classes.names[(size_t)i] = "Class_" + std::to_string(i);
                    state.editing_class = -1;
                }
            } else {
                if (ImGui::Selectable(classes.names[(size_t)i].c_str(),
                                      i == state.current_class,
                                      ImGuiSelectableFlags_AllowDoubleClick)) {
                    state.current_class = i;
                    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                        state.editing_class = i;
                        state.focus_edit = true;
                    }
                }
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal))
                    ImGui::SetTooltip("Double-click to rename");
            }
            ImGui::PopID();
        }
        if (ImGui::Button("+ Add Class")) {
            add_box_class(state, classes);
            state.editing_class = state.current_class;   // name it straight away
            state.focus_edit = true;
        }
        },
        nullptr, ImVec2(300, 350));
}
