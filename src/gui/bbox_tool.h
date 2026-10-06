#pragma once
// bbox_tool.h — Axis-aligned bounding box labeling tool
//
// Shift+drag draws a new bbox: press with Shift held, drag, let go. Bboxes are
// stored in the unified AnnotationMap (CameraAnnotation extras), on the
// instance being edited, which takes the box's class (YOLO needs one per box;
// ids stay unique across classes, so the CSVs need no class column). The class list is
// pm.annotation_config.label_info, saved with the labels (annotations.json).

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

    // Moving a box's edges: the edge(s) under the pointer (or being dragged),
    // as BoxEdge bits, on which camera and instance.
    int edge_mask = 0;
    int edge_cam = -1;
    int edge_instance = -1;
    bool resizing = false;      // dragging edge_mask's edges
    double grab_dx = 0, grab_dy = 0;   // kEdgeMove: pointer to the box's top-left

    // Right-click menu on a box: which camera, frame and instance it is for.
    bool open_menu = false;     // open it this frame (in that camera's plot)
    int menu_cam = -1;
    u32 menu_frame = 0;
    int menu_instance = -1;
};

// A box's edges, as bits, in screen terms (T is the edge drawn on top);
// kEdgeMove is all of them at once -- the box moved whole, by its label.
enum BoxEdge { kEdgeL = 1, kEdgeR = 2, kEdgeT = 4, kEdgeB = 8, kEdgeMove = 16 };

// Where a box's label is drawn (plot coords; ImPlot centres text on it).
inline ImPlotPoint box_label_anchor(double x1, double top) {
    return ImPlotPoint(x1 + 4, top - 4);
}

// The resize cursor for a set of edges: <> for a side, up-down for top or
// bottom, a diagonal at a corner.
inline ImGuiMouseCursor box_edge_cursor(int mask) {
    if (mask & kEdgeMove) return ImGuiMouseCursor_ResizeAll;
    const bool h = mask & (kEdgeL | kEdgeR), v = mask & (kEdgeT | kEdgeB);
    if (h && v)
        return ((mask & kEdgeL) && (mask & kEdgeT)) || ((mask & kEdgeR) && (mask & kEdgeB))
                   ? ImGuiMouseCursor_ResizeNWSE
                   : ImGuiMouseCursor_ResizeNESW;
    return h ? ImGuiMouseCursor_ResizeEW : ImGuiMouseCursor_ResizeNS;
}

// Whether camera view cam shows the resize cursor: ImPlot's crosshairs hide
// the system cursor, so the view leaves them off then (set at BeginPlot, from
// the frame before).
inline bool bbox_shows_resize_cursor(const BBoxToolState &state, int cam) {
    return state.enabled && state.edge_mask && state.edge_cam == cam;
}

// A class's colour, the same every session: the first cyan, the rest spread
// round the hue circle by the golden ratio.
inline ImVec4 default_box_class_color(int i) {
    if (i <= 0) return ImVec4(0.3f, 1.0f, 1.0f, 1.0f);
    float hue = i * 0.618033f;
    hue -= std::floor(hue);
    return (ImVec4)ImColor::HSV(hue, 0.85f, 0.95f);
}

// A class's colour: the one picked for it, else its default.
inline ImVec4 box_class_color(const LabelInfo &classes, int i) {
    if (i >= 0 && classes.has_color((size_t)i)) {
        const auto &c = classes.colors[(size_t)i];
        return ImVec4(c[0], c[1], c[2], 1.0f);
    }
    return default_box_class_color(i);
}

// The colour a box is drawn in: with one class, its instance's (as in the
// Instances list) so instances tell apart; with several, its class's.
// inst_index is the instance's place in the frame's list.
inline ImVec4 box_draw_color(const LabelInfo &classes, int category,
                             int instance_id, int inst_index) {
    return classes.names.size() > 1
               ? box_class_color(classes, category)
               : instance_color(classes, instance_id, inst_index);
}

// The colour of a box about to be drawn on camera cam_idx: the selected
// class's, or with one class the colour of the instance it will go on -- the
// one being edited, or the next if that one already has a box here (as
// box_target decides).
inline ImVec4 new_box_color(const BBoxToolState &state, const LabelInfo &classes,
                            const AnnotationMap &amap, u32 frame, int cam_idx,
                            int active_instance) {
    auto it = amap.find(frame);
    if (it == amap.end() || it->second.empty())
        return box_draw_color(classes, state.current_class, 0, 0);
    const FrameInstances &fis = it->second;
    const int idx = active_instance > 0 && active_instance < (int)fis.size()
                        ? active_instance : 0;
    const FrameAnnotation &editing = fis[(size_t)idx];
    if (cam_idx < (int)editing.cameras.size() &&
        editing.cameras[(size_t)cam_idx].has_bbox()) {
        int next_id = 0;
        for (const auto &fa : fis) next_id = std::max(next_id, fa.instance_id + 1);
        return box_draw_color(classes, state.current_class, next_id, (int)fis.size());
    }
    return box_draw_color(classes, state.current_class, editing.instance_id, idx);
}

// A box's label: the instance's name ("#1" until named), and its class once
// there is more than one to tell apart ("rat #1").
inline std::string box_label(const LabelInfo &classes, const FrameAnnotation &fa) {
    std::string s = classes.instance_name(fa.instance_id);
    if (classes.names.size() > 1) {
        const int ci = fa.category_id;
        s = (ci >= 0 && ci < (int)classes.names.size() ? classes.names[(size_t)ci]
                                                       : std::string("?")) +
            " " + s;
    }
    return s;
}

// Adds Class_<n>, n its index, to the list and makes it current.
inline void add_box_class(BBoxToolState &state, LabelInfo &classes) {
    state.current_class = (int)classes.names.size();
    classes.names.push_back("Class_" + std::to_string(classes.names.size()));
}

// The class a new box gets: the current one, after making Class_0 if the list
// is empty (a new project's is).
inline int box_class_for_new(BBoxToolState &state, LabelInfo &classes) {
    if (classes.names.empty()) add_box_class(state, classes);
    state.current_class = std::clamp(state.current_class, 0, (int)classes.names.size() - 1);
    return state.current_class;
}

// The instance a new box goes on, made the one being edited (active_instance,
// an index into the frame's list): the one being edited, unless it already
// has this kind of box on this camera -- then the next instance, with the
// next unused id, so drawing box after box labels instance after instance.
// A frame with none gets a first instance. `has` says whether an instance
// already has the box on that camera (an axis-aligned box or an OBB).
template <typename HasBox>
inline FrameAnnotation &box_target(AnnotationMap &amap, u32 frame, int cam_idx,
                                   int &active_instance, int num_nodes,
                                   int num_cameras, HasBox has) {
    auto it = amap.find(frame);
    if (it == amap.end() || it->second.empty()) {
        active_instance = 0;
        return get_or_create_frame(amap, frame, num_nodes, num_cameras);
    }
    FrameInstances &fis = it->second;
    if (active_instance < 0 || active_instance >= (int)fis.size()) active_instance = 0;
    FrameAnnotation &editing = fis[(size_t)active_instance];
    if (cam_idx >= (int)editing.cameras.size() || !has(editing.cameras[(size_t)cam_idx]))
        return editing;
    int next_id = 0;
    for (const auto &fa : fis) next_id = std::max(next_id, fa.instance_id + 1);
    FrameAnnotation &fresh = get_or_create_frame(amap, frame, num_nodes,
                                                 num_cameras, next_id);
    active_instance = (int)fis.size() - 1;
    return fresh;
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
                               const LabelInfo &classes,
                               const AnnotationMap &amap, u32 frame,
                               int cam_idx, int active_instance,
                               int img_w, int img_h) {
    (void)img_w;
    // Draw in-progress bbox (while shift-dragging), in its own view only
    if (state.drawing && cam_idx == state.drawing_cam) {
        ImPlotPoint mouse = ImPlot::GetPlotMousePos();
        double dxs[] = {state.start_x, mouse.x, mouse.x, state.start_x, state.start_x};
        double dys[] = {state.start_y, state.start_y, mouse.y, mouse.y, state.start_y};
        ImPlotSpec nspec;
        nspec.LineColor =
            new_box_color(state, classes, amap, frame, cam_idx, active_instance);
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
        ImVec4 color = box_draw_color(classes, ci, fa.instance_id, (int)inst);
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
            const ImPlotPoint at = box_label_anchor(x1, y2_plot);
            ImPlot::PlotText(box_label(classes, fa).c_str(), at.x, at.y);
        }
    }
}

// True while the camera views must not pan: a left drag with Shift held is
// drawing a box, not moving the image (ImPlot pans on left drag whatever the
// modifiers).
inline bool bbox_blocks_pan(const BBoxToolState &state) {
    return state.enabled && (state.drawing || state.resizing || state.edge_mask ||
                             ImGui::GetIO().KeyShift);
}

// The pointer while a box can be drawn (Shift over a view) or is being drawn:
// crosshairs with a small box beside them, in the class colour. Replaces
// ImPlot's "not allowed" cursor, which it shows for a drag on the views
// bbox_blocks_pan locks. Call inside the camera's plot.
inline void bbox_draw_cursor(const BBoxToolState &state,
                             const LabelInfo &classes, const AnnotationMap &amap,
                             u32 frame, int cam_idx, int active_instance) {
    if (!state.enabled) return;
    const bool here = state.drawing ? cam_idx == state.drawing_cam
                                    : ImGui::GetIO().KeyShift && ImPlot::IsPlotHovered();
    if (!here) return;
    ImGui::SetMouseCursor(ImGuiMouseCursor_None);
    ImDrawList *dl = ImPlot::GetPlotDrawList();
    const ImVec2 m = ImGui::GetIO().MousePos;
    const ImVec2 lo = ImPlot::GetPlotPos();
    const ImVec2 hi(lo.x + ImPlot::GetPlotSize().x, lo.y + ImPlot::GetPlotSize().y);
    const ImU32 col = ImGui::GetColorU32(
        new_box_color(state, classes, amap, frame, cam_idx, active_instance));
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
                              LabelInfo &classes,
                              AnnotationMap &amap, u32 frame, int cam_idx,
                              int &active_instance, int num_nodes,
                              int num_cameras, int img_w, int img_h) {
    if (!state.enabled) return;
    // A drag belongs to the view it started in; the others leave it alone.
    if (state.drawing && cam_idx != state.drawing_cam) return;
    if (state.resizing && cam_idx != state.edge_cam) return;
    // This view's edge hover is found afresh below (or not at all, if the
    // pointer has left it).
    if (!state.resizing && state.edge_cam == cam_idx) {
        state.edge_mask = 0;
        state.edge_cam = -1;
    }
    if (!state.drawing && !state.resizing && !ImPlot::IsPlotHovered()) return;

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
        auto &fa = box_target(amap, frame, cam_idx, active_instance, num_nodes,
                              num_cameras,
                              [](const CameraAnnotation &c) { return c.has_bbox(); });
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

    // Moving edges: follow the pointer until the button is let go.
    if (state.resizing) {
        ImGui::SetMouseCursor(box_edge_cursor(state.edge_mask));
        auto it = amap.find(frame);
        FrameAnnotation *fa = it != amap.end() && state.edge_instance >= 0 &&
                                      state.edge_instance < (int)it->second.size()
                                  ? &it->second[(size_t)state.edge_instance]
                                  : nullptr;
        if (!fa || cam_idx >= (int)fa->cameras.size() ||
            !fa->cameras[cam_idx].has_bbox()) {
            state.resizing = false;
            return;
        }
        auto &e = fa->cameras[cam_idx].get_extras();
        // Plot coords: y up, so the edge drawn on top is the larger y.
        double l = e.bbox_x, r = e.bbox_x + e.bbox_w;
        double b = img_h - e.bbox_y - e.bbox_h, t = img_h - e.bbox_y;
        int &m = state.edge_mask;
        if (m & kEdgeMove) {   // whole box, keeping its size and the grab point
            const double w = r - l, h = t - b;
            l = std::clamp(mx - state.grab_dx, 0.0, std::max(0.0, img_w - w));
            t = std::clamp(my + state.grab_dy, h, (double)img_h);
            r = l + w;
            b = t - h;
            m = kEdgeMove;
        }
        if (m & kEdgeL) l = mx;
        if (m & kEdgeR) r = mx;
        if (m & kEdgeB) b = my;
        if (m & kEdgeT) t = my;
        // Dragged past the opposite edge: carry on as that edge.
        if (l > r) { std::swap(l, r); if (m & (kEdgeL | kEdgeR)) m ^= kEdgeL | kEdgeR; }
        if (b > t) { std::swap(b, t); if (m & (kEdgeT | kEdgeB)) m ^= kEdgeT | kEdgeB; }
        e.bbox_x = l;
        e.bbox_w = std::max(r - l, 1.0);
        e.bbox_y = img_h - t;
        e.bbox_h = std::max(t - b, 1.0);
        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            state.resizing = false;
            state.edge_mask = 0;
            state.edge_cam = -1;
        }
        return;
    }

    // An edge under the pointer (within a few pixels; a corner when near two),
    // or the box's label: the resize / move cursor, and a press starts moving
    // it. Not with Shift, which draws a new box, nor over a keypoint, which
    // drags the keypoint.
    if (!ImGui::GetIO().KeyShift && !keypoint_hovered_now()) {
        auto it = amap.find(frame);
        if (it != amap.end()) {
            const ImVec2 mp = ImGui::GetIO().MousePos;
            const float tol = 5.0f;
            double best_area = 0;
            for (size_t inst = 0; inst < it->second.size(); ++inst) {
                const auto &fa = it->second[inst];
                if (cam_idx >= (int)fa.cameras.size() || !fa.cameras[cam_idx].has_bbox())
                    continue;
                const auto &e = *fa.cameras[cam_idx].extras;
                const ImVec2 tl = ImPlot::PlotToPixels(e.bbox_x, img_h - e.bbox_y);
                const ImVec2 br = ImPlot::PlotToPixels(e.bbox_x + e.bbox_w,
                                                       img_h - e.bbox_y - e.bbox_h);
                if (mp.x < tl.x - tol || mp.x > br.x + tol || mp.y < tl.y - tol ||
                    mp.y > br.y + tol)
                    continue;
                int mask = 0;
                if (std::fabs(mp.x - tl.x) <= tol) mask |= kEdgeL;
                else if (std::fabs(mp.x - br.x) <= tol) mask |= kEdgeR;
                if (std::fabs(mp.y - tl.y) <= tol) mask |= kEdgeT;
                else if (std::fabs(mp.y - br.y) <= tol) mask |= kEdgeB;
                // The label moves the box whole; it wins over the edges.
                if (state.show_ids) {
                    const ImPlotPoint a = box_label_anchor(e.bbox_x, img_h - e.bbox_y);
                    const ImVec2 c = ImPlot::PlotToPixels(a.x, a.y);
                    const ImVec2 ts = ImGui::CalcTextSize(
                        box_label(classes, fa).c_str());
                    if (std::fabs(mp.x - c.x) <= ts.x * 0.5f + 2 &&
                        std::fabs(mp.y - c.y) <= ts.y * 0.5f + 2)
                        mask = kEdgeMove;
                }
                if (!mask) continue;
                const double area = e.bbox_w * e.bbox_h;
                if (!state.edge_mask || area < best_area) {
                    state.edge_mask = mask;
                    state.edge_cam = cam_idx;
                    state.edge_instance = (int)inst;
                    best_area = area;
                }
            }
        }
        if (state.edge_mask) {
            ImGui::SetMouseCursor(box_edge_cursor(state.edge_mask));
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                state.resizing = true;
                active_instance = state.edge_instance;   // the box being shaped
                const auto &e = *it->second[(size_t)state.edge_instance]
                                     .cameras[cam_idx].extras;
                state.grab_dx = mx - e.bbox_x;              // pointer - left
                state.grab_dy = (img_h - e.bbox_y) - my;    // top - pointer
                return;
            }
        }
    }

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
        // Right-click: the box's menu (a keypoint keeps its own).
        if (!keypoint_hovered_now() && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
            state.open_menu = true;
            state.menu_cam = cam_idx;
            state.menu_frame = frame;
            state.menu_instance = state.hovered_instance;
        }
        auto &fa = it->second[(size_t)state.hovered_instance];
        // Either way that instance becomes the one being edited, ready to
        // draw its box again.
        if (!keypoint_hovered_now() && box_key(ImGuiKey_R)) {
            fa.cameras[cam_idx].get_extras().has_bbox = false;
            active_instance = state.hovered_instance;
            state.hovered = false;
        } else if (!keypoint_hovered_now() && box_key(ImGuiKey_F)) {
            for (auto &cam : fa.cameras)
                if (cam.has_bbox()) cam.get_extras().has_bbox = false;
            active_instance = state.hovered_instance;
            state.hovered = false;
        }
    }
}

// The right-click menu on a box. Call every frame inside each camera's plot,
// after bbox_handle_input (the popup lives in the plot's ID scope).
inline void bbox_draw_menu(BBoxToolState &state, LabelInfo &classes,
                           AnnotationMap &amap, u32 frame, int cam_idx,
                           int &active_instance) {
    if (cam_idx != state.menu_cam) return;
    if (state.open_menu) {
        ImGui::OpenPopup("##box_menu");
        state.open_menu = false;
    }
    if (!ImGui::BeginPopup("##box_menu")) return;
    auto it = amap.find(frame);
    const bool valid = frame == state.menu_frame && it != amap.end() &&
                       state.menu_instance >= 0 &&
                       state.menu_instance < (int)it->second.size() &&
                       cam_idx < (int)it->second[(size_t)state.menu_instance].cameras.size();
    if (!valid) {   // the frame moved on under it
        ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
        return;
    }
    FrameAnnotation &fa = it->second[(size_t)state.menu_instance];
    ImGui::TextDisabled("%s", box_label(classes, fa).c_str());
    if (ImGui::MenuItem("Edit this instance", nullptr,
                        active_instance == state.menu_instance))
        active_instance = state.menu_instance;
    if (classes.names.size() > 1 && ImGui::BeginMenu("Class")) {
        for (int c = 0; c < (int)classes.names.size(); ++c) {
            ImGui::PushID(c);
            if (ImGui::MenuItem(classes.names[(size_t)c].c_str(), nullptr,
                                fa.category_id == c))
                fa.category_id = c;
            ImGui::PopID();
        }
        ImGui::EndMenu();
    }
    ImGui::Separator();
    if (ImGui::MenuItem("Delete box", "R")) {
        fa.cameras[(size_t)cam_idx].get_extras().has_bbox = false;
        active_instance = state.menu_instance;
    }
    if (ImGui::MenuItem("Delete on all cameras", "F")) {
        for (auto &cam : fa.cameras)
            if (cam.has_bbox()) cam.get_extras().has_bbox = false;
        active_instance = state.menu_instance;
    }
    ImGui::EndPopup();
}

// Settings panel for the bbox tool
inline void DrawBBoxToolWindow(BBoxToolState &state, AppContext &ctx) {
    DrawPanel("Bbox Tool", state.show,
        [&]() {
        auto &classes = ctx.pm.annotation_config.label_info;
        ImGui::Checkbox("Enable Bbox Drawing", &state.enabled);
        ImGui::Checkbox("Show IDs", &state.show_ids);

        ImGui::Separator();
        ImGui::TextWrapped("Shift+drag: draw a box of the selected class on "
                           "the instance being edited; if it already has a "
                           "box on this camera, the box starts the next "
                           "instance. Esc cancels.");
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
