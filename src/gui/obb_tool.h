#pragma once
// obb_tool.h — Oriented bounding box labeling tool
//
// 3-click construction: axis point 1, axis point 2, perpendicular corner.
// OBBs are stored in AnnotationMap Camera2D (cx, cy, w, h, angle).
// Follows the same class/instance system as bbox_tool.h.

#include "imgui.h"
#include "implot.h"
#include "annotation.h"
#include "app_context.h"
#include "gui/panel.h"
#include "gui/bbox_tool.h" // shares class names/colors with bbox tool
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

// OBB construction state machine
enum class OBBDrawState {
    Idle,         // not drawing
    FirstPoint,   // placed axis_point1, waiting for axis_point2
    SecondPoint,  // placed axis_point2, waiting for corner
};

struct OBBToolState {
    bool show = false;
    bool enabled = false;

    // Construction state
    OBBDrawState draw_state = OBBDrawState::Idle;
    double ax1_x = 0, ax1_y = 0;   // axis point 1 (ImPlot coords)
    double ax2_x = 0, ax2_y = 0;   // axis point 2 (ImPlot coords)

    // Hover state
    int hovered_instance = -1;
    int hovered_cam = -1;
};

// Calculate OBB properties from 3 points in ImPlot coordinates.
// axis_point1, axis_point2 define the long axis; corner_point defines width.
// Returns: center, width, height, angle (radians from ImPlot X-axis).
inline void calculate_obb_from_3_points(
    double ax1_x, double ax1_y, double ax2_x, double ax2_y,
    double corner_x, double corner_y,
    double &cx, double &cy, double &w, double &h, double &angle) {

    // Edge vector along the main axis
    double ex = ax2_x - ax1_x;
    double ey = ax2_y - ax1_y;
    double axis_len = std::sqrt(ex * ex + ey * ey);
    if (axis_len < 1e-6) { cx = cy = w = h = angle = 0; return; }

    // Perpendicular vector
    double px = -ey;
    double py = ex;
    double plen = std::sqrt(px * px + py * py);
    px /= plen; py /= plen;

    // Project corner onto perp direction to get width/2
    double dx = corner_x - ax1_x;
    double dy = corner_y - ax1_y;
    double proj = dx * px + dy * py;

    w = axis_len;
    h = std::abs(proj);
    angle = std::atan2(ey, ex);

    // Center is midpoint of axis, offset by half the perpendicular extent
    double sign = (proj >= 0) ? 1.0 : -1.0;
    cx = (ax1_x + ax2_x) / 2.0 + sign * (h / 2.0) * px;
    cy = (ax1_y + ax2_y) / 2.0 + sign * (h / 2.0) * py;
}

// Get 4 corners of an OBB in ImPlot coords
inline void obb_get_corners(double cx, double cy, double w, double h,
                             double angle, double out_x[5], double out_y[5]) {
    double ca = std::cos(angle), sa = std::sin(angle);
    double hw = w / 2.0, hh = h / 2.0;

    // Local corners: (-hw,-hh), (hw,-hh), (hw,hh), (-hw,hh)
    double lx[] = {-hw, hw, hw, -hw};
    double ly[] = {-hh, -hh, hh, hh};

    for (int i = 0; i < 4; ++i) {
        out_x[i] = cx + lx[i] * ca - ly[i] * sa;
        out_y[i] = cy + lx[i] * sa + ly[i] * ca;
    }
    out_x[4] = out_x[0]; // close the loop
    out_y[4] = out_y[0];
}

// Check if point is inside an OBB (ImPlot coords)
inline bool obb_contains(double cx, double cy, double w, double h,
                          double angle, double px, double py) {
    double ca = std::cos(-angle), sa = std::sin(-angle);
    double dx = px - cx, dy = py - cy;
    double lx = dx * ca - dy * sa;
    double ly = dx * sa + dy * ca;
    return std::abs(lx) <= w / 2.0 && std::abs(ly) <= h / 2.0;
}

// Draw OBB overlays on a camera's ImPlot view
inline void obb_draw_overlays(OBBToolState &state, const BBoxToolState &bbox_state,
                               const AnnotationMap &amap, u32 frame,
                               int cam_idx, int img_w, int img_h) {
    auto it = amap.find(frame);
    if (it == amap.end()) goto draw_construction;

    for (int i = 0; i < (int)it->second.instances.size(); ++i) {
        const auto &inst = it->second.instances[i];
        if (cam_idx >= (int)inst.cameras.size()) continue;
        const auto &cam = inst.cameras[cam_idx];
        if (!cam.has_obb) continue;

        // OBB stored in image coords; convert center Y to ImPlot
        double plot_cx = cam.obb_cx;
        double plot_cy = img_h - cam.obb_cy;
        double angle = -cam.obb_angle; // flip angle for Y inversion

        int ci = inst.category_id;
        ImVec4 color = (ci < (int)bbox_state.class_colors.size())
                           ? bbox_state.class_colors[ci]
                           : ImVec4(1, 1, 1, 1);
        if (i != state.hovered_instance || cam_idx != state.hovered_cam)
            color.w *= 0.6f;

        double xs[5], ys[5];
        obb_get_corners(plot_cx, plot_cy, cam.obb_w, cam.obb_h, angle, xs, ys);

        ImPlot::PushStyleColor(ImPlotCol_Line, color);
        ImPlot::PlotLine("##obb", xs, ys, 5);
        ImPlot::PopStyleColor();

        if (bbox_state.show_ids) {
            char label[64];
            snprintf(label, sizeof(label), "%s #%d (OBB)",
                     (ci < (int)bbox_state.class_names.size())
                         ? bbox_state.class_names[ci].c_str() : "?",
                     inst.instance_id);
            ImPlot::PlotText(label, plot_cx, plot_cy);
        }
    }

draw_construction:
    // Draw construction preview
    if (state.draw_state == OBBDrawState::FirstPoint) {
        // Show axis_point1 + line to mouse
        ImPlotPoint mouse = ImPlot::GetPlotMousePos();
        double xs[] = {state.ax1_x, mouse.x};
        double ys[] = {state.ax1_y, mouse.y};
        ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1, 0.3f, 0.3f, 1));
        ImPlot::PlotLine("##obb_ax", xs, ys, 2);
        ImPlot::PopStyleColor();
        // Marker at first point
        ImPlot::PushStyleColor(ImPlotCol_MarkerFill, ImVec4(1, 0, 0, 1));
        ImPlot::PlotScatter("##obb_p1", &state.ax1_x, &state.ax1_y, 1);
        ImPlot::PopStyleColor();
    }
    else if (state.draw_state == OBBDrawState::SecondPoint) {
        // Show both axis points + axis line + rectangle preview
        ImPlotPoint mouse = ImPlot::GetPlotMousePos();

        // Axis line
        double ax_xs[] = {state.ax1_x, state.ax2_x};
        double ax_ys[] = {state.ax1_y, state.ax2_y};
        ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.3f, 1, 0.3f, 1));
        ImPlot::PlotLine("##obb_axis", ax_xs, ax_ys, 2);
        ImPlot::PopStyleColor();

        // Preview rectangle
        double cx, cy, w, h, angle;
        calculate_obb_from_3_points(state.ax1_x, state.ax1_y,
                                     state.ax2_x, state.ax2_y,
                                     mouse.x, mouse.y,
                                     cx, cy, w, h, angle);
        if (w > 0 && h > 0) {
            double xs[5], ys[5];
            obb_get_corners(cx, cy, w, h, angle, xs, ys);
            ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1, 1, 0, 0.7f));
            ImPlot::PlotLine("##obb_preview", xs, ys, 5);
            ImPlot::PopStyleColor();
        }

        // Markers
        double mx[] = {state.ax1_x, state.ax2_x};
        double my[] = {state.ax1_y, state.ax2_y};
        ImPlot::PushStyleColor(ImPlotCol_MarkerFill, ImVec4(0, 1, 0, 1));
        ImPlot::PlotScatter("##obb_pts", mx, my, 2);
        ImPlot::PopStyleColor();
    }
}

// Handle OBB input on a focused camera view
inline void obb_handle_input(OBBToolState &state, BBoxToolState &bbox_state,
                              AnnotationMap &amap, u32 frame, int cam_idx,
                              int num_nodes, int num_cameras,
                              int img_w, int img_h) {
    if (!state.enabled) return;
    if (!ImPlot::IsPlotHovered()) return;

    ImPlotPoint mouse = ImPlot::GetPlotMousePos();
    double mx = std::clamp(mouse.x, 0.0, (double)img_w);
    double my = std::clamp(mouse.y, 0.0, (double)img_h);

    // W key advances the OBB state machine
    if (ImGui::IsKeyPressed(ImGuiKey_W)) {
        switch (state.draw_state) {
        case OBBDrawState::Idle:
            state.ax1_x = mx; state.ax1_y = my;
            state.draw_state = OBBDrawState::FirstPoint;
            break;

        case OBBDrawState::FirstPoint:
            state.ax2_x = mx; state.ax2_y = my;
            state.draw_state = OBBDrawState::SecondPoint;
            break;

        case OBBDrawState::SecondPoint: {
            // Compute final OBB
            double cx, cy, w, h, angle;
            calculate_obb_from_3_points(state.ax1_x, state.ax1_y,
                                         state.ax2_x, state.ax2_y,
                                         mx, my, cx, cy, w, h, angle);
            if (w < 3 || h < 3) { state.draw_state = OBBDrawState::Idle; break; }

            // Convert to image coords (Y-flip)
            double img_cx = cx;
            double img_cy = img_h - cy;
            double img_angle = -angle;

            // Store in AnnotationMap
            auto &fa = get_or_create_frame(amap, frame, num_nodes, num_cameras);
            InstanceAnnotation *target = nullptr;
            for (auto &inst : fa.instances) {
                if (inst.category_id == bbox_state.current_class &&
                    inst.instance_id == bbox_state.current_instance) {
                    target = &inst;
                    break;
                }
            }
            if (!target) {
                fa.instances.push_back(make_instance(num_nodes, num_cameras,
                                                      bbox_state.current_instance,
                                                      bbox_state.current_class));
                target = &fa.instances.back();
            }

            if (cam_idx < (int)target->cameras.size()) {
                auto &cam = target->cameras[cam_idx];
                cam.obb_cx = img_cx;
                cam.obb_cy = img_cy;
                cam.obb_w = w;
                cam.obb_h = h;
                cam.obb_angle = img_angle;
                cam.has_obb = true;
            }

            state.draw_state = OBBDrawState::Idle;
            break;
        }
        }
    }

    // Escape: cancel construction
    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
        state.draw_state = OBBDrawState::Idle;
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
            if (!cam.has_obb) continue;

            double plot_cx = cam.obb_cx;
            double plot_cy = img_h - cam.obb_cy;
            double plot_angle = -cam.obb_angle;

            if (obb_contains(plot_cx, plot_cy, cam.obb_w, cam.obb_h,
                              plot_angle, mx, my)) {
                state.hovered_instance = i;
                state.hovered_cam = cam_idx;
                break;
            }
        }
    }

    // T key: delete hovered OBB from this camera
    if (state.hovered_instance >= 0 && ImGui::IsKeyPressed(ImGuiKey_T)) {
        auto &inst = amap[frame].instances[state.hovered_instance];
        if (cam_idx < (int)inst.cameras.size())
            inst.cameras[cam_idx].has_obb = false;
        state.hovered_instance = -1;
    }
}

// Settings panel for the OBB tool
inline void DrawOBBToolWindow(OBBToolState &state, AppContext &ctx) {
    drawPanel("OBB Tool", state.show,
        [&]() {
        ImGui::Checkbox("Enable OBB Drawing", &state.enabled);

        ImGui::Separator();
        const char *state_labels[] = {"Idle", "Axis Point 1 placed", "Axis Point 2 placed"};
        ImGui::Text("State: %s", state_labels[(int)state.draw_state]);

        ImGui::Separator();
        ImGui::TextWrapped("W (3x): place axis point 1, axis point 2, corner");
        ImGui::TextWrapped("Escape: cancel construction");
        ImGui::TextWrapped("T: delete hovered OBB");
        },
        nullptr, ImVec2(300, 250));
}
