#pragma once
// midline_tool.h — Single-view midline labeling tool
//
// Speeds up labeling of midline structures (e.g. the 4-keypoint proboscis) by
// letting the user label the keypoints in ONE side camera and draw a 2-click
// line in a top/line camera. The line fixes the plane the midline lies in; each
// side-view ray is intersected with that plane to solve 3D, then reprojected
// into all other views. See solve_midline_constraint() in gui_keypoints.h and
// the red_math midline helpers.
//
// Plane modes (per-frame, via fa.midline.force_vertical):
//   • preimage (default): intersect side rays with the line's true
//     back-projected plane — ~3× more accurate on a well-calibrated rig.
//   • force-vertical: extrude the line's horizontal footprint along world up;
//     independent of top-cam extrinsics but adds systematic tilt error.

#include "imgui.h"
#include "implot.h"
#include "annotation.h"
#include "app_context.h"
#include "gui/gui_keypoints.h"
#include "gui/panel.h"
#include "gui/toast.h"
#include <string>
#include <vector>

struct MidlineToolState {
    bool show = false;
    bool enabled = false;          // master toggle for line-drawing mode
    int  side_cam_idx = -1;        // side camera the keypoints are labeled in
    int  line_cam_idx = -1;        // camera the 2-click line is drawn in
    bool force_vertical = false;   // false: preimage plane (default)
    bool inited = false;           // one-time default cam selection done

    int  pending_click = 0;        // next click sets endpoint 0 (p1) or 1 (p2)
    std::string status;            // last solve/placement message
    double last_min_sin = 1.0;     // last conditioning (worst ray/plane sine)
};

// Copy the tool's current camera/mode selection into a frame's constraint.
inline void midline_sync_to_frame(const MidlineToolState &state,
                                  FrameAnnotation &fa) {
    fa.midline.keypoint_camera_id = state.side_cam_idx;
    fa.midline.line_camera_id = state.line_cam_idx;
    fa.midline.force_vertical = state.force_vertical;
}

// Handle line-drawing clicks on a focused camera view (the line camera only).
inline void midline_handle_input(MidlineToolState &state, AnnotationMap &amap,
                                 u32 frame, int cam_idx, int num_nodes,
                                 int num_cameras, int /*img_w*/, int img_h) {
    if (!state.enabled) return;
    if (cam_idx != state.line_cam_idx) return;   // draw only in the line camera
    if (!ImPlot::IsPlotHovered()) return;
    if (ImGui::GetIO().KeyShift) return;          // leave shift for bbox/zoom

    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        ImPlotPoint mouse = ImPlot::GetPlotMousePos();
        auto &fa = get_or_create_frame(amap, frame, num_nodes, num_cameras);
        midline_sync_to_frame(state, fa);
        if (state.pending_click == 0) {
            fa.midline.p1x = mouse.x; fa.midline.p1y = mouse.y;
            fa.midline.p2x = mouse.x; fa.midline.p2y = mouse.y;
            fa.midline.has_line = false;          // need the second click
            state.pending_click = 1;
        } else {
            fa.midline.p2x = mouse.x; fa.midline.p2y = mouse.y;
            fa.midline.has_line = true;
            state.pending_click = 0;
        }
    }
}

// Draw the drawn line + endpoints on the line camera's view.
inline void midline_draw_overlay(MidlineToolState &state, const AnnotationMap &amap,
                                 u32 frame, int cam_idx) {
    if (cam_idx != state.line_cam_idx) return;
    auto it = amap.find(frame);
    if (it == amap.end()) return;
    const auto &m = it->second.midline;
    if (m.line_camera_id != cam_idx) return;
    // Show the first endpoint even before the segment is complete.
    bool has_p1 = m.has_line || state.pending_click == 1;
    if (!has_p1) return;

    ImVec4 col(1.0f, 0.55f, 0.1f, 1.0f);
    if (m.has_line) {
        double xs[2]{m.p1x, m.p2x}, ys[2]{m.p1y, m.p2y};
        ImPlotSpec seg_spec;
        seg_spec.LineColor = col;
        ImPlot::PlotLine("##midline_seg", xs, ys, 2, seg_spec);
    }
    double ex[2]{m.p1x, m.p2x}, ey[2]{m.p1y, m.p2y};
    // ImPlot v1.0: marker styling and item colors both moved into ImPlotSpec
    // (SetNextMarkerStyle and ImPlotCol_Marker* are gone).
    ImPlotSpec mspec;
    mspec.Marker = ImPlotMarker_Circle;
    mspec.MarkerSize = 5.0f;
    mspec.MarkerFillColor = col;
    mspec.MarkerLineColor = col;
    ImPlot::PlotScatter("##midline_ends", ex, ey, m.has_line ? 2 : 1, mspec);
}

// Settings panel for the midline tool.
inline void DrawMidlineToolWindow(MidlineToolState &state, AppContext &ctx) {
    auto &pm = ctx.pm;
    const auto &names = pm.camera_names;

    // One-time defaults: line cam = Cam2012630 if present, side cam = first
    // near-horizontal camera (best conditioned) or camera 0.
    if (!state.inited && !names.empty()) {
        state.line_cam_idx = 0;
        for (int i = 0; i < (int)names.size(); i++)
            if (names[i] == "Cam2012630") { state.line_cam_idx = i; break; }
        state.side_cam_idx = (state.line_cam_idx == 0 && names.size() > 1) ? 1 : 0;
        state.inited = true;
    }

    DrawPanel("Midline Tool", state.show, [&]() {
        ImGui::Checkbox("Enable line drawing", &state.enabled);
        ImGui::TextDisabled("Label the keypoints in the side camera, draw the "
                            "line in the line camera, then Solve.");
        ImGui::Separator();

        auto cam_combo = [&](const char *label, int *idx) {
            const char *preview = (*idx >= 0 && *idx < (int)names.size())
                                      ? names[*idx].c_str() : "(none)";
            if (ImGui::BeginCombo(label, preview)) {
                for (int i = 0; i < (int)names.size(); i++) {
                    bool sel = (*idx == i);
                    if (ImGui::Selectable(names[i].c_str(), sel)) *idx = i;
                    if (sel) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
        };
        cam_combo("Side (keypoint) camera", &state.side_cam_idx);
        cam_combo("Line camera", &state.line_cam_idx);
        if (state.side_cam_idx == state.line_cam_idx)
            ImGui::TextColored(ImVec4(1, 0.7f, 0.2f, 1),
                               "Side and line camera should differ.");

        ImGui::Checkbox("Force vertical plane", &state.force_vertical);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Off (default): intersect the line's true "
                              "back-projected plane — more accurate.\n"
                              "On: extrude the footprint along world up (+z) — "
                              "independent of top-cam calibration but less "
                              "accurate on a well-calibrated rig.");

        ImGui::Separator();
        if (ImGui::Button("Solve this frame", ImVec2(-1, 0))) {
            u32 f = (u32)ctx.current_frame_num;
            auto it = ctx.annotations.find(f);
            if (it == ctx.annotations.end()) {
                state.status = "No annotations on this frame";
                ctx.toasts.push(state.status, Toast::Warning, 3.0f);
            } else {
                midline_sync_to_frame(state, it->second);
                std::string status; double min_sin = 1.0;
                bool ok = solve_midline_constraint(it->second, &ctx.skeleton,
                                                   pm.camera_params, ctx.scene,
                                                   status, min_sin);
                state.status = status;
                state.last_min_sin = min_sin;
                ctx.toasts.push(status, ok ? (min_sin < 0.15 ? Toast::Warning
                                                             : Toast::Info)
                                           : Toast::Warning,
                                4.0f);
            }
        }
        if (ImGui::Button("Clear line", ImVec2(-1, 0))) {
            u32 f = (u32)ctx.current_frame_num;
            auto it = ctx.annotations.find(f);
            if (it != ctx.annotations.end()) it->second.midline = MidlineConstraint{};
            state.pending_click = 0;
            state.status.clear();
        }

        if (!state.status.empty()) {
            ImGui::Separator();
            bool warn = state.status.rfind("WARNING", 0) == 0;
            ImGui::TextColored(warn ? ImVec4(1, 0.4f, 0.3f, 1)
                                    : ImVec4(0.5f, 1, 0.5f, 1),
                               "%s", state.status.c_str());
        }

        ImGui::Separator();
        ImGui::TextWrapped("Left-click twice in the line camera to draw the "
                           "line along the midline's footprint.");
        ImGui::TextWrapped("Keypoints: label them in the side camera as usual "
                           "(W key at the cursor).");
    }, nullptr, ImVec2(320, 380));
}
