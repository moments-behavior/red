#pragma once
// crop_designer.h — interactive sensor-ROI crop preview for the
// Cropped-Sensor Refinement wizard.
//
// Draws a draggable, FIXED-SIZE rectangle (shared crop_w x crop_h, snapped to
// multiples of 16 like orange's GUI) on each full-frame camera view,
// simulating what orange will record with that ROI. Dragging any handle moves
// the rect (size is imposed after the drag); offsets are clamped to the
// sensor. Also shows how many clicked posts fall inside the crop — posts
// outside would be invisible in the real cropped recording.
//
// Coordinates: ImPlot is y-up, sensor OffsetY is from the top:
//   offset_y = img_h - rect_top   (same convention as bbox_tool.h)

#include "calib_tool_state.h"
#include "annotation.h"
#include "imgui.h"
#include "implot.h"
#include <algorithm>
#include <cstdio>

inline int crop_snap16(int v) { return (v / 16) * 16; }

inline void crop_designer_draw(CalibrationToolState &state,
                               const AnnotationMap &annotations, int cam_idx,
                               int img_w, int img_h) {
    auto &cs = state.cropped;
    if (state.project.subtype !=
            CalibrationTool::CalibSubtype::CroppedRefinement ||
        !cs.designer_enabled)
        return;
    if (cam_idx < 0 || cam_idx >= (int)cs.crop_spec.cameras.size())
        return;
    if (img_w <= 0 || img_h <= 0) return;

    auto &crop = cs.crop_spec.cameras[cam_idx];

    // Shared dims rule the wizard: keep this row in sync.
    int w = std::min(cs.crop_w, img_w);
    int h = std::min(cs.crop_h, img_h);
    crop.width = w;
    crop.height = h;
    crop.offset_x = std::clamp(crop.offset_x, 0, img_w - w);
    crop.offset_y = std::clamp(crop.offset_y, 0, img_h - h);

    // Sensor (y-down) -> ImPlot (y-up)
    double x1 = crop.offset_x;
    double x2 = crop.offset_x + w;
    double y2 = img_h - crop.offset_y;        // rect top
    double y1 = img_h - (crop.offset_y + h);  // rect bottom

    ImVec4 col(1.0f, 0.65f, 0.1f, 1.0f);  // orange, fittingly
    bool clicked = false, hovered = false, held = false;
    ImPlot::DragRect(90000 + cam_idx, &x1, &y1, &x2, &y2, col,
                     ImPlotDragToolFlags_NoFit, &clicked, &hovered, &held);

    // Fixed size: whatever the user dragged, re-impose w x h around the
    // dragged center, snap offsets to 16, clamp to the sensor.
    double cx = 0.5 * (x1 + x2);
    double cy = 0.5 * (y1 + y2);
    int ox = (int)std::lround(cx - 0.5 * w);
    int oy_img = (int)std::lround((img_h - cy) - 0.5 * h);  // y-flip to top-left
    ox = std::clamp(crop_snap16(ox), 0, std::max(0, img_w - w));
    oy_img = std::clamp(crop_snap16(oy_img), 0, std::max(0, img_h - h));
    crop.offset_x = ox;
    crop.offset_y = oy_img;

    // Count clicked posts inside/outside the crop (frame 0, this camera).
    int inside = 0, total = 0;
    auto it = annotations.find(0);
    if (it != annotations.end() &&
        cam_idx < (int)it->second.cameras.size()) {
        const auto &kps = it->second.cameras[cam_idx].keypoints;
        for (int k = 0; k < state.project.posts_num && k < (int)kps.size();
             k++) {
            const auto &kp = kps[k];
            if (!kp.labeled || kp.x >= UNLABELED * 0.9 ||
                kp.y >= UNLABELED * 0.9)
                continue;
            total++;
            // kp is in ImPlot coords (y-up); convert to sensor y-down.
            double px = kp.x, py = img_h - kp.y;
            if (px >= ox && px < ox + w && py >= oy_img && py < oy_img + h)
                inside++;
        }
    }

    char label[128];
    if (total > 0)
        snprintf(label, sizeof(label), "crop %d,%d  %dx%d  posts %d/%d%s",
                 ox, oy_img, w, h, inside, total,
                 inside < total ? " OUTSIDE!" : "");
    else
        snprintf(label, sizeof(label), "crop %d,%d  %dx%d", ox, oy_img, w, h);

    ImVec4 txt_col = (total > 0 && inside < total)
        ? ImVec4(1.0f, 0.35f, 0.35f, 1.0f)
        : col;
    ImPlot::Annotation((double)ox, (double)(img_h - oy_img), txt_col,
                       ImVec2(4, 4), true, "%s", label);
}
