#pragma once
#include "implot.h"
#include "implot_internal.h"
#include "render.h"
#include "skeleton.h"
#include "camera.h"
#include "red_math.h"
#include "gui/gui_keypoints.h"
#include <iostream>
#include <vector>

static void triangulate_bounding_boxes(KeyPoints *keypoints,
                                       SkeletonContext *skeleton,
                                       std::vector<CameraParams> camera_params,
                                       RenderScene *scene,
                                       int current_frame_num) {
    if (!skeleton->has_bbox || keypoints->bbox2d_list.empty()) {
        return;
    }

    try {
        std::vector<int> source_cameras;
        for (int cam_id = 0;
             cam_id < scene->num_cams && cam_id < keypoints->bbox2d_list.size();
             cam_id++) {
            for (const auto &bbox : keypoints->bbox2d_list[cam_id]) {
                if (bbox.state == RectTwoPoints && bbox.confidence >= 1.0f) {
                    source_cameras.push_back(cam_id);
                    break;
                }
            }
        }

        if (source_cameras.size() != 2) {
            std::cout << "Triangulation requires exactly 2 cameras with "
                         "user-drawn bounding boxes. Found: "
                      << source_cameras.size() << " cameras." << std::endl;
            return;
        }

        int cam1_id = source_cameras[0];
        int cam2_id = source_cameras[1];

        BoundingBox *bbox1 = nullptr;
        BoundingBox *bbox2 = nullptr;

        for (auto &bbox : keypoints->bbox2d_list[cam1_id]) {
            if (bbox.state == RectTwoPoints && bbox.confidence >= 1.0f) {
                bbox1 = &bbox;
                break;
            }
        }

        for (auto &bbox : keypoints->bbox2d_list[cam2_id]) {
            if (bbox.state == RectTwoPoints && bbox.confidence >= 1.0f) {
                bbox2 = &bbox;
                break;
            }
        }

        if (!bbox1 || !bbox2) {
            std::cout
                << "Could not find user-drawn bounding boxes in source cameras."
                << std::endl;
            return;
        }

        double center1_x = (bbox1->rect->X.Min + bbox1->rect->X.Max) / 2.0;
        double center1_y = (bbox1->rect->Y.Min + bbox1->rect->Y.Max) / 2.0;

        double center2_x = (bbox2->rect->X.Min + bbox2->rect->X.Max) / 2.0;
        double center2_y = (bbox2->rect->Y.Min + bbox2->rect->Y.Max) / 2.0;

        std::cout << "Center 1: (" << center1_x << ", " << center1_y << ")"
                  << std::endl;
        std::cout << "Center 2: (" << center2_x << ", " << center2_y << ")"
                  << std::endl;

        Eigen::Vector2d pt1(center1_x,
                             (double)scene->image_height[cam1_id] - center1_y);
        Eigen::Vector2d pt2(center2_x,
                             (double)scene->image_height[cam2_id] - center2_y);

        Eigen::Vector2d pt1_undist = red_math::undistortPoint(
            pt1, camera_params[cam1_id].k, camera_params[cam1_id].dist_coeffs);
        Eigen::Vector2d pt2_undist = red_math::undistortPoint(
            pt2, camera_params[cam2_id].k, camera_params[cam2_id].dist_coeffs);

        std::vector<Eigen::Vector2d> undist_pts = {pt1_undist, pt2_undist};
        std::vector<Eigen::Matrix<double, 3, 4>> proj_mats = {
            camera_params[cam1_id].projection_mat,
            camera_params[cam2_id].projection_mat};

        Eigen::Vector3d triangulated_center =
            red_math::triangulatePoints(undist_pts, proj_mats);

        std::cout << "Triangulated 3D center: ("
                  << triangulated_center(0) << ", "
                  << triangulated_center(1) << ", "
                  << triangulated_center(2) << ")" << std::endl;

        double width1 = bbox1->rect->X.Max - bbox1->rect->X.Min;
        double height1 = bbox1->rect->Y.Max - bbox1->rect->Y.Min;
        double width2 = bbox2->rect->X.Max - bbox2->rect->X.Min;
        double height2 = bbox2->rect->Y.Max - bbox2->rect->Y.Min;

        double long1 = std::max(width1, height1);
        double short1 = std::min(width1, height1);
        double long2 = std::max(width2, height2);
        double short2 = std::min(width2, height2);

        double avg_long_side = (long1 + long2) / 2.0;
        double avg_short_side = (short1 + short2) / 2.0;

        std::cout << "Average long side: " << avg_long_side
                  << ", Average short side: " << avg_short_side << std::endl;

        for (int target_cam = 0; target_cam < scene->num_cams; target_cam++) {
            if (target_cam == cam1_id || target_cam == cam2_id) {
                continue;
            }

            if (!is_in_camera_fov(
                    triangulated_center, camera_params[target_cam].rvec,
                    camera_params[target_cam].tvec, camera_params[target_cam].k,
                    scene->image_width[target_cam],
                    scene->image_height[target_cam])) {
                std::cout << "3D center not in FOV of camera " << target_cam
                          << std::endl;
                continue;
            }

            auto reproj = red_math::projectPoint(
                triangulated_center, camera_params[target_cam].rvec,
                camera_params[target_cam].tvec, camera_params[target_cam].k,
                camera_params[target_cam].dist_coeffs);

            double proj_x = reproj(0);
            double proj_y = (double)scene->image_height[target_cam] -
                            reproj(1);

            std::cout << "Reprojected center in camera " << target_cam << ": ("
                      << proj_x << ", " << proj_y << ")" << std::endl;

            double half_width = avg_long_side / 2.0;
            double half_height = avg_short_side / 2.0;

            BoundingBox new_bbox;
            new_bbox.rect =
                new ImPlotRect(proj_x - half_width, proj_x + half_width,
                               proj_y - half_height, proj_y + half_height);
            new_bbox.state = RectTwoPoints;
            new_bbox.class_id = bbox1->class_id;
            new_bbox.confidence = 1.0f;
            new_bbox.has_bbox_keypoints = false;
            new_bbox.bbox_keypoints2d = nullptr;
            new_bbox.active_kp_id = nullptr;

            keypoints->bbox2d_list[target_cam].push_back(new_bbox);

            std::cout << "Added reprojected bounding box to camera "
                      << target_cam << " at (" << proj_x - half_width << ", "
                      << proj_y - half_height << ") to (" << proj_x + half_width
                      << ", " << proj_y + half_height << ")" << std::endl;
        }

        std::cout << "Bounding box triangulation and reprojection completed "
                     "successfully."
                  << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "Error in bounding box triangulation: " << e.what()
                  << std::endl;
    }
}

bool MyDragRect(int n_id, double *x_min, double *y_min, double *x_max,
                double *y_max, const ImVec4 &col, ImPlotDragToolFlags flags,
                bool *out_clicked, bool *out_hovered, bool *out_held) {
    ImGui::PushID("#IMPLOT_DRAG_RECT");
    IM_ASSERT_USER_ERROR(
        GImPlot->CurrentPlot != nullptr,
        "DragRect() needs to be called between BeginPlot() and EndPlot()!");
    ImPlot::SetupLock();

    if (!ImHasFlag(flags, ImPlotDragToolFlags_NoFit) &&
        ImPlot::FitThisFrame()) {
        ImPlot::FitPoint(ImPlotPoint(*x_min, *y_min));
        ImPlot::FitPoint(ImPlotPoint(*x_max, *y_max));
    }

    const bool input = !ImHasFlag(flags, ImPlotDragToolFlags_NoInputs);
    const bool show_curs = !ImHasFlag(flags, ImPlotDragToolFlags_NoCursors);
    const bool no_delay = !ImHasFlag(flags, ImPlotDragToolFlags_Delayed);
    bool h[] = {true, false, true, false};
    double *x[] = {x_min, x_max, x_max, x_min};
    double *y[] = {y_min, y_min, y_max, y_max};
    ImVec2 p[4];
    for (int i = 0; i < 4; ++i)
        p[i] = ImPlot::PlotToPixels(*x[i], *y[i], IMPLOT_AUTO, IMPLOT_AUTO);
    ImVec2 pc = ImPlot::PlotToPixels(
        (*x_min + *x_max) / 2, (*y_min + *y_max) / 2, IMPLOT_AUTO, IMPLOT_AUTO);
    ImRect rect(ImMin(p[0], p[2]), ImMax(p[0], p[2]));
    ImRect rect_grab = rect;
    float DRAG_GRAB_HALF_SIZE = 4.0f;
    rect_grab.Expand(DRAG_GRAB_HALF_SIZE);

    ImGuiMouseCursor cur[4];
    if (show_curs) {
        cur[0] = (rect.Min.x == p[0].x && rect.Min.y == p[0].y) ||
                         (rect.Max.x == p[0].x && rect.Max.y == p[0].y)
                     ? ImGuiMouseCursor_ResizeNWSE
                     : ImGuiMouseCursor_ResizeNESW;
        cur[1] = cur[0] == ImGuiMouseCursor_ResizeNWSE
                     ? ImGuiMouseCursor_ResizeNESW
                     : ImGuiMouseCursor_ResizeNWSE;
        cur[2] = cur[1] == ImGuiMouseCursor_ResizeNWSE
                     ? ImGuiMouseCursor_ResizeNESW
                     : ImGuiMouseCursor_ResizeNWSE;
        cur[3] = cur[2] == ImGuiMouseCursor_ResizeNWSE
                     ? ImGuiMouseCursor_ResizeNESW
                     : ImGuiMouseCursor_ResizeNWSE;
    }

    ImVec4 color = ImPlot::IsColorAuto(col)
                       ? ImGui::GetStyleColorVec4(ImGuiCol_Text)
                       : col;
    ImU32 col32 = ImGui::ColorConvertFloat4ToU32(color);
    color.w *= 0.25f;
    ImU32 col32_a = ImGui::ColorConvertFloat4ToU32(color);
    const ImGuiID id = ImGui::GetCurrentWindow()->GetID(n_id);

    bool modified = false;
    bool clicked = false, hovered = false, held = false;
    ImRect b_rect(pc.x - DRAG_GRAB_HALF_SIZE, pc.y - DRAG_GRAB_HALF_SIZE,
                  pc.x + DRAG_GRAB_HALF_SIZE, pc.y + DRAG_GRAB_HALF_SIZE);

    ImGui::KeepAliveID(id);
    if (input) {
        // middle point
        clicked = ImGui::ButtonBehavior(b_rect, id, &hovered, &held);
        if (out_clicked)
            *out_clicked = clicked;
        if (out_hovered)
            *out_hovered = hovered;
        if (out_held)
            *out_held = held;
    }

    if ((hovered || held) && show_curs)
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
    if (held && ImGui::IsMouseDragging(0)) {
        for (int i = 0; i < 4; ++i) {
            ImVec2 md = ImGui::GetIO().MouseDelta;
            ImVec2 sum(p[i].x + md.x, p[i].y + md.y);

            ImPlotPoint pp =
                ImPlot::PixelsToPlot(sum, IMPLOT_AUTO, IMPLOT_AUTO);
            *y[i] = pp.y;
            *x[i] = pp.x;
        }
        modified = true;
    }

    for (int i = 0; i < 4; ++i) {
        // points
        b_rect =
            ImRect(p[i].x - DRAG_GRAB_HALF_SIZE, p[i].y - DRAG_GRAB_HALF_SIZE,
                   p[i].x + DRAG_GRAB_HALF_SIZE, p[i].y + DRAG_GRAB_HALF_SIZE);
        ImGuiID p_id = id + i + 1;
        ImGui::KeepAliveID(p_id);
        if (input) {
            clicked = ImGui::ButtonBehavior(b_rect, p_id, &hovered, &held);
            if (out_clicked)
                *out_clicked = *out_clicked || clicked;
            if (out_hovered)
                *out_hovered = *out_hovered || hovered;
            if (out_held)
                *out_held = *out_held || held;
        }
        if ((hovered || held) && show_curs)
            ImGui::SetMouseCursor(cur[i]);

        if (held && ImGui::IsMouseDragging(0)) {
            *x[i] = ImPlot::GetPlotMousePos(IMPLOT_AUTO, IMPLOT_AUTO).x;
            *y[i] = ImPlot::GetPlotMousePos(IMPLOT_AUTO, IMPLOT_AUTO).y;
            modified = true;
        }

        // edges
        ImVec2 e_min = ImMin(p[i], p[(i + 1) % 4]);
        ImVec2 e_max = ImMax(p[i], p[(i + 1) % 4]);
        b_rect = h[i] ? ImRect(e_min.x + DRAG_GRAB_HALF_SIZE,
                               e_min.y - DRAG_GRAB_HALF_SIZE,
                               e_max.x - DRAG_GRAB_HALF_SIZE,
                               e_max.y + DRAG_GRAB_HALF_SIZE)
                      : ImRect(e_min.x - DRAG_GRAB_HALF_SIZE,
                               e_min.y + DRAG_GRAB_HALF_SIZE,
                               e_max.x + DRAG_GRAB_HALF_SIZE,
                               e_max.y - DRAG_GRAB_HALF_SIZE);
        ImGuiID e_id = id + i + 5;
        ImGui::KeepAliveID(e_id);
        if (input) {
            clicked = ImGui::ButtonBehavior(b_rect, e_id, &hovered, &held);
            if (out_clicked)
                *out_clicked = *out_clicked || clicked;
            if (out_hovered)
                *out_hovered = *out_hovered || hovered;
            if (out_held)
                *out_held = *out_held || held;
        }
        if ((hovered || held) && show_curs)
            h[i] ? ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS)
                 : ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
        if (held && ImGui::IsMouseDragging(0)) {
            if (h[i])
                *y[i] = ImPlot::GetPlotMousePos(IMPLOT_AUTO, IMPLOT_AUTO).y;
            else
                *x[i] = ImPlot::GetPlotMousePos(IMPLOT_AUTO, IMPLOT_AUTO).x;
            modified = true;
        }
        if (hovered && ImGui::IsMouseDoubleClicked(0)) {
            ImPlotRect b = ImPlot::GetPlotLimits(IMPLOT_AUTO, IMPLOT_AUTO);
            if (h[i])
                *y[i] = ((y[i] == y_min && *y_min < *y_max) ||
                         (y[i] == y_max && *y_max < *y_min))
                            ? b.Y.Min
                            : b.Y.Max;
            else
                *x[i] = ((x[i] == x_min && *x_min < *x_max) ||
                         (x[i] == x_max && *x_max < *x_min))
                            ? b.X.Min
                            : b.X.Max;
            modified = true;
        }
    }

    const bool mouse_inside = rect_grab.Contains(ImGui::GetMousePos());
    const bool mouse_clicked = ImGui::IsMouseClicked(0);
    const bool mouse_down = ImGui::IsMouseDown(0);
    if (input && mouse_inside) {
        if (out_clicked)
            *out_clicked = *out_clicked || mouse_clicked;
        if (out_hovered)
            *out_hovered = true;
        if (out_held)
            *out_held = *out_held || mouse_down;
    }

    ImPlot::PushPlotClipRect();
    ImDrawList &DrawList = *ImPlot::GetPlotDrawList();
    if (modified && no_delay) {
        for (int i = 0; i < 4; ++i)
            p[i] = ImPlot::PlotToPixels(*x[i], *y[i], IMPLOT_AUTO, IMPLOT_AUTO);
        pc = ImPlot::PlotToPixels((*x_min + *x_max) / 2, (*y_min + *y_max) / 2,
                                  IMPLOT_AUTO, IMPLOT_AUTO);
        rect = ImRect(ImMin(p[0], p[2]), ImMax(p[0], p[2]));
    }
    DrawList.AddRectFilled(rect.Min, rect.Max, col32_a);
    DrawList.AddRect(rect.Min, rect.Max, col32);
    if (input && (modified || mouse_inside)) {
        // DrawList.AddCircleFilled(pc, DRAG_GRAB_HALF_SIZE, col32);
        for (int i = 0; i < 4; ++i)
            DrawList.AddCircleFilled(p[i], DRAG_GRAB_HALF_SIZE, col32);
    }
    ImPlot::PopPlotClipRect();
    ImGui::PopID();
    return modified;
}

bool MyDragRect(int id, ImPlotRect *bounds, const ImVec4 &col,
                ImPlotDragToolFlags flags, bool *out_clicked, bool *out_hovered,
                bool *out_held) {
    return MyDragRect(id, &bounds->X.Min, &bounds->Y.Min, &bounds->X.Max,
                      &bounds->Y.Max, col, flags, out_clicked, out_hovered,
                      out_held);
}

static void gui_plot_bbox_keypoints(BoundingBox *bbox,
                                    SkeletonContext *skeleton, int view_idx,
                                    int num_cams, bool is_active,
                                    bool &is_saved, int bbox_id) {
    if (!bbox->has_bbox_keypoints || !skeleton->has_skeleton)
        return;

    float pt_size = 4.0f;
    for (u32 node = 0; node < skeleton->num_nodes; node++) {
        if (bbox->bbox_keypoints2d[view_idx][node].is_labeled) {
            ImVec4 node_color;
            ImPlotDragToolFlags flag;
            if (is_active) {
                flag = ImPlotDragToolFlags_None;
                if (bbox->active_kp_id[view_idx] == node) {
                    node_color = (ImVec4)ImColor::HSV(
                        0.2, 0.9f,
                        0.9f); // Different active color for bbox keypoints
                } else {
                    node_color = skeleton->node_colors[node];
                }
            } else {
                flag = ImPlotDragToolFlags_NoInputs;
                // Gray out inactive bbox keypoints
                node_color = ImVec4(0.5f, 0.5f, 0.5f, 0.7f);
            }

            int id =
                10000 + bbox_id * 1000 + skeleton->num_nodes * view_idx + node;
            bool drag_point_clicked = false;
            bool drag_point_hovered = false;
            bool drag_point_modified = false;

            drag_point_modified = ImPlot::DragPoint(
                id, &bbox->bbox_keypoints2d[view_idx][node].position.x,
                &bbox->bbox_keypoints2d[view_idx][node].position.y, node_color,
                pt_size, flag, &drag_point_clicked, &drag_point_hovered);

            if (drag_point_modified && is_active) {
                constrain_keypoint_to_bbox(
                    &bbox->bbox_keypoints2d[view_idx][node], bbox->rect);
                is_saved = false;
            }

            if (drag_point_hovered && is_active) {
                if (ImGui::IsKeyPressed(ImGuiKey_R,
                                        false)) { // delete hovered keypoint
                    bbox->bbox_keypoints2d[view_idx][node].position = {1E7,
                                                                       1E7};
                    bbox->bbox_keypoints2d[view_idx][node].is_labeled = false;
                    bbox->active_kp_id[view_idx] = node;
                    is_saved = false;
                }

                if (ImGui::IsKeyPressed(ImGuiKey_F, false)) {
                    for (int cam_idx = 0; cam_idx < num_cams; cam_idx++) {
                        bbox->bbox_keypoints2d[cam_idx][node].position = {1E7,
                                                                          1E7};
                        bbox->bbox_keypoints2d[cam_idx][node].is_labeled =
                            false;
                        bbox->active_kp_id[cam_idx] = node;
                    }
                    is_saved = false;
                }
            }

            if (drag_point_clicked && is_active) {
                bbox->active_kp_id[view_idx] = node;
            }
        }
    }

    // Draw skeleton edges within bbox
    for (u32 edge = 0; edge < skeleton->num_edges; edge++) {
        auto [a, b] = skeleton->edges[edge];

        if (bbox->bbox_keypoints2d[view_idx][a].is_labeled &&
            bbox->bbox_keypoints2d[view_idx][b].is_labeled) {
            double xs[2]{bbox->bbox_keypoints2d[view_idx][a].position.x,
                         bbox->bbox_keypoints2d[view_idx][b].position.x};
            double ys[2]{bbox->bbox_keypoints2d[view_idx][a].position.y,
                         bbox->bbox_keypoints2d[view_idx][b].position.y};

            // Gray out edges for inactive bboxes
            if (is_active) {
                ImPlot::SetNextLineStyle(ImVec4(0.8f, 0.8f, 0.2f, 0.8f), 1.5f);
            } else {
                ImPlot::SetNextLineStyle(ImVec4(0.4f, 0.4f, 0.4f, 0.5f),
                                         1.0f); // Grayed out edges
            }
            ImPlot::PlotLine("##bbox_line", xs, ys, 2);
        }
    }
}
