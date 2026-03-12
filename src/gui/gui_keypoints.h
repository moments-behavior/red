#pragma once
#include "annotation.h"
#include "implot.h"
#include "render.h"
#include "skeleton.h"
#include "camera.h"
#include "red_math.h"
#include <sstream>
#include <vector>

static void gui_plot_keypoints(FrameAnnotation &fa, SkeletonContext *skeleton,
                               int view_idx, int num_cams) {
    if (view_idx >= (int)fa.cameras.size()) return;
    auto &cam = fa.cameras[view_idx];

    float pt_size = 6.0f;
    for (u32 node = 0; node < skeleton->num_nodes; node++) {
        if (node >= (u32)cam.keypoints.size()) break;
        if (cam.keypoints[node].labeled) {
            ImVec4 node_color;
            if (cam.active_id == node) {
                node_color = (ImVec4)ImColor::HSV(0.8, 1.0f, 1.0f);
                node_color.w = 0.9;
                pt_size = 8.0f;
            } else {
                node_color = skeleton->node_colors.at(node);
                node_color.w = 0.9;
                pt_size = 6.0f;
            }
            int id = skeleton->num_nodes * view_idx + node;
            bool drag_point_clicked;
            bool drag_point_hovered;
            bool drag_point_modified;
            drag_point_modified = ImPlot::DragPoint(
                id, &cam.keypoints[node].x,
                &cam.keypoints[node].y, node_color,
                pt_size, ImPlotDragToolFlags_None, &drag_point_clicked,
                &drag_point_hovered);
            if (drag_point_modified) {
                fa.kp3d[node].triangulated = false;
            }
            if (drag_point_hovered) {
                if (fa.kp3d[node].triangulated) {

                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(2);
                    oss << "(" << fa.kp3d[node].x << ", "
                        << fa.kp3d[node].y << ", "
                        << fa.kp3d[node].z << ")";
                    std::string label = oss.str();
                    ImVec2 mouse_pos = ImGui::GetMousePos();
                    ImVec2 textPos = ImVec2(mouse_pos.x + 10, mouse_pos.y + 10);
                    ImGui::GetForegroundDrawList()->AddText(
                        textPos, IM_COL32(220, 20, 60, 255), label.c_str());
                }

                if (ImGui::IsKeyPressed(ImGuiKey_R,
                                        false)) // delete active keypoint
                {
                    cam.keypoints[node] = Keypoint2D{}; // reset all fields
                    cam.active_id = node;
                }

                if (ImGui::IsKeyPressed(
                        ImGuiKey_F,
                        false)) // Delete active keypoints from all the views
                {
                    for (int cam_idx = 0; cam_idx < num_cams; cam_idx++) {
                        if (cam_idx >= (int)fa.cameras.size()) break;
                        fa.cameras[cam_idx].keypoints[node] = Keypoint2D{};
                        fa.cameras[cam_idx].active_id = node;
                    }
                }
            }

            if (drag_point_clicked) {
                cam.active_id = node;
            }
        }
    }

    for (u32 edge = 0; edge < skeleton->num_edges; edge++) {
        auto [a, b] = skeleton->edges[edge];

        if (a < (u32)cam.keypoints.size() && b < (u32)cam.keypoints.size() &&
            cam.keypoints[a].labeled && cam.keypoints[b].labeled) {
            double xs[2]{cam.keypoints[a].x, cam.keypoints[b].x};
            double ys[2]{cam.keypoints[a].y, cam.keypoints[b].y};
            ImPlot::PlotLine("##line", xs, ys, 2);
        }
    }
}

bool is_in_camera_fov(const Eigen::Vector3d &point_world,
                      const Eigen::Vector3d &rvec,
                      const Eigen::Vector3d &tvec,
                      const Eigen::Matrix3d &K, int image_width,
                      int image_height) {
    auto pt2d = red_math::projectPointNoDist(point_world, rvec, tvec, K);
    double x = pt2d(0);
    double y = image_height - pt2d(1);
    if (x > 0 && x < image_width && y > 0 && y < image_height) {
        return true;
    }
    return false;
}

static void reprojection(FrameAnnotation &fa, SkeletonContext *skeleton,
                         const std::vector<CameraParams> &camera_params,
                         RenderScene *scene) {

    bool telecentric = !camera_params.empty() && camera_params[0].telecentric;

    for (u32 node = 0; node < skeleton->num_nodes; node++) {

        u32 num_views_labeled{0};
        for (u32 view_idx = 0; view_idx < scene->num_cams; view_idx++) {
            if (view_idx < (u32)fa.cameras.size() &&
                node < (u32)fa.cameras[view_idx].keypoints.size() &&
                fa.cameras[view_idx].keypoints[node].labeled) {
                num_views_labeled++;
            }
        }

        if (num_views_labeled >= 2) {

            std::vector<Eigen::Vector2d> undist_pts;
            std::vector<Eigen::Matrix<double, 3, 4>> proj_mats;

            for (u32 view_idx = 0; view_idx < scene->num_cams; view_idx++) {
                if (view_idx >= (u32)fa.cameras.size()) continue;
                if (node >= (u32)fa.cameras[view_idx].keypoints.size()) continue;
                if (fa.cameras[view_idx].keypoints[node].labeled) {
                    Eigen::Vector2d pt(
                        fa.cameras[view_idx].keypoints[node].x,
                        (double)scene->image_height[view_idx] -
                            fa.cameras[view_idx].keypoints[node].y);

                    Eigen::Vector2d pt_undist;
                    if (telecentric) {
                        pt_undist = red_math::undistortPointTelecentric(
                            pt, camera_params[view_idx].k,
                            camera_params[view_idx].dist_coeffs);
                    } else {
                        pt_undist = red_math::undistortPoint(
                            pt, camera_params[view_idx].k,
                            camera_params[view_idx].dist_coeffs);
                    }

                    undist_pts.push_back(pt_undist);
                    proj_mats.push_back(
                        camera_params[view_idx].projection_mat);
                }
            }

            Eigen::Vector3d pt3d =
                red_math::triangulatePoints(undist_pts, proj_mats);

            fa.kp3d[node].x = pt3d(0);
            fa.kp3d[node].y = pt3d(1);
            fa.kp3d[node].z = pt3d(2);
            fa.kp3d[node].triangulated = true;

            for (u32 view_idx = 0; view_idx < scene->num_cams; view_idx++) {
                if (view_idx >= (u32)fa.cameras.size()) continue;
                if (node >= (u32)fa.cameras[view_idx].keypoints.size()) continue;

                if (telecentric) {
                    // Telecentric reprojection
                    auto reproj = red_math::projectPointTelecentric(
                        pt3d,
                        camera_params[view_idx].projection_mat,
                        camera_params[view_idx].k,
                        camera_params[view_idx].dist_coeffs);
                    double x = reproj(0);
                    double y = double(scene->image_height[view_idx]) -
                               reproj(1);
                    if (x > 0 && x < scene->image_width[view_idx] && y > 0 &&
                        y < scene->image_height[view_idx]) {
                        fa.cameras[view_idx].keypoints[node].x = x;
                        fa.cameras[view_idx].keypoints[node].y = y;
                        fa.cameras[view_idx].keypoints[node].labeled = true;
                    }
                } else {
                    // Perspective reprojection
                    if (is_in_camera_fov(pt3d, camera_params[view_idx].rvec,
                                         camera_params[view_idx].tvec,
                                         camera_params[view_idx].k,
                                         scene->image_width[view_idx],
                                         scene->image_height[view_idx])) {
                        auto reproj = red_math::projectPoint(
                            pt3d, camera_params[view_idx].rvec,
                            camera_params[view_idx].tvec,
                            camera_params[view_idx].k,
                            camera_params[view_idx].dist_coeffs);
                        double x = reproj(0);
                        double y = double(scene->image_height[view_idx]) -
                                   reproj(1);
                        if (x > 0 && x < scene->image_width[view_idx] &&
                            y > 0 && y < scene->image_height[view_idx]) {
                            fa.cameras[view_idx].keypoints[node].x = x;
                            fa.cameras[view_idx].keypoints[node].y = y;
                            fa.cameras[view_idx].keypoints[node].labeled = true;
                        }
                    }
                }
            }
        }
    }
}
