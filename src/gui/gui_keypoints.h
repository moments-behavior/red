#pragma once
#include "annotation.h"
#include "implot.h"
#include "render.h"
#include "skeleton.h"
#include "camera.h"
#include "red_math.h"
#include <ceres/ceres.h>
#include <cmath>
#include <sstream>
#include <vector>

inline void gui_plot_keypoints(FrameAnnotation &fa, SkeletonContext *skeleton,
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

inline bool is_in_camera_fov(const Eigen::Vector3d &point_world,
                      const Eigen::Matrix3d &R,
                      const Eigen::Vector3d &tvec,
                      const Eigen::Matrix3d &K, int image_width,
                      int image_height) {
    // Check point is in front of camera
    Eigen::Vector3d cam_pt = R * point_world + tvec;
    if (cam_pt(2) <= 0) return false;
    // Use matrix-based projection (safe for det(R)=-1)
    Eigen::Matrix<double, 5, 1> zero_dist = Eigen::Matrix<double, 5, 1>::Zero();
    auto pt2d = red_math::projectPointR(point_world, R, tvec, K, zero_dist);
    double x = pt2d(0);
    double y = image_height - pt2d(1);
    return (x > 0 && x < image_width && y > 0 && y < image_height);
}

inline void reprojection(FrameAnnotation &fa, SkeletonContext *skeleton,
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
                    // Perspective reprojection (matrix-based, safe for det(R)=-1)
                    if (is_in_camera_fov(pt3d, camera_params[view_idx].r,
                                         camera_params[view_idx].tvec,
                                         camera_params[view_idx].k,
                                         scene->image_width[view_idx],
                                         scene->image_height[view_idx])) {
                        auto reproj = red_math::projectPointR(
                            pt3d, camera_params[view_idx].r,
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

// ─────────────────────────────────────────────────────────────────────────────
// Refine 3D via bundle adjustment (per-keypoint Levenberg-Marquardt on 3D
// position only — camera params stay fixed). Unlike reprojection() which is a
// one-shot closed-form DLT, this iterates under a Cauchy robust loss so a
// single noisy 2D label pulls the 3D less. After refinement, re-projects
// every keypoint onto every camera, same as reprojection().
// ─────────────────────────────────────────────────────────────────────────────

// Per-(camera, observation) reprojection error for Ceres autodiff.
// Camera intrinsics + extrinsics are captured by value (fixed during BA).
struct Refine3DReprojErr {
    Refine3DReprojErr(const Eigen::Vector2d &obs_pixel, const CameraParams &cam)
        : obs_(obs_pixel), R_(cam.r), t_(cam.tvec), K_(cam.k),
          dist_(cam.dist_coeffs) {}

    template <typename T>
    bool operator()(const T *const xyz, T *residuals) const {
        Eigen::Matrix<T, 3, 1> X(xyz[0], xyz[1], xyz[2]);
        Eigen::Matrix<T, 3, 3> R = R_.cast<T>();
        Eigen::Matrix<T, 3, 1> t = t_.cast<T>();
        Eigen::Matrix<T, 3, 1> Xc = R * X + t;

        if (ceres::abs(Xc(2)) < T(1e-8)) {
            residuals[0] = T(1e6);
            residuals[1] = T(1e6);
            return true;
        }
        T xp = Xc(0) / Xc(2);
        T yp = Xc(1) / Xc(2);
        T r2 = xp * xp + yp * yp;
        T r4 = r2 * r2;
        T r6 = r4 * r2;
        T k1 = T(dist_(0)), k2 = T(dist_(1));
        T p1 = T(dist_(2)), p2 = T(dist_(3));
        T k3 = T(dist_(4));
        T radial = T(1) + k1 * r2 + k2 * r4 + k3 * r6;
        T xpp = xp * radial + T(2) * p1 * xp * yp + p2 * (r2 + T(2) * xp * xp);
        T ypp = yp * radial + p1 * (r2 + T(2) * yp * yp) + T(2) * p2 * xp * yp;

        T fx = T(K_(0, 0)), fy = T(K_(1, 1));
        T cx = T(K_(0, 2)), cy = T(K_(1, 2));
        T u = xpp * fx + cx;
        T v = ypp * fy + cy;
        residuals[0] = u - T(obs_(0));
        residuals[1] = v - T(obs_(1));
        return true;
    }

    Eigen::Vector2d obs_;
    Eigen::Matrix3d R_;
    Eigen::Vector3d t_;
    Eigen::Matrix3d K_;
    Eigen::Matrix<double, 5, 1> dist_;
};

// Run BA on each keypoint independently. Writes refined 3D into fa.kp3d and
// reprojects onto every camera so the 2D overlay reflects the new 3D.
// Telecentric cameras fall back to the closed-form reprojection() path since
// we haven't wired up a telecentric cost functor yet.
inline void refine_3d_ba(FrameAnnotation &fa, SkeletonContext *skeleton,
                         const std::vector<CameraParams> &camera_params,
                         RenderScene *scene) {
    if (camera_params.empty() || !skeleton || !scene) return;
    if (camera_params[0].telecentric) {
        reprojection(fa, skeleton, camera_params, scene);
        return;
    }

    int nodes_refined = 0;
    double err_before_sum = 0.0, err_after_sum = 0.0;
    int obs_total = 0;

    for (u32 node = 0; node < (u32)skeleton->num_nodes; node++) {
        if (node >= (u32)fa.kp3d.size()) break;

        // Collect 2D observations for this node across all cameras.
        // Keypoints in the annotation are in ImPlot space (Y=0 at bottom);
        // convert to image space (Y=0 at top) for projection math.
        std::vector<int> cam_idx;
        std::vector<Eigen::Vector2d> obs_pix;
        for (u32 ci = 0; ci < scene->num_cams; ci++) {
            if (ci >= (u32)fa.cameras.size()) continue;
            if (node >= (u32)fa.cameras[ci].keypoints.size()) continue;
            const auto &kp = fa.cameras[ci].keypoints[node];
            if (!kp.labeled) continue;
            Eigen::Vector2d px(kp.x,
                               (double)scene->image_height[ci] - kp.y);
            cam_idx.push_back((int)ci);
            obs_pix.push_back(px);
        }
        if (obs_pix.size() < 2) continue;

        // Initial 3D: use the already-triangulated point if present,
        // otherwise do a DLT from the 2D observations to seed the LM.
        Eigen::Vector3d X;
        if (fa.kp3d[node].triangulated) {
            X = Eigen::Vector3d(fa.kp3d[node].x, fa.kp3d[node].y,
                                fa.kp3d[node].z);
        } else {
            std::vector<Eigen::Vector2d> undist;
            std::vector<Eigen::Matrix<double, 3, 4>> Ps;
            for (size_t i = 0; i < obs_pix.size(); i++) {
                int ci = cam_idx[i];
                undist.push_back(red_math::undistortPoint(
                    obs_pix[i], camera_params[ci].k,
                    camera_params[ci].dist_coeffs));
                Ps.push_back(camera_params[ci].projection_mat);
            }
            X = red_math::triangulatePoints(undist, Ps);
        }

        // Pre-refinement RMS reprojection error for reporting.
        double sq_before = 0.0;
        for (size_t i = 0; i < obs_pix.size(); i++) {
            int ci = cam_idx[i];
            Eigen::Vector2d proj = red_math::projectPoint(
                X, camera_params[ci].rvec, camera_params[ci].tvec,
                camera_params[ci].k, camera_params[ci].dist_coeffs);
            sq_before += (proj - obs_pix[i]).squaredNorm();
        }

        // Build and solve the per-node problem.
        double xyz[3] = {X(0), X(1), X(2)};
        ceres::Problem problem;
        problem.AddParameterBlock(xyz, 3);
        for (size_t i = 0; i < obs_pix.size(); i++) {
            int ci = cam_idx[i];
            ceres::CostFunction *cost =
                new ceres::AutoDiffCostFunction<Refine3DReprojErr, 2, 3>(
                    new Refine3DReprojErr(obs_pix[i], camera_params[ci]));
            // Cauchy loss with scale 2 px: residuals under ~2 px count
            // linearly, larger ones get down-weighted (robust to a single
            // bad 2D label).
            problem.AddResidualBlock(cost, new ceres::CauchyLoss(2.0), xyz);
        }
        ceres::Solver::Options opts;
        opts.max_num_iterations = 25;
        opts.linear_solver_type = ceres::DENSE_QR;
        opts.minimizer_progress_to_stdout = false;
        opts.logging_type = ceres::SILENT;
        ceres::Solver::Summary summary;
        ceres::Solve(opts, &problem, &summary);

        X = Eigen::Vector3d(xyz[0], xyz[1], xyz[2]);

        // Post-refinement RMS error for reporting.
        double sq_after = 0.0;
        for (size_t i = 0; i < obs_pix.size(); i++) {
            int ci = cam_idx[i];
            Eigen::Vector2d proj = red_math::projectPoint(
                X, camera_params[ci].rvec, camera_params[ci].tvec,
                camera_params[ci].k, camera_params[ci].dist_coeffs);
            sq_after += (proj - obs_pix[i]).squaredNorm();
        }

        err_before_sum += sq_before;
        err_after_sum += sq_after;
        obs_total += (int)obs_pix.size();

        // Commit refined 3D.
        fa.kp3d[node].x = X(0);
        fa.kp3d[node].y = X(1);
        fa.kp3d[node].z = X(2);
        fa.kp3d[node].triangulated = true;
        nodes_refined++;

        // Reproject onto every camera (same policy as reprojection()).
        for (u32 view_idx = 0; view_idx < scene->num_cams; view_idx++) {
            if (view_idx >= (u32)fa.cameras.size()) continue;
            if (node >= (u32)fa.cameras[view_idx].keypoints.size()) continue;
            if (!is_in_camera_fov(X, camera_params[view_idx].r,
                                  camera_params[view_idx].tvec,
                                  camera_params[view_idx].k,
                                  scene->image_width[view_idx],
                                  scene->image_height[view_idx]))
                continue;
            Eigen::Vector2d proj = red_math::projectPointR(
                X, camera_params[view_idx].r,
                camera_params[view_idx].tvec,
                camera_params[view_idx].k,
                camera_params[view_idx].dist_coeffs);
            double x = proj(0);
            double y = (double)scene->image_height[view_idx] - proj(1);
            if (x > 0 && x < scene->image_width[view_idx] &&
                y > 0 && y < scene->image_height[view_idx]) {
                fa.cameras[view_idx].keypoints[node].x = x;
                fa.cameras[view_idx].keypoints[node].y = y;
                fa.cameras[view_idx].keypoints[node].labeled = true;
            }
        }
    }

    if (obs_total > 0) {
        double rms_before = std::sqrt(err_before_sum / obs_total);
        double rms_after = std::sqrt(err_after_sum / obs_total);
        fprintf(stderr,
                "[Refine3D] refined %d nodes (%d total obs): rms reproj "
                "%.3f → %.3f px\n",
                nodes_refined, obs_total, rms_before, rms_after);
    }
}
