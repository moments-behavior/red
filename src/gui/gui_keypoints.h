#pragma once
#include "annotation.h"
#include "implot.h"
#include "render.h"
#include "skeleton.h"
#include "camera.h"
#include "red_math.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <vector>

inline void gui_plot_keypoints(FrameAnnotation &fa, SkeletonContext *skeleton,
                               int view_idx, int num_cams,
                               ImVec4 active_color = ImVec4(1, 1, 1, 1)) {
    if (view_idx >= (int)fa.cameras.size()) return;
    auto &cam = fa.cameras[view_idx];

    float pt_size = 6.0f;
    for (u32 node = 0; node < skeleton->num_nodes; node++) {
        if (node >= (u32)cam.keypoints.size()) break;
        if (cam.keypoints[node].labeled) {
            ImVec4 node_color;
            if (cam.active_id == node) {
                node_color = active_color; // active keypoint: user-selected color
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
                fa.kp3d[node].clear();
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

// Project a single 3D point into a camera's image, returning ImPlot coords
// (Y=0 at bottom). Returns false if the point falls outside the image. Mirrors
// the per-camera reprojection in reprojection() (telecentric vs pinhole).
inline bool reproject_3d_to_cam(const Eigen::Vector3d &pt3d,
                                const CameraParams &cp, int W, int H,
                                double &out_x, double &out_y) {
    if (cp.telecentric) {
        auto rp = red_math::projectPointTelecentric(
            pt3d, cp.projection_mat, cp.k, cp.dist_coeffs, cp.dist_center);
        out_x = rp(0);
        out_y = (double)H - rp(1);
    } else {
        if (!is_in_camera_fov(pt3d, cp.r, cp.tvec, cp.k, W, H)) return false;
        auto rp = red_math::projectPointR(pt3d, cp.r, cp.tvec, cp.k, cp.dist_coeffs);
        out_x = rp(0);
        out_y = (double)H - rp(1);
    }
    return (out_x > 0 && out_x < W && out_y > 0 && out_y < H);
}

// ── Single-view midline solve ──────────────────────────────────────────────
// World up-axis for this rig. Verified 3 ways on the Feeding 3D export +
// Posts39a/July6_dlt_linear calibration (body-above-feet, roam-spread, and
// foot-cloud floor normal all point to +z). Used only by the force-vertical
// plane mode and the sanity check below; the default preimage-plane solve does
// not need it.
static const Eigen::Vector3d MIDLINE_WORLD_UP = Eigen::Vector3d(0, 0, 1);

// Undistort one keypoint (ImPlot coords, y-up) into the projection frame the
// camera math expects (y flipped to image coords), mirroring reprojection().
inline Eigen::Vector2d midline_undistort_px(double implot_x, double implot_y,
                                            const CameraParams &cp, int H,
                                            bool telecentric) {
    Eigen::Vector2d pt(implot_x, (double)H - implot_y);
    if (telecentric)
        return red_math::undistortPointTelecentric(pt, cp.k, cp.dist_coeffs,
                                                    cp.dist_center);
    return red_math::undistortPoint(pt, cp.k, cp.dist_coeffs);
}

inline red_math::Ray3D midline_backproject(const Eigen::Vector2d &undist_px,
                                           const CameraParams &cp,
                                           bool telecentric) {
    if (telecentric)
        return red_math::backprojectRayTelecentric(cp.projection_mat, undist_px);
    return red_math::backprojectRayPinhole(cp.r, cp.tvec, cp.k, undist_px);
}

// Solve 3D for a midline structure from ONE side camera's keypoints plus the
// 2-click line drawn in the line camera (fa.midline). Writes fa.kp3d for every
// node labeled in the side camera and reprojects into all OTHER views for
// verification (mirrors reprojection()'s per-camera telecentric/pinhole flow).
// `status` is filled for the UI; `min_sin` returns the worst ray/plane angle
// sine (small ⇒ side camera edge-on to the plane, ill-conditioned).
// Returns true if at least one keypoint was solved.
inline bool solve_midline_constraint(FrameAnnotation &fa,
                                     SkeletonContext *skeleton,
                                     const std::vector<CameraParams> &cp,
                                     RenderScene *scene, std::string &status,
                                     double &min_sin) {
    min_sin = 1.0;
    const MidlineConstraint &m = fa.midline;
    int nc = (int)scene->num_cams;
    if (cp.empty()) { status = "No calibration loaded"; return false; }
    if (!m.has_line) { status = "Draw the 2-click line in the line camera"; return false; }
    if (m.keypoint_camera_id < 0 || m.keypoint_camera_id >= nc ||
        m.line_camera_id < 0 || m.line_camera_id >= nc) {
        status = "Pick a side camera and a line camera"; return false;
    }
    const int side = m.keypoint_camera_id, line = m.line_camera_id;
    if (side >= (int)fa.cameras.size()) { status = "Side camera has no annotations"; return false; }
    const bool telecentric = cp[0].telecentric;

    // --- Build the constraint plane from the drawn line ---
    Eigen::Vector2d e1 = midline_undistort_px(m.p1x, m.p1y, cp[line],
                                              scene->image_height[line], telecentric);
    Eigen::Vector2d e2 = midline_undistort_px(m.p2x, m.p2y, cp[line],
                                              scene->image_height[line], telecentric);
    if ((e2 - e1).norm() < 1e-6) { status = "Line endpoints coincide"; return false; }

    red_math::Plane3D plane;
    if (m.force_vertical) {
        red_math::Ray3D r1 = midline_backproject(e1, cp[line], telecentric);
        red_math::Ray3D r2 = midline_backproject(e2, cp[line], telecentric);
        double cond = 0;
        if (!red_math::verticalPlaneFromFootprint(r1.anchor, r2.anchor,
                                                  MIDLINE_WORLD_UP, plane, cond)) {
            status = "Degenerate line: horizontal footprint too short "
                     "(line drawn end-on, or line camera not top-down)";
            return false;
        }
    } else {
        Eigen::Vector2d n_img; double off;
        red_math::imageLineThroughPoints(e1, e2, n_img, off);
        plane = telecentric
                    ? red_math::preimagePlaneTelecentric(cp[line].projection_mat, n_img, off)
                    : red_math::preimagePlanePinhole(cp[line].r, cp[line].tvec,
                                                     cp[line].k, n_img, off);
    }
    if (plane.normal.norm() < 1e-9) { status = "Could not build a plane from the line"; return false; }

    // --- Intersect each side-view ray with the plane ---
    int n_solved = 0;
    for (u32 node = 0; node < skeleton->num_nodes; node++) {
        if (node >= (u32)fa.cameras[side].keypoints.size()) break;
        const auto &kp = fa.cameras[side].keypoints[node];
        if (!kp.labeled) continue;
        Eigen::Vector2d pu = midline_undistort_px(kp.x, kp.y, cp[side],
                                                  scene->image_height[side], telecentric);
        red_math::Ray3D ray = midline_backproject(pu, cp[side], telecentric);
        Eigen::Vector3d X; double s = 0;
        if (!red_math::intersectRayPlane(ray, plane, X, s)) continue;
        min_sin = std::min(min_sin, s);
        fa.kp3d[node].x = X(0);
        fa.kp3d[node].y = X(1);
        fa.kp3d[node].z = X(2);
        fa.kp3d[node].set_triangulated();
        // Single manual side label drives it → treat as reviewed iff manual.
        fa.kp3d[node].reviewed = (kp.source == LabelSource::Manual);
        n_solved++;

        // Reproject into every OTHER view for verification (keep the side
        // camera's manual label as the source of truth).
        for (int v = 0; v < nc; v++) {
            if (v == side) continue;
            if (v >= (int)fa.cameras.size()) continue;
            if (node >= (u32)fa.cameras[v].keypoints.size()) continue;
            double rx, ry;
            if (reproject_3d_to_cam(X, cp[v], scene->image_width[v],
                                    scene->image_height[v], rx, ry)) {
                fa.cameras[v].keypoints[node].x = rx;
                fa.cameras[v].keypoints[node].y = ry;
                fa.cameras[v].keypoints[node].labeled = true;
                fa.cameras[v].keypoints[node].source = LabelSource::Predicted;
            }
        }
    }

    if (n_solved == 0) {
        status = "No labeled keypoints in the side camera to solve";
        return false;
    }

    double ang_deg = std::asin(std::min(1.0, std::max(0.0, min_sin))) * 180.0 / M_PI;
    std::ostringstream oss;
    oss << (m.force_vertical ? "Solved " : "Solved ") << n_solved
        << (m.force_vertical ? " pt(s) [force-vertical]" : " pt(s) [preimage plane]")
        << ", side/plane angle " << std::fixed << std::setprecision(1) << ang_deg << "°";
    if (min_sin < 0.15)
        status = "WARNING edge-on: " + oss.str() +
                 " — side camera nearly parallel to the plane; pick a more head-on side camera";
    else
        status = oss.str();
    return true;
}

inline void reprojection(FrameAnnotation &fa, SkeletonContext *skeleton,
                         const std::vector<CameraParams> &camera_params,
                         RenderScene *scene) {

    // 2D / uncalibrated projects have no projection matrices: there is nothing
    // to triangulate or reproject, and indexing camera_params[] would be out of
    // bounds. Bail so the per-camera 2D labels stand on their own. (Guards every
    // reprojection() call site at once — T-key, JARVIS, the Triangulate button.)
    if (camera_params.empty())
        return;

    bool telecentric = camera_params[0].telecentric;

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
                            camera_params[view_idx].dist_coeffs,
                            camera_params[view_idx].dist_center);
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
            // Reviewed=true iff every contributing 2D label was Manual.
            // Mixed (manual + predicted) contributions count as un-reviewed
            // until the user explicitly approves the resulting 3D point.
            bool all_manual = true;
            for (u32 view_idx = 0; view_idx < scene->num_cams; view_idx++) {
                if (view_idx >= (u32)fa.cameras.size()) continue;
                if (node >= (u32)fa.cameras[view_idx].keypoints.size()) continue;
                const auto &kp2d = fa.cameras[view_idx].keypoints[node];
                if (kp2d.labeled && kp2d.source != LabelSource::Manual) {
                    all_manual = false;
                    break;
                }
            }
            fa.kp3d[node].set_triangulated();
            fa.kp3d[node].reviewed = all_manual;

            for (u32 view_idx = 0; view_idx < scene->num_cams; view_idx++) {
                if (view_idx >= (u32)fa.cameras.size()) continue;
                if (node >= (u32)fa.cameras[view_idx].keypoints.size()) continue;

                if (telecentric) {
                    // Telecentric reprojection
                    auto reproj = red_math::projectPointTelecentric(
                        pt3d,
                        camera_params[view_idx].projection_mat,
                        camera_params[view_idx].k,
                        camera_params[view_idx].dist_coeffs,
                        camera_params[view_idx].dist_center);
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
