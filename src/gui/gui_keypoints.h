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

// Per-animal tint. Applied to every node so two animals are told apart at a
// glance even when they share a skeleton -- which they always do, since the
// keypoint axis is per session, not per animal.
inline ImVec4 instance_tint(int instance) {
    static const ImVec4 kTints[] = {
        {1.00f, 1.00f, 1.00f, 1.0f},  // 0: untinted, the animal being labelled
        {1.00f, 0.55f, 0.35f, 1.0f},
        {0.45f, 0.80f, 1.00f, 1.0f},
        {0.60f, 1.00f, 0.55f, 1.0f},
        {1.00f, 0.75f, 0.95f, 1.0f},
        {1.00f, 0.90f, 0.40f, 1.0f},
    };
    constexpr int n = (int)(sizeof(kTints) / sizeof(kTints[0]));
    return kTints[instance <= 0 ? 0 : 1 + ((instance - 1) % (n - 1))];
}

// `instance` separates one animal's draggable points from another's: ImPlot
// keys DragPoint by id, so without it five animals would share one point per
// node and dragging any would move them together.
// Returns true if the user touched one of this instance's points this frame
// (clicked or dragged). The caller uses that to make the animal you just
// grabbed the one the Labeling Tool is editing -- otherwise you drag animal 2
// and every panel keeps reporting animal 0.
// Which keypoint the right-click context menu is acting on. gui_plot_keypoints
// runs once per (view, instance), so the menu has to remember which call owns
// it -- otherwise every view would draw the same popup.
struct KeypointMenuTarget {
    int view = -1;
    int instance = -1;
    u32 node = 0;
};
inline KeypointMenuTarget &keypoint_menu_target() {
    static KeypointMenuTarget t;
    return t;
}

inline bool reproject_3d_to_cam(const Eigen::Vector3d &pt3d,
                                const CameraParams &cp, int W, int H,
                                double &out_x, double &out_y);

// Draws only. Every position it needs is already in the keypoints, including
// an occluded node's -- reprojection() refreshes those -- so it takes no
// calibration and does no projecting of its own.
inline bool gui_plot_keypoints(FrameAnnotation &fa, SkeletonContext *skeleton,
                               int view_idx, int num_cams,
                               ImVec4 active_color = ImVec4(1, 1, 1, 1),
                               int instance = 0, bool is_active = true,
                               bool show_names = false) {
    if (view_idx >= (int)fa.cameras.size()) return false;
    auto &cam = fa.cameras[view_idx];
    bool touched = false;
    // Whether the cursor is on a marker. R means "delete what you are pointing
    // at", and that is only knowable here: the shortcut block in red.cpp runs
    // before this function each frame, so it cannot see the hover state.
    bool any_point_hovered = false;

    // Where each node is drawn in this view, and why. A labelled node is at
    // its own 2D; an occluded one has none, so it falls back to where the
    // frame's 3D projects. Computed once because both the marker and the
    // skeleton edges need it -- edges used to require both ends `labeled`, so
    // occluding one node cut every limb through it and the skeleton fell apart.
    struct NodeDraw { double x = 0, y = 0; bool has = false; bool inferred = false; };
    // Reused rather than allocated: this runs once per camera per animal per
    // frame -- eighty times on a sixteen-camera, five-animal rig -- and a
    // fresh vector each time would be eighty allocations a frame for a buffer
    // that is always the same size. ImGui is single-threaded, so one is safe.
    static std::vector<NodeDraw> draw_pos;
    draw_pos.assign((size_t)skeleton->num_nodes, NodeDraw{});
    for (u32 n = 0; n < skeleton->num_nodes; n++) {
        if (n >= (u32)cam.keypoints.size()) break;
        const Keypoint2D &kp = cam.keypoints[n];
        if (kp.usable()) {
            draw_pos[n] = {kp.x, kp.y, true, false};
        } else if (kp.is_occluded() && kp.x != UNLABELED && kp.y != UNLABELED) {
            // Just the stored position. set_occluded keeps x/y, and the
            // reprojection below refreshes it for occluded nodes too, so this
            // is already where the solve says the hidden part is -- no reason
            // to project it a second time here.
            draw_pos[n] = {kp.x, kp.y, true, true};
        }
    }

    float pt_size = 6.0f;
    // The ACTIVE node is drawn last. Two keypoints of one animal can sit on
    // the same pixel -- and legitimately do, when the anatomy overlaps in a
    // view -- and ImGui gives a contested hover to whichever item was
    // submitted last. Drawing 0..n-1 in order meant the higher node index
    // always won, so the other one could not be grabbed at all. Now A/D/Q/E
    // choose which node is reachable, and hit-testing agrees with the
    // highlight that already marks the active node as foremost.
    auto draw_node = [&](u32 node) {
        if (node >= (u32)cam.keypoints.size()) return;

        // An occluded keypoint has no 2D coordinates -- that is what the
        // assessment means -- but the frame's 3D for that node usually still
        // exists, solved from the cameras that CAN see it. Drawing nothing
        // made an assessed node look identical to one nobody has reached yet.
        // A cross marks where that 3D lands in this view: not something to
        // drag, a note saying "it is here, you just cannot see it".
        if (draw_pos[node].has && draw_pos[node].inferred) {
            const ImVec2 px = ImPlot::PlotToPixels(draw_pos[node].x,
                                                   draw_pos[node].y);
            // Marked as the active node the same way a live point is: the
            // user's active colour, drawn bigger and heavier. Without it an
            // occluded node could be selected with no sign of it, so there
            // was no way to tell which node M was about to act on -- and M is
            // a toggle, so that is the difference between marking and
            // unmarking. `is_active` below is the ANIMAL, a separate thing.
            const bool node_is_active = (cam.active_id == node);
            ImVec4 c = node_is_active
                           ? active_color
                           : (node < skeleton->node_colors.size()
                                  ? skeleton->node_colors.at(node)
                                  : ImVec4(1, 1, 1, 1));
            c.w = is_active ? 0.85f : 0.45f;
            const ImU32 col = ImGui::ColorConvertFloat4ToU32(c);
            const float r = node_is_active ? 7.0f : 5.0f;
            const float thick = node_is_active ? 2.4f : 1.6f;
            ImDrawList *dl = ImPlot::GetPlotDrawList();
            dl->AddLine(ImVec2(px.x - r, px.y - r), ImVec2(px.x + r, px.y + r),
                        col, thick);
            dl->AddLine(ImVec2(px.x - r, px.y + r), ImVec2(px.x + r, px.y - r),
                        col, thick);

            // The cross is a target, not just a note. Marking a point occluded
            // used to put it out of reach entirely -- no hover, no tooltip, no
            // menu, no drag -- and the only way back was to make the node
            // active from the keyboard and place it again. An invisible
            // DragPoint (alpha 0, the same trick the ring uses) gives it
            // every interaction a visible point has. Safe to share the id
            // formula: usable() is has_pos && !occluded, so only one of the two
            // branches ever runs for a node.
            double hx = draw_pos[node].x, hy = draw_pos[node].y;
            bool occ_clicked = false, occ_hovered = false;
            const int occ_id = (skeleton->num_nodes * num_cams) * instance +
                               skeleton->num_nodes * view_idx + node;
            const bool occ_dragged = ImPlot::DragPoint(
                occ_id, &hx, &hy, ImVec4(0, 0, 0, 0), r + 2.0f,
                ImPlotDragToolFlags_None, &occ_clicked, &occ_hovered);

            Keypoint2D &okp = cam.keypoints[node];
            if (occ_dragged) {
                // Dragging it onto the part takes the assessment back: you
                // could see it after all, and you have just said where. Same
                // as dragging a derived point -- it becomes yours, and the
                // old solve it no longer agrees with goes.
                okp.x = hx;
                okp.y = hy;
                okp.set_manual();
                if (node < fa.kp3d.size()) fa.kp3d[node].clear();
                touched = true;
            }
            if (occ_clicked) {
                cam.active_id = node;
                touched = true;
            }
            if (occ_hovered) {
                any_point_hovered = true;
                std::string label;
                if (node < skeleton->node_names.size())
                    label = skeleton->node_names[node];
                if (!label.empty()) label += ": ";
                label += "occluded";
                ImVec2 mouse_pos = ImGui::GetMousePos();
                ImGui::GetForegroundDrawList()->AddText(
                    ImVec2(mouse_pos.x + 10, mouse_pos.y + 10),
                    IM_COL32(220, 20, 60, 255), label.c_str());

                if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
                    KeypointMenuTarget &t = keypoint_menu_target();
                    t.view = view_idx;
                    t.instance = instance;
                    t.node = node;
                    ImGui::OpenPopup("##keypoint_menu");
                }

                // M here takes the assessment back. set_occluded touched only
                // visibility, so this restores the whole point -- author,
                // position and origin were never lost, and a point you placed
                // returns to Observed rather than to "not judged".
                if (ImGui::IsKeyPressed(ImGuiKey_M, false)) {
                    okp.clear_occluded();
                    touched = true;
                }

                if (ImGui::IsKeyPressed(ImGuiKey_R, false)) {
                    okp = Keypoint2D{};
                    cam.active_id = node;
                    touched = true;
                }
            }
        }

        if (cam.keypoints[node].usable()) {
            ImVec4 node_color;
            if (cam.active_id == node) {
                node_color = active_color; // active keypoint: user-selected color
                pt_size = 8.0f;
            } else {
                node_color = skeleton->node_colors.at(node);
                pt_size = 6.0f;
            }
            node_color.w = 0.9f;

            // Tint by animal, and dim the ones that are not being edited.
            const ImVec4 tint = instance_tint(instance);
            node_color.x *= tint.x; node_color.y *= tint.y; node_color.z *= tint.z;
            if (!is_active) { node_color.w *= 0.55f; pt_size *= 0.8f; }
            int id = (skeleton->num_nodes * num_cams) * instance +
                     skeleton->num_nodes * view_idx + node;
            bool drag_point_clicked;
            bool drag_point_hovered;
            bool drag_point_modified;
            // Every point is a circle. What the fill says is the one thing
            // that changes what happens next: FILLED will feed the next
            // solve, RING will not.
            //
            // The same predicate reprojection() counts with, so the overlay
            // cannot disagree with the solve about which points are inputs --
            // that was the question the old triangle was standing in for, one
            // step removed, by showing provenance and leaving you to work out
            // the consequence.
            //
            // Provenance itself is a per-node question and the keypoints
            // table answers it properly: author, coordinate origin and
            // visibility as separate columns, rather than one shape trying to
            // carry a three-valued field.
            const Keypoint2D &kpn = cam.keypoints[node];
            const bool feeds_solve = kpn.usable() && kpn.is_manual();

            // DragPoint only ever draws a filled circle in the colour given,
            // so a transparent one keeps the hit-testing and drag while the
            // ring goes on by hand.
            ImVec4 marker_color = node_color;
            if (!feeds_solve) marker_color.w = 0.0f;

            drag_point_modified = ImPlot::DragPoint(
                id, &cam.keypoints[node].x,
                &cam.keypoints[node].y, marker_color,
                pt_size, ImPlotDragToolFlags_None, &drag_point_clicked,
                &drag_point_hovered);

            if (!feeds_solve) {
                const ImVec2 c = ImPlot::PlotToPixels(kpn.x, kpn.y);
                ImPlot::GetPlotDrawList()->AddCircle(
                    c, pt_size, ImGui::ColorConvertFloat4ToU32(node_color),
                    0, 2.0f);
            }

            if (drag_point_modified) {
                // A drag turns a projected point back into a user annotation.
                cam.keypoints[node].set_manual();
                fa.kp3d[node].clear();
                touched = true;
            }

            // Draw the skeleton name just to the right of each point when
            // requested. PlotToPixels keeps the label attached while the user
            // pans or zooms the image.
            if (show_names && node < skeleton->node_names.size()) {
                const std::string &name = skeleton->node_names[node];
                if (!name.empty()) {
                    const ImVec2 point_px = ImPlot::PlotToPixels(
                        cam.keypoints[node].x, cam.keypoints[node].y);
                    const float font_size = ImGui::GetFontSize() * 0.75f;
                    ImDrawList *draw_list = ImPlot::GetPlotDrawList();
                    const ImVec2 text_pos(point_px.x + 9.0f,
                                          point_px.y - font_size * 0.5f);
                    draw_list->AddText(ImGui::GetFont(), font_size,
                                       text_pos,
                                       ImGui::ColorConvertFloat4ToU32(node_color),
                                       name.c_str());
                }
            }

            if (drag_point_hovered) {
                any_point_hovered = true;
                std::string label;
                if (node < skeleton->node_names.size())
                    label = skeleton->node_names[node];
                if (fa.kp3d[node].exist) {
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(2);
                    oss << "(" << fa.kp3d[node].x << ", "
                        << fa.kp3d[node].y << ", "
                        << fa.kp3d[node].z << ")";
                    if (!label.empty()) label += ": ";
                    label += oss.str();
                }
                if (!label.empty()) {
                    ImVec2 mouse_pos = ImGui::GetMousePos();
                    ImVec2 textPos = ImVec2(mouse_pos.x + 10, mouse_pos.y + 10);
                    ImGui::GetForegroundDrawList()->AddText(
                        textPos, IM_COL32(220, 20, 60, 255), label.c_str());
                }

                if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
                    KeypointMenuTarget &t = keypoint_menu_target();
                    t.view = view_idx;
                    t.instance = instance;
                    t.node = node;
                    ImGui::OpenPopup("##keypoint_menu");
                }

                if (ImGui::IsKeyPressed(ImGuiKey_M, false)) {
                    // Clear the 3D only if this point was an INPUT to it.
                    // Triangulation uses manual points alone, so occluding a
                    // reprojection -- the common case, a solve landing where
                    // the body hides the part -- says nothing about a solve it
                    // never fed. Authorship is untouched either way.
                    const bool fed_solve = cam.keypoints[node].usable() &&
                                           cam.keypoints[node].is_manual();
                    cam.keypoints[node].set_occluded();
                    if (fed_solve && node < fa.kp3d.size())
                        fa.kp3d[node].clear();
                    // Advance, same as the active-node path. M means "mark
                    // this one and move on" wherever it is pressed.
                    cam.active_id =
                        (node < skeleton->num_nodes - 1) ? node + 1 : node;
                }

                if (ImGui::IsKeyPressed(ImGuiKey_R,
                                        false)) // delete the hovered keypoint
                {
                    cam.keypoints[node] = Keypoint2D{}; // reset all fields
                    cam.active_id = node;
                }

                if (ImGui::IsKeyPressed(
                        ImGuiKey_F,
                        false)) // Delete this keypoint from all the views
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
                touched = true;
            }
        }
    };
    for (u32 node = 0; node < skeleton->num_nodes; node++)
        if (node != cam.active_id)
            draw_node(node);
    if (cam.active_id < skeleton->num_nodes)
        draw_node(cam.active_id);

    // The context menu for the keypoint that was right-clicked. Drawn after
    // the node loop so it is not nested inside a DragPoint's item scope.
    {
        KeypointMenuTarget &t = keypoint_menu_target();
        if (t.view == view_idx && t.instance == instance &&
            t.node < (u32)cam.keypoints.size() && t.node < fa.kp3d.size()) {
            if (ImGui::BeginPopup("##keypoint_menu")) {
                Keypoint2D &kp = cam.keypoints[t.node];
                if (t.node < skeleton->node_names.size())
                    ImGui::TextDisabled("%s",
                                        skeleton->node_names[t.node].c_str());
                ImGui::Separator();

                // Only Manual points feed triangulation, so this is what
                // promotes a reprojection you have judged correct into an
                // input for the next solve.
                ImGui::BeginDisabled(kp.is_manual());
                if (ImGui::MenuItem("Accept as manual")) {
                    // Accepting the position as your own: it is yours now, and
                    // the numbers are the ones you accepted rather than a
                    // pending solve's.
                    kp.set_manual();
                }
                ImGui::EndDisabled();
                if (kp.is_manual() && ImGui::IsItemHovered(
                                         ImGuiHoveredFlags_AllowWhenDisabled))
                    ImGui::SetTooltip("Already a manual label");

                ImGui::Separator();

                // Visibility, all three states spelled out, with the one you
                // are already in shown as selected. A judgement you make now
                // and then, not something to type -- which is why there is no
                // shortcut for any of them.
                ImGui::TextDisabled("Can you see it here?");
                if (ImGui::MenuItem("Visible", nullptr, kp.is_observed())) {
                    kp.vis = Keypoint2D::Vis::Observed;
                }
                if (ImGui::MenuItem("Occluded", nullptr, kp.is_occluded())) {
                    // Clearing the 3D only if this point was an INPUT to it.
                    // Triangulation uses manual points alone, so occluding a
                    // reprojection -- the common case, a solve landing where
                    // the body hides the part -- says nothing about a solve it
                    // never fed.
                    const bool fed_solve = kp.usable() && kp.is_manual();
                    kp.set_occluded();
                    if (fed_solve) fa.kp3d[t.node].clear();
                }
                if (ImGui::MenuItem("Not judged", nullptr,
                                    kp.vis == Keypoint2D::Vis::Unknown)) {
                    kp.vis = Keypoint2D::Vis::Unknown;
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Delete")) {
                    kp = Keypoint2D{};
                    cam.active_id = t.node;
                }
                ImGui::EndPopup();
            }
        }
    }

    // M handles both cases, which widens what PR #26 wrote. Its comment said M
    // was "intentionally tied to the active node, not to hovering an existing
    // point", reasoning that an occluded point has no coordinates and so has
    // no marker to aim at. That holds for the node-with-no-marker case, which
    // is exactly this fallback -- but the common case is the other one: a
    // point is drawn, you can see it, and you have decided it is occluded.
    // Requiring it to be made active first is a step for nothing.
    //
    // With no marker under the cursor there is nothing to point at, so R and M
    // fall back to the active node -- neither then needs pixel-perfect
    // placement. Both live here rather than with the other shortcuts in
    // red.cpp because that block runs BEFORE this function each frame and so
    // cannot know what the cursor is on.
    const bool plot_keys_ok = is_active && !any_point_hovered &&
                              ImPlot::IsPlotHovered() &&
                              !ImGui::GetIO().WantTextInput &&
                              cam.active_id < cam.keypoints.size();

    if (plot_keys_ok && ImGui::IsKeyPressed(ImGuiKey_R, false)) {
        cam.keypoints[cam.active_id] = Keypoint2D{};
    }

    // With nothing under the cursor, M acts on the active node. This is the
    // case the menu cannot reach: an unlabelled node has no marker to
    // right-click, so without this there is no way to mark it occluded at all.
    //
    // A toggle, like the hovered paths. Only MARKING advances, so M can be
    // tapped down a skeleton; unmarking stays put, because you are undoing
    // this node rather than moving past it.
    if (plot_keys_ok && ImGui::IsKeyPressed(ImGuiKey_M, false)) {
        Keypoint2D &akp = cam.keypoints[cam.active_id];
        if (akp.is_occluded()) {
            akp.clear_occluded();
        } else {
            const bool fed_solve = akp.usable() && akp.is_manual();
            akp.set_occluded();
            if (fed_solve && cam.active_id < fa.kp3d.size())
                fa.kp3d[cam.active_id].clear();
            if (cam.active_id < skeleton->num_nodes - 1)
                cam.active_id++;
        }
    }

    for (u32 edge = 0; edge < skeleton->num_edges; edge++) {
        auto [a, b] = skeleton->edges[edge];

        if (a >= (u32)draw_pos.size() || b >= (u32)draw_pos.size()) continue;
        if (!draw_pos[a].has || !draw_pos[b].has) continue;

        double xs[2]{draw_pos[a].x, draw_pos[b].x};
        double ys[2]{draw_pos[a].y, draw_pos[b].y};
        if (draw_pos[a].inferred || draw_pos[b].inferred) {
            // A limb reaching an occluded node is DASHED: the segment is real,
            // but one end is inferred from the 3D rather than seen in this
            // camera. Dashes rather than a fainter line, for the same reason
            // the markers use shape -- a washed-out line has to compete with
            // whatever the frame shows underneath it.
            //
            // Drawn by hand: ImPlot::PlotLine has no dash pattern.
            const ImVec2 p0 = ImPlot::PlotToPixels(xs[0], ys[0]);
            const ImVec2 p1 = ImPlot::PlotToPixels(xs[1], ys[1]);
            const float dx = p1.x - p0.x, dy = p1.y - p0.y;
            const float len = std::sqrt(dx * dx + dy * dy);
            if (len > 0.5f) {
                const float ux = dx / len, uy = dy / len;
                const ImU32 col = ImGui::ColorConvertFloat4ToU32(
                    ImVec4(0.85f, 0.85f, 0.85f, is_active ? 0.9f : 0.45f));
                ImDrawList *dl = ImPlot::GetPlotDrawList();
                const float dash = 6.0f, gap = 4.0f;
                for (float t = 0.0f; t < len; t += dash + gap) {
                    const float e = std::min(t + dash, len);
                    dl->AddLine(ImVec2(p0.x + ux * t, p0.y + uy * t),
                                ImVec2(p0.x + ux * e, p0.y + uy * e), col, 1.6f);
                }
            }
        } else {
            ImPlot::PlotLine("##line", xs, ys, 2);
        }
    }
    return touched;
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
        if (!kp.usable()) continue;
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
        n_solved++;

        // Reproject into every OTHER view for verification (keep the side
        // camera's manual label as the source of truth).
        for (int v = 0; v < nc; v++) {
            if (v == side) continue;
            if (v >= (int)fa.cameras.size()) continue;
            if (node >= (u32)fa.cameras[v].keypoints.size()) continue;
            if (fa.cameras[v].keypoints[node].is_occluded()) continue;
            double rx, ry;
            if (reproject_3d_to_cam(X, cp[v], scene->image_width[v],
                                    scene->image_height[v], rx, ry)) {
                fa.cameras[v].keypoints[node].x = rx;
                fa.cameras[v].keypoints[node].y = ry;
                fa.cameras[v].keypoints[node].vis = Keypoint2D::Vis::Unknown;
                fa.cameras[v].keypoints[node].set_reprojected();
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
                fa.cameras[view_idx].keypoints[node].usable() &&
                fa.cameras[view_idx].keypoints[node].is_manual()) {
                num_views_labeled++;
            }
        }

        if (num_views_labeled >= 2) {

            std::vector<Eigen::Vector2d> undist_pts;
            std::vector<Eigen::Matrix<double, 3, 4>> proj_mats;

            for (u32 view_idx = 0; view_idx < scene->num_cams; view_idx++) {
                if (view_idx >= (u32)fa.cameras.size()) continue;
                if (node >= (u32)fa.cameras[view_idx].keypoints.size()) continue;
                if (fa.cameras[view_idx].keypoints[node].usable() &&
                    fa.cameras[view_idx].keypoints[node].is_manual()) {
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

            // Always solve from the current 2D observations, even if this
            // keypoint already has a triangulated 3D value.
            Eigen::Vector3d pt3d =
                red_math::triangulatePoints(undist_pts, proj_mats);

            fa.kp3d[node].x = pt3d(0);
            fa.kp3d[node].y = pt3d(1);
            fa.kp3d[node].z = pt3d(2);
            fa.kp3d[node].set_triangulated();

            for (u32 view_idx = 0; view_idx < scene->num_cams; view_idx++) {
                if (view_idx >= (u32)fa.cameras.size()) continue;
                if (node >= (u32)fa.cameras[view_idx].keypoints.size()) continue;

                // Refresh every camera's coordinates, but retain the visible
                // status of a point that was explicitly user-annotated. A
                // refreshed user annotation remains visible; derived values
                // are marked projected and are excluded from the next solve.
                auto &kp2d = fa.cameras[view_idx].keypoints[node];
                // An occluded node gets its coordinates refreshed too -- the
                // overlay draws the cross straight from x/y, so this is what
                // keeps it on the part as the solve moves. Only the position
                // and its origin are touched: `exist` stays false and the
                // assessment stands, because where the part is and whether you
                // can see it are different questions.
                const bool coords_only = kp2d.is_occluded();
                // `manual` alone: an occluded point is the one place a manual
                // point has no coordinates, and it is excluded just above.
                if (!coords_only && !kp2d.is_manual()) kp2d = Keypoint2D{};

                // Land the projection first, decide what it means second.
                // Both camera models answer the same question -- is there a
                // pixel in THIS image for that 3D point -- and the answer has
                // to be acted on in one place, because "no" is not the same as
                // "leave whatever was there".
                double x = 0.0, y = 0.0;
                bool in_frame = false;
                if (telecentric) {
                    auto reproj = red_math::projectPointTelecentric(
                        pt3d,
                        camera_params[view_idx].projection_mat,
                        camera_params[view_idx].k,
                        camera_params[view_idx].dist_coeffs,
                        camera_params[view_idx].dist_center);
                    x = reproj(0);
                    y = double(scene->image_height[view_idx]) - reproj(1);
                    in_frame = x > 0 && x < scene->image_width[view_idx] &&
                               y > 0 && y < scene->image_height[view_idx];
                } else if (is_in_camera_fov(pt3d, camera_params[view_idx].r,
                                            camera_params[view_idx].tvec,
                                            camera_params[view_idx].k,
                                            scene->image_width[view_idx],
                                            scene->image_height[view_idx])) {
                    // Matrix-based, safe for det(R)=-1.
                    auto reproj = red_math::projectPointR(
                        pt3d, camera_params[view_idx].r,
                        camera_params[view_idx].tvec,
                        camera_params[view_idx].k,
                        camera_params[view_idx].dist_coeffs);
                    x = reproj(0);
                    y = double(scene->image_height[view_idx]) - reproj(1);
                    in_frame = x > 0 && x < scene->image_width[view_idx] &&
                               y > 0 && y < scene->image_height[view_idx];
                }

                if (coords_only) {
                    // The assessment and its author stand either way; only the
                    // position is in question. Out of frame there is nowhere to
                    // put the cross, and the old coordinates are a place the
                    // solve has already left -- so drop them rather than draw a
                    // marker at a spot nothing is at any more. The next solve
                    // that lands in frame puts them back.
                    // has_pos follows the coordinates, which is the whole
                    // point of it being its own flag: an occluded node the
                    // solve can place HAS a position (the cross is drawn from
                    // it), and one the solve cannot place does not.
                    if (in_frame) {
                        kp2d.x = x;
                        kp2d.y = y;
                        kp2d.has_pos = true;
                        kp2d.reprojected = true;
                    } else {
                        kp2d.x = UNLABELED;
                        kp2d.y = UNLABELED;
                        kp2d.has_pos = false;
                        kp2d.reprojected = false;
                    }
                    continue;
                }

                if (in_frame) {
                    kp2d.x = x;
                    kp2d.y = y;
                    kp2d.vis = Keypoint2D::Vis::Unknown;
                    // Coordinate origin only. Authorship is untouched: a
                    // hand-placed point stays manual through a refresh, and one
                    // nobody placed simply has no author -- it was not
                    // predicted by anything, it was computed.
                    kp2d.set_reprojected();
                }
            }
        }
    }
}
