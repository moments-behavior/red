#pragma once
#include "implot.h"
#include "skeleton.h"
#include <cmath>

static void calculate_obb_properties(OrientedBoundingBox *obb) {
    if (obb->state < OBBSecondAxisPoint)
        return;

    // The first two points are vertices of the OBB (one edge)
    ImVec2 vertex1 = obb->axis_point1;
    ImVec2 vertex2 = obb->axis_point2;

    // Calculate the vector along the edge defined by the first two points
    ImVec2 edge_vector = {vertex2.x - vertex1.x, vertex2.y - vertex1.y};

    // Calculate perpendicular vector (90 degrees rotated)
    ImVec2 perp_vector = {-edge_vector.y, edge_vector.x};

    if (obb->state >= OBBThirdPoint) {
        // We have the third point - calculate the final OBB
        ImVec2 mouse_point = obb->corner_point;

        // Project the mouse point onto the perpendicular direction to find the
        // height
        ImVec2 to_mouse = {mouse_point.x - vertex1.x,
                           mouse_point.y - vertex1.y};

        // Calculate the perpendicular distance (this becomes the height of the
        // rectangle)
        float perp_dot =
            to_mouse.x * perp_vector.x + to_mouse.y * perp_vector.y;
        float perp_length_sq =
            perp_vector.x * perp_vector.x + perp_vector.y * perp_vector.y;

        if (perp_length_sq > 0) {
            float height = fabsf(perp_dot) / sqrtf(perp_length_sq);

            // Normalize the perpendicular vector
            float perp_length = sqrtf(perp_length_sq);
            ImVec2 perp_unit = {perp_vector.x / perp_length,
                                perp_vector.y / perp_length};

            // Determine which side of the edge the mouse is on
            float side_sign = perp_dot > 0 ? 1.0f : -1.0f;

            // Calculate the four vertices of the rectangle
            // vertex1 and vertex2 are already defined
            ImVec2 vertex3 = {vertex2.x + height * perp_unit.x * side_sign,
                              vertex2.y + height * perp_unit.y * side_sign};
            ImVec2 vertex4 = {vertex1.x + height * perp_unit.x * side_sign,
                              vertex1.y + height * perp_unit.y * side_sign};

            // Calculate center, width, height, and rotation
            obb->center.x =
                (vertex1.x + vertex2.x + vertex3.x + vertex4.x) / 4.0f;
            obb->center.y =
                (vertex1.y + vertex2.y + vertex3.y + vertex4.y) / 4.0f;

            obb->width = sqrtf(edge_vector.x * edge_vector.x +
                               edge_vector.y * edge_vector.y);
            obb->height = height;
            obb->rotation = atan2f(edge_vector.y, edge_vector.x);
        }
    }
}

static void calculate_obb_preview(OrientedBoundingBox *obb, ImVec2 mouse_pos) {
    if (obb->state < OBBSecondAxisPoint)
        return;

    // Temporarily set the corner point to mouse position for preview
    // calculation
    ImVec2 original_corner = obb->corner_point;
    obb->corner_point = mouse_pos;

    // Calculate properties for preview
    calculate_obb_properties(obb);

    // Restore original corner point
    obb->corner_point = original_corner;
}

static bool is_point_near(ImVec2 point1, ImVec2 point2,
                          float threshold = 15.0f) {
    float dx = point1.x - point2.x;
    float dy = point1.y - point2.y;
    float distance_sq = dx * dx + dy * dy;
    return distance_sq < (threshold * threshold);
}

void draw_obb(OrientedBoundingBox &obb, bool is_active,
              ImVec4 class_color = ImVec4(1, 1, 1, 0.7f),
              ImVec2 mouse_pos = ImVec2(0, 0), bool show_preview = false) {
    if (obb.state == OBBNull)
        return;

    // Draw construction points during creation
    if (obb.state < OBBComplete) {
        // Draw first vertex
        if (obb.state >= OBBFirstAxisPoint) {
            double x1 = obb.axis_point1.x, y1 = obb.axis_point1.y;
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 6,
                                       ImVec4(1, 0, 0, 1), IMPLOT_AUTO,
                                       ImVec4(1, 0, 0, 1));
            ImPlot::PlotScatter("##obb_vertex1", &x1, &y1, 1);

            if (show_preview && obb.state == OBBFirstAxisPoint) {
                double xs_preview[2] = {obb.axis_point1.x, mouse_pos.x};
                double ys_preview[2] = {obb.axis_point1.y, mouse_pos.y};
                ImPlot::SetNextLineStyle(ImVec4(1, 0, 0, 0.6f), 2.0f);
                ImPlot::PlotLine("##obb_preview_line", xs_preview, ys_preview,
                                 2);
            }
        }

        // Draw second vertex and the edge between them
        if (obb.state >= OBBSecondAxisPoint) {
            double x2 = obb.axis_point2.x, y2 = obb.axis_point2.y;
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 6,
                                       ImVec4(0, 1, 0, 1), IMPLOT_AUTO,
                                       ImVec4(0, 1, 0, 1));
            ImPlot::PlotScatter("##obb_vertex2", &x2, &y2, 1);

            // Draw the edge line between first two vertices
            double xs[2] = {obb.axis_point1.x, obb.axis_point2.x};
            double ys[2] = {obb.axis_point1.y, obb.axis_point2.y};
            ImPlot::SetNextLineStyle(ImVec4(0, 1, 0, 0.8f), 3.0f);
            ImPlot::PlotLine("##obb_edge_base", xs, ys, 2);

            // Show preview rectangle if mouse position is provided
            if (show_preview && obb.state == OBBSecondAxisPoint) {
                // Create a temporary OBB for preview calculation
                OrientedBoundingBox preview_obb = obb;
                preview_obb.corner_point = mouse_pos;
                preview_obb.state = OBBThirdPoint;
                calculate_obb_properties(&preview_obb);

                // Draw preview rectangle with dashed/transparent style
                if (preview_obb.width > 0 && preview_obb.height > 0) {
                    float cos_rot = cosf(preview_obb.rotation);
                    float sin_rot = sinf(preview_obb.rotation);
                    float half_width = preview_obb.width / 2.0f;
                    float half_height = preview_obb.height / 2.0f;

                    ImVec2 corners[4];
                    ImVec2 local_corners[4] = {{-half_width, -half_height},
                                               {half_width, -half_height},
                                               {half_width, half_height},
                                               {-half_width, half_height}};

                    for (int i = 0; i < 4; i++) {
                        corners[i].x = preview_obb.center.x +
                                       local_corners[i].x * cos_rot -
                                       local_corners[i].y * sin_rot;
                        corners[i].y = preview_obb.center.y +
                                       local_corners[i].x * sin_rot +
                                       local_corners[i].y * cos_rot;
                    }

                    // Draw preview rectangle with transparent style
                    ImVec4 preview_color = ImVec4(1, 1, 0, 0.4f);
                    for (int i = 0; i < 4; i++) {
                        int next = (i + 1) % 4;
                        double xs_prev[2] = {corners[i].x, corners[next].x};
                        double ys_prev[2] = {corners[i].y, corners[next].y};
                        ImPlot::SetNextLineStyle(preview_color, 1.5f);
                        ImPlot::PlotLine("##obb_preview", xs_prev, ys_prev, 2);
                    }
                }
            }
        }

        // Draw third point if we have it
        if (obb.state >= OBBThirdPoint) {
            double x3 = obb.corner_point.x, y3 = obb.corner_point.y;
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 6,
                                       ImVec4(0, 0, 1, 1), IMPLOT_AUTO,
                                       ImVec4(0, 0, 1, 1));
            ImPlot::PlotScatter("##obb_corner", &x3, &y3, 1);
        }
    }

    // Draw final OBB when complete
    if (obb.state == OBBComplete || obb.state == OBBThirdPoint) {
        // Calculate the four corners of the OBB
        float cos_rot = cosf(obb.rotation);
        float sin_rot = sinf(obb.rotation);
        float half_width = obb.width / 2.0f;
        float half_height = obb.height / 2.0f;

        ImVec2 corners[4];
        ImVec2 local_corners[4] = {{-half_width, -half_height},
                                   {half_width, -half_height},
                                   {half_width, half_height},
                                   {-half_width, half_height}};

        for (int i = 0; i < 4; i++) {
            corners[i].x = obb.center.x + local_corners[i].x * cos_rot -
                           local_corners[i].y * sin_rot;
            corners[i].y = obb.center.y + local_corners[i].x * sin_rot +
                           local_corners[i].y * cos_rot;
        }

        // Draw the rectangle with class-based color
        ImVec4 color = is_active ? ImVec4(0, 1, 1, 0.9f) : class_color;
        float line_width = obb.state == OBBComplete ? 2.5f : 2.0f;

        for (int i = 0; i < 4; i++) {
            int next = (i + 1) % 4;
            double xs[2] = {corners[i].x, corners[next].x};
            double ys[2] = {corners[i].y, corners[next].y};
            ImPlot::SetNextLineStyle(color, line_width);
            ImPlot::PlotLine("##obb_final", xs, ys, 2);
        }
    }
}

// Helper function to check if a point is near a line segment
bool is_point_near_line_segment(ImVec2 point, ImVec2 line_start,
                                ImVec2 line_end, float threshold = 5.0f) {
    // Calculate distance from point to line segment
    ImVec2 line_vec = {line_end.x - line_start.x, line_end.y - line_start.y};
    ImVec2 point_vec = {point.x - line_start.x, point.y - line_start.y};

    float line_length_sq = line_vec.x * line_vec.x + line_vec.y * line_vec.y;
    if (line_length_sq == 0)
        return false;

    float t =
        (point_vec.x * line_vec.x + point_vec.y * line_vec.y) / line_length_sq;
    t = fmaxf(0.0f, fminf(1.0f, t)); // Clamp to [0,1]

    ImVec2 closest = {line_start.x + t * line_vec.x,
                      line_start.y + t * line_vec.y};
    float dist_sq = (point.x - closest.x) * (point.x - closest.x) +
                    (point.y - closest.y) * (point.y - closest.y);

    return dist_sq <= threshold * threshold;
}

// Function to check if a point is inside an oriented bounding box
bool is_point_inside_obb(ImVec2 point, const OrientedBoundingBox &obb) {
    if (obb.state != OBBComplete)
        return false;

    // Calculate the four corners of the OBB
    float cos_rot = cosf(obb.rotation);
    float sin_rot = sinf(obb.rotation);
    float half_width = obb.width / 2.0f;
    float half_height = obb.height / 2.0f;

    ImVec2 corners[4];
    ImVec2 local_corners[4] = {{-half_width, -half_height},
                               {half_width, -half_height},
                               {half_width, half_height},
                               {-half_width, half_height}};

    for (int i = 0; i < 4; i++) {
        corners[i].x = obb.center.x + local_corners[i].x * cos_rot -
                       local_corners[i].y * sin_rot;
        corners[i].y = obb.center.y + local_corners[i].x * sin_rot +
                       local_corners[i].y * cos_rot;
    }

    // Use ray casting algorithm to check if point is inside polygon
    bool inside = false;
    for (int i = 0, j = 3; i < 4; j = i++) {
        if (((corners[i].y > point.y) != (corners[j].y > point.y)) &&
            (point.x < (corners[j].x - corners[i].x) *
                               (point.y - corners[i].y) /
                               (corners[j].y - corners[i].y) +
                           corners[i].x)) {
            inside = !inside;
        }
    }

    return inside;
}

// Function to handle OBB manipulation for resizing
void handle_obb_dragging(OrientedBoundingBox &obb, ImVec2 mouse_pos,
                         bool is_active_drag) {
    if (obb.state != OBBComplete)
        return;

    // Calculate the four corners of the OBB
    float cos_rot = cosf(obb.rotation);
    float sin_rot = sinf(obb.rotation);
    float half_width = obb.width / 2.0f;
    float half_height = obb.height / 2.0f;

    ImVec2 corners[4];
    ImVec2 local_corners[4] = {{-half_width / 2, -half_height},
                               {half_width / 2, -half_height},
                               {half_width / 2, half_height},
                               {-half_width / 2, half_height}};

    for (int i = 0; i < 4; i++) {
        corners[i].x = obb.center.x + local_corners[i].x * cos_rot -
                       local_corners[i].y * sin_rot;
        corners[i].y = obb.center.y + local_corners[i].x * sin_rot +
                       local_corners[i].y * cos_rot;
    }

    // Only actually modify the OBB if actively dragging
    if (is_active_drag) {
        static ImVec2 last_mouse_pos = mouse_pos;
        static bool first_drag = true;

        if (first_drag) {
            last_mouse_pos = mouse_pos;
            first_drag = false;
            return;
        }

        // Calculate mouse movement
        ImVec2 mouse_delta = {mouse_pos.x - last_mouse_pos.x,
                              mouse_pos.y - last_mouse_pos.y};

        // Check which side is being dragged and resize accordingly
        for (int i = 0; i < 4; i++) {
            int next = (i + 1) % 4;
            if (is_point_near_line_segment(mouse_pos, corners[i], corners[next],
                                           8.0f)) {
                // Calculate side direction and perpendicular direction
                ImVec2 side_vec = {corners[next].x - corners[i].x,
                                   corners[next].y - corners[i].y};
                float side_length =
                    sqrtf(side_vec.x * side_vec.x + side_vec.y * side_vec.y);

                if (side_length > 0) {
                    // Normalize side vector
                    side_vec.x /= side_length;
                    side_vec.y /= side_length;

                    // Calculate perpendicular direction (pointing outward from
                    // center)
                    ImVec2 perp_dir = {-side_vec.y, side_vec.x};

                    // Check if perpendicular points away from center
                    ImVec2 side_center = {
                        (corners[i].x + corners[next].x) / 2.0f,
                        (corners[i].y + corners[next].y) / 2.0f};
                    ImVec2 to_center = {obb.center.x - side_center.x,
                                        obb.center.y - side_center.y};
                    if (perp_dir.x * to_center.x + perp_dir.y * to_center.y >
                        0) {
                        perp_dir.x = -perp_dir.x;
                        perp_dir.y = -perp_dir.y;
                    }

                    // Project mouse movement onto perpendicular direction
                    float perp_movement =
                        mouse_delta.x * perp_dir.x + mouse_delta.y * perp_dir.y;

                    // Update width or height based on which side is being
                    // dragged
                    if (i == 0 || i == 2) { // Top/bottom sides (height)
                        obb.height =
                            fmaxf(5.0f, obb.height + 2.0f * perp_movement);
                    } else { // Left/right sides (width)
                        obb.width =
                            fmaxf(5.0f, obb.width + 2.0f * perp_movement);
                    }
                }
                break;
            }
        }

        last_mouse_pos = mouse_pos;
    } else {
        // Reset drag state when not actively dragging
        static bool reset_needed = true;
        if (reset_needed) {
            static bool first_drag = true;
            first_drag = true;
            reset_needed = false;
        }
    }
}
