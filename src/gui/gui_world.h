#pragma once
#include "implot.h"
#include "camera.h"
#include "red_math.h"
#include "types.h"
#include <vector>

void world_coordinates_projection_points(CameraParams *cvp, int image_height,
                                         double *axis_x_values,
                                         double *axis_y_values, float scale) {
    std::vector<Eigen::Vector3d> world_coordinates = {
        {0.0, 0.0, 0.0},
        {scale * 1.0, 0.0, 0.0},
        {0.0, scale * 1.0, 0.0},
        {0.0, 0.0, scale * 1.0}};

    auto img_pts = red_math::projectPoints(world_coordinates, cvp->rvec,
                                           cvp->tvec, cvp->k,
                                           cvp->dist_coeffs);

    for (int i = 0; i < 4; i++) {
        axis_x_values[i] = img_pts[i](0);
        axis_y_values[i] = image_height - img_pts[i](1);
    }
}

static void gui_plot_world_coordinates(CameraParams *cvp, int cam_id,
                                       int image_height) {
    double axis_x_values[4];
    double axis_y_values[4];
    world_coordinates_projection_points(cvp, image_height, axis_x_values,
                                        axis_y_values, 50);
    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 6.0,
                               ImVec4(1.0, 1.0, 1.0, 1.0));
    ImPlot::SetNextLineStyle(ImVec4(1.0, 1.0, 1.0, 1.0), 3.0);
    std::string name = "World Origin";

    float one_axis_x[2];
    float one_axis_y[2];

    std::vector<triple_f> node_colors = {{1.0f, 1.0f, 1.0f},
                                         {1.0f, 0.0f, 0.0f},
                                         {0.0f, 1.0f, 0.0f},
                                         {0.0f, 0.0f, 1.0f}};

    for (u32 edge = 0; edge < 3; edge++) {
        double xs[2]{axis_x_values[0], axis_x_values[edge + 1]};
        double ys[2]{axis_y_values[0], axis_y_values[edge + 1]};

        double vec2_x = axis_x_values[edge + 1] - axis_x_values[0];
        double vec2_y = axis_y_values[edge + 1] - axis_y_values[0];

        double vec2_norm_x = -vec2_y;
        double vec2_norm_y = vec2_x;

        double arrow_end_1_x =
            axis_x_values[edge + 1] - vec2_x / 2 + vec2_norm_x / 2;
        double arrow_end_1_y =
            axis_y_values[edge + 1] - vec2_y / 2 + vec2_norm_y / 2;

        double arrow_end_2_x =
            axis_x_values[edge + 1] - vec2_x / 2 - vec2_norm_x / 2;
        double arrow_end_2_y =
            axis_y_values[edge + 1] - vec2_y / 2 - vec2_norm_y / 2;

        ImVec4 my_color;
        my_color.w = 0.8f;
        my_color.x = node_colors[edge + 1].x;
        my_color.y = node_colors[edge + 1].y;
        my_color.z = node_colors[edge + 1].z;

        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 6.0, my_color);
        ImPlot::SetNextLineStyle(my_color, 3.0);
        ImPlot::PlotLine(name.c_str(), xs, ys, 2, ImPlotLineFlags_Segments);

        xs[0] = axis_x_values[edge + 1];
        xs[1] = arrow_end_1_x;
        ys[0] = axis_y_values[edge + 1];
        ys[1] = arrow_end_1_y;
        ImPlot::PlotLine(name.c_str(), xs, ys, 2, ImPlotLineFlags_Segments);

        xs[0] = axis_x_values[edge + 1];
        xs[1] = arrow_end_2_x;
        ys[0] = axis_y_values[edge + 1];
        ys[1] = arrow_end_2_y;
        ImPlot::PlotLine(name.c_str(), xs, ys, 2, ImPlotLineFlags_Segments);
    }
}

void gui_arena_projection_points(CameraParams *cvp, int image_height,
                                 float *arena_x, float *arena_y, int n) {
    float radius = 1473.0f;
    std::vector<Eigen::Vector3d> inPts;

    for (int i = 0; i <= n; i++) {
        float angle = (3.14159265358979323846 * 2) * (float(i) / float(n - 1));
        inPts.push_back(Eigen::Vector3d(sin(angle) * radius,
                                        cos(angle) * radius, 0.0));
    }
    // Only project n points (not n+1)
    inPts.resize(n);

    auto img_pts = red_math::projectPoints(inPts, cvp->rvec, cvp->tvec,
                                           cvp->k, cvp->dist_coeffs);

    for (int i = 0; i < n; i++) {
        arena_x[i] = (float)img_pts[i](0);
        arena_y[i] = (float)(image_height - img_pts[i](1));
    }
}

static void gui_plot_perimeter(CameraParams *cvp, int image_height) {
    float arena_x[100];
    float arena_y[100];
    gui_arena_projection_points(cvp, image_height, arena_x, arena_y, 100);
    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 6.0,
                               ImVec4(1.0, 1.0, 1.0, 1.0));
    ImPlot::SetNextLineStyle(ImVec4(1.0, 1.0, 1.0, 1.0), 3.0);
    std::string name = "arena";
    ImPlot::PlotLine(name.c_str(), arena_x, arena_y, 100);
}
