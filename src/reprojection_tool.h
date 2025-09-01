#pragma once
#include "render.h"
#include "skeleton.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits> // for quiet_NaN
#include <limits>
#include <string>
#include <vector>

enum class PlotMode { ByCamera, ByNode };
struct ReprojectionTool {
    bool show_reprojection_error = false;
    PlotMode mode = PlotMode::ByNode;
};
static bool show_error_bars = true; // toggle SD/SEM whiskers
static bool use_sem = false;        // SD vs SEM

// --- Helpers ---------------------------------------------------------------
static inline std::string AbbrevLabel(const std::string &s) {
    if (s.size() >= 3 && (s[0] == 'C' || s[0] == 'c') &&
        (s[1] == 'a' || s[1] == 'A') && (s[2] == 'm' || s[2] == 'M')) {
        std::string digits;
        for (int i = (int)s.size() - 1;
             i >= 0 && std::isdigit((unsigned char)s[(size_t)i]); --i)
            digits.push_back(s[(size_t)i]);
        std::reverse(digits.begin(), digits.end());
        if (!digits.empty()) {
            if (digits.size() > 5)
                digits = digits.substr(digits.size() - 5);
            while (digits.size() < 5)
                digits = '0' + digits;
            return std::string("C") + digits;
        }
    }
    if (s.size() > 10) {
        std::string tail;
        for (int i = (int)s.size() - 1; i >= 0 && (int)tail.size() < 4; --i)
            if (std::isalnum((unsigned char)s[(size_t)i]))
                tail.push_back(s[(size_t)i]);
        std::reverse(tail.begin(), tail.end());
        return std::string(1, (char)std::toupper((unsigned char)s[0])) + tail;
    }
    return s;
}

static inline void MakeAbbrevLabels(const std::vector<int> &cat_idx,
                                    const std::vector<std::string> &full_labels,
                                    std::vector<std::string> &abbrev_storage,
                                    std::vector<const char *> &abbrev_ptrs) {
    abbrev_storage.clear();
    abbrev_ptrs.clear();
    abbrev_storage.reserve(cat_idx.size());
    abbrev_ptrs.reserve(cat_idx.size());
    for (int idx : cat_idx) {
        abbrev_storage.push_back(AbbrevLabel(full_labels[(size_t)idx]));
        abbrev_ptrs.push_back(abbrev_storage.back().c_str());
    }
}

// values: [cam][node]
void PlotBarsWithScatter(
    const std::vector<std::vector<double>> &values,
    const std::vector<std::string> &camera_names,
    const std::vector<std::string> &node_names, PlotMode mode,
    const std::vector<ImVec4> &node_colors, // point colors by node
    const char *title = "Bars + Scatter", float bar_width = 0.6f,
    float jitter = 0.0f, float marker_size = 4.0f,
    bool show_error_bars = true, // SD/SEM whiskers
    bool use_sem = false) {      // if true: SEM instead of SD

    if (values.empty() || values[0].empty())
        return;
    const int n_cams = (int)values.size();
    const int n_nodes = (int)values[0].size();

    for (int c = 0; c < n_cams; ++c)
        if ((int)values[c].size() != n_nodes)
            return;
    if ((int)camera_names.size() != n_cams)
        return;
    if ((int)node_names.size() != n_nodes)
        return;

    const bool by_camera = (mode == PlotMode::ByCamera);
    const int n_cat = by_camera ? n_cams : n_nodes;  // categories (bars)
    const int n_item = by_camera ? n_nodes : n_cams; // points per category

    // --- Filter categories that have at least one finite value ---------------
    std::vector<int> cat_idx;
    cat_idx.reserve(n_cat);
    for (int i = 0; i < n_cat; ++i) {
        bool any_finite = false;
        for (int k = 0; k < n_item; ++k) {
            const double v = by_camera ? values[i][k] : values[k][i];
            if (std::isfinite(v)) {
                any_finite = true;
                break;
            }
        }
        if (any_finite)
            cat_idx.push_back(i);
    }
    if (cat_idx.empty())
        return;

    // Labels (full for tooltip, abbreviated for ticks)
    std::vector<std::string> full_labels_storage;
    full_labels_storage.reserve(cat_idx.size());
    if (by_camera) {
        for (int idx : cat_idx)
            full_labels_storage.push_back(camera_names[(size_t)idx]);
    } else {
        for (int idx : cat_idx)
            full_labels_storage.push_back(node_names[(size_t)idx]);
    }

    std::vector<std::string> abbrev_storage;
    std::vector<const char *> abbrev_ptrs;
    if (by_camera)
        MakeAbbrevLabels(cat_idx, camera_names, abbrev_storage, abbrev_ptrs);
    else
        MakeAbbrevLabels(cat_idx, node_names, abbrev_storage, abbrev_ptrs);

    // --- Begin plot (no legend) ---------------------------------------------
    ImPlotFlags pflags = ImPlotFlags_NoLegend;
    if (ImPlot::BeginPlot(title, ImVec2(-1, 0), pflags)) {
        ImPlot::SetupAxis(ImAxis_X1, by_camera ? "Camera" : "Keypoint");
        ImPlot::SetupAxis(ImAxis_Y1, "Reprojection Error (px)");
        ImPlot::SetupAxisTicks(ImAxis_X1, 0, (int)cat_idx.size() - 1,
                               (int)cat_idx.size(), abbrev_ptrs.data());

        // --- Compute bar means + SD/SEM for filtered categories --------------
        std::vector<float> x_pos;
        x_pos.reserve(cat_idx.size());
        std::vector<float> bar_vals;
        bar_vals.reserve(cat_idx.size());
        std::vector<float> err_minus;
        err_minus.reserve(cat_idx.size());
        std::vector<float> err_plus;
        err_plus.reserve(cat_idx.size());

        for (int i = 0; i < (int)cat_idx.size(); ++i) {
            const int src = cat_idx[i];
            double sum = 0.0, sum2 = 0.0;
            int cnt = 0;
            for (int k = 0; k < n_item; ++k) {
                const double v = by_camera ? values[src][k] : values[k][src];
                if (!std::isfinite(v))
                    continue;
                sum += v;
                sum2 += v * v;
                ++cnt;
            }
            if (cnt > 0) {
                const double mu = sum / cnt;
                double sd = 0.0;
                const double var = std::max(0.0, (sum2 / cnt) - mu * mu);
                sd = std::sqrt(var);
                if (use_sem && cnt > 1)
                    sd /= std::sqrt((double)cnt);

                x_pos.push_back((float)i);
                bar_vals.push_back((float)mu);
                err_minus.push_back((float)sd);
                err_plus.push_back((float)sd);
            }
        }

        if (!bar_vals.empty()) {
            ImPlot::PlotBars("##Mean", x_pos.data(), bar_vals.data(),
                             (int)bar_vals.size(), bar_width, 0.0f);
            if (show_error_bars) {
                const float cap_size = 5.0f; // pixels
                ImPlot::PlotErrorBars("##Err", x_pos.data(), bar_vals.data(),
                                      err_minus.data(), err_plus.data(),
                                      (int)bar_vals.size(), cap_size);
            }
        }

        // --- Scatter with node-based colors ---------------------------------
        const ImVec4 outline_col = ImVec4(0, 0, 0, 1);
        if (mode == PlotMode::ByNode) {
            for (int i = 0; i < (int)cat_idx.size(); ++i) {
                const int node_idx = cat_idx[i];
                std::vector<float> xs;
                xs.reserve(n_item);
                std::vector<float> ys;
                ys.reserve(n_item);
                for (int cam = 0; cam < n_item; ++cam) {
                    const double v = values[cam][node_idx];
                    if (!std::isfinite(v))
                        continue;
                    float x = (float)i;
                    if (jitter > 0 && n_item > 1) {
                        const float t = (float)cam / (float)(n_item - 1);
                        x += (t * 2.0f - 1.0f) * jitter;
                    }
                    xs.push_back(x);
                    ys.push_back((float)v);
                }
                if (!xs.empty()) {
                    ImVec4 fill_col = (node_idx < (int)node_colors.size())
                                          ? node_colors[(size_t)node_idx]
                                          : ImVec4(1, 1, 1, 1);
                    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, marker_size,
                                               fill_col, 1.0f, outline_col);
                    ImPlot::PlotScatter("##pts_node", xs.data(), ys.data(),
                                        (int)xs.size());
                }
            }
        } else { // ByCamera
            for (int i = 0; i < (int)cat_idx.size(); ++i) {
                const int cam_idx = cat_idx[i];
                for (int node = 0; node < n_item; ++node) {
                    const double v = values[cam_idx][node];
                    if (!std::isfinite(v))
                        continue;

                    float x = (float)i;
                    if (jitter > 0 && n_item > 1) {
                        const float t = (float)node / (float)(n_item - 1);
                        x += (t * 2.0f - 1.0f) * jitter;
                    }
                    const float X[1] = {x};
                    const float Y[1] = {(float)v};

                    ImVec4 fill_col = (node < (int)node_colors.size())
                                          ? node_colors[(size_t)node]
                                          : ImVec4(1, 1, 1, 1);
                    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, marker_size,
                                               fill_col, 1.0f, outline_col);
                    ImPlot::PlotScatter("##pt", X, Y, 1);
                }
            }
        }

        // --- Tooltip with full label for nearest category --------------------
        if (ImPlot::IsPlotHovered()) {
            const ImPlotPoint mp =
                ImPlot::GetPlotMousePos(ImAxis_X1, ImAxis_Y1);
            const int nearest = (int)std::round(mp.x);
            if (nearest >= 0 && nearest < (int)cat_idx.size()) {
                if (std::abs(mp.x - nearest) < 0.45f) {
                    ImGui::BeginTooltip();
                    ImGui::TextUnformatted(
                        full_labels_storage[(size_t)nearest].c_str());
                    ImGui::EndTooltip();
                }
            }
        }

        ImPlot::EndPlot();
    }
}

inline void DrawReprojectionWindow(KeyPoints *frame_keypoints,
                                   const std::vector<std::string> &camera_names,
                                   render_scene *scene,
                                   SkeletonContext &skeleton,
                                   ReprojectionTool &rp_tool) {
    if (!rp_tool.show_reprojection_error)
        return;

    if (ImGui::Begin("Reprojection##TOOL", &rp_tool.show_reprojection_error)) {

        const bool by_cam = (rp_tool.mode == PlotMode::ByCamera);
        const char *label = by_cam ? "By Camera (click to switch)"
                                   : "By Keypoint (click to switch)";
        if (ImGui::Button(label)) {
            rp_tool.mode = by_cam ? PlotMode::ByNode : PlotMode::ByCamera;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Switch to %s",
                              by_cam ? "By Keypoint" : "By Camera");
        }

        std::vector<std::vector<double>> values; // [cam][node]

        // Initialize [cams × nodes] with NaN (plotters
        // typically skip NaN)
        const double NaN = std::numeric_limits<double>::quiet_NaN();
        values.assign(scene->num_cams,
                      std::vector<double>(skeleton.num_nodes, NaN));

        // 4) Fill reprojection errors where available
        for (u32 node = 0; node < skeleton.num_nodes; ++node) {
            for (u32 cam = 0; cam < scene->num_cams; ++cam) {
                const auto &kp2d = frame_keypoints->kp2d[cam][node];
                if (kp2d.last_is_labeled) {
                    double dx = kp2d.position.x - kp2d.last_position.x;
                    double dy = kp2d.position.y - kp2d.last_position.y;
                    double reproj_error = std::sqrt(dx * dx + dy * dy);
                    values[cam][node] = reproj_error;
                }
            }
        }

        // 5) Plot in both groupings
        if (rp_tool.mode == PlotMode::ByCamera) {
            PlotBarsWithScatter(values, camera_names, skeleton.node_names,
                                PlotMode::ByCamera, skeleton.node_colors,
                                "Reproj Error by Camera", 0.6f, 0.15f, 4.0f,
                                show_error_bars, use_sem);
        } else {
            PlotBarsWithScatter(values, camera_names, skeleton.node_names,
                                PlotMode::ByNode, skeleton.node_colors,
                                "Reproj Error by Keypoint", 0.6f, 0.15f, 4.0f,
                                show_error_bars, use_sem);
        }
    }
    ImGui::End();
}
