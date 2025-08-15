#ifndef RED_GUI
#define RED_GUI
#include "implot.h"
#include "implot_internal.h"
#include "render.h"
#include "skeleton.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <opencv2/sfm.hpp>
#include <regex>
#include <sstream>
#include <thread>
#include <vector>

struct ProjectContext {
    std::string root_dir;
    std::vector<std::string> input_file_names;
    std::vector<std::string> camera_names;
};

int load_bboxes(std::map<u32, KeyPoints *> &keypoints_map,
                SkeletonContext *skeleton, std::string bbox_file, int cam_idx,
                render_scene *scene, std::string &error_message);

int load_bbox_keypoints(std::map<u32, KeyPoints *> &keypoints_map,
                        SkeletonContext *skeleton, std::string bbox_kp_file,
                        int cam_idx, render_scene *scene,
                        std::string &error_message);

int load_obb(std::map<u32, KeyPoints *> &keypoints_map,
             SkeletonContext *skeleton, std::string obb_file, int cam_idx,
             render_scene *scene, std::string &error_message,
             std::vector<std::string> &class_names);
static void draw_cv_contours(std::vector<cv::Rect> boxes,
                             std::vector<std::string> labels,
                             std::vector<int> class_ids, int image_height) {
    for (int i = 0; i < boxes.size(); i++) {
        double x[5] = {(double)boxes[i].x, (double)boxes[i].x,
                       (double)boxes[i].x + boxes[i].width,
                       (double)boxes[i].x + boxes[i].width, (double)boxes[i].x};
        double y[5] = {(double)image_height - boxes[i].y,
                       (double)image_height - boxes[i].y - boxes[i].height,
                       (double)image_height - boxes[i].y - boxes[i].height,
                       (double)image_height - boxes[i].y,
                       (double)image_height - boxes[i].y};

        if (class_ids[i] == 0) {
            ImPlot::SetNextLineStyle(ImVec4(1.0, 0.0, 1.0, 1.0), 3.0);
        } else {
            ImPlot::SetNextLineStyle(ImVec4(0.5, 1.0, 1.0, 1.0), 3.0);
        }

        ImPlot::PlotLine(labels[i].c_str(), &x[0], &y[0], 5);
    }
}

static void gui_plot_keypoints(KeyPoints *keypoints, SkeletonContext *skeleton,
                               int view_idx, int num_cams) {
    float pt_size = 6.0f;
    for (u32 node = 0; node < skeleton->num_nodes; node++) {
        if (keypoints->keypoints2d[view_idx][node].is_labeled) {
            ImVec4 node_color;
            if (keypoints->active_id[view_idx] == node) {
                node_color = (ImVec4)ImColor::HSV(0.8, 1.0f, 1.0f);
                node_color.w = 0.9;
                pt_size = 8.0f;
            } else {
                node_color = skeleton->node_colors.at(node);
                node_color.w = 0.9;
                pt_size = 6.0f;
            }
            int id = skeleton->num_nodes * view_idx + node;
            static bool drag_point_clicked;
            static bool drag_point_hovered;
            static bool drag_point_modified;
            drag_point_modified = ImPlot::DragPoint(
                id, &keypoints->keypoints2d[view_idx][node].position.x,
                &keypoints->keypoints2d[view_idx][node].position.y, node_color,
                pt_size, ImPlotDragToolFlags_None, &drag_point_clicked,
                &drag_point_hovered);
            if (drag_point_modified) {
                keypoints->keypoints2d[view_idx][node].is_triangulated = false;
            }
            if (drag_point_hovered) {
                if (keypoints->keypoints2d[view_idx][node].is_triangulated) {

                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(2);
                    oss << "(" << keypoints->keypoints3d[node].x << ", "
                        << keypoints->keypoints3d[node].y << ", "
                        << keypoints->keypoints3d[node].z << ")";
                    std::string label = oss.str();
                    ImVec2 mouse_pos = ImGui::GetMousePos();
                    ImVec2 textPos = ImVec2(mouse_pos.x + 10, mouse_pos.y + 10);
                    ImGui::GetForegroundDrawList()->AddText(
                        textPos, IM_COL32(220, 20, 60, 255), label.c_str());
                }

                if (ImGui::IsKeyPressed(ImGuiKey_R,
                                        false)) // delete active keypoint
                {
                    keypoints->keypoints2d[view_idx][node].position = {1E7,
                                                                       1E7};
                    keypoints->keypoints2d[view_idx][node].is_labeled = false;
                    keypoints->keypoints2d[view_idx][node].is_triangulated =
                        false;
                    keypoints->active_id[view_idx] = node;
                }

                if (ImGui::IsKeyPressed(
                        ImGuiKey_F,
                        false)) // Delete active keypoints from all the views
                {
                    for (int cam_idx = 0; cam_idx < num_cams; cam_idx++) {
                        keypoints->keypoints2d[cam_idx][node].position = {1E7,
                                                                          1E7};
                        keypoints->keypoints2d[cam_idx][node].is_labeled =
                            false;
                        keypoints->keypoints2d[cam_idx][node].is_triangulated =
                            false;
                        keypoints->active_id[cam_idx] = node;
                    }
                }
            }

            if (drag_point_clicked) {
                keypoints->active_id[view_idx] = node;
            }
        }
    }

    for (u32 edge = 0; edge < skeleton->num_edges; edge++) {
        auto [a, b] = skeleton->edges[edge];

        if (keypoints->keypoints2d[view_idx][a].is_labeled &&
            keypoints->keypoints2d[view_idx][b].is_labeled) {
            double xs[2]{keypoints->keypoints2d[view_idx][a].position.x,
                         keypoints->keypoints2d[view_idx][b].position.x};
            double ys[2]{keypoints->keypoints2d[view_idx][a].position.y,
                         keypoints->keypoints2d[view_idx][b].position.y};
            ImPlot::PlotLine("##line", xs, ys, 2);
        }
    }
}

static void gui_plot_bbox_from_keypoints(KeyPoints *keypoints,
                                         SkeletonContext *skeleton,
                                         int view_idx, int top_left_idx,
                                         int bottom_right_idx) {
    if (keypoints->keypoints2d[view_idx][top_left_idx].is_labeled &&
        keypoints->keypoints2d[view_idx][bottom_right_idx].is_labeled) {
        double xs[5]{
            keypoints->keypoints2d[view_idx][top_left_idx].position.x,
            keypoints->keypoints2d[view_idx][bottom_right_idx].position.x,
            keypoints->keypoints2d[view_idx][bottom_right_idx].position.x,
            keypoints->keypoints2d[view_idx][top_left_idx].position.x,
            keypoints->keypoints2d[view_idx][top_left_idx].position.x};

        double ys[5]{
            keypoints->keypoints2d[view_idx][top_left_idx].position.y,
            keypoints->keypoints2d[view_idx][top_left_idx].position.y,
            keypoints->keypoints2d[view_idx][bottom_right_idx].position.y,
            keypoints->keypoints2d[view_idx][bottom_right_idx].position.y,
            keypoints->keypoints2d[view_idx][top_left_idx].position.y};

        ImPlot::SetNextLineStyle(ImVec4(0.5, 1.0, 1.0, 1.0), 3.0);
        ImPlot::PlotLine("##line", xs, ys, 5);
    }
}

bool is_in_camera_fov(cv::Mat point_world, const cv::Mat &rvec,
                      const cv::Mat &tvec, const cv::Mat &K, int image_width,
                      int image_height) {
    cv::Mat image_pts;
    cv::Mat dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);
    cv::projectPoints(point_world, rvec, tvec, K, dist_coeffs, image_pts);
    double x = image_pts.at<double>(0, 0);
    double y = image_height - image_pts.at<double>(0, 1);
    if (x > 0 && x < image_width && y > 0 && y < image_height) {
        return true;
    }
    return false;
}

static void reprojection(KeyPoints *keypoints, SkeletonContext *skeleton,
                         std::vector<CameraParams> camera_params,
                         render_scene *scene) {

    for (u32 node = 0; node < skeleton->num_nodes; node++) {

        u32 num_views_labeled{0};
        for (u32 view_idx = 0; view_idx < scene->num_cams; view_idx++) {
            if (keypoints->keypoints2d[view_idx][node].is_labeled) {
                num_views_labeled++;
            }
        }

        if (num_views_labeled >= 2) {

            std::vector<cv::Mat> sfmPoints2d;
            std::vector<cv::Mat> projection_matrices;
            cv::Mat output;

            for (u32 view_idx = 0; view_idx < scene->num_cams; view_idx++) {
                if (keypoints->keypoints2d[view_idx][node].is_labeled) {

                    cv::Mat point =
                        (cv::Mat_<double>(2, 1)
                             << keypoints->keypoints2d[view_idx][node]
                                    .position.x,
                         (double)scene->image_height[view_idx] -
                             keypoints->keypoints2d[view_idx][node].position.y);
                    cv::Mat pointUndistort;
                    cv::undistortPoints(
                        point, pointUndistort, camera_params[view_idx].k,
                        camera_params[view_idx].dist_coeffs, cv::noArray(),
                        camera_params[view_idx].k);

                    sfmPoints2d.push_back(pointUndistort.reshape(1, 2));
                    projection_matrices.push_back(
                        camera_params[view_idx].projection_mat);
                }
            }

            cv::sfm::triangulatePoints(sfmPoints2d, projection_matrices,
                                       output);

            keypoints->keypoints3d[node].x = output.at<double>(0);
            keypoints->keypoints3d[node].y = output.at<double>(1);
            keypoints->keypoints3d[node].z = output.at<double>(2);

            for (u32 view_idx = 0; view_idx < scene->num_cams; view_idx++) {

                if (is_in_camera_fov(output, camera_params[view_idx].rvec,
                                     camera_params[view_idx].tvec,
                                     camera_params[view_idx].k,
                                     scene->image_width[view_idx],
                                     scene->image_height[view_idx])) {
                    cv::Mat imagePts;
                    cv::projectPoints(
                        output, camera_params[view_idx].rvec,
                        camera_params[view_idx].tvec, camera_params[view_idx].k,
                        camera_params[view_idx].dist_coeffs, imagePts);
                    double x = imagePts.at<double>(0, 0);
                    double y = double(scene->image_height[view_idx]) -
                               imagePts.at<double>(0, 1);
                    if (x > 0 && x < scene->image_width[view_idx] && y > 0 &&
                        y < scene->image_height[view_idx]) {
                        keypoints->keypoints2d[view_idx][node].position.x = x;
                        keypoints->keypoints2d[view_idx][node].position.y = y;
                        keypoints->keypoints2d[view_idx][node].is_labeled =
                            true;
                        keypoints->keypoints2d[view_idx][node].is_triangulated =
                            true;
                    }
                }
            }
        }
    }
}

const std::string current_date_time() {
    time_t now = time(0);
    struct tm tstruct;
    char buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y:%m:%d:%X", &tstruct);

    std::string delimiter = ":";

    std::string s(buf);
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back(token);
    }

    res.push_back(s.substr(pos_start));
    std::string final_string;

    for (int i = 0; i < res.size(); i++) {
        if (i != 0) {
            final_string += "_";
        }
        final_string += res[i];
    }
    return final_string.c_str();
}

void save_keypoints_depreciated(std::map<u32, KeyPoints *> keypoints_map,
                                SkeletonContext *skeleton, std::string root_dir,
                                int num_cameras,
                                std::vector<std::string> &camera_names,
                                bool *input_is_imgs,
                                const std::vector<std::string> &input_files) {
    std::string now = current_date_time();
    std::string filename =
        root_dir + "/worldKeyPoints/keypoints_" + now + ".csv";
    std::ofstream output_file(filename);
    std::vector<std::ofstream> output2d_files;

    for (uint i = 0; i < num_cameras; i++) {
        std::string filename_cam = root_dir + "/" + camera_names[i] + "/" +
                                   camera_names[i] + "_" + now + ".csv";
        std::ofstream output_file_cam(filename_cam);
        output2d_files.push_back(std::move(output_file_cam));
    }

    output_file << skeleton->name << ",\n";
    for (uint i = 0; i < num_cameras; i++) {
        output2d_files[i] << skeleton->name << ",\n";
    }

    std::map<u32, KeyPoints *>::iterator it = keypoints_map.begin();
    while (it != keypoints_map.end()) {
        uint frame = it->first;
        KeyPoints *keypoints = it->second;
        // write frame number
        if (*input_is_imgs) {
            output_file << input_files[frame] << ",";
        } else {
            output_file << frame << ",";
        }
        // fore each labeled keypoint, write idx, xpos, ypos, zpos
        for (uint i = 0; i < skeleton->num_nodes; i++) {
            if (i == skeleton->num_nodes - 1) {
                // last keypoints (RJ added extra "," at end of row)
                output_file << i << "," << keypoints->keypoints3d[i].x << ","
                            << keypoints->keypoints3d[i].y << ","
                            << keypoints->keypoints3d[i].z << ",";
            } else {
                output_file << i << "," << keypoints->keypoints3d[i].x << ","
                            << keypoints->keypoints3d[i].y << ","
                            << keypoints->keypoints3d[i].z << ",";
            }
        }
        output_file << "\n";

        for (int cam = 0; cam < num_cameras; cam++) {
            if (*input_is_imgs) {
                output2d_files[cam] << input_files[frame] << ",";
            } else {
                output2d_files[cam] << frame << ",";
            }
            for (int node = 0; node < skeleton->num_nodes; node++) {
                if (node == skeleton->num_nodes - 1) {
                    // last keypoints (RJ added extra "," at end of row)
                    output2d_files[cam]
                        << node << ","
                        << keypoints->keypoints2d[cam][node].position.x << ","
                        << keypoints->keypoints2d[cam][node].position.y << ",";
                } else {
                    output2d_files[cam]
                        << node << ","
                        << keypoints->keypoints2d[cam][node].position.x << ","
                        << keypoints->keypoints2d[cam][node].position.y << ",";
                }
            }
            output2d_files[cam] << "\n";
        }

        it++;
    }

    output_file.close();
    std::cout << filename << " created" << std::endl;

    for (uint i = 0; i < num_cameras; i++) {
        output2d_files[i].close();
    }
}

void save_keypoints(std::map<u32, KeyPoints *> keypoints_map,
                    SkeletonContext *skeleton, std::string root_dir,
                    int num_cameras, std::vector<std::string> &camera_names,
                    bool *input_is_imgs,
                    const std::vector<std::string> &input_files) {
    std::string now = current_date_time();
    std::string save_folder = root_dir + "/" + now;
    std::filesystem::create_directories(save_folder);
    std::string filename = save_folder + "/keypoints3d.csv";

    std::ofstream output3d_file(filename);
    std::vector<std::ofstream> output2d_files;

    for (uint i = 0; i < num_cameras; i++) {
        std::string filename_cam = save_folder + "/" + camera_names[i] + ".csv";
        std::ofstream output_file_cam(filename_cam);
        output2d_files.push_back(std::move(output_file_cam));
    }

    output3d_file << skeleton->name << "\n";
    for (uint i = 0; i < num_cameras; i++) {
        output2d_files[i] << skeleton->name << "\n";
    }

    std::map<u32, KeyPoints *>::iterator it = keypoints_map.begin();
    while (it != keypoints_map.end()) {
        uint frame = it->first;
        KeyPoints *keypoints = it->second;

        // Write frame number
        if (*input_is_imgs) {
            output3d_file << input_files[frame];
        } else {
            output3d_file << frame;
        }

        // Write each labeled keypoint
        for (uint i = 0; i < skeleton->num_nodes; i++) {
            output3d_file << "," << i << "," << keypoints->keypoints3d[i].x
                          << "," << keypoints->keypoints3d[i].y << ","
                          << keypoints->keypoints3d[i].z;
        }
        output3d_file << "\n";

        for (int cam = 0; cam < num_cameras; cam++) {
            if (*input_is_imgs) {
                output2d_files[cam] << input_files[frame];
            } else {
                output2d_files[cam] << frame;
            }
            for (int node = 0; node < skeleton->num_nodes; node++) {
                output2d_files[cam]
                    << "," << node << ","
                    << keypoints->keypoints2d[cam][node].position.x << ","
                    << keypoints->keypoints2d[cam][node].position.y;
            }
            output2d_files[cam] << "\n";
        }

        it++;
    }
    output3d_file.close();
    for (uint i = 0; i < num_cameras; i++) {
        output2d_files[i].close();
    }
}

// Save bounding boxes to CSV files
void save_bboxes(std::map<u32, KeyPoints *> keypoints_map,
                 SkeletonContext *skeleton, std::string root_dir,
                 int num_cameras, std::vector<std::string> &camera_names,
                 bool *input_is_imgs,
                 const std::vector<std::string> &input_files) {
    std::string now = current_date_time();
    std::string save_folder = root_dir + "/" + now;
    std::filesystem::create_directories(save_folder);

    // Create keypoints3d.csv (empty for bbox-only workflows)
    std::string keypoints3d_filename = save_folder + "/keypoints3d.csv";
    std::ofstream keypoints3d_file(keypoints3d_filename);
    keypoints3d_file << skeleton->name << "\n";
    keypoints3d_file.close();

    std::vector<std::ofstream> bbox_files;

    for (uint i = 0; i < num_cameras; i++) {
        std::string filename_cam =
            save_folder + "/" + camera_names[i] + "_bboxes.csv";
        std::ofstream output_file_cam(filename_cam);
        bbox_files.push_back(std::move(output_file_cam));
    }

    // Write header for bbox files
    for (uint i = 0; i < num_cameras; i++) {
        bbox_files[i] << skeleton->name << "\n";
        bbox_files[i]
            << "frame,bbox_id,class_id,confidence,x_min,y_min,x_max,y_max\n";
    }

    std::map<u32, KeyPoints *>::iterator it = keypoints_map.begin();
    while (it != keypoints_map.end()) {
        uint frame = it->first;
        KeyPoints *keypoints = it->second;

        for (int cam = 0; cam < num_cameras; cam++) {
            bool frame_written = false;

            for (size_t bbox_idx = 0;
                 bbox_idx < keypoints->bbox2d_list[cam].size(); bbox_idx++) {
                const auto &bbox = keypoints->bbox2d_list[cam][bbox_idx];

                // Only save completed bounding boxes
                if (bbox.state == RectTwoPoints && bbox.rect != nullptr) {
                    if (!frame_written) {
                        // Write frame identifier only once per camera per frame
                        frame_written = true;
                    }

                    // Write frame number or filename
                    if (*input_is_imgs) {
                        bbox_files[cam] << input_files[frame];
                    } else {
                        bbox_files[cam] << frame;
                    }

                    // Write bbox data:
                    // frame,bbox_id,class_id,confidence,x_min,y_min,x_max,y_max
                    bbox_files[cam]
                        << "," << bbox_idx << "," << bbox.class_id << ","
                        << bbox.confidence << "," << bbox.rect->X.Min << ","
                        << bbox.rect->Y.Min << "," << bbox.rect->X.Max << ","
                        << bbox.rect->Y.Max << "\n";
                }
            }
        }
        it++;
    }

    for (uint i = 0; i < num_cameras; i++) {
        bbox_files[i].close();
    }
}

// Save bounding boxes with keypoints to CSV files
void save_bbox_keypoints(std::map<u32, KeyPoints *> keypoints_map,
                         SkeletonContext *skeleton, std::string root_dir,
                         int num_cameras,
                         std::vector<std::string> &camera_names,
                         bool *input_is_imgs,
                         const std::vector<std::string> &input_files) {
    std::string now = current_date_time();
    std::string save_folder = root_dir + "/" + now;
    std::filesystem::create_directories(save_folder);

    // Create keypoints3d.csv (empty for bbox workflows without 3D
    // triangulation)
    std::string keypoints3d_filename = save_folder + "/keypoints3d.csv";
    std::ofstream keypoints3d_file(keypoints3d_filename);
    keypoints3d_file << skeleton->name << "\n";
    keypoints3d_file.close();

    // Create camera CSV files for 2D keypoints (from bbox keypoints)
    std::vector<std::ofstream> cam_files;
    for (uint i = 0; i < num_cameras; i++) {
        std::string cam_filename = save_folder + "/" + camera_names[i] + ".csv";
        std::ofstream cam_file(cam_filename);
        cam_file << skeleton->name << "\n";
        cam_files.push_back(std::move(cam_file));
    }

    std::vector<std::ofstream> bbox_kp_files;

    for (uint i = 0; i < num_cameras; i++) {
        std::string filename_cam =
            save_folder + "/" + camera_names[i] + "_bbox_keypoints.csv";
        std::ofstream output_file_cam(filename_cam);
        bbox_kp_files.push_back(std::move(output_file_cam));
    }

    // Write header for bbox keypoint files
    for (uint i = 0; i < num_cameras; i++) {
        bbox_kp_files[i] << skeleton->name << "\n";
        bbox_kp_files[i] << "frame,bbox_id,class_id,confidence,x_min,y_min,x_"
                            "max,y_max,keypoint_id,kp_x,kp_y,is_labeled\n";
    }

    std::map<u32, KeyPoints *>::iterator it = keypoints_map.begin();
    while (it != keypoints_map.end()) {
        uint frame = it->first;
        KeyPoints *keypoints = it->second;

        for (int cam = 0; cam < num_cameras; cam++) {
            for (size_t bbox_idx = 0;
                 bbox_idx < keypoints->bbox2d_list[cam].size(); bbox_idx++) {
                const auto &bbox = keypoints->bbox2d_list[cam][bbox_idx];

                // Only save completed bounding boxes with keypoints
                if (bbox.state == RectTwoPoints && bbox.rect != nullptr &&
                    bbox.has_bbox_keypoints &&
                    bbox.bbox_keypoints2d != nullptr) {

                    // Write 2D keypoints to camera CSV files (for loading
                    // compatibility)
                    for (int kp_id = 0; kp_id < skeleton->num_nodes; kp_id++) {
                        if (bbox.bbox_keypoints2d[cam][kp_id].is_labeled) {
                            // Write frame number or filename
                            if (*input_is_imgs) {
                                cam_files[cam] << input_files[frame];
                            } else {
                                cam_files[cam] << frame;
                            }

                            cam_files[cam]
                                << "," << kp_id << ","
                                << bbox.bbox_keypoints2d[cam][kp_id].position.x
                                << ","
                                << bbox.bbox_keypoints2d[cam][kp_id].position.y
                                << "\n";
                        }
                    }

                    for (int kp_id = 0; kp_id < skeleton->num_nodes; kp_id++) {
                        // Write frame number or filename
                        if (*input_is_imgs) {
                            bbox_kp_files[cam] << input_files[frame];
                        } else {
                            bbox_kp_files[cam] << frame;
                        }

                        // Write bbox and keypoint data
                        bbox_kp_files[cam]
                            << "," << bbox_idx << "," << bbox.class_id << ","
                            << bbox.confidence << "," << bbox.rect->X.Min << ","
                            << bbox.rect->Y.Min << "," << bbox.rect->X.Max
                            << "," << bbox.rect->Y.Max << "," << kp_id << ","
                            << bbox.bbox_keypoints2d[cam][kp_id].position.x
                            << ","
                            << bbox.bbox_keypoints2d[cam][kp_id].position.y
                            << ","
                            << (bbox.bbox_keypoints2d[cam][kp_id].is_labeled
                                    ? 1
                                    : 0)
                            << "\n";
                    }
                }
            }
        }
        it++;
    }

    // Close all files
    for (uint i = 0; i < num_cameras; i++) {
        cam_files[i].close();
        bbox_kp_files[i].close();
    }
}

void save_obb(std::map<u32, KeyPoints *> keypoints_map,
              SkeletonContext *skeleton, std::string root_dir,
              std::vector<std::string> &camera_names, int num_cameras,
              std::vector<std::string> *input_files, bool *input_is_imgs,
              std::vector<std::string> &class_names) {
    if (!skeleton->has_obb)
        return;

    std::string now = current_date_time();
    std::string save_folder = root_dir + "/" + now;
    std::filesystem::create_directories(save_folder);

    // Save class names file
    std::string class_names_file = save_folder + "/class_names.txt";
    std::ofstream class_file(class_names_file);
    for (const auto &class_name : class_names) {
        class_file << class_name << "\n";
    }
    class_file.close();

    std::vector<std::ofstream> obb_files;

    for (uint i = 0; i < num_cameras; i++) {
        std::string filename_cam =
            save_folder + "/" + camera_names[i] + "_obb.csv";
        std::ofstream output_file_cam(filename_cam);
        obb_files.push_back(std::move(output_file_cam));
    }

    // Write header for OBB files
    for (uint i = 0; i < num_cameras; i++) {
        obb_files[i] << skeleton->name << "\n";
        obb_files[i] << "frame,obb_id,class_id,corner_x1,corner_y1,corner_x2,"
                        "corner_y2,corner_x3,corner_y3,corner_x4,corner_y4\n";
    }

    std::map<u32, KeyPoints *>::iterator it = keypoints_map.begin();
    while (it != keypoints_map.end()) {
        u32 frame = it->first;
        KeyPoints *keypoints = it->second;

        if (keypoints != nullptr) {
            for (uint cam = 0; cam < num_cameras; cam++) {
                for (size_t obb_idx = 0;
                     obb_idx < keypoints->obb2d_list[cam].size(); obb_idx++) {
                    const auto &obb = keypoints->obb2d_list[cam][obb_idx];

                    if (obb.state == OBBComplete) {
                        // Get the four corners of the OBB
                        ImVec2 corners[4];
                        get_obb_corners(&obb, corners);

                        // Write frame number or filename
                        if (*input_is_imgs) {
                            obb_files[cam] << (*input_files)[frame];
                        } else {
                            obb_files[cam] << frame;
                        }

                        // Write simplified OBB data: frame, obb_id, class_id,
                        // and four corner coordinates
                        obb_files[cam]
                            << "," << obb_idx << "," << obb.class_id << ","
                            << corners[0].x << "," << corners[0].y << ","
                            << corners[1].x << "," << corners[1].y << ","
                            << corners[2].x << "," << corners[2].y << ","
                            << corners[3].x << "," << corners[3].y << "\n";
                    }
                }
            }
        }
        it++;
    }

    // Close all files
    for (uint i = 0; i < num_cameras; i++) {
        obb_files[i].close();
    }
}

void load_2d_keypoints_depreciated(std::map<u32, KeyPoints *> &keypoints_map,
                                   SkeletonContext *skeleton,
                                   std::string root_dir, int cam_idx,
                                   std::string camera_name,
                                   render_scene *scene) {
    std::string labeled_data_dir = root_dir + "/" + camera_name;
    std::vector<std::string> filenames;

    for (const auto &entry :
         std::filesystem::directory_iterator(labeled_data_dir)) {
        filenames.push_back(entry.path());
    }

    if (filenames.size() == 0) {
        std::cout << "No files in directory for " << camera_name << std::endl;
        return;
    };

    sort(filenames.begin(), filenames.end());
    std::string mostRecentFile = filenames.back();
    std::cout << "mostRecentFile: " << mostRecentFile << std::endl;

    std::ifstream fin;
    fin.open(mostRecentFile);
    if (fin.fail())
        throw mostRecentFile; // the exception being checked

    std::string line;
    std::string delimeter = ",";
    size_t pos = 0;
    std::string token;

    // read csv file with cam parameters and tokenize line for this camera
    int lineNum = 0;
    while (!fin.eof()) {
        fin >> line;
        while ((pos = line.find(delimeter)) != std::string::npos) {
            token = line.substr(0, pos);
            if (lineNum == 0) {
                if (token.compare(skeleton->name) != 0) {
                    std::cout << "Failed loading, skeleton doesn't match.\n"
                              << skeleton->name << ":" << token << std::endl;
                    return;
                }
                line.erase(0, pos + delimeter.length());
            } else {
                uint frame_num = stoul(token);
                if (keypoints_map.find(frame_num) == keypoints_map.end()) {
                    KeyPoints *keypoints =
                        (KeyPoints *)malloc(sizeof(KeyPoints));
                    allocate_keypoints(keypoints, scene, skeleton);
                    keypoints_map[frame_num] = keypoints;
                }
                line.erase(0, pos + delimeter.length());

                while ((pos = line.find(delimeter)) != std::string::npos) {
                    token = line.substr(0, pos);
                    int node = stoi(token); // get the node index
                    line.erase(0, pos + delimeter.length());

                    pos = line.find(delimeter);
                    token = line.substr(0, pos);
                    double x = stod(token);
                    line.erase(0, pos + delimeter.length());

                    pos = line.find(delimeter);
                    token = line.substr(0, pos);
                    double y = stod(token);
                    line.erase(0, pos + delimeter.length());

                    keypoints_map[frame_num]
                        ->keypoints2d[cam_idx][node]
                        .position.x = x;
                    keypoints_map[frame_num]
                        ->keypoints2d[cam_idx][node]
                        .position.y = y;

                    if (x == 1E7 || y == 1E7) {
                        keypoints_map[frame_num]
                            ->keypoints2d[cam_idx][node]
                            .is_labeled = false;
                    } else {
                        keypoints_map[frame_num]
                            ->keypoints2d[cam_idx][node]
                            .is_labeled = true;
                    }
                    // std::cout << "frame: " << frame_num << "  node: " << node
                    // << "  x: " <<
                    // keypoints_map[frame_num]->keypoints2d[cam_idx][node].position.x
                    // << "  y: " <<
                    // keypoints_map[frame_num]->keypoints2d[cam_idx][node].position.y
                    // << std::endl;
                }
            }
        }
        lineNum++;
    }
    fin.close();
}

int load_keypoints_depreciated(std::map<u32, KeyPoints *> &keypoints_map,
                               SkeletonContext *skeleton, std::string root_dir,
                               render_scene *scene,
                               std::vector<std::string> &camera_names,
                               std::string &error_message) {

    if (scene->num_cams > 1) {
        if (!std::filesystem::exists(root_dir + "/worldKeyPoints")) {
            error_message =
                "'worldKeyPoints' directory is missing from: " + root_dir;
            return 1;
        }

        std::string label3d_dir = root_dir + "/worldKeyPoints/";
        std::vector<std::string> filenames;

        for (const auto &entry :
             std::filesystem::directory_iterator(label3d_dir)) {
            filenames.push_back(entry.path());
        }

        if (filenames.size() == 0) {
            error_message = "Failed loading, no files in directory.";
            return 1;
        };

        sort(filenames.begin(), filenames.end());
        std::string mostRecentFile = filenames.back();
        std::cout << "mostRecentFile: " << mostRecentFile << std::endl;

        std::ifstream fin;
        fin.open(mostRecentFile);
        if (fin.fail())
            throw mostRecentFile;
        std::string line;
        std::string delimeter = ",";
        size_t pos = 0;
        std::string token;

        int lineNum = 0;
        while (!fin.eof()) {
            fin >> line;
            while ((pos = line.find(delimeter)) != std::string::npos) {
                token = line.substr(0, pos);
                if (lineNum == 0) {
                    if (token.compare(skeleton->name) != 0) {
                        error_message = "Failed loading 3d keypoints, skeleton "
                                        "doesn't match.\n";
                        error_message += skeleton->name + ":" + token;
                        return 1;
                    }
                    line.erase(0, pos + delimeter.length());
                } else {
                    uint frame_num = stoul(token);
                    if (keypoints_map.find(frame_num) == keypoints_map.end()) {
                        KeyPoints *keypoints =
                            (KeyPoints *)malloc(sizeof(KeyPoints));
                        allocate_keypoints(keypoints, scene, skeleton);
                        keypoints_map[frame_num] = keypoints;
                    }
                    line.erase(0, pos + delimeter.length());

                    while ((pos = line.find(delimeter)) != std::string::npos) {
                        token = line.substr(0, pos);
                        int node = stoi(token); // get the node index
                        line.erase(0, pos + delimeter.length());

                        pos = line.find(delimeter);
                        token = line.substr(0, pos);
                        double x = stod(token);
                        line.erase(0, pos + delimeter.length());

                        pos = line.find(delimeter);
                        token = line.substr(0, pos);
                        double y = stod(token);
                        line.erase(0, pos + delimeter.length());

                        pos = line.find(delimeter);
                        token = line.substr(0, pos);
                        double z = stod(token);
                        line.erase(0, pos + delimeter.length());

                        keypoints_map[frame_num]->keypoints3d[node].x = x;
                        keypoints_map[frame_num]->keypoints3d[node].y = y;
                        keypoints_map[frame_num]->keypoints3d[node].z = z;

                        if (x == 1E7 || y == 1E7 || z == 1E7) {
                            for (int cam_idx = 0; cam_idx < scene->num_cams;
                                 cam_idx++) {
                                keypoints_map[frame_num]
                                    ->keypoints2d[cam_idx][node]
                                    .is_triangulated = false;
                            }
                        } else {
                            for (int cam_idx = 0; cam_idx < scene->num_cams;
                                 cam_idx++) {
                                keypoints_map[frame_num]
                                    ->keypoints2d[cam_idx][node]
                                    .is_triangulated = true;
                            }
                        }

                        // std::cout << "frame: " << frame_num << "  node: " << node << "  x: " << keypoints_map[frame_num]->keypoints3d[node].x \
                        //  << "  y: " << keypoints_map[frame_num]->keypoints3d[node].y << "  z: " << keypoints_map[frame_num]->keypoints3d[node].z << std::endl;
                    }
                }
            }
            lineNum++;
        }
        fin.close();
    }

    // for (int i=0; i<scene->num_cams; i++) {
    //     load_2d_keypoints(keypoints_map, skeleton, root_dir, i,
    //     camera_names[i], scene);
    // }

    auto handles = std::vector<std::thread>();
    for (int i = 0; i < scene->num_cams; i++) {
        handles.push_back(std::thread(&load_2d_keypoints_depreciated,
                                      std::ref(keypoints_map), skeleton,
                                      root_dir, i, camera_names[i], scene));
    }

    for (auto &handle : handles) {
        handle.join();
    }
    return 0;
}

int load_2d_keypoints(std::map<u32, KeyPoints *> &keypoints_map,
                      SkeletonContext *skeleton, std::string kp2d_file,
                      int cam_idx, render_scene *scene,
                      std::string &error_message) {

    std::ifstream fin(kp2d_file);
    if (!fin) {
        error_message = "Failed to open: " + kp2d_file;
        return 1;
    }

    std::string line;
    std::string delimeter = ",";
    size_t pos = 0;
    std::string token;

    // read csv file with cam parameters and tokenize line for this camera
    int lineNum = 0;
    while (!fin.eof()) {
        fin >> line;
        while ((pos = line.find(delimeter)) != std::string::npos) {
            token = line.substr(0, pos);
            if (lineNum == 0) {
                std::cout << token << std::endl;
                std::cout << skeleton->name << std::endl;
                if (token.compare(skeleton->name) != 0) {
                    error_message = "Failed loading, skeleton doesn't match.\n";
                    error_message += skeleton->name + ":" + token;
                    return 1;
                }
                line.erase(0, pos + delimeter.length());
            } else {
                uint frame_num = stoul(token);
                if (keypoints_map.find(frame_num) == keypoints_map.end()) {
                    KeyPoints *keypoints =
                        (KeyPoints *)malloc(sizeof(KeyPoints));
                    allocate_keypoints(keypoints, scene, skeleton);
                    keypoints_map[frame_num] = keypoints;
                }
                line.erase(0, pos + delimeter.length());

                while ((pos = line.find(delimeter)) != std::string::npos) {
                    token = line.substr(0, pos);
                    int node = stoi(token); // get the node index
                    line.erase(0, pos + delimeter.length());

                    pos = line.find(delimeter);
                    token = line.substr(0, pos);
                    double x = stod(token);
                    line.erase(0, pos + delimeter.length());

                    pos = line.find(delimeter);
                    token = line.substr(0, pos);
                    double y = stod(token);
                    line.erase(0, pos + delimeter.length());

                    keypoints_map[frame_num]
                        ->keypoints2d[cam_idx][node]
                        .position.x = x;
                    keypoints_map[frame_num]
                        ->keypoints2d[cam_idx][node]
                        .position.y = y;

                    if (x == 1E7 || y == 1E7) {
                        keypoints_map[frame_num]
                            ->keypoints2d[cam_idx][node]
                            .is_labeled = false;
                    } else {
                        keypoints_map[frame_num]
                            ->keypoints2d[cam_idx][node]
                            .is_labeled = true;
                    }
                    // std::cout << "frame: " << frame_num << "  node: " << node
                    // << "  x: " <<
                    // keypoints_map[frame_num]->keypoints2d[cam_idx][node].position.x
                    // << "  y: " <<
                    // keypoints_map[frame_num]->keypoints2d[cam_idx][node].position.y
                    // << std::endl;
                }
            }
        }
        lineNum++;
    }
    fin.close();
    return 0;
}

int find_most_recent_labels(std::string root_dir, std::string &most_recent_file,
                            std::string &error_message) {
    std::regex datetime_regex(R"(^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$)");

    std::vector<std::string> filenames;
    for (const auto &entry : std::filesystem::directory_iterator(root_dir)) {
        if (!entry.is_directory())
            continue;

        std::string folder_name = entry.path().filename().string();

        if (std::regex_match(folder_name, datetime_regex)) {
            filenames.push_back(entry.path());
        }
    }

    if (filenames.empty()) {
        error_message = "Failed loading, no date-time named folders found.";
        error_message +=
            "\nIf you are loading an old format, please check 'Old Format'. "
            "Once loaded, please save it to convert to the new format.";
        return 1;
    }
    sort(filenames.begin(), filenames.end());
    most_recent_file = filenames.back();
    std::cout << most_recent_file << std::endl;
    return 0;
}

int load_keypoints(std::string keypoints_folder,
                   std::map<u32, KeyPoints *> &keypoints_map,
                   SkeletonContext *skeleton, render_scene *scene,
                   std::vector<std::string> &camera_names,
                   std::string &error_message,
                   std::vector<std::string> &class_names) {

    // Intelligent loading based on skeleton configuration
    bool has_skeleton = skeleton->has_skeleton;
    bool has_bbox = skeleton->has_bbox;

    // Load 3D keypoints if skeleton supports them and multi-camera setup
    if (has_skeleton && scene->num_cams > 1) {
        std::string kp_3d = keypoints_folder + "/keypoints3d.csv";
        std::ifstream fin(kp_3d);
        if (!fin) {
            error_message = "Failed to open: " + kp_3d;
            return 1;
        }

        std::string line;
        std::string delimeter = ",";
        size_t pos = 0;
        std::string token;

        int line_num = 0;
        while (!fin.eof()) {
            fin >> line;
            while ((pos = line.find(delimeter)) != std::string::npos) {
                token = line.substr(0, pos);
                if (line_num == 0) {
                    if (token.compare(skeleton->name) != 0) {
                        error_message = "3D keypoints failed loading, skeleton "
                                        "doesn't match.";
                        error_message += skeleton->name + ":" + token;
                        return 1;
                    }
                    line.erase(0, pos + delimeter.length());
                } else {
                    uint frame_num = stoul(token);
                    if (keypoints_map.find(frame_num) == keypoints_map.end()) {
                        KeyPoints *keypoints =
                            (KeyPoints *)malloc(sizeof(KeyPoints));
                        allocate_keypoints(keypoints, scene, skeleton);
                        keypoints_map[frame_num] = keypoints;
                    }
                    line.erase(0, pos + delimeter.length());

                    while ((pos = line.find(delimeter)) != std::string::npos) {
                        token = line.substr(0, pos);
                        int node = stoi(token); // get the node index
                        line.erase(0, pos + delimeter.length());

                        pos = line.find(delimeter);
                        token = line.substr(0, pos);
                        double x = stod(token);
                        line.erase(0, pos + delimeter.length());

                        pos = line.find(delimeter);
                        token = line.substr(0, pos);
                        double y = stod(token);
                        line.erase(0, pos + delimeter.length());

                        pos = line.find(delimeter);
                        token = line.substr(0, pos);
                        double z = stod(token);
                        line.erase(0, pos + delimeter.length());

                        keypoints_map[frame_num]->keypoints3d[node].x = x;
                        keypoints_map[frame_num]->keypoints3d[node].y = y;
                        keypoints_map[frame_num]->keypoints3d[node].z = z;

                        if (x == 1E7 || y == 1E7 || z == 1E7) {
                            for (int cam_idx = 0; cam_idx < scene->num_cams;
                                 cam_idx++) {
                                keypoints_map[frame_num]
                                    ->keypoints2d[cam_idx][node]
                                    .is_triangulated = false;
                            }
                        } else {
                            for (int cam_idx = 0; cam_idx < scene->num_cams;
                                 cam_idx++) {
                                keypoints_map[frame_num]
                                    ->keypoints2d[cam_idx][node]
                                    .is_triangulated = true;
                            }
                        }
                    }
                }
            }
            line_num++;
        }
        fin.close();
    }

    // Load appropriate files based on skeleton configuration
    std::vector<std::thread> handles;
    std::vector<std::promise<int>> promises(scene->num_cams);
    std::vector<std::future<int>> results;
    std::vector<std::string> error_messages(scene->num_cams);

    for (int i = 0; i < scene->num_cams; i++) {
        results.push_back(promises[i].get_future());

        // Determine which files to load based on skeleton configuration
        if (has_skeleton && has_bbox) {
            std::string bbox_kp_file = keypoints_folder + "/" +
                                       camera_names[i] + "_bbox_keypoints.csv";
            std::string kp2d_file =
                keypoints_folder + "/" + camera_names[i] + ".csv";

            handles.emplace_back(
                [&keypoints_map, skeleton, bbox_kp_file, kp2d_file, i, scene,
                 &error_messages, &promises](int cam_idx) {
                    int ret = load_bbox_keypoints(keypoints_map, skeleton,
                                                  bbox_kp_file, i, scene,
                                                  error_messages[i]);
                    if (ret != 0) {
                        // Fallback to regular 2D keypoints
                        ret = load_2d_keypoints(keypoints_map, skeleton,
                                                kp2d_file, i, scene,
                                                error_messages[i]);
                    }
                    promises[cam_idx].set_value(ret);
                },
                i);
        } else if (skeleton->has_obb) {
            std::string obb_file =
                keypoints_folder + "/" + camera_names[i] + "_obb.csv";

            handles.emplace_back(
                [&keypoints_map, skeleton, obb_file, i, scene, &error_messages,
                 &promises, &class_names](int cam_idx) {
                    int ret = load_obb(keypoints_map, skeleton, obb_file, i,
                                       scene, error_messages[i], class_names);
                    promises[cam_idx].set_value(ret);
                },
                i);
        } else if (has_bbox && !has_skeleton) {
            std::string bbox_kp_file = keypoints_folder + "/" +
                                       camera_names[i] + "_bbox_keypoints.csv";
            std::string bbox_file =
                keypoints_folder + "/" + camera_names[i] + "_bboxes.csv";

            handles.emplace_back(
                [&keypoints_map, skeleton, bbox_kp_file, bbox_file, i, scene,
                 &error_messages, &promises](int cam_idx) {
                    // Try bbox keypoints first
                    int ret = load_bbox_keypoints(keypoints_map, skeleton,
                                                  bbox_kp_file, i, scene,
                                                  error_messages[i]);
                    if (ret != 0) {
                        // Fallback to bbox only
                        ret = load_bboxes(keypoints_map, skeleton, bbox_file, i,
                                          scene, error_messages[i]);
                    }
                    promises[cam_idx].set_value(ret);
                },
                i);
        } else if (has_skeleton && !has_bbox) {
            std::string kp2d_file =
                keypoints_folder + "/" + camera_names[i] + ".csv";

            handles.emplace_back(
                [&keypoints_map, skeleton, kp2d_file, i, scene, &error_messages,
                 &promises](int cam_idx) {
                    int ret =
                        load_2d_keypoints(keypoints_map, skeleton, kp2d_file, i,
                                          scene, error_messages[i]);
                    promises[cam_idx].set_value(ret);
                },
                i);
        } else {
            handles.emplace_back(
                [&error_messages, &promises](int cam_idx) {
                    error_messages[cam_idx] =
                        "Skeleton configuration has neither skeleton nor bbox "
                        "support";
                    promises[cam_idx].set_value(1);
                },
                i);
        }
    }

    // Join threads
    for (auto &t : handles)
        t.join();

    bool has_error = false;
    for (int i = 0; i < results.size(); ++i) {
        int status = results[i].get();
        if (status != 0) {
            error_message +=
                camera_names[i] + " error: " + error_messages[i] + "\n";
            has_error = true;
        }
    }

    if (has_error) {
        return 1;
    }

    return 0;
}

int load_bboxes(std::map<u32, KeyPoints *> &keypoints_map,
                SkeletonContext *skeleton, std::string bbox_file, int cam_idx,
                render_scene *scene, std::string &error_message) {
    std::ifstream fin;
    fin.open(bbox_file);
    if (fin.fail()) {
        error_message = "Could not open file: " + bbox_file;
        return 1;
    }

    std::string line;
    std::string delimeter = ",";
    size_t pos = 0;
    std::string token;
    int line_num = 0;

    while (std::getline(fin, line)) {
        if (line.empty())
            continue;

        if (line_num == 0) {
            // Check skeleton name
            if (line.compare(skeleton->name) != 0) {
                error_message =
                    "Skeleton doesn't match. Expected: " + skeleton->name +
                    " Got: " + line;
                fin.close();
                return 1;
            }
        } else if (line_num == 1) {
            // Skip header line
        } else {
            std::vector<std::string> tokens;
            std::stringstream ss(line);
            std::string item;

            while (std::getline(ss, item, ',')) {
                tokens.push_back(item);
            }

            if (tokens.size() >= 8) {
                uint frame_num;
                try {
                    frame_num = std::stoul(tokens[0]);
                } catch (const std::exception &e) {
                    continue;
                }

                int bbox_id = std::stoi(tokens[1]);
                int class_id = std::stoi(tokens[2]);
                float confidence = std::stof(tokens[3]);
                double x_min = std::stod(tokens[4]);
                double y_min = std::stod(tokens[5]);
                double x_max = std::stod(tokens[6]);
                double y_max = std::stod(tokens[7]);

                // Create keypoints entry if it doesn't exist
                if (keypoints_map.find(frame_num) == keypoints_map.end()) {
                    KeyPoints *keypoints =
                        (KeyPoints *)malloc(sizeof(KeyPoints));
                    allocate_keypoints(keypoints, scene, skeleton);
                    keypoints_map[frame_num] = keypoints;
                }

                // Create bounding box
                BoundingBox bbox;
                bbox.rect = new ImPlotRect(x_min, x_max, y_min, y_max);
                bbox.state = RectTwoPoints;
                bbox.class_id = class_id;
                bbox.confidence = confidence;
                bbox.has_bbox_keypoints = false;
                bbox.bbox_keypoints2d = nullptr;
                bbox.active_kp_id = nullptr;

                while (keypoints_map[frame_num]->bbox2d_list[cam_idx].size() <=
                       bbox_id) {
                    BoundingBox default_bbox;
                    default_bbox.rect = nullptr;
                    default_bbox.state = RectNull;
                    default_bbox.class_id = -1;
                    default_bbox.confidence = 0.0f;
                    default_bbox.has_bbox_keypoints = false;
                    default_bbox.bbox_keypoints2d = nullptr;
                    default_bbox.active_kp_id = nullptr;
                    keypoints_map[frame_num]->bbox2d_list[cam_idx].push_back(
                        default_bbox);
                }

                if (bbox_id <
                    keypoints_map[frame_num]->bbox2d_list[cam_idx].size()) {
                    keypoints_map[frame_num]->bbox2d_list[cam_idx][bbox_id] =
                        bbox;
                } else {
                    keypoints_map[frame_num]->bbox2d_list[cam_idx].push_back(
                        bbox);
                }
            }
        }
        line_num++;
    }

    fin.close();
    return 0;
}

// Load bounding boxes with keypoints from CSV files
int load_bbox_keypoints(std::map<u32, KeyPoints *> &keypoints_map,
                        SkeletonContext *skeleton, std::string bbox_kp_file,
                        int cam_idx, render_scene *scene,
                        std::string &error_message) {
    std::ifstream fin;
    fin.open(bbox_kp_file);
    if (fin.fail()) {
        error_message = "Could not open file: " + bbox_kp_file;
        return 1;
    }

    std::string line;
    int line_num = 0;

    while (std::getline(fin, line)) {
        if (line.empty())
            continue;

        if (line_num == 0) {
            // Check skeleton name
            if (line.compare(skeleton->name) != 0) {
                error_message =
                    "Skeleton doesn't match. Expected: " + skeleton->name +
                    " Got: " + line;
                fin.close();
                return 1;
            }
        } else if (line_num == 1) {
            // Skip header line
        } else {
            std::vector<std::string> tokens;
            std::stringstream ss(line);
            std::string item;

            while (std::getline(ss, item, ',')) {
                tokens.push_back(item);
            }

            if (tokens.size() >= 12) {
                uint frame_num;
                try {
                    frame_num = std::stoul(tokens[0]);
                } catch (const std::exception &e) {
                    continue;
                }

                int bbox_id = std::stoi(tokens[1]);
                int class_id = std::stoi(tokens[2]);
                float confidence = std::stof(tokens[3]);
                double x_min = std::stod(tokens[4]);
                double y_min = std::stod(tokens[5]);
                double x_max = std::stod(tokens[6]);
                double y_max = std::stod(tokens[7]);
                int keypoint_id = std::stoi(tokens[8]);
                double kp_x = std::stod(tokens[9]);
                double kp_y = std::stod(tokens[10]);
                bool is_labeled = std::stoi(tokens[11]) != 0;

                // Create keypoints entry if it doesn't exist
                if (keypoints_map.find(frame_num) == keypoints_map.end()) {
                    KeyPoints *keypoints =
                        (KeyPoints *)malloc(sizeof(KeyPoints));
                    allocate_keypoints(keypoints, scene, skeleton);
                    keypoints_map[frame_num] = keypoints;
                }

                // Ensure the bbox list is large enough
                while (keypoints_map[frame_num]->bbox2d_list[cam_idx].size() <=
                       bbox_id) {
                    BoundingBox default_bbox;
                    default_bbox.rect = nullptr;
                    default_bbox.state = RectNull;
                    default_bbox.class_id = -1;
                    default_bbox.confidence = 0.0f;
                    default_bbox.has_bbox_keypoints = false;
                    default_bbox.bbox_keypoints2d = nullptr;
                    default_bbox.active_kp_id = nullptr;
                    keypoints_map[frame_num]->bbox2d_list[cam_idx].push_back(
                        default_bbox);
                }

                // Get reference to the bounding box
                BoundingBox *bbox =
                    &keypoints_map[frame_num]->bbox2d_list[cam_idx][bbox_id];

                // Initialize bounding box if it's the first keypoint
                if (keypoint_id == 0 || bbox->rect == nullptr) {
                    bbox->rect = new ImPlotRect(x_min, x_max, y_min, y_max);
                    bbox->state = RectTwoPoints;
                    bbox->class_id = class_id;
                    bbox->confidence = confidence;

                    // Allocate keypoints if not already done
                    if (!bbox->has_bbox_keypoints) {
                        allocate_bbox_keypoints(bbox, scene, skeleton);
                    }
                }

                // Set keypoint data
                if (keypoint_id < skeleton->num_nodes &&
                    bbox->bbox_keypoints2d != nullptr) {
                    bbox->bbox_keypoints2d[cam_idx][keypoint_id].position.x =
                        kp_x;
                    bbox->bbox_keypoints2d[cam_idx][keypoint_id].position.y =
                        kp_y;
                    bbox->bbox_keypoints2d[cam_idx][keypoint_id].is_labeled =
                        is_labeled;
                    bbox->bbox_keypoints2d[cam_idx][keypoint_id]
                        .is_triangulated = false;
                }
            }
        }
        line_num++;
    }

    fin.close();
    return 0;
}

// Load oriented bounding boxes from CSV files
int load_obb(std::map<u32, KeyPoints *> &keypoints_map,
             SkeletonContext *skeleton, std::string obb_file, int cam_idx,
             render_scene *scene, std::string &error_message,
             std::vector<std::string> &class_names) {
    std::ifstream fin;
    fin.open(obb_file);
    if (fin.fail()) {
        error_message = "Could not open file: " + obb_file;
        return 1;
    }

    std::string line;
    int line_num = 0;

    while (std::getline(fin, line)) {
        if (line.empty())
            continue;

        if (line_num == 0) {
            // Check skeleton name
            if (line.compare(skeleton->name) != 0) {
                error_message =
                    "Skeleton doesn't match. Expected: " + skeleton->name +
                    " Got: " + line;
                fin.close();
                return 1;
            }
        } else if (line_num == 1) {
            // Skip header line
        } else {
            std::vector<std::string> tokens;
            std::stringstream ss(line);
            std::string item;

            while (std::getline(ss, item, ',')) {
                tokens.push_back(item);
            }

            // Check for new simplified format (11 fields: frame, obb_id,
            // class_id, 8 corner coordinates)
            if (tokens.size() >= 11) {
                uint frame_num;
                try {
                    frame_num = std::stoul(tokens[0]);
                } catch (const std::exception &e) {
                    continue;
                }

                int obb_id = std::stoi(tokens[1]);
                int class_id = std::stoi(tokens[2]);

                // Parse the four corners
                ImVec2 corners[4];
                corners[0] = ImVec2(std::stof(tokens[3]),
                                    std::stof(tokens[4])); // corner 1
                corners[1] = ImVec2(std::stof(tokens[5]),
                                    std::stof(tokens[6])); // corner 2
                corners[2] = ImVec2(std::stof(tokens[7]),
                                    std::stof(tokens[8])); // corner 3
                corners[3] = ImVec2(std::stof(tokens[9]),
                                    std::stof(tokens[10])); // corner 4

                // Create keypoints entry if it doesn't exist
                if (keypoints_map.find(frame_num) == keypoints_map.end()) {
                    KeyPoints *keypoints =
                        (KeyPoints *)malloc(sizeof(KeyPoints));
                    allocate_keypoints(keypoints, scene, skeleton);
                    keypoints_map[frame_num] = keypoints;
                }

                // Ensure the OBB list is large enough
                while (keypoints_map[frame_num]->obb2d_list[cam_idx].size() <=
                       obb_id) {
                    OrientedBoundingBox default_obb;
                    default_obb.axis_point1 = ImVec2(0, 0);
                    default_obb.axis_point2 = ImVec2(0, 0);
                    default_obb.corner_point = ImVec2(0, 0);
                    default_obb.center = ImVec2(0, 0);
                    default_obb.width = 0;
                    default_obb.height = 0;
                    default_obb.rotation = 0;
                    default_obb.state = OBBNull;
                    default_obb.class_id = -1;
                    default_obb.confidence = 0.0f;
                    keypoints_map[frame_num]->obb2d_list[cam_idx].push_back(
                        default_obb);
                }

                // Get reference to the oriented bounding box
                OrientedBoundingBox *obb =
                    &keypoints_map[frame_num]->obb2d_list[cam_idx][obb_id];

                // Set OBB data from corners using the helper function
                set_obb_from_corners(obb, corners, class_id);
            }
        }
        line_num++;
    }

    fin.close();
    return 0;
}

void world_coordinates_projection_points(CameraParams *cvp, int image_height,
                                         double *axis_x_values,
                                         double *axis_y_values, float scale) {
    std::vector<cv::Point3f> world_coordinates;
    world_coordinates.push_back(cv::Point3f(0.0f, 0.0f, 0.0f));
    world_coordinates.push_back(cv::Point3f(scale * 1.0f, 0.0f, 0.0f));
    world_coordinates.push_back(cv::Point3f(0.0f, scale * 1.0f, 0.0f));
    world_coordinates.push_back(cv::Point3f(0.0f, 0.0f, scale * 1.0f));

    std::vector<cv::Point2f> img_pts;
    cv::projectPoints(world_coordinates, cvp->rvec, cvp->tvec, cvp->k,
                      cvp->dist_coeffs, img_pts);

    for (int i = 0; i < 4; i++) {
        axis_x_values[i] = img_pts.at(i).x;
        axis_y_values[i] = image_height - img_pts.at(i).y;
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
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;

    float radius = 1473.0f;
    std::vector<cv::Point3f> inPts;

    for (int i = 0; i <= n; i++) {
        float angle = (3.14159265358979323846 * 2) * (float(i) / float(n - 1));
        x.push_back(sin(angle) * radius);
        y.push_back(cos(angle) * radius);
        z.push_back(0.0f);
    }

    for (int i = 0; i < n; i++) {
        cv::Point3f p;
        p.x = x[i];
        p.y = y[i];
        p.z = z[i];
        inPts.push_back(p);
    }

    std::vector<cv::Point2f> img_pts;
    cv::projectPoints(inPts, cvp->rvec, cvp->tvec, cvp->k, cvp->dist_coeffs,
                      img_pts);

    for (int i = 0; i < n; i++) {
        arena_x[i] = img_pts.at(i).x;
        arena_y[i] = image_height - img_pts.at(i).y;
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

static void triangulate_bounding_boxes(KeyPoints *keypoints,
                                       SkeletonContext *skeleton,
                                       std::vector<CameraParams> camera_params,
                                       render_scene *scene,
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

        std::vector<cv::Mat> sfmPoints2d;
        std::vector<cv::Mat> projection_matrices;

        cv::Mat point1 = (cv::Mat_<double>(2, 1) << center1_x,
                          (double)scene->image_height[cam1_id] - center1_y);
        cv::Mat point2 = (cv::Mat_<double>(2, 1) << center2_x,
                          (double)scene->image_height[cam2_id] - center2_y);

        cv::Mat point1_undistort, point2_undistort;
        cv::undistortPoints(point1, point1_undistort, camera_params[cam1_id].k,
                            camera_params[cam1_id].dist_coeffs, cv::noArray(),
                            camera_params[cam1_id].k);
        cv::undistortPoints(point2, point2_undistort, camera_params[cam2_id].k,
                            camera_params[cam2_id].dist_coeffs, cv::noArray(),
                            camera_params[cam2_id].k);

        sfmPoints2d.push_back(point1_undistort.reshape(1, 2));
        sfmPoints2d.push_back(point2_undistort.reshape(1, 2));
        projection_matrices.push_back(camera_params[cam1_id].projection_mat);
        projection_matrices.push_back(camera_params[cam2_id].projection_mat);

        cv::Mat triangulated_center;
        cv::sfm::triangulatePoints(sfmPoints2d, projection_matrices,
                                   triangulated_center);

        std::cout << "Triangulated 3D center: ("
                  << triangulated_center.at<double>(0) << ", "
                  << triangulated_center.at<double>(1) << ", "
                  << triangulated_center.at<double>(2) << ")" << std::endl;

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

            cv::Mat reprojected_points;
            cv::projectPoints(
                triangulated_center, camera_params[target_cam].rvec,
                camera_params[target_cam].tvec, camera_params[target_cam].k,
                camera_params[target_cam].dist_coeffs, reprojected_points);

            double proj_x = reprojected_points.at<double>(0, 0);
            double proj_y = (double)scene->image_height[target_cam] -
                            reprojected_points.at<double>(0, 1);

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

    } catch (const cv::Exception &e) {
        std::cerr << "OpenCV error in bounding box triangulation: " << e.what()
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
#endif
