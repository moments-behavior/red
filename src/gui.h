#ifndef RED_GUI
#define RED_GUI
#include "render.h"
#include "skeleton.h"
#include <filesystem>
#include <fstream>
#include <future>
#include <thread>
#include <vector>

struct ProjectContext {
    std::string root_dir;
    std::vector<std::string> input_file_names;
    std::vector<std::string> camera_names;
};

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
                    oss << "(" << keypoints->keypoints3d->x << ", "
                        << keypoints->keypoints3d->y << ", "
                        << keypoints->keypoints3d->z << ")";
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

    for (int i = 0; i < filenames.size(); i++) {
        std::cout << filenames.at(i) << std::endl;
    }

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
                std::cout << token << std::endl;
                std::cout << skeleton->name << std::endl;
                if (token.compare(skeleton->name) != 0) {
                    std::cout << "Failed loading, skeleton doesn't match."
                              << std::endl;
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
                        error_message =
                            "Failed loading, skeleton doesn't match.";
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
                    error_message = "Failed loading, skeleton doesn't match.";
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

int load_keypoints(std::map<u32, KeyPoints *> &keypoints_map,
                   SkeletonContext *skeleton, std::string root_dir,
                   render_scene *scene, std::vector<std::string> &camera_names,
                   std::string &error_message) {

    std::vector<std::string> filenames;
    for (const auto &entry : std::filesystem::directory_iterator(root_dir)) {
        filenames.push_back(entry.path());
    }

    if (filenames.size() == 0) {
        error_message = "Failed loading, no files in directory.";
        return 1;
    };

    sort(filenames.begin(), filenames.end());
    std::string most_recent_file = filenames.back();
    std::cout << "Most recent file: " << most_recent_file << std::endl;

    if (scene->num_cams > 1) {
        // load 3d keypoints
        std::string kp_3d = most_recent_file + "/keypoints3d.csv";
        std::ifstream fin(kp_3d);
        if (!fin) {
            error_message = "File open failed, skipping.";
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
                        error_message =
                            "Failed loading, skeleton doesn't match.";
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
            line_num++;
        }
        fin.close();
    }

    std::vector<std::thread> handles;
    std::vector<std::promise<int>> promises(scene->num_cams);
    std::vector<std::future<int>> results;
    std::vector<std::string> error_messages(scene->num_cams);

    for (int i = 0; i < scene->num_cams; i++) {
        std::string kp2d = most_recent_file + "/" + camera_names[i] + ".csv";
        results.push_back(promises[i].get_future());

        handles.emplace_back(
            [&keypoints_map, skeleton, kp2d, i, scene, &error_messages,
             &promises](int cam_idx) {
                int ret = load_2d_keypoints(keypoints_map, skeleton, kp2d, i,
                                            scene, error_messages[i]);
                promises[cam_idx].set_value(ret);
            },
            i);
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

#endif
