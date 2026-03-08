#pragma once
#include "implot.h"
#include "render.h"
#include "skeleton.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

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
            output3d_file << "," << i << "," << keypoints->kp3d[i].position.x
                          << "," << keypoints->kp3d[i].position.y << ","
                          << keypoints->kp3d[i].position.z;
        }
        output3d_file << "\n";

        for (int cam = 0; cam < num_cameras; cam++) {
            if (*input_is_imgs) {
                output2d_files[cam] << input_files[frame];
            } else {
                output2d_files[cam] << frame;
            }
            for (int node = 0; node < skeleton->num_nodes; node++) {
                output2d_files[cam] << "," << node << ","
                                    << keypoints->kp2d[cam][node].position.x
                                    << ","
                                    << keypoints->kp2d[cam][node].position.y;
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
void get_obb_corners(const OrientedBoundingBox *obb, ImVec2 corners[4]) {
    if (obb->state != OBBComplete) {
        // If OBB is not complete, set all corners to zero
        for (int i = 0; i < 4; i++) {
            corners[i] = ImVec2(0, 0);
        }
        return;
    }

    // Use the stored properties to calculate corners
    float cos_rot = cosf(obb->rotation);
    float sin_rot = sinf(obb->rotation);

    // Half-width and half-height
    float half_w = obb->width * 0.5f;
    float half_h = obb->height * 0.5f;

    // Calculate the four corners relative to center, then translate
    // Corner order: bottom-left, bottom-right, top-right, top-left (in local
    // coordinates)
    ImVec2 local_corners[4] = {
        {-half_w, -half_h}, // bottom-left
        {half_w, -half_h},  // bottom-right
        {half_w, half_h},   // top-right
        {-half_w, half_h}   // top-left
    };

    // Rotate and translate each corner
    for (int i = 0; i < 4; i++) {
        float x = local_corners[i].x;
        float y = local_corners[i].y;

        // Apply rotation
        float rotated_x = x * cos_rot - y * sin_rot;
        float rotated_y = x * sin_rot + y * cos_rot;

        // Translate to world position
        corners[i].x = rotated_x + obb->center.x;
        corners[i].y = rotated_y + obb->center.y;
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
                                   RenderScene *scene) {
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

                    keypoints_map[frame_num]->kp2d[cam_idx][node].position.x =
                        x;
                    keypoints_map[frame_num]->kp2d[cam_idx][node].position.y =
                        y;

                    if (x == 1E7 || y == 1E7) {
                        keypoints_map[frame_num]
                            ->kp2d[cam_idx][node]
                            .is_labeled = false;
                    } else {
                        keypoints_map[frame_num]
                            ->kp2d[cam_idx][node]
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
                               RenderScene *scene,
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

                        keypoints_map[frame_num]->kp3d[node].position.x = x;
                        keypoints_map[frame_num]->kp3d[node].position.y = y;
                        keypoints_map[frame_num]->kp3d[node].position.z = z;

                        if (x == 1E7 || y == 1E7 || z == 1E7) {
                            keypoints_map[frame_num]
                                ->kp3d[node]
                                .is_triangulated = false;
                        } else {
                            keypoints_map[frame_num]
                                ->kp3d[node]
                                .is_triangulated = true;
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
                      int cam_idx, RenderScene *scene,
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

                    keypoints_map[frame_num]->kp2d[cam_idx][node].position.x =
                        x;
                    keypoints_map[frame_num]->kp2d[cam_idx][node].position.y =
                        y;

                    if (x == 1E7 || y == 1E7) {
                        keypoints_map[frame_num]
                            ->kp2d[cam_idx][node]
                            .is_labeled = false;
                    } else {
                        keypoints_map[frame_num]
                            ->kp2d[cam_idx][node]
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
    return 0;
}
int load_bboxes(std::map<u32, KeyPoints *> &keypoints_map,
                SkeletonContext *skeleton, std::string bbox_file, int cam_idx,
                RenderScene *scene, std::string &error_message) {
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
                        int cam_idx, RenderScene *scene,
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
                }
            }
        }
        line_num++;
    }

    fin.close();
    return 0;
}
void set_obb_from_corners(OrientedBoundingBox *obb, const ImVec2 corners[4],
                          int class_id) {
    obb->axis_point1 = corners[0];
    obb->axis_point2 = corners[1];
    obb->corner_point = corners[3];

    obb->center.x =
        (corners[0].x + corners[1].x + corners[2].x + corners[3].x) / 4.0f;
    obb->center.y =
        (corners[0].y + corners[1].y + corners[2].y + corners[3].y) / 4.0f;

    float dx = corners[1].x - corners[0].x;
    float dy = corners[1].y - corners[0].y;
    obb->width = sqrtf(dx * dx + dy * dy);

    dx = corners[3].x - corners[0].x;
    dy = corners[3].y - corners[0].y;
    obb->height = sqrtf(dx * dx + dy * dy);

    obb->rotation =
        atan2f(corners[1].y - corners[0].y, corners[1].x - corners[0].x);

    obb->state = OBBComplete;
    obb->class_id = class_id;
    obb->confidence = 1.0f;
}
// Load oriented bounding boxes from CSV files
int load_obb(std::map<u32, KeyPoints *> &keypoints_map,
             SkeletonContext *skeleton, std::string obb_file, int cam_idx,
             RenderScene *scene, std::string &error_message,
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
int load_keypoints(std::string keypoints_folder,
                   std::map<u32, KeyPoints *> &keypoints_map,
                   SkeletonContext *skeleton, RenderScene *scene,
                   std::vector<std::string> &camera_names,
                   std::string &error_message) {

    // Load 3D keypoints if multi-camera setup
    if (skeleton->has_skeleton && scene->num_cams > 1) {
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

                        keypoints_map[frame_num]->kp3d[node].position.x = x;
                        keypoints_map[frame_num]->kp3d[node].position.y = y;
                        keypoints_map[frame_num]->kp3d[node].position.z = z;

                        if (x == 1E7 || y == 1E7 || z == 1E7) {
                            keypoints_map[frame_num]
                                ->kp3d[node]
                                .is_triangulated = false;
                        } else {
                            keypoints_map[frame_num]
                                ->kp3d[node]
                                .is_triangulated = true;
                        }
                    }
                }
            }
            line_num++;
        }
        fin.close();
    }

    // Load 2D keypoints per camera
    std::vector<std::thread> handles;
    std::vector<std::promise<int>> promises(scene->num_cams);
    std::vector<std::future<int>> results;
    std::vector<std::string> error_messages(scene->num_cams);

    for (int i = 0; i < scene->num_cams; i++) {
        results.push_back(promises[i].get_future());

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
