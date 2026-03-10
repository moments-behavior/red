#pragma once
#include "implot.h"
#include "render.h"
#include "skeleton.h"
#include "json.hpp"
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
