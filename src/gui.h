#ifndef RED_GUI
#define RED_GUI
#include "render.h"
#include "skeleton.h"
#include <imfilebrowser.h>
#include <fstream>
#include <filesystem>

struct ProjectContext{
    std::string root_dir;
    std::vector<std::string> input_file_names;
    std::vector<std::string> camera_names;
};


static void gui_plot_keypoints(KeyPoints *keypoints, SkeletonContext *skeleton, int view_idx, int num_cams)
{
    float pt_size = 6.0f;
    for (u32 node=0; node < skeleton->num_nodes; node++){
        if (keypoints->keypoints2d[view_idx][node].is_labeled){
            ImVec4 node_color; 
            if (keypoints->active_id[view_idx]==node) {
                node_color = (ImVec4)ImColor::HSV(0.8, 0.9f, 0.9f);
            } else {
                node_color.w = 1.0f; 
                node_color.x = skeleton->node_colors.at(node).x;
                node_color.y = skeleton->node_colors.at(node).y;
                node_color.z = skeleton->node_colors.at(node).z;
            }
            int id = skeleton->num_nodes * view_idx + node;
            static bool drag_point_clicked;
            static bool drag_point_hovered;
            static bool drag_point_modified;
            drag_point_modified = ImPlot::DragPoint(id, &keypoints->keypoints2d[view_idx][node].position.x, &keypoints->keypoints2d[view_idx][node].position.y, node_color, pt_size, ImPlotDragToolFlags_None, &drag_point_clicked, &drag_point_hovered);
            if (drag_point_modified) {
                keypoints->keypoints2d[view_idx][node].is_triangulated = false;
            }
            if (drag_point_hovered) {
                if (ImGui::IsKeyPressed(ImGuiKey_R, false))  // delete active keypoint
                {
                    keypoints->keypoints2d[view_idx][node].position = {1E7,  1E7};
                    keypoints->keypoints2d[view_idx][node].is_labeled = false;                                        
                    keypoints->keypoints2d[view_idx][node].is_triangulated = false;
                    keypoints->active_id[view_idx] = node;
                }
                
                if (ImGui::IsKeyPressed(ImGuiKey_F, false)) // Delete active keypoints from all the views
                {
                    for (int cam_idx =0; cam_idx < num_cams; cam_idx++) {
                        keypoints->keypoints2d[cam_idx][node].position = {1E7,  1E7};
                        keypoints->keypoints2d[cam_idx][node].is_labeled = false;                                        
                        keypoints->keypoints2d[cam_idx][node].is_triangulated = false;
                        keypoints->active_id[cam_idx] = node;
                    }
                }
            }

            if (drag_point_clicked) {
                keypoints->active_id[view_idx] = node;
            }
        }
    }

    for (u32 edge=0; edge < skeleton->num_edges; edge++)
    {
        auto[a,b] = skeleton->edges[edge];

        if (keypoints->keypoints2d[view_idx][a].is_labeled && keypoints->keypoints2d[view_idx][b].is_labeled)
        {
            double xs[2] {keypoints->keypoints2d[view_idx][a].position.x, keypoints->keypoints2d[view_idx][b].position.x};
            double ys[2] {keypoints->keypoints2d[view_idx][a].position.y, keypoints->keypoints2d[view_idx][b].position.y};
            ImPlot::PlotLine("##line", xs, ys, 2);
        }
    }

}

static void gui_plot_bbox_from_keypoints(KeyPoints *keypoints, SkeletonContext *skeleton, int view_idx, int top_left_idx, int bottom_right_idx)
{
    if (keypoints->keypoints2d[view_idx][top_left_idx].is_labeled && keypoints->keypoints2d[view_idx][bottom_right_idx].is_labeled) {
        double xs[5] {keypoints->keypoints2d[view_idx][top_left_idx].position.x,
                    keypoints->keypoints2d[view_idx][bottom_right_idx].position.x, 
                    keypoints->keypoints2d[view_idx][bottom_right_idx].position.x,
                    keypoints->keypoints2d[view_idx][top_left_idx].position.x,
                    keypoints->keypoints2d[view_idx][top_left_idx].position.x
                    };

        double ys[5] {keypoints->keypoints2d[view_idx][top_left_idx].position.y,
                    keypoints->keypoints2d[view_idx][top_left_idx].position.y,
                    keypoints->keypoints2d[view_idx][bottom_right_idx].position.y,
                    keypoints->keypoints2d[view_idx][bottom_right_idx].position.y,
                    keypoints->keypoints2d[view_idx][top_left_idx].position.y
                    };

        ImPlot::SetNextLineStyle(ImVec4(0.5, 1.0, 1.0,1.0), 3.0);
        ImPlot::PlotLine("##line", xs, ys, 5); 
    }
}

std::string current_date_time() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d_%X", &tstruct);
    return buf;
}


void save_keypoints(std::map<u32, KeyPoints*> keypoints_map, SkeletonContext* skeleton, std::string root_dir, int num_cameras, std::vector<std::string>& camera_names)
{
    std::string now = current_date_time();
    std::string filename = root_dir + "/worldKeyPoints/keypoints_" + now;
    std::ofstream output_file(filename);
    std::vector<std::ofstream> output2d_files;

    for (u32 i = 0; i < num_cameras; i++) {
        std::string filename_cam = root_dir + "/" + camera_names[i] + "/" + camera_names[i] + "_" + now;
        std::ofstream output_file_cam(filename_cam);
        output2d_files.push_back(std::move(output_file_cam));
    }

    output_file << skeleton->name << ",\n";
    for (u32 i = 0; i < num_cameras; i++) {
        output2d_files[i] << skeleton->name << ",\n";
    }

    std::map<u32, KeyPoints*>::iterator it = keypoints_map.begin();
    while (it != keypoints_map.end())
    {
        u32 frame = it->first;
        KeyPoints* keypoints = it->second;
        // write frame number
        output_file << frame << ",";
        // fore each labeled keypoint, write idx, xpos, ypos, zpos
        for (u32 i = 0; i < skeleton->num_nodes; i++)
        {   
            if (i == skeleton->num_nodes - 1) {
                // last keypoints (RJ added extra "," at end of row)
                output_file << i << "," << keypoints->keypoints3d[i].x << "," << keypoints->keypoints3d[i].y << "," << keypoints->keypoints3d[i].z << ",";
            } else {
                output_file << i << "," << keypoints->keypoints3d[i].x << "," << keypoints->keypoints3d[i].y << "," << keypoints->keypoints3d[i].z << ",";
            }
        }
        output_file << "\n";

        for (int cam = 0; cam < num_cameras; cam++) {
            output2d_files[cam] << frame << ",";
            for (int node = 0; node < skeleton->num_nodes; node++) {
                if (node == skeleton->num_nodes - 1) {
                    // last keypoints (RJ added extra "," at end of row)
                    output2d_files[cam] << node << "," << keypoints->keypoints2d[cam][node].position.x << "," << keypoints->keypoints2d[cam][node].position.y << ",";
                } else {
                    output2d_files[cam] << node << "," << keypoints->keypoints2d[cam][node].position.x << "," << keypoints->keypoints2d[cam][node].position.y << ",";
                }
            }
            output2d_files[cam] << "\n";
        }

        it++;
    }

    output_file.close();
    std::cout << filename << " created"  << std::endl; 

    for (u32 i = 0; i < num_cameras; i++) {
        output2d_files[i].close();
    }
}


void load_2d_keypoints(std::map<u32, KeyPoints*>& keypoints_map, SkeletonContext* skeleton, std::string root_dir, int cam_idx, std::string camera_name, render_scene *scene) {
    std::string labeled_data_dir = root_dir  + "/" + camera_name;
    std::vector<std::string> filenames;

    for (const auto & entry : std::filesystem::directory_iterator(labeled_data_dir))
    {
        #ifdef _WIN32
            filenames.push_back(entry.path().string());
        #else
            filenames.push_back(entry.path());
        #endif 
    }

    if (filenames.size() == 0)
    {
        std::cout << "no files in labeled_data_dir for " << camera_name << std::endl;
        return;
    };

    for (int i=0; i<filenames.size(); i++)
    {
        std::cout << filenames.at(i) << std::endl;
    }

    sort(filenames.begin(), filenames.end());
    std::string mostRecentFile = filenames.back();
    std::cout << "mostRecentFile: " << mostRecentFile << std::endl;

    std::ifstream fin;
    fin.open(mostRecentFile);
    if (fin.fail()) throw mostRecentFile;  // the exception being checked

    std::string line;
    std::string delimeter = ",";
    size_t pos = 0;
    std::string token;

    // read csv file with cam parameters and tokenize line for this camera
    int lineNum = 0;
    while(!fin.eof()){
        fin >> line;

        while ((pos = line.find(delimeter)) != std::string::npos)
        {
            token = line.substr(0, pos);
            if (lineNum == 0)
            {
                if (token.compare(skeleton->name) != 0) {
                    std::cout << "Failed loading, skeleton doesn't match." << std::endl;
                    return;
                }                       
                line.erase(0, pos + delimeter.length());
            }
            else
            {
                u32 frame_num = stoul(token);
                if (keypoints_map.find(frame_num)==keypoints_map.end()) {
                    KeyPoints* keypoints = (KeyPoints *)malloc(sizeof(KeyPoints));
                    allocate_keypoints(keypoints, scene, skeleton);
                    keypoints_map[frame_num] = keypoints; 
                }
                line.erase(0, pos + delimeter.length());

                while ((pos = line.find(delimeter)) != std::string::npos)
                {
                    token = line.substr(0, pos);
                    int node = stoi(token);   // get the node index
                    line.erase(0, pos + delimeter.length());

                    pos = line.find(delimeter);
                    token = line.substr(0, pos);
                    double x = stod(token);
                    line.erase(0, pos + delimeter.length());

                    pos = line.find(delimeter);
                    token = line.substr(0, pos);
                    double y = stod(token);
                    line.erase(0, pos + delimeter.length());

                    keypoints_map[frame_num]->keypoints2d[cam_idx][node].position.x = x;
                    keypoints_map[frame_num]->keypoints2d[cam_idx][node].position.y = y;

                    if (x == 1E7 || y == 1E7)  
                    {
                        keypoints_map[frame_num]->keypoints2d[cam_idx][node].is_labeled = false;
                    } else {
                        keypoints_map[frame_num]->keypoints2d[cam_idx][node].is_labeled = true;
                    }
                    // std::cout << "frame: " << frame_num << "  node: " << node << "  x: " << keypoints_map[frame_num]->keypoints2d[cam_idx][node].position.x << "  y: " << keypoints_map[frame_num]->keypoints2d[cam_idx][node].position.y << std::endl;
                }
            }
        }
        lineNum++;
    }
    fin.close();
}

void load_keypoints(std::map<u32, KeyPoints*>& keypoints_map, SkeletonContext* skeleton, std::string root_dir, render_scene *scene, std::vector<std::string>& camera_names) {
    std::string label3d_dir = root_dir + "/worldKeyPoints/";
    std::vector<std::string> filenames;

    for (const auto & entry : std::filesystem::directory_iterator(label3d_dir))
    {
        #ifdef _WIN32
            filenames.push_back(entry.path().string());
        #else
            filenames.push_back(entry.path());
        #endif 
    }

    if (filenames.size() == 0)
    {
        std::cout << "Failed loading, no files in labeled_data_dir" << std::endl;
        return;
    };

    sort(filenames.begin(), filenames.end());
    std::string mostRecentFile = filenames.back();
    std::cout << "mostRecentFile: " << mostRecentFile << std::endl;

    std::ifstream fin;
    fin.open(mostRecentFile);
    if (fin.fail()) throw mostRecentFile;  
    std::string line;
    std::string delimeter = ",";
    size_t pos = 0;
    std::string token;

    int lineNum = 0;
    while(!fin.eof()) {
        fin >> line;
        while ((pos = line.find(delimeter)) != std::string::npos)
        {
            token = line.substr(0, pos);
            if (lineNum == 0)
            {
                if (token.compare(skeleton->name) != 0) {
                    std::cout << "Failed loading, skeleton doesn't match." << std::endl;
                    return;
                }                       
                line.erase(0, pos + delimeter.length());
            }
            else
            {
                u32 frame_num = stoul(token);
                if (keypoints_map.find(frame_num)==keypoints_map.end()) {
                    KeyPoints* keypoints = (KeyPoints *)malloc(sizeof(KeyPoints));
                    allocate_keypoints(keypoints, scene, skeleton);
                    keypoints_map[frame_num] = keypoints; 
                }
                line.erase(0, pos + delimeter.length());

                while ((pos = line.find(delimeter)) != std::string::npos)
                {
                    token = line.substr(0, pos);
                    int node = stoi(token);   // get the node index
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
                    
                    if (x == 1E7 || y == 1E7 || z==1E7) {
                        for (int cam_idx=0; cam_idx<scene->num_cams; cam_idx++) {
                            keypoints_map[frame_num]->keypoints2d[cam_idx][node].is_triangulated = false;
                        }
                    }
                    else {
                        for (int cam_idx=0; cam_idx<scene->num_cams; cam_idx++) {
                            keypoints_map[frame_num]->keypoints2d[cam_idx][node].is_triangulated = true;
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

    // for (int i=0; i<scene->num_cams; i++) {
    //     load_2d_keypoints(keypoints_map, skeleton, root_dir, i, camera_names[i], scene);
    // }

    auto handles = std::vector<std::thread>();
    for (int i=0; i<scene->num_cams; i++) {
        handles.push_back(std::thread(&load_2d_keypoints, std::ref(keypoints_map), skeleton, root_dir, i, camera_names[i], scene));
    }

    for (auto &handle : handles)
    {
        handle.join();
    }
}

#endif
