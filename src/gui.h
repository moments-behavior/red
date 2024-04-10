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

static void draw_cv_contours(std::vector<cv::Rect> boxes, std::vector<std::string> labels, std::vector<int> class_ids)
{
    for (int i=0; i<boxes.size(); i++)
    {
        double x[5] = {(double)boxes[i].x, (double)boxes[i].x, (double)boxes[i].x + boxes[i].width, (double)boxes[i].x + boxes[i].width, (double)boxes[i].x};
        double y[5] = {(double)2200 - boxes[i].y, (double)2200 - boxes[i].y - boxes[i].height, (double)2200 - boxes[i].y - boxes[i].height, (double)2200 - boxes[i].y, (double)2200 - boxes[i].y};
        
        if(class_ids[i] == 0){
            ImPlot::SetNextLineStyle(ImVec4(1.0, 0.0, 1.0,1.0), 3.0);
        } else{
            ImPlot::SetNextLineStyle(ImVec4(0.5, 1.0, 1.0,1.0), 3.0);}

        ImPlot::PlotLine(labels[i].c_str(), &x[0], &y[0], 5); 
    }
}

static void gui_plot_keypoints(KeyPoints *keypoints, SkeletonContext *skeleton, int view_idx)
{
    float pt_size = 6.0f;
    for (u32 node=0; node < skeleton->num_nodes; node++){
        if (keypoints->keypoints2d[view_idx][node].is_labeled){
            ImVec4 node_color; 
            node_color.w = 1.0f; 
            node_color.x = skeleton->node_colors.at(node).x;
            node_color.y = skeleton->node_colors.at(node).y;
            node_color.z = skeleton->node_colors.at(node).z;
            
            int id = skeleton->num_nodes * view_idx + node;
            ImPlot::DragPoint(id, &keypoints->keypoints2d[view_idx][node].position.x, &keypoints->keypoints2d[view_idx][node].position.y, node_color, pt_size);
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


static void reprojection(KeyPoints *keypoints, SkeletonContext *skeleton, std::vector<CameraParams> camera_params, int num_cams)
{
   
    for (u32 node=0; node < skeleton->num_nodes; node++){

        u32 num_views_labeled {0}; 
        for (u32 view_idx = 0; view_idx < num_cams; view_idx++){
            if(keypoints->keypoints2d[view_idx][node].is_labeled) {num_views_labeled++;}
        }

        if (num_views_labeled >= 2){
            
            std::vector<cv::Mat> sfmPoints2d;
            std::vector<cv::Mat> projection_matrices;
            cv::Mat output;

            for (u32 view_idx = 0; view_idx < num_cams; view_idx++)
            {
                if(keypoints->keypoints2d[view_idx][node].is_labeled)
                {
                    cv::Mat point = (cv::Mat_<float>(2, 1) << keypoints->keypoints2d[view_idx][node].position.x, (float)2200 - keypoints->keypoints2d[view_idx][node].position.y);
                    cv::Mat pointUndistort;
                    cv::undistortPoints(point, pointUndistort, camera_params[view_idx].k, camera_params[view_idx].dist_coeffs, cv::noArray(), camera_params[view_idx].k);
                    
                    sfmPoints2d.push_back(pointUndistort.reshape(1, 2));
                    projection_matrices.push_back(camera_params[view_idx].projection_mat);
                }
            }

            cv::sfm::triangulatePoints(sfmPoints2d, projection_matrices, output);
            output.convertTo(output, CV_32F);

            keypoints->keypoints3d[node].x = output.at<float>(0);
            keypoints->keypoints3d[node].y = output.at<float>(1);
            keypoints->keypoints3d[node].z = output.at<float>(2);

            for (u32 view_idx = 0; view_idx < num_cams; view_idx++)
            {
                cv::Mat imagePts;
                cv::projectPoints(output, camera_params[view_idx].rvec, camera_params[view_idx].tvec, camera_params[view_idx].k, camera_params[view_idx].dist_coeffs, imagePts);
                double x = imagePts.at<float>(0, 0);
                double y = float(2200) - imagePts.at<float>(0, 1);
                keypoints->keypoints2d[view_idx][node].position.x = x;
                keypoints->keypoints2d[view_idx][node].position.y = y;
                keypoints->keypoints2d[view_idx][node].is_labeled = true;
            }
        }

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

    for (uint i = 0; i < num_cameras; i++) {
        std::string filename_cam = root_dir + "/" + camera_names[i] + "/" + camera_names[i] + "_" + now;
        std::ofstream output_file_cam(filename_cam);
        output2d_files.push_back(std::move(output_file_cam));
    }

    output_file << skeleton->name << ",\n";
    for (uint i = 0; i < num_cameras; i++) {
        output2d_files[i] << skeleton->name << ",\n";
    }

    std::map<u32, KeyPoints*>::iterator it = keypoints_map.begin();
    while (it != keypoints_map.end())
    {
        uint frame = it->first;
        KeyPoints* keypoints = it->second;
        // write frame number
        output_file << frame << ",";
        // fore each labeled keypoint, write idx, xpos, ypos, zpos
        for (uint i = 0; i < skeleton->num_nodes; i++)
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

    for (uint i = 0; i < num_cameras; i++) {
        output2d_files[i].close();
    }
}


void load_2d_keypoints(std::map<u32, KeyPoints*>& keypoints_map, SkeletonContext* skeleton, std::string root_dir, int cam_idx, std::string camera_name, render_scene *scene) {
    std::string labeled_data_dir = root_dir  + "/" + camera_name;
    std::vector<std::string> filenames;

    for (const auto & entry : std::filesystem::directory_iterator(labeled_data_dir))
    {
        filenames.push_back(entry.path());
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
                uint frame_num = stoul(token);
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
                    keypoints_map[frame_num]->keypoints2d[cam_idx][node].is_labeled = true;
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
        filenames.push_back(entry.path());
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
                uint frame_num = stoul(token);
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

void world_coordinates_projection_points(CameraParams* cvp, double* axis_x_values, double* axis_y_values, float scale)
{
    std::vector<cv::Point3f> world_coordinates;
    world_coordinates.push_back(cv::Point3f(0.0f, 0.0f, 0.0f));
    world_coordinates.push_back(cv::Point3f(scale * 1.0f, 0.0f, 0.0f));
    world_coordinates.push_back(cv::Point3f(0.0f, scale * 1.0f, 0.0f));
    world_coordinates.push_back(cv::Point3f(0.0f, 0.0f, scale * 1.0f));

    std::vector<cv::Point2f> img_pts;
    cv::projectPoints(world_coordinates, cvp->rvec, cvp->tvec, cvp->k, cvp->dist_coeffs, img_pts);
    
    for (int i = 0; i < 4; i++){
        axis_x_values[i] = img_pts.at(i).x;
        axis_y_values[i] = 2200 - img_pts.at(i).y;
    }
}

static void gui_plot_world_coordinates(CameraParams* cvp, int cam_id)
{
    double axis_x_values[4]; double axis_y_values[4]; 
    world_coordinates_projection_points(cvp, axis_x_values, axis_y_values, 50);
    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 6.0, ImVec4(1.0, 1.0, 1.0,1.0));
    ImPlot::SetNextLineStyle(ImVec4(1.0, 1.0, 1.0,1.0), 3.0);
    std::string name = "World Origin";
    
    float one_axis_x[2];
    float one_axis_y[2];

    std::vector<triple_f> node_colors = {
        {1.0f, 1.0f, 1.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f}};
                
    for (u32 edge=0; edge < 3; edge++)
    {
        double xs[2] {axis_x_values[0], axis_x_values[edge+1]};
        double ys[2] {axis_y_values[0], axis_y_values[edge+1]};
        
        ImVec4 my_color; 
        my_color.w = 1.0f; 
        my_color.x = node_colors[edge+1].x;
        my_color.y = node_colors[edge+1].y;
        my_color.z = node_colors[edge+1].z;

        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 6.0, my_color);
        ImPlot::SetNextLineStyle(my_color, 3.0);
        ImPlot::PlotLine(name.c_str(), xs, ys, 2, ImPlotLineFlags_Segments);
    }
    
}


void gui_arena_projection_points(CameraParams* cvp, float* arena_x, float* arena_y, int n)
{
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;

    float radius = 1473.0f;
    std::vector<cv::Point3f> inPts;

    for (int i=0; i<=n; i++)
    {
        float angle = (3.14159265358979323846 * 2) * (float(i) / float(n-1));
        x.push_back(sin(angle) * radius);
        y.push_back(cos(angle) * radius);
        z.push_back(0.0f);
    }

    for (int i=0; i<n; i++)
    {
        cv::Point3f p;
        p.x = x[i];
        p.y = y[i];
        p.z = z[i];
        inPts.push_back(p);
    }

    std::vector<cv::Point2f> img_pts;
    cv::projectPoints(inPts, cvp->rvec, cvp->tvec, cvp->k, cvp->dist_coeffs, img_pts);

    for (int i = 0; i < n; i++){
        arena_x[i] = img_pts.at(i).x;
        arena_y[i] = 2200 - img_pts.at(i).y;
    }
}


static void gui_plot_perimeter(CameraParams* cvp)
{
    float arena_x[100]; float arena_y[100]; 
    gui_arena_projection_points(cvp, arena_x, arena_y, 100);
    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 6.0, ImVec4(1.0, 1.0, 1.0,1.0));
    ImPlot::SetNextLineStyle(ImVec4(1.0, 1.0, 1.0,1.0), 3.0);
    std::string name = "arena";
    ImPlot::PlotLine(name.c_str(), arena_x, arena_y, 100);    
}

#endif
