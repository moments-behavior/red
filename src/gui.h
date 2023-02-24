#ifndef RED_GUI
#define RED_GUI
#include "render.h"
#include "skeleton.h"
#include <imfilebrowser.h>
#include <fstream>

struct ProjectContext{
    std::string root_dir;
    std::vector<std::string> input_file_names;
    std::vector<std::string> camera_names;
};


static void gui_plot_keypoints(KeyPoints *keypoints, SkeletonContext *skeleton, int view_idx)
{
    for (u32 node=0; node < skeleton->num_nodes; node++){
        if (keypoints->keypoints2d[view_idx][node].is_labeled){
            ImVec4 node_color; 
            node_color.w = 1.0f; 
            node_color.x = skeleton->node_colors.at(node).x;
            node_color.y = skeleton->node_colors.at(node).y;
            node_color.z = skeleton->node_colors.at(node).z;
            
            int id = skeleton->num_nodes * view_idx + node;
            ImPlot::DragPoint(id, &keypoints->keypoints2d[view_idx][node].position.x, &keypoints->keypoints2d[view_idx][node].position.y, node_color);
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

            keypoints->keypoints3d->x = output.at<float>(0);
            keypoints->keypoints3d->y = output.at<float>(1);
            keypoints->keypoints3d->z = output.at<float>(2);

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


void save_keypoints(std::map<u32, KeyPoints*> keypoints_map, SkeletonContext* skeleton, std::string root_dir)
{
    std::string now = current_date_time();
    std::string filename = root_dir + "/labeled_data/worldKeyPoints/keypoints_" + now;
    std::ofstream output_file(filename);

    output_file << skeleton->name << ",\n";

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
            output_file << i << "," << keypoints->keypoints3d->x << "," << keypoints->keypoints3d->y << "," << keypoints->keypoints3d->z << ",";
        }
        output_file << "\n";
        it++;
    }

    output_file.close();
    std::cout << filename << " created"  << std::endl; 
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
