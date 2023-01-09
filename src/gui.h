#ifndef RED_GUI
#define RED_GUI
#include "render.h"
#include "skeleton.h"
#include <imfilebrowser.h>

struct ProjectContext{
    std::string root_dir;
    std::vector<std::string> input_file_names;
    std::vector<std::string> camera_names;
};


static void gui_label_one_view(KeyPoints *keypoints, SkeletonContext *skeleton, int view_idx)
{   
    if (ImPlot::IsPlotHovered())
    {
        if (ImGui::IsKeyPressed(ImGuiKey_W, false)){
            // labeling sequentially each view
            ImPlotPoint mouse = ImPlot::GetPlotMousePos();
            keypoints->keypoints2d[view_idx][keypoints->active_id[view_idx]].position = {mouse.x,  mouse.y};
            keypoints->keypoints2d[view_idx][keypoints->active_id[view_idx]].is_labeled = true;
            if(keypoints->active_id[view_idx] < (skeleton->num_nodes - 1)) { keypoints->active_id[view_idx]++; }
            std::cout << "active_id: " << keypoints->active_id[view_idx] << "view_id:" << view_idx << std::endl;
        }
    }
}


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

#endif
