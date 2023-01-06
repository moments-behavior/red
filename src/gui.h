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


static void gui_label_one_view(tuple_d* keypoints, u32& active_id, SkeletonContext* skeleton)
{   
    if (ImPlot::IsPlotHovered())
    {
        if (ImGui::IsKeyPressed(ImGuiKey_W, false)){
            ImPlotPoint mouse = ImPlot::GetPlotMousePos();
            keypoints[active_id] = {mouse.x,  mouse.y};
        }

        if(active_id < (skeleton->num_nodes - 1)) { active_id++; }
    }
}


static void gui_plot_keypoints(){
    
}

#endif
