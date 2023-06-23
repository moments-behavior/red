#ifndef RED_SKELETON
#define RED_SKELETON
#include <vector>
#include <string>
#include "types.h"
#include <map>


struct KeyPoints2D{
    tuple_d position; 
    bool is_labeled;
};

struct KeyPoints{
    triple_d* keypoints3d;
    KeyPoints2D** keypoints2d; 
    u32* active_id; 
};

struct SkeletonContext {
    int num_nodes;
    int num_edges;
    std::vector<triple_f> node_colors; 
    std::vector<tuple_i> edges;
    std::vector<std::string> node_names;
    std::string name;
};

enum SkeletonPrimitive { 
    Rat7Target=0,
    Rat10Target2=1,
    RatTarget=2,
    CalibrationFourCorners=3
};

std::map<std::string, SkeletonPrimitive> skeleton_get_all()
{
    std::map<std::string, SkeletonPrimitive> skeleton_all = {
        {"Rat7Target", Rat7Target},
        {"Rat10Target2", Rat10Target2},
        {"RatTarget", RatTarget},
        {"CalibrationFourCorners", CalibrationFourCorners}
    };
    return skeleton_all;
}


void skeleton_initialize(SkeletonContext* skeleton, SkeletonPrimitive skeleton_type)
{
    switch (skeleton_type){
        case CalibrationFourCorners:
            skeleton->num_nodes = 3;
            skeleton->num_edges = 2;
             
            skeleton->node_names = {"BottomLeft", "BottomRight", "TopLeft"};

            for (int i = 0; i < skeleton->num_nodes; i++) {
                ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 0.8f, 0.8f);
                skeleton->node_colors.push_back({color.x, color.y, color.z});
            }

            skeleton->edges ={
                {0, 1},
                {1, 2},
                };
            break;


        case Rat7Target:
            skeleton->num_nodes = 9;
            skeleton->num_edges = 8;
            skeleton->node_names = {"Snout", "EarL", "EarR", "Neck", "SpineF", "SpineM", "SpineL", "TailBase", "Target"};

            for (int i = 0; i < skeleton->num_nodes; i++) {
                ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 1.0f, 1.0f);
                skeleton->node_colors.push_back({color.x, color.y, color.z});
            }

            skeleton->edges ={
                {0, 1},
                {0, 2},
                {1, 3},
                {2, 3},
                {3, 4},
                {4, 5},
                {5, 6},
                {6, 7} 
                };
            break;

        case RatTarget:
            skeleton->num_nodes = 2;
            skeleton->num_edges = 1;
            skeleton->node_names = {"Snout", "Target"};

            for (int i = 0; i < skeleton->num_nodes; i++) {
                ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 1.0f, 1.0f);
                skeleton->node_colors.push_back({color.x, color.y, color.z});
            }
            skeleton->edges ={
                {0, 1}
            };

            break;


        case Rat10Target2:
                skeleton->num_nodes = 12;
                skeleton->num_edges = 10;
                skeleton->node_names = {"Snout", "EarL", "EarR", "Neck", "SpineL", "TailBase", "HandL", "HandR", "FootL", "FootR", "Target1", "Target2"};

                for (int i = 0; i < skeleton->num_nodes; i++) {
                    ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 1.0f, 1.0f);
                    skeleton->node_colors.push_back({color.x, color.y, color.z});
                }
                
                skeleton->edges ={
                    {0, 1},
                    {0, 2},
                    {0, 3},
                    {1, 2},
                    {3, 4},
                    {4, 5},
                    {3, 6},
                    {3, 7},
                    {4, 8},
                    {4, 9}};
        
    }
};


void allocate_keypoints(KeyPoints *keypoints, render_scene *scene, SkeletonContext* skeleton)
{
    // allocate memory for storing keypoints
    keypoints->active_id = (u32 *)malloc(sizeof(u32) * scene->num_cams);
    keypoints->keypoints3d = (triple_d *)malloc(sizeof(triple_d) * skeleton->num_nodes); 
    keypoints->keypoints2d = (KeyPoints2D **)malloc(sizeof(KeyPoints2D*) * scene->num_cams);
    for (u32 j=0; j < scene->num_cams; j++) {
        keypoints->keypoints2d[j] = (KeyPoints2D *)malloc(sizeof(KeyPoints2D) * skeleton->num_nodes);
    }
    
    // initialize to big number 
    for (u32 j=0; j < scene->num_cams; j++) {
        keypoints->active_id[j] = 0;
        for (u32 k=0; k < skeleton->num_nodes; k++){
            keypoints->keypoints2d[j][k].is_labeled = false;
            keypoints->keypoints2d[j][k].position.x = 1E7;
            keypoints->keypoints2d[j][k].position.y = 1E7;
        }
    }

    for (u32 k=0; k < skeleton->num_nodes; k++) {
        keypoints->keypoints3d[k].x = 1E7;
        keypoints->keypoints3d[k].y = 1E7;
        keypoints->keypoints3d[k].z = 1E7;
    }
}

#endif
