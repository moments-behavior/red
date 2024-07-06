#ifndef RED_SKELETON
#define RED_SKELETON
#include <vector>
#include <string>
#include "types.h"
#include <map>


struct KeyPoints2D{
    tuple_d position; 
    bool is_labeled;
    bool is_triangulated;
};

enum RectState {
    RectNull,
    RectOnePoint,
    RectTwoPoints
};

struct BoudingBox{
    ImPlotRect* rect;
    RectState state;
};

struct KeyPoints{
    triple_d* keypoints3d;
    KeyPoints2D** keypoints2d; 
    u32* active_kp_id;
    BoudingBox* bbox2d;
    bool has_labels;
};

struct Animals{
    KeyPoints* keypoints;
    u32 active_id;
    u32 max_number;
    ImColor* colors;
};

struct SkeletonContext {
    int num_nodes;
    int num_edges;
    std::vector<triple_f> node_colors; 
    std::vector<tuple_i> edges;
    std::vector<std::string> node_names;
    std::string name;
    bool has_bbox; 
    bool has_skeleton;
};

enum SkeletonPrimitive { 
    Rat7Target,
    Rat10Target2,
    RatTarget,
    Rat3Target,
    Rat4Target,
    Rat6Target,
    Rat6,
    Rat4Box,
    Rat4Box3Ball,
    Table3Corners,
    Rat22,
    Rat20,
    Rat20Target,
    BoundingBox
};

std::map<std::string, SkeletonPrimitive> skeleton_get_all()
{
    std::map<std::string, SkeletonPrimitive> skeleton_all = {
        {"Rat7Target", Rat7Target},
        {"Rat10Target2", Rat10Target2},
        {"RatTarget", RatTarget},
        {"Rat3Target", Rat3Target},
        {"Rat4Target", Rat4Target},
        {"Rat6Target", Rat6Target},
        {"Rat6", Rat6},
        {"Rat4Box", Rat4Box},
        {"Rat4Box3Ball", Rat4Box3Ball},
        {"Table3Corners", Table3Corners},
        {"Rat22", Rat22},
        {"Rat20", Rat20},
        {"Rat20Target", Rat20Target},
        {"BoundingBox", BoundingBox}
    };
    return skeleton_all;
}


void skeleton_initialize(SkeletonContext* skeleton, SkeletonPrimitive skeleton_type)
{
    switch (skeleton_type){
        case Table3Corners:
            skeleton->num_nodes = 3;
            skeleton->num_edges = 2;
             
            skeleton->node_names = {"BottomLeft", "BottomRight", "TopLeft"};

            for (int i = 0; i < skeleton->num_nodes; i++) {
                ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 0.8f, 0.8f);
                skeleton->node_colors.push_back({color.x, color.y, color.z});
            }

            skeleton->edges ={
                {0, 1},
                {0, 2},
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

        case Rat3Target:
            skeleton->num_nodes = 4;
            skeleton->num_edges = 2;
            skeleton->node_names = {"Snout", "EarL", "EarR", "Target"};

            for (int i = 0; i < skeleton->num_nodes; i++) {
                ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 1.0f, 1.0f);
                skeleton->node_colors.push_back({color.x, color.y, color.z});
            }
            skeleton->edges ={
                {0, 1},
                {0, 2}
            };
            break;

        case Rat4Target:
            skeleton->num_nodes = 5;
            skeleton->num_edges = 3;
            skeleton->node_names = {"EarR", "EarL", "Snout", "Tail", "Target"};

            for (int i = 0; i < skeleton->num_nodes; i++) {
                ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 1.0f, 1.0f);
                skeleton->node_colors.push_back({color.x, color.y, color.z});
            }
            skeleton->edges ={
                {0, 2},
                {1, 2},
                {2, 3}
            };
            break;

        case Rat6Target:
            skeleton->num_nodes = 7;
            skeleton->num_edges = 6;
            skeleton->node_names = {"Snout", "EarL", "EarR", "Neck", "SpineL", "TailBase", "Target"};

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
                {4, 5}
                };
            break;

        case Rat6:
            skeleton->num_nodes = 6;
            skeleton->num_edges = 6;
            skeleton->node_names = {"Snout", "EarL", "EarR", "Neck", "SpineL", "TailBase"};

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
                {4, 5}
                };
            break;

        case Rat4Box:
            skeleton->num_nodes = 6;
            skeleton->num_edges = 3;
            skeleton->node_names = {"EarR", "EarL", "Snout", "Tail", "TopLeft", "BottomRight"};

            for (int i = 0; i < skeleton->num_nodes; i++) {
                ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 1.0f, 1.0f);
                skeleton->node_colors.push_back({color.x, color.y, color.z});
            }

            skeleton->edges ={
                {0, 2},
                {1, 2},
                {2, 3}
                };
            break;

        case Rat4Box3Ball:
            skeleton->num_nodes = 9;
            skeleton->num_edges = 3;
            skeleton->node_names = {"EarR", "EarL", "Snout", "Tail", "TopLeft", "BottomRight", "Ball0", "Ball1", "Ball2"};

            for (int i = 0; i < skeleton->num_nodes; i++) {
                ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 1.0f, 1.0f);
                skeleton->node_colors.push_back({color.x, color.y, color.z});
            }

            skeleton->edges ={
                {0, 2},
                {1, 2},
                {2, 3}
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
            break;

        case Rat20:
            skeleton->num_nodes = 20;
            skeleton->num_edges = 20;
            skeleton->node_names = {"Snout", "EarL", "EarR", "Neck", "SpineL", "TailBase",
                                    "ShoulderL", "ElbowL", "WristL", "HandL",
                                    "ShoulderR", "ElbowR", "WristR", "HandR",
                                    "KneeL", "AnkleL", "FootL",
                                    "KneeR", "AnkleR", "FootR"};

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
                {3, 6},
                {6, 7},
                {7, 8},
                {8, 9},
                {3, 10},
                {10, 11},
                {11, 12},
                {12, 13},
                {4, 14},
                {14, 15},
                {15, 16},
                {4, 17},
                {17, 18},
                {18, 19}};
            break;

        case Rat20Target:
            skeleton->num_nodes = 21;
            skeleton->num_edges = 20;
            skeleton->node_names = {"Snout", "EarL", "EarR", "Neck", "SpineL", "TailBase",
                                    "ShoulderL", "ElbowL", "WristL", "HandL",
                                    "ShoulderR", "ElbowR", "WristR", "HandR",
                                    "KneeL", "AnkleL", "FootL",
                                    "KneeR", "AnkleR", "FootR", "Target"};

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
                {3, 6},
                {6, 7},
                {7, 8},
                {8, 9},
                {3, 10},
                {10, 11},
                {11, 12},
                {12, 13},
                {4, 14},
                {14, 15},
                {15, 16},
                {4, 17},
                {17, 18},
                {18, 19}};
            break;

        case Rat22:
            skeleton->num_nodes = 22;
            skeleton->num_edges = 22;
            skeleton->node_names = {"Snout", "EarL", "EarR", "Neck", "SpineL", "TailBase",
                                    "ShoulderL", "ElbowL", "WristL", "HandL",
                                    "ShoulderR", "ElbowR", "WristR", "HandR",
                                    "HipL", "KneeL", "AnkleL", "FootL",
                                    "HipR", "KneeR", "AnkleR", "FootR"};

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
                {3, 6},
                {6, 7},
                {7, 8},
                {8, 9},
                {3, 10},
                {10, 11},
                {11, 12},
                {12, 13},
                {4, 14},
                {14, 15},
                {15, 16},
                {16, 17},
                {4, 18},
                {18, 19},
                {19, 20},
                {20, 21}};
                break;
        case BoundingBox:
            skeleton->has_bbox = true;
            skeleton->has_skeleton = false;
            break;
    }
};

void allocate_keypoints(Animals *animals, render_scene *scene, SkeletonContext* skeleton, u32 number_animals) 
{
    animals->keypoints = (KeyPoints *)malloc(sizeof(KeyPoints) * number_animals);
    animals->colors = new ImColor[number_animals];
    animals->max_number = number_animals;
    animals->active_id = 0;
    
    for (u32 animal_idx=0; animal_idx < number_animals; animal_idx++) {
        animals->colors[animal_idx] = ImColor::HSV(animal_idx / (float)number_animals, 0.8f, 0.8f);
        KeyPoints* keypoints = &animals->keypoints[animal_idx]; 
        keypoints->has_labels = false;
        if (skeleton->has_bbox) {
            keypoints->bbox2d = (BoudingBox *)malloc(sizeof(BoudingBox) * scene->num_cams);
            for (u32 j=0; j < scene->num_cams; j++) {
                keypoints->bbox2d[j].rect = NULL;
                keypoints->bbox2d[j].state = RectNull;
            }
        }

        if (skeleton->has_skeleton) {
            keypoints->active_kp_id = (u32 *)malloc(sizeof(u32) * scene->num_cams);
            keypoints->keypoints3d = (triple_d *)malloc(sizeof(triple_d) * skeleton->num_nodes); 
            keypoints->keypoints2d = (KeyPoints2D **)malloc(sizeof(KeyPoints2D*) * scene->num_cams);
            for (u32 j=0; j < scene->num_cams; j++) {
                keypoints->keypoints2d[j] = (KeyPoints2D *)malloc(sizeof(KeyPoints2D) * skeleton->num_nodes);
            }
            
            // initialize to big number 
            for (u32 j=0; j < scene->num_cams; j++) {
                keypoints->active_kp_id[j] = 0;
                for (u32 k=0; k < skeleton->num_nodes; k++){
                    keypoints->keypoints2d[j][k].is_labeled = false;
                    keypoints->keypoints2d[j][k].is_triangulated = false;
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
    }    
}

void deallocate_keypoints(KeyPoints *keypoints, render_scene *scene, SkeletonContext* skeleton)
{
    // clean delete
}


#endif
