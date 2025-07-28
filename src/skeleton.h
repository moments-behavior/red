#ifndef RED_SKELETON
#define RED_SKELETON
#include <vector>
#include <string>
#include "types.h"
#include <map>
#include "json.hpp"
using json = nlohmann::json;

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

struct BoundingBox{
    ImPlotRect* rect;
    RectState state;
    int class_id;
    float confidence;  
};

struct KeyPoints{
    triple_d* keypoints3d;
    KeyPoints2D** keypoints2d; 
    u32* active_kp_id;
    std::vector<std::vector<BoundingBox>> bbox2d_list;  
    BoundingBox* bbox2d;  
    bool has_labels;
    ImVec4 animal_color;
    u32 counter;
};

struct Animals{
    KeyPoints* keypoints;
    u32 active_id;
    u32 max_number;
};

struct SkeletonContext {
    int num_nodes;
    int num_edges;
    std::vector<ImVec4> node_colors; 
    std::vector<tuple_i> edges;
    std::vector<std::string> node_names;
    std::string name;
    bool has_bbox; 
    bool has_skeleton;
};

enum SkeletonPrimitive { 
    SP_FISH6,
    SP_FISH12,
    SP_BBOX,
    SP_SIMPLE_BBOX_SKELETON,
    SP_LOAD
};

std::map<std::string, SkeletonPrimitive> skeleton_get_all()
{
    std::map<std::string, SkeletonPrimitive> skeleton_all = {
        {"Fish6", SP_FISH6},
        {"Fish12", SP_FISH12},
        {"BoundingBox", SP_BBOX},
        {"Simple BBox+Skeleton", SP_SIMPLE_BBOX_SKELETON},
        {"Load from JSON", SP_LOAD}
    };
    return skeleton_all;
}

void load_skeleton_json(std::string file_name, SkeletonContext* skeleton)
{
    std::ifstream f(file_name);
    json s_config = json::parse(f);
    skeleton->name = s_config["name"];
    skeleton->has_skeleton = s_config["has_skeleton"];
    skeleton->has_bbox = s_config["has_bbox"];
    skeleton->num_nodes = s_config["num_nodes"];
    skeleton->num_edges = s_config["num_edges"];

    for(int i=0; i<s_config["node_names"].size(); i++) {
        skeleton->node_names.push_back(s_config["node_names"][i]);
    }

    for(int i=0; i<s_config["edges"].size(); i++) {
        tuple_i edge_start_end = {s_config["edges"][i][0], s_config["edges"][i][1]};
        skeleton->edges.push_back(edge_start_end);
    }
}


void skeleton_initialize(std::string name, std::string root_dir, SkeletonContext* skeleton, SkeletonPrimitive skeleton_type)
{
    switch (skeleton_type){
        case SP_FISH6:
            skeleton->name = name;
            skeleton->has_skeleton = true;
            skeleton->has_bbox = false;
            skeleton->num_nodes = 6;
            skeleton->num_edges = 5;
            skeleton->node_names = {"EyeL", "EyeR", "Mid1", "SB_Ant", "Mid2", "SB_Post"};

            for (int i = 0; i < skeleton->num_nodes; i++) {
                ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 1.0f, 1.0f);
                skeleton->node_colors.push_back(color);
            }
            
            skeleton->edges ={
                {0, 2},
                {1, 2},
                {2, 3},
                {3, 4},
                {4, 5}};
                break;

        case SP_FISH12:
            skeleton->has_skeleton = true;
            skeleton->has_bbox = false;
            skeleton->num_nodes = 12;
            skeleton->num_edges = 11;
            skeleton->node_names = {"EyeL", "EyeR", "Mid1", "SB_Ant", "Mid2", "SB_Post", "Tail1", "Tail2", "Tail3", "Tail4", "Tail5", "Tail6"};

            for (int i = 0; i < skeleton->num_nodes; i++) {
                ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 1.0f, 1.0f);
                skeleton->node_colors.push_back(color);
            }

            skeleton->edges = {
                {0, 2},
                {1, 2},
                {2, 3},
                {3, 4},
                {4, 5},
                {5, 6},
                {6, 7},
                {7, 8},
                {8, 9},
                {9, 10},
                {10, 11}};
                break;

        case SP_BBOX:
            skeleton->name = name;
            skeleton->has_bbox = true;
            skeleton->has_skeleton = false;
            break;
        
        case SP_SIMPLE_BBOX_SKELETON:
            skeleton->name = name;
            skeleton->has_skeleton = true;
            skeleton->has_bbox = true;
            skeleton->num_nodes = 2;
            skeleton->num_edges = 1;
            skeleton->node_names = {"Point1", "Point2"};

            for (int i = 0; i < skeleton->num_nodes; i++) {
                ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 1.0f, 1.0f);
                skeleton->node_colors.push_back(color);
            }

            skeleton->edges = {
                {0, 1}};
                break;

        case SP_LOAD:
            // This case is now handled by file dialog in main application
            skeleton->name = name;
            break;
    }
};

void allocate_keypoints(Animals *animals, render_scene *scene, SkeletonContext* skeleton, u32 number_animals) 
{
    animals->keypoints = (KeyPoints *)malloc(sizeof(KeyPoints) * number_animals);
    animals->max_number = number_animals;
    animals->active_id = 0;
    
    for (u32 animal_idx=0; animal_idx < number_animals; animal_idx++) {
        KeyPoints* keypoints = &animals->keypoints[animal_idx];
        keypoints->animal_color = (ImVec4)ImColor::HSV(animal_idx / (float)number_animals, 0.8f, 0.8f);
        keypoints->has_labels = false;
        
        // Initialize the bbox2d_list using placement new to properly construct the vector
        new (&keypoints->bbox2d_list) std::vector<std::vector<BoundingBox>>();
        
        if (skeleton->has_bbox) {
            // Initialize the multiple bounding box list
            keypoints->bbox2d_list.resize(scene->num_cams);
            
            // Keep backward compatibility
            keypoints->bbox2d = (BoundingBox *)malloc(sizeof(BoundingBox) * scene->num_cams);
            for (u32 j=0; j < scene->num_cams; j++) {
                keypoints->bbox2d[j].rect = NULL;
                keypoints->bbox2d[j].state = RectNull;
                keypoints->bbox2d[j].class_id = -1;
                keypoints->bbox2d[j].confidence = 0.0f;
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
            keypoints->counter = 0;
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


void reinitalize_keypoint_active_animal(Animals *animals, render_scene *scene, SkeletonContext* skeleton) {
    KeyPoints* keypoints = &animals->keypoints[animals->active_id];
    if (keypoints->has_labels) {
        if (skeleton->has_bbox) {
            for (u32 j=0; j < scene->num_cams; j++) {
                if (keypoints->bbox2d[j].rect != NULL) {
                    delete keypoints->bbox2d[j].rect;
                }
                keypoints->bbox2d[j].state = RectNull;
            }
        }

        if (skeleton->has_skeleton) {
            keypoints->counter = 0;
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

        if (animals->active_id > 1) {
            animals->active_id = animals->active_id - 1;
        } else {
            animals->active_id = 0;
        }
        keypoints->has_labels = false;
    }
}

void delete_label_per_animal(KeyPoints* keypoints, render_scene *scene, SkeletonContext* skeleton) {
    keypoints->has_labels = false;
    
    // Properly destroy the bbox2d_list vector
    keypoints->bbox2d_list.~vector<std::vector<BoundingBox>>();
    
    if (skeleton->has_bbox) {
        if (keypoints->bbox2d->rect != NULL) {
            delete(keypoints->bbox2d->rect);
        }
        free(keypoints->bbox2d);
    } 

    if (skeleton->has_skeleton) {
        for (u32 j=0; j < scene->num_cams; j++) {
            free(keypoints->keypoints2d[j]);
        }
        free(keypoints->keypoints3d);
        free(keypoints->keypoints2d);
        free(keypoints->active_kp_id);
    }
}

void delete_all_labels(Animals *animals, render_scene *scene, SkeletonContext* skeleton, u32 number_animals)
{
    // clean delete
    for (u32 animal_id=0; animal_id < number_animals; animal_id++) {
        KeyPoints* keypoints = &animals->keypoints[animal_id];
        delete_label_per_animal(keypoints, scene, skeleton);
    }
    free(animals->keypoints);
    free(animals);
}

#endif
