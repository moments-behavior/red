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
    Fish6,
    BoundingBox
};

std::map<std::string, SkeletonPrimitive> skeleton_get_all()
{
    std::map<std::string, SkeletonPrimitive> skeleton_all = {
        {"Fish6", Fish6},
        {"BoundingBox", BoundingBox}
    };
    return skeleton_all;
}


void skeleton_initialize(SkeletonContext* skeleton, SkeletonPrimitive skeleton_type)
{
    switch (skeleton_type){
        case Fish6:
            skeleton->has_skeleton = true;
            skeleton->has_bbox = false;
            skeleton->num_nodes = 6;
            skeleton->num_edges = 6;
            skeleton->node_names = {"Snout", "EyeL", "EarR", "Neck", "Internal", "TailEnd"};

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
                {4, 5}};
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
    if (skeleton->has_bbox) {
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
    keypoints->has_labels = false;
}

void delete_all_labels(Animals *animals, render_scene *scene, SkeletonContext* skeleton, u32 number_animals)
{
    // clean delete
    for (u32 animal_id=0; animal_id < number_animals; animal_id++) {
        KeyPoints* keypoints = &animals->keypoints[animal_id];
        delete_label_per_animal(keypoints, scene, skeleton);
    }
    free(animals->keypoints);
    delete animals->colors;
    free(animals);
}


#endif
