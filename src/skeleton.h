#ifndef RED_SKELETON
#define RED_SKELETON
#include "imgui.h"
#include "implot.h"
#include "json.hpp"
#include "types.h"
#include "render.h"
#include <fstream>
#include <map>
#include <string>
#include <vector>

using json = nlohmann::json;

struct KeyPoints2D {
    tuple_d position;
    bool is_labeled;
    bool is_triangulated;
};

enum RectState {
    RectNull,
    RectOnePoint,
    RectTwoPoints
};

enum OBBState {
    OBBNull,
    OBBFirstAxisPoint,
    OBBSecondAxisPoint, 
    OBBThirdPoint,
    OBBComplete
};

struct BoundingBox {
    ImPlotRect *rect;
    RectState state;
    int class_id;
    int id;  
    float confidence;

    KeyPoints2D **bbox_keypoints2d; // Per-camera keypoints for this bbox
    bool has_bbox_keypoints;
    u32 *active_kp_id; // Active keypoint ID per camera for this bbox
};

struct OrientedBoundingBox {
    ImVec2 axis_point1;      // First axis point
    ImVec2 axis_point2;      // Second axis point 
    ImVec2 corner_point;     // Third point to define width/height
    ImVec2 center;           // Center of the OBB
    float width;             // Width of the OBB
    float height;            // Height of the OBB  
    float rotation;          // Rotation angle in radians
    OBBState state;          // Current drawing state
    int class_id;
    int id;  
    float confidence;
};

struct KeyPoints {
    triple_d *keypoints3d;
    KeyPoints2D **keypoints2d;
    u32 *active_id;
    std::vector<std::vector<BoundingBox>> bbox2d_list;
    std::vector<std::vector<OrientedBoundingBox>> obb2d_list;
};

struct SkeletonContext {
    int num_nodes;
    int num_edges;
    std::vector<ImVec4> node_colors;
    std::vector<tuple_i> edges;
    std::vector<std::string> node_names;
    std::string name;
    bool has_bbox;
    bool has_obb; 
    bool has_skeleton;
};

enum SkeletonPrimitive {
    Target,
    Rat7Target,
    Rat10Target2,
    RatTarget,
    Rat3Target,
    Rat4Target,
    Rat6Target,
    Rat6Target2,
    Rat6,
    Rat4,
    Rat4Box,
    Rat4Box3Ball,
    Table3Corners,
    Rat22,
    Rat20,
    Rat24,
    Rat20Target,
    Rat24Target,
    SP_BBOX,
    SP_OBB,
    SP_SIMPLE_BBOX_SKELETON,
    SP_LOAD
};



std::map<std::string, SkeletonPrimitive> skeleton_get_all() {
    std::map<std::string, SkeletonPrimitive> skeleton_all = {
        {"Target", Target},
        {"Rat7Target", Rat7Target},
        {"Rat10Target2", Rat10Target2},
        {"RatTarget", RatTarget},
        {"Rat3Target", Rat3Target},
        {"Rat4Target", Rat4Target},
        {"Rat6Target", Rat6Target},
        {"Rat6Target2", Rat6Target2},
        {"Rat6", Rat6},
        {"Rat4", Rat4},
        {"Rat4Box", Rat4Box},
        {"Rat4Box3Ball", Rat4Box3Ball},
        {"Table3Corners", Table3Corners},
        {"Rat22", Rat22},
        {"Rat20", Rat20},
        {"Rat24", Rat24},
        {"Rat20Target", Rat20Target},
        {"Rat24Target", Rat24Target},
        {"BoundingBox", SP_BBOX},
        {"OrientedBoundingBox", SP_OBB},
        {"Simple BBox+Skeleton", SP_SIMPLE_BBOX_SKELETON},
        {"Load from json", SP_LOAD}};
    return skeleton_all;
}

void load_skeleton_json(std::string file_name, SkeletonContext *skeleton) {
    std::ifstream f(file_name);
    json s_config = json::parse(f);
    skeleton->name = file_name;
    skeleton->has_skeleton = s_config.contains("has_skeleton") ? s_config["has_skeleton"].get<bool>() : true;
    skeleton->has_bbox = s_config.contains("has_bbox") ? s_config["has_bbox"].get<bool>() : false;
    skeleton->has_obb = s_config.contains("has_obb") ? s_config["has_obb"].get<bool>() : false;
    skeleton->num_nodes = s_config["num_nodes"];
    skeleton->num_edges = s_config["num_edges"];

    for (int i = 0; i < s_config["node_names"].size(); i++) {
        skeleton->node_names.push_back(s_config["node_names"][i]);
    }

    for (int i = 0; i < s_config["edges"].size(); i++) {
        tuple_i edge_start_end = {s_config["edges"][i][0],
                                  s_config["edges"][i][1]};
        skeleton->edges.push_back(edge_start_end);
    }
}

void skeleton_initialize(std::string name, std::string skeleton_file_name,
                         SkeletonContext *skeleton,
                         SkeletonPrimitive skeleton_type) {
    skeleton->has_skeleton = true;
    skeleton->has_bbox = false;
    skeleton->has_obb = false;
    
    switch (skeleton_type) {
    case Table3Corners:
        skeleton->name = name;
        skeleton->has_skeleton = true;
        skeleton->has_bbox = false;
        skeleton->has_obb = false;
        skeleton->num_nodes = 3;
        skeleton->num_edges = 2;

        skeleton->node_names = {"BottomLeft", "BottomRight", "TopLeft"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                0.8f, 0.8f);
            skeleton->node_colors.push_back(color);
        }

        skeleton->edges = {
            {0, 1},
            {0, 2},
        };
        break;

    case Target:
        skeleton->name = name;
        skeleton->has_skeleton = true;
        skeleton->has_bbox = false;
        skeleton->num_nodes = 1;
        skeleton->num_edges = 0;
        skeleton->node_names = {"Target"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }

        break;

    case Rat7Target:
        skeleton->name = name;
        skeleton->has_skeleton = true;
        skeleton->has_bbox = false;
        skeleton->num_nodes = 9;
        skeleton->num_edges = 8;
        skeleton->node_names = {"Snout",  "EarL",     "EarR",
                                "Neck",   "SpineF",   "SpineM",
                                "SpineL", "TailBase", "Target"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }

        skeleton->edges = {{0, 1}, {0, 2}, {1, 3}, {2, 3},
                           {3, 4}, {4, 5}, {5, 6}, {6, 7}};
        break;

    case RatTarget:
        skeleton->name = name;
        skeleton->has_skeleton = true;
        skeleton->has_bbox = false;
        skeleton->num_nodes = 2;
        skeleton->num_edges = 1;
        skeleton->node_names = {"Snout", "Target"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }
        skeleton->edges = {{0, 1}};

        break;

    case Rat3Target:
        skeleton->name = name;
        skeleton->has_skeleton = true;
        skeleton->has_bbox = false;
        skeleton->num_nodes = 4;
        skeleton->num_edges = 2;
        skeleton->node_names = {"Snout", "EarL", "EarR", "Target"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }
        skeleton->edges = {{0, 1}, {0, 2}};
        break;

    case Rat4Target:
        skeleton->name = name;
        skeleton->num_nodes = 5;
        skeleton->num_edges = 4;
        skeleton->node_names = {"Snout", "EarL", "EarR", "Tail", "Target"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }
        skeleton->edges = {{0, 1}, {0, 2}, {1, 3}, {2, 3}};
        break;

    case Rat4:
        skeleton->name = name;
        skeleton->num_nodes = 4;
        skeleton->num_edges = 4;
        skeleton->node_names = {"Snout", "EarL", "EarR", "Tail"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }
        skeleton->edges = {{0, 1}, {0, 2}, {1, 3}, {2, 3}};
        break;

    case Rat6Target:
        skeleton->name = name;
        skeleton->num_nodes = 7;
        skeleton->num_edges = 6;
        skeleton->node_names = {"Snout",  "EarL",     "EarR",  "Neck",
                                "SpineL", "TailBase", "Target"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }

        skeleton->edges = {{0, 1}, {0, 2}, {1, 3}, {2, 3}, {3, 4}, {4, 5}};
        break;

    case Rat6Target2:
        skeleton->name = name;
        skeleton->num_nodes = 8;
        skeleton->num_edges = 6;
        skeleton->node_names = {"Snout",  "EarL",     "EarR",    "Neck",
                                "SpineL", "TailBase", "Target1", "Target2"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }

        skeleton->edges = {{0, 1}, {0, 2}, {1, 3}, {2, 3}, {3, 4}, {4, 5}};
        break;

    case Rat6:
        skeleton->name = name;
        skeleton->num_nodes = 6;
        skeleton->num_edges = 6;
        skeleton->node_names = {"Snout", "EarL",   "EarR",
                                "Neck",  "SpineL", "TailBase"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }

        skeleton->edges = {{0, 1}, {0, 2}, {1, 3}, {2, 3}, {3, 4}, {4, 5}};
        break;

    case Rat4Box:
        skeleton->name = name;
        skeleton->num_nodes = 6;
        skeleton->num_edges = 3;
        skeleton->node_names = {"EarR", "EarL",    "Snout",
                                "Tail", "TopLeft", "BottomRight"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }

        skeleton->edges = {{0, 2}, {1, 2}, {2, 3}};
        break;

    case Rat4Box3Ball:
        skeleton->name = name;
        skeleton->num_nodes = 9;
        skeleton->num_edges = 3;
        skeleton->node_names = {"EarR",  "EarL",    "Snout",
                                "Tail",  "TopLeft", "BottomRight",
                                "Ball0", "Ball1",   "Ball2"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }

        skeleton->edges = {{0, 2}, {1, 2}, {2, 3}};
        break;

    case Rat10Target2:
        skeleton->name = name;
        skeleton->num_nodes = 12;
        skeleton->num_edges = 10;
        skeleton->node_names = {"Snout",  "EarL",     "EarR",    "Neck",
                                "SpineL", "TailBase", "HandL",   "HandR",
                                "FootL",  "FootR",    "Target1", "Target2"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }

        skeleton->edges = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {3, 4},
                           {4, 5}, {3, 6}, {3, 7}, {4, 8}, {4, 9}};
        break;

    case Rat20:
        skeleton->name = name;
        skeleton->num_nodes = 20;
        skeleton->num_edges = 20;
        skeleton->node_names = {"Snout",  "EarL",     "EarR",      "Neck",
                                "SpineL", "TailBase", "ShoulderL", "ElbowL",
                                "WristL", "HandL",    "ShoulderR", "ElbowR",
                                "WristR", "HandR",    "KneeL",     "AnkleL",
                                "FootL",  "KneeR",    "AnkleR",    "FootR"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }

        skeleton->edges = {{0, 1},   {0, 2},   {1, 3},   {2, 3},   {3, 4},
                           {4, 5},   {3, 6},   {6, 7},   {7, 8},   {8, 9},
                           {3, 10},  {10, 11}, {11, 12}, {12, 13}, {4, 14},
                           {14, 15}, {15, 16}, {4, 17},  {17, 18}, {18, 19}};
        break;

    case Rat24:
        skeleton->name = name;
        skeleton->num_nodes = 24;
        skeleton->num_edges = 24;
        skeleton->node_names = {"Snout",   "EarL",     "EarR",      "Neck",
                                "SpineL",  "TailBase", "ShoulderL", "ElbowL",
                                "WristL",  "HandL",    "ShoulderR", "ElbowR",
                                "WristR",  "HandR",    "KneeL",     "AnkleL",
                                "FootL",   "KneeR",    "AnkleR",    "FootR",
                                "TailTip", "TailMid",  "Tail1Q",    "Tail3Q"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }

        skeleton->edges = {{0, 1},   {0, 2},   {1, 3},   {2, 3},   {3, 4},
                           {4, 5},   {3, 6},   {6, 7},   {7, 8},   {8, 9},
                           {3, 10},  {10, 11}, {11, 12}, {12, 13}, {4, 14},
                           {14, 15}, {15, 16}, {4, 17},  {17, 18}, {18, 19},
                           {5, 22},  {22, 21}, {21, 23}, {23, 20}};
        break;

    case Rat20Target:
        skeleton->name = name;
        skeleton->num_nodes = 21;
        skeleton->num_edges = 20;
        skeleton->node_names = {
            "Snout",     "EarL",   "EarR",   "Neck",   "SpineL",    "TailBase",
            "ShoulderL", "ElbowL", "WristL", "HandL",  "ShoulderR", "ElbowR",
            "WristR",    "HandR",  "KneeL",  "AnkleL", "FootL",     "KneeR",
            "AnkleR",    "FootR",  "Target"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }

        skeleton->edges = {{0, 1},   {0, 2},   {1, 3},   {2, 3},   {3, 4},
                           {4, 5},   {3, 6},   {6, 7},   {7, 8},   {8, 9},
                           {3, 10},  {10, 11}, {11, 12}, {12, 13}, {4, 14},
                           {14, 15}, {15, 16}, {4, 17},  {17, 18}, {18, 19}};
        break;

    case Rat24Target:
        skeleton->name = name;
        skeleton->num_nodes = 25;
        skeleton->num_edges = 24;
        skeleton->node_names = {
            "Snout",     "EarL",      "EarR",   "Neck",   "SpineL",
            "TailBase",  "ShoulderL", "ElbowL", "WristL", "HandL",
            "ShoulderR", "ElbowR",    "WristR", "HandR",  "KneeL",
            "AnkleL",    "FootL",     "KneeR",  "AnkleR", "FootR",
            "TailTip",   "TailMid",   "Tail1Q", "Tail3Q", "Target"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }

        skeleton->edges = {{0, 1},   {0, 2},   {1, 3},   {2, 3},   {3, 4},
                           {4, 5},   {3, 6},   {6, 7},   {7, 8},   {8, 9},
                           {3, 10},  {10, 11}, {11, 12}, {12, 13}, {4, 14},
                           {14, 15}, {15, 16}, {4, 17},  {17, 18}, {18, 19},
                           {5, 22},  {22, 21}, {21, 23}, {23, 20}};
        break;

    case Rat22:
        skeleton->name = name;
        skeleton->num_nodes = 22;
        skeleton->num_edges = 22;
        skeleton->node_names = {
            "Snout",     "EarL",   "EarR",   "Neck",  "SpineL",    "TailBase",
            "ShoulderL", "ElbowL", "WristL", "HandL", "ShoulderR", "ElbowR",
            "WristR",    "HandR",  "HipL",   "KneeL", "AnkleL",    "FootL",
            "HipR",      "KneeR",  "AnkleR", "FootR"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }

        skeleton->edges = {{0, 1},   {0, 2},   {1, 3},   {2, 3},   {3, 4},
                           {4, 5},   {3, 6},   {6, 7},   {7, 8},   {8, 9},
                           {3, 10},  {10, 11}, {11, 12}, {12, 13}, {4, 14},
                           {14, 15}, {15, 16}, {16, 17}, {4, 18},  {18, 19},
                           {19, 20}, {20, 21}};
        break;

    case SP_BBOX:
        skeleton->name = name;
        skeleton->has_bbox = true;
        skeleton->has_obb = false;
        skeleton->has_skeleton = false;
        break;
    
    case SP_OBB:
        skeleton->name = name;
        skeleton->has_bbox = false;
        skeleton->has_obb = true;
        skeleton->has_skeleton = false;
        break;
    
    case SP_SIMPLE_BBOX_SKELETON:
        skeleton->name = name;
        skeleton->has_skeleton = true;
        skeleton->has_bbox = true;
        skeleton->has_obb = false;
        skeleton->num_nodes = 2;
        skeleton->num_edges = 1;
        skeleton->node_names = {"Point1", "Point2"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }

        skeleton->edges = {{0, 1}};
        break;

    case SP_LOAD:
        load_skeleton_json(skeleton_file_name, skeleton);
        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }
        break;
    }
};

void allocate_keypoints(KeyPoints *keypoints, render_scene *scene,
                        SkeletonContext *skeleton) {
    // allocate memory for storing keypoints
    keypoints->active_id = (u32 *)malloc(sizeof(u32) * scene->num_cams);
    
    new (&keypoints->bbox2d_list) std::vector<std::vector<BoundingBox>>();
    new (&keypoints->obb2d_list) std::vector<std::vector<OrientedBoundingBox>>();
    
    if (skeleton->has_bbox) {
        keypoints->bbox2d_list.resize(scene->num_cams);
        for (u32 j = 0; j < scene->num_cams; j++) {
            // Initialize with a default BoundingBox
            BoundingBox default_bbox;
            default_bbox.rect = NULL;
            default_bbox.state = RectNull;
            default_bbox.class_id = -1;
            default_bbox.confidence = 0.0f;
            default_bbox.has_bbox_keypoints = false;
            default_bbox.bbox_keypoints2d = nullptr;
            default_bbox.active_kp_id = nullptr;
            keypoints->bbox2d_list[j].push_back(default_bbox);
        }
    } else {
        keypoints->bbox2d_list.resize(scene->num_cams);
    }

    if (skeleton->has_obb) {
        keypoints->obb2d_list.resize(scene->num_cams);
        for (u32 j = 0; j < scene->num_cams; j++) {
            // Initialize with a default OrientedBoundingBox
            OrientedBoundingBox default_obb;
            default_obb.axis_point1 = ImVec2(0, 0);
            default_obb.axis_point2 = ImVec2(0, 0);
            default_obb.corner_point = ImVec2(0, 0);
            default_obb.center = ImVec2(0, 0);
            default_obb.width = 0;
            default_obb.height = 0;
            default_obb.rotation = 0;
            default_obb.state = OBBNull;
            default_obb.class_id = -1;
            default_obb.confidence = 0.0f;
            keypoints->obb2d_list[j].push_back(default_obb);
        }
    } else {
        keypoints->obb2d_list.resize(scene->num_cams);
    }

    if (skeleton->has_skeleton) {
        keypoints->keypoints3d =
            (triple_d *)malloc(sizeof(triple_d) * skeleton->num_nodes);
        keypoints->keypoints2d =
            (KeyPoints2D **)malloc(sizeof(KeyPoints2D *) * scene->num_cams);
        for (u32 j = 0; j < scene->num_cams; j++) {
            keypoints->keypoints2d[j] =
                (KeyPoints2D *)malloc(sizeof(KeyPoints2D) * skeleton->num_nodes);
        }

        // initialize to big number
        for (u32 j = 0; j < scene->num_cams; j++) {
            keypoints->active_id[j] = 0;
            for (u32 k = 0; k < skeleton->num_nodes; k++) {
                keypoints->keypoints2d[j][k].is_labeled = false;
                keypoints->keypoints2d[j][k].is_triangulated = false;
                keypoints->keypoints2d[j][k].position.x = 1E7;
                keypoints->keypoints2d[j][k].position.y = 1E7;
            }
        }

        for (u32 k = 0; k < skeleton->num_nodes; k++) {
            keypoints->keypoints3d[k].x = 1E7;
            keypoints->keypoints3d[k].y = 1E7;
            keypoints->keypoints3d[k].z = 1E7;
        }
    } else {
        keypoints->keypoints3d = nullptr;
        keypoints->keypoints2d = nullptr;
        // Still need to initialize active_id array even without skeleton
        for (u32 j = 0; j < scene->num_cams; j++) {
            keypoints->active_id[j] = 0;
        }
    }
}

void free_bbox_keypoints(BoundingBox* bbox, render_scene *scene) {
    if (!bbox->has_bbox_keypoints) return;
    
    if (bbox->bbox_keypoints2d) {
        for (u32 j = 0; j < scene->num_cams; j++) {
            if (bbox->bbox_keypoints2d[j]) {
                free(bbox->bbox_keypoints2d[j]);
            }
        }
        free(bbox->bbox_keypoints2d);
        bbox->bbox_keypoints2d = nullptr;
    }
    if (bbox->active_kp_id) {
        free(bbox->active_kp_id);
        bbox->active_kp_id = nullptr;
    }
    bbox->has_bbox_keypoints = false;
}

void free_keypoints(KeyPoints *keypoints, render_scene *scene) {
    if (!keypoints)
        return;

    if (keypoints->keypoints2d) {
        for (u32 j = 0; j < scene->num_cams; ++j) {
            free(keypoints->keypoints2d[j]);
        }
        free(keypoints->keypoints2d);
    }

    free(keypoints->active_id);
    free(keypoints->keypoints3d);
    
    // Free bounding boxes
    for (u32 j = 0; j < scene->num_cams; j++) {
        for (auto& bbox : keypoints->bbox2d_list[j]) {
            if (bbox.rect) {
                delete bbox.rect;
            }
            free_bbox_keypoints(&bbox, scene);
        }
    }
    
    keypoints->bbox2d_list.clear();
    keypoints->obb2d_list.clear();
    
    free(keypoints); // finally free the KeyPoints struct itself
}

void allocate_bbox_keypoints(BoundingBox* bbox, render_scene *scene, SkeletonContext* skeleton) {
    if (!skeleton->has_skeleton) {
        bbox->has_bbox_keypoints = false;
        return;
    }
    
    bbox->has_bbox_keypoints = true;
    bbox->active_kp_id = (u32 *)malloc(sizeof(u32) * scene->num_cams);
    bbox->bbox_keypoints2d = (KeyPoints2D **)malloc(sizeof(KeyPoints2D*) * scene->num_cams);
    
    for (u32 j = 0; j < scene->num_cams; j++) {
        bbox->bbox_keypoints2d[j] = (KeyPoints2D *)malloc(sizeof(KeyPoints2D) * skeleton->num_nodes);
        bbox->active_kp_id[j] = 0;
        
        for (u32 k = 0; k < skeleton->num_nodes; k++) {
            bbox->bbox_keypoints2d[j][k].is_labeled = false;
            bbox->bbox_keypoints2d[j][k].is_triangulated = false;
            bbox->bbox_keypoints2d[j][k].position.x = 1E7;
            bbox->bbox_keypoints2d[j][k].position.y = 1E7;
        }
    }
}

void constrain_keypoint_to_bbox(KeyPoints2D* keypoint, ImPlotRect* bbox_rect) {
    if (!bbox_rect || !keypoint->is_labeled) return;
    
    if (keypoint->position.x < bbox_rect->X.Min) {
        keypoint->position.x = bbox_rect->X.Min;
    } else if (keypoint->position.x > bbox_rect->X.Max) {
        keypoint->position.x = bbox_rect->X.Max;
    }
    
    if (keypoint->position.y < bbox_rect->Y.Min) {
        keypoint->position.y = bbox_rect->Y.Min;
    } else if (keypoint->position.y > bbox_rect->Y.Max) {
        keypoint->position.y = bbox_rect->Y.Max;
    }
}

bool is_point_in_bbox(double x, double y, ImPlotRect* bbox_rect) {
    if (!bbox_rect) return false;
    return (x >= bbox_rect->X.Min && x <= bbox_rect->X.Max && 
            y >= bbox_rect->Y.Min && y <= bbox_rect->Y.Max);
}

// Scale keypoints to maintain relative positions within a bounding box
void scale_bbox_keypoints(BoundingBox* bbox, render_scene *scene, SkeletonContext* skeleton, 
                         ImPlotRect* old_rect, ImPlotRect* new_rect) {
    if (!bbox->has_bbox_keypoints || !old_rect || !new_rect) return;
    
    double old_width = old_rect->X.Max - old_rect->X.Min;
    double old_height = old_rect->Y.Max - old_rect->Y.Min;
    double new_width = new_rect->X.Max - new_rect->X.Min;
    double new_height = new_rect->Y.Max - new_rect->Y.Min;
    
    if (old_width <= 0 || old_height <= 0) return;

    double old_min_x = std::min(old_rect->X.Min, old_rect->X.Max);
    double old_min_y = std::min(old_rect->Y.Min, old_rect->Y.Max);
    double old_max_x = std::max(old_rect->X.Min, old_rect->X.Max);
    double old_max_y = std::max(old_rect->Y.Min, old_rect->Y.Max);

    double new_min_x = std::min(new_rect->X.Min, new_rect->X.Max); 
    double new_min_y = std::min(new_rect->Y.Min, new_rect->Y.Max);
    double new_max_x = std::max(new_rect->X.Min, new_rect->X.Max);
    double new_max_y = std::max(new_rect->Y.Min, new_rect->Y.Max);

    double normalized_old_width = old_max_x - old_min_x;
    double normalized_old_height = old_max_y - old_min_y;
    
    double normalized_new_width = new_max_x - new_min_x;
    double normalized_new_height = new_max_y - new_min_y;

    if (normalized_old_width <= 0 || normalized_old_height <= 0) return;

    
    double scale_x = normalized_new_width / normalized_old_width;
    double scale_y = normalized_new_height / normalized_old_height;
    
    for (u32 j = 0; j < scene->num_cams; j++) {
        for (u32 node = 0; node < skeleton->num_nodes; node++) {
            if (bbox->bbox_keypoints2d[j][node].is_labeled) {
                // Get relative position in old bbox
                double rel_x = (bbox->bbox_keypoints2d[j][node].position.x - old_min_x) / normalized_old_width;
                double rel_y = (bbox->bbox_keypoints2d[j][node].position.y - old_min_y) / normalized_old_height;
                
                // Scale to new bbox
                bbox->bbox_keypoints2d[j][node].position.x = new_min_x + rel_x * normalized_new_width;
                bbox->bbox_keypoints2d[j][node].position.y = new_min_y + rel_y * normalized_new_height;
                
                constrain_keypoint_to_bbox(&bbox->bbox_keypoints2d[j][node], new_rect);
            }
        }
    }
}

void free_all_keypoints(std::map<u32, KeyPoints *> &keypoints_map,
                        render_scene *scene) {
    for (auto &[frame, kp] : keypoints_map)
        free_keypoints(kp, scene);
    keypoints_map.clear();
}

// Render keypoints within a bounding box
void gui_plot_bbox_keypoints(BoundingBox* bbox, SkeletonContext *skeleton, int view_idx, int num_cams, bool is_active, bool& is_saved, int bbox_id) {
    if (!bbox->has_bbox_keypoints || !skeleton->has_skeleton) return;
    
    float pt_size = 4.0f; 
    for (u32 node = 0; node < skeleton->num_nodes; node++) {
        if (bbox->bbox_keypoints2d[view_idx][node].is_labeled) {
            ImVec4 node_color; 
            ImPlotDragToolFlags flag;
            if (is_active) {
                flag = ImPlotDragToolFlags_None;
                if (bbox->active_kp_id[view_idx] == node) {
                    node_color = (ImVec4)ImColor::HSV(0.2, 0.9f, 0.9f); // Different active color for bbox keypoints
                } else {
                    node_color = skeleton->node_colors[node];
                }
            } else {
                flag = ImPlotDragToolFlags_NoInputs;
                // Gray out inactive bbox keypoints
                node_color = ImVec4(0.5f, 0.5f, 0.5f, 0.7f);
            }
            
            int id = 10000 + bbox_id * 1000 + skeleton->num_nodes * view_idx + node;
            bool drag_point_clicked = false;
            bool drag_point_hovered = false;
            bool drag_point_modified = false;
            
            drag_point_modified = ImPlot::DragPoint(id, 
                &bbox->bbox_keypoints2d[view_idx][node].position.x, 
                &bbox->bbox_keypoints2d[view_idx][node].position.y, 
                node_color, pt_size, flag, &drag_point_clicked, &drag_point_hovered);
                
            if (drag_point_modified && is_active) {
                constrain_keypoint_to_bbox(&bbox->bbox_keypoints2d[view_idx][node], bbox->rect);
                bbox->bbox_keypoints2d[view_idx][node].is_triangulated = false;
                is_saved = false;
            }
            
            if (drag_point_hovered && is_active) {
                if (ImGui::IsKeyPressed(ImGuiKey_R, false)) { // delete hovered keypoint
                    bbox->bbox_keypoints2d[view_idx][node].position = {1E7, 1E7};
                    bbox->bbox_keypoints2d[view_idx][node].is_labeled = false;                                        
                    bbox->bbox_keypoints2d[view_idx][node].is_triangulated = false;
                    bbox->active_kp_id[view_idx] = node;
                    is_saved = false;
                }
                
                if (ImGui::IsKeyPressed(ImGuiKey_F, false)) { 
                    for (int cam_idx = 0; cam_idx < num_cams; cam_idx++) {
                        bbox->bbox_keypoints2d[cam_idx][node].position = {1E7, 1E7};
                        bbox->bbox_keypoints2d[cam_idx][node].is_labeled = false;                                        
                        bbox->bbox_keypoints2d[cam_idx][node].is_triangulated = false;
                        bbox->active_kp_id[cam_idx] = node;
                    }
                    is_saved = false;
                }
            }

            if (drag_point_clicked && is_active) {
                bbox->active_kp_id[view_idx] = node;
            }
        }
    }

    // Draw skeleton edges within bbox
    for (u32 edge = 0; edge < skeleton->num_edges; edge++) {
        auto[a, b] = skeleton->edges[edge];

        if (bbox->bbox_keypoints2d[view_idx][a].is_labeled && bbox->bbox_keypoints2d[view_idx][b].is_labeled) {
            double xs[2] {bbox->bbox_keypoints2d[view_idx][a].position.x, bbox->bbox_keypoints2d[view_idx][b].position.x};
            double ys[2] {bbox->bbox_keypoints2d[view_idx][a].position.y, bbox->bbox_keypoints2d[view_idx][b].position.y};
            
            // Gray out edges for inactive bboxes
            if (is_active) {
                ImPlot::SetNextLineStyle(ImVec4(0.8f, 0.8f, 0.2f, 0.8f), 1.5f); 
            } else {
                ImPlot::SetNextLineStyle(ImVec4(0.4f, 0.4f, 0.4f, 0.5f), 1.0f);  // Grayed out edges
            }
            ImPlot::PlotLine("##bbox_line", xs, ys, 2);
        }
    }
}

// Oriented Bounding Box utility functions
float calculate_distance(ImVec2 p1, ImVec2 p2) {
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    return sqrtf(dx*dx + dy*dy);
}

float calculate_angle(ImVec2 p1, ImVec2 p2) {
    return atan2f(p2.y - p1.y, p2.x - p1.x);
}

void calculate_obb_properties(OrientedBoundingBox* obb) {
    if (obb->state < OBBSecondAxisPoint) return;
    
    // The first two points are vertices of the OBB (one edge)
    ImVec2 vertex1 = obb->axis_point1;
    ImVec2 vertex2 = obb->axis_point2;
    
    // Calculate the vector along the edge defined by the first two points
    ImVec2 edge_vector = {
        vertex2.x - vertex1.x,
        vertex2.y - vertex1.y
    };
    
    // Calculate perpendicular vector (90 degrees rotated)
    ImVec2 perp_vector = {-edge_vector.y, edge_vector.x};
    
    if (obb->state >= OBBThirdPoint) {
        // We have the third point - calculate the final OBB
        ImVec2 mouse_point = obb->corner_point;
        
        // Project the mouse point onto the perpendicular direction to find the height
        ImVec2 to_mouse = {
            mouse_point.x - vertex1.x,
            mouse_point.y - vertex1.y
        };
        
        // Calculate the perpendicular distance (this becomes the height of the rectangle)
        float perp_dot = to_mouse.x * perp_vector.x + to_mouse.y * perp_vector.y;
        float perp_length_sq = perp_vector.x * perp_vector.x + perp_vector.y * perp_vector.y;
        
        if (perp_length_sq > 0) {
            float height = fabsf(perp_dot) / sqrtf(perp_length_sq);
            
            // Normalize the perpendicular vector
            float perp_length = sqrtf(perp_length_sq);
            ImVec2 perp_unit = {perp_vector.x / perp_length, perp_vector.y / perp_length};
            
            // Determine which side of the edge the mouse is on
            float side_sign = perp_dot > 0 ? 1.0f : -1.0f;
            
            // Calculate the four vertices of the rectangle
            // vertex1 and vertex2 are already defined
            ImVec2 vertex3 = {
                vertex2.x + height * perp_unit.x * side_sign,
                vertex2.y + height * perp_unit.y * side_sign
            };
            ImVec2 vertex4 = {
                vertex1.x + height * perp_unit.x * side_sign,
                vertex1.y + height * perp_unit.y * side_sign
            };
            
            // Calculate center, width, height, and rotation
            obb->center.x = (vertex1.x + vertex2.x + vertex3.x + vertex4.x) / 4.0f;
            obb->center.y = (vertex1.y + vertex2.y + vertex3.y + vertex4.y) / 4.0f;
            
            obb->width = sqrtf(edge_vector.x * edge_vector.x + edge_vector.y * edge_vector.y);
            obb->height = height;
            obb->rotation = atan2f(edge_vector.y, edge_vector.x);
        }
    }
}

// New function to calculate preview OBB based on mouse position
void calculate_obb_preview(OrientedBoundingBox* obb, ImVec2 mouse_pos) {
    if (obb->state < OBBSecondAxisPoint) return;
    
    // Temporarily set the corner point to mouse position for preview calculation
    ImVec2 original_corner = obb->corner_point;
    obb->corner_point = mouse_pos;
    
    // Calculate properties for preview
    calculate_obb_properties(obb);
    
    // Restore original corner point
    obb->corner_point = original_corner;
}

// Helper function to check if a point is near another point (for dragging hit detection)
bool is_point_near(ImVec2 point1, ImVec2 point2, float threshold = 15.0f) {
    float dx = point1.x - point2.x;
    float dy = point1.y - point2.y;
    float distance_sq = dx * dx + dy * dy;
    return distance_sq < (threshold * threshold);
}

// Helper function to get the four corners of an OBB for saving/loading
void get_obb_corners(const OrientedBoundingBox* obb, ImVec2 corners[4]) {
    if (obb->state != OBBComplete) {
        // If OBB is not complete, set all corners to zero
        for (int i = 0; i < 4; i++) {
            corners[i] = ImVec2(0, 0);
        }
        return;
    }
    
    // Use the stored properties to calculate corners
    float cos_rot = cosf(obb->rotation);
    float sin_rot = sinf(obb->rotation);
    
    // Half-width and half-height
    float half_w = obb->width * 0.5f;
    float half_h = obb->height * 0.5f;
    
    // Calculate the four corners relative to center, then translate
    // Corner order: bottom-left, bottom-right, top-right, top-left (in local coordinates)
    ImVec2 local_corners[4] = {
        {-half_w, -half_h},  // bottom-left
        { half_w, -half_h},  // bottom-right  
        { half_w,  half_h},  // top-right
        {-half_w,  half_h}   // top-left
    };
    
    // Rotate and translate each corner
    for (int i = 0; i < 4; i++) {
        float x = local_corners[i].x;
        float y = local_corners[i].y;
        
        // Apply rotation
        float rotated_x = x * cos_rot - y * sin_rot;
        float rotated_y = x * sin_rot + y * cos_rot;
        
        // Translate to world position
        corners[i].x = rotated_x + obb->center.x;
        corners[i].y = rotated_y + obb->center.y;
    }
}

void set_obb_from_corners(OrientedBoundingBox* obb, const ImVec2 corners[4], int class_id) {
    obb->axis_point1 = corners[0];
    obb->axis_point2 = corners[1];
    obb->corner_point = corners[3];
    
    obb->center.x = (corners[0].x + corners[1].x + corners[2].x + corners[3].x) / 4.0f;
    obb->center.y = (corners[0].y + corners[1].y + corners[2].y + corners[3].y) / 4.0f;
    
    float dx = corners[1].x - corners[0].x;
    float dy = corners[1].y - corners[0].y;
    obb->width = sqrtf(dx * dx + dy * dy);
    
    dx = corners[3].x - corners[0].x;
    dy = corners[3].y - corners[0].y;
    obb->height = sqrtf(dx * dx + dy * dy);
    
    obb->rotation = atan2f(corners[1].y - corners[0].y, corners[1].x - corners[0].x);
    
    obb->state = OBBComplete;
    obb->class_id = class_id;
    obb->confidence = 1.0f; 
}

void draw_obb(OrientedBoundingBox& obb, bool is_active, ImVec4 class_color = ImVec4(1, 1, 1, 0.7f), ImVec2 mouse_pos = ImVec2(0, 0), bool show_preview = false) {
    if (obb.state == OBBNull) return;
    
    // Draw construction points during creation
    if (obb.state < OBBComplete) {
        // Draw first vertex
        if (obb.state >= OBBFirstAxisPoint) {
            double x1 = obb.axis_point1.x, y1 = obb.axis_point1.y;
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 6, ImVec4(1, 0, 0, 1), IMPLOT_AUTO, ImVec4(1, 0, 0, 1));
            ImPlot::PlotScatter("##obb_vertex1", &x1, &y1, 1);
            
            if (show_preview && obb.state == OBBFirstAxisPoint) {
                double xs_preview[2] = {obb.axis_point1.x, mouse_pos.x};
                double ys_preview[2] = {obb.axis_point1.y, mouse_pos.y};
                ImPlot::SetNextLineStyle(ImVec4(1, 0, 0, 0.6f), 2.0f);
                ImPlot::PlotLine("##obb_preview_line", xs_preview, ys_preview, 2);
            }
        }
        
        // Draw second vertex and the edge between them
        if (obb.state >= OBBSecondAxisPoint) {
            double x2 = obb.axis_point2.x, y2 = obb.axis_point2.y;
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 6, ImVec4(0, 1, 0, 1), IMPLOT_AUTO, ImVec4(0, 1, 0, 1));
            ImPlot::PlotScatter("##obb_vertex2", &x2, &y2, 1);
            
            // Draw the edge line between first two vertices
            double xs[2] = {obb.axis_point1.x, obb.axis_point2.x};
            double ys[2] = {obb.axis_point1.y, obb.axis_point2.y};
            ImPlot::SetNextLineStyle(ImVec4(0, 1, 0, 0.8f), 3.0f);
            ImPlot::PlotLine("##obb_edge_base", xs, ys, 2);
            
            // Show preview rectangle if mouse position is provided
            if (show_preview && obb.state == OBBSecondAxisPoint) {
                // Create a temporary OBB for preview calculation
                OrientedBoundingBox preview_obb = obb;
                preview_obb.corner_point = mouse_pos;
                preview_obb.state = OBBThirdPoint;
                calculate_obb_properties(&preview_obb);
                
                // Draw preview rectangle with dashed/transparent style
                if (preview_obb.width > 0 && preview_obb.height > 0) {
                    float cos_rot = cosf(preview_obb.rotation);
                    float sin_rot = sinf(preview_obb.rotation);
                    float half_width = preview_obb.width / 2.0f;
                    float half_height = preview_obb.height / 2.0f;
                    
                    ImVec2 corners[4];
                    ImVec2 local_corners[4] = {
                        {-half_width, -half_height},
                        { half_width, -half_height},
                        { half_width,  half_height},
                        {-half_width,  half_height}
                    };
                    
                    for (int i = 0; i < 4; i++) {
                        corners[i].x = preview_obb.center.x + local_corners[i].x * cos_rot - local_corners[i].y * sin_rot;
                        corners[i].y = preview_obb.center.y + local_corners[i].x * sin_rot + local_corners[i].y * cos_rot;
                    }
                    
                    // Draw preview rectangle with transparent style
                    ImVec4 preview_color = ImVec4(1, 1, 0, 0.4f);
                    for (int i = 0; i < 4; i++) {
                        int next = (i + 1) % 4;
                        double xs_prev[2] = {corners[i].x, corners[next].x};
                        double ys_prev[2] = {corners[i].y, corners[next].y};
                        ImPlot::SetNextLineStyle(preview_color, 1.5f);
                        ImPlot::PlotLine("##obb_preview", xs_prev, ys_prev, 2);
                    }
                }
            }
        }
        
        // Draw third point if we have it
        if (obb.state >= OBBThirdPoint) {
            double x3 = obb.corner_point.x, y3 = obb.corner_point.y;
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 6, ImVec4(0, 0, 1, 1), IMPLOT_AUTO, ImVec4(0, 0, 1, 1));
            ImPlot::PlotScatter("##obb_corner", &x3, &y3, 1);
        }
    }
    
    // Draw final OBB when complete
    if (obb.state == OBBComplete || obb.state == OBBThirdPoint) {
        // Calculate the four corners of the OBB
        float cos_rot = cosf(obb.rotation);
        float sin_rot = sinf(obb.rotation);
        float half_width = obb.width / 2.0f;
        float half_height = obb.height / 2.0f;
        
        ImVec2 corners[4];
        ImVec2 local_corners[4] = {
            {-half_width, -half_height},
            { half_width, -half_height},
            { half_width,  half_height},
            {-half_width,  half_height}
        };
        
        for (int i = 0; i < 4; i++) {
            corners[i].x = obb.center.x + local_corners[i].x * cos_rot - local_corners[i].y * sin_rot;
            corners[i].y = obb.center.y + local_corners[i].x * sin_rot + local_corners[i].y * cos_rot;
        }
        
        // Draw the rectangle with class-based color
        ImVec4 color = is_active ? ImVec4(0, 1, 1, 0.9f) : class_color;
        float line_width = obb.state == OBBComplete ? 2.5f : 2.0f;
        
        for (int i = 0; i < 4; i++) {
            int next = (i + 1) % 4;
            double xs[2] = {corners[i].x, corners[next].x};
            double ys[2] = {corners[i].y, corners[next].y};
            ImPlot::SetNextLineStyle(color, line_width);
            ImPlot::PlotLine("##obb_final", xs, ys, 2);
        }
    }
}

// Helper function to check if a point is near a line segment
bool is_point_near_line_segment(ImVec2 point, ImVec2 line_start, ImVec2 line_end, float threshold = 5.0f) {
    // Calculate distance from point to line segment
    ImVec2 line_vec = {line_end.x - line_start.x, line_end.y - line_start.y};
    ImVec2 point_vec = {point.x - line_start.x, point.y - line_start.y};
    
    float line_length_sq = line_vec.x * line_vec.x + line_vec.y * line_vec.y;
    if (line_length_sq == 0) return false;
    
    float t = (point_vec.x * line_vec.x + point_vec.y * line_vec.y) / line_length_sq;
    t = fmaxf(0.0f, fminf(1.0f, t));  // Clamp to [0,1]
    
    ImVec2 closest = {line_start.x + t * line_vec.x, line_start.y + t * line_vec.y};
    float dist_sq = (point.x - closest.x) * (point.x - closest.x) + (point.y - closest.y) * (point.y - closest.y);
    
    return dist_sq <= threshold * threshold;
}

// Function to check if a point is inside an oriented bounding box
bool is_point_inside_obb(ImVec2 point, const OrientedBoundingBox& obb) {
    if (obb.state != OBBComplete) return false;
    
    // Calculate the four corners of the OBB
    float cos_rot = cosf(obb.rotation);
    float sin_rot = sinf(obb.rotation);
    float half_width = obb.width / 2.0f;
    float half_height = obb.height / 2.0f;
    
    ImVec2 corners[4];
    ImVec2 local_corners[4] = {
        {-half_width, -half_height},
        { half_width, -half_height},
        { half_width,  half_height},
        {-half_width,  half_height}
    };
    
    for (int i = 0; i < 4; i++) {
        corners[i].x = obb.center.x + local_corners[i].x * cos_rot - local_corners[i].y * sin_rot;
        corners[i].y = obb.center.y + local_corners[i].x * sin_rot + local_corners[i].y * cos_rot;
    }
    
    // Use ray casting algorithm to check if point is inside polygon
    bool inside = false;
    for (int i = 0, j = 3; i < 4; j = i++) {
        if (((corners[i].y > point.y) != (corners[j].y > point.y)) &&
            (point.x < (corners[j].x - corners[i].x) * (point.y - corners[i].y) / (corners[j].y - corners[i].y) + corners[i].x)) {
            inside = !inside;
        }
    }
    
    return inside;
}

// Function to handle OBB manipulation for resizing
void handle_obb_dragging(OrientedBoundingBox& obb, ImVec2 mouse_pos, bool is_active_drag) {
    if (obb.state != OBBComplete) return;
    
    // Calculate the four corners of the OBB
    float cos_rot = cosf(obb.rotation);
    float sin_rot = sinf(obb.rotation);
    float half_width = obb.width / 2.0f;
    float half_height = obb.height / 2.0f;
    
    ImVec2 corners[4];
    ImVec2 local_corners[4] = {
        {-half_width/2, -half_height},
        { half_width/2, -half_height},
        { half_width/2,  half_height},
        {-half_width/2,  half_height}
    };
    
    for (int i = 0; i < 4; i++) {
        corners[i].x = obb.center.x + local_corners[i].x * cos_rot - local_corners[i].y * sin_rot;
        corners[i].y = obb.center.y + local_corners[i].x * sin_rot + local_corners[i].y * cos_rot;
    }
    
    // Only actually modify the OBB if actively dragging
    if (is_active_drag) {
        static ImVec2 last_mouse_pos = mouse_pos;
        static bool first_drag = true;
        
        if (first_drag) {
            last_mouse_pos = mouse_pos;
            first_drag = false;
            return;
        }
        
        // Calculate mouse movement
        ImVec2 mouse_delta = {mouse_pos.x - last_mouse_pos.x, mouse_pos.y - last_mouse_pos.y};
        
        // Check which side is being dragged and resize accordingly
        for (int i = 0; i < 4; i++) {
            int next = (i + 1) % 4;
            if (is_point_near_line_segment(mouse_pos, corners[i], corners[next], 8.0f)) {
                // Calculate side direction and perpendicular direction
                ImVec2 side_vec = {corners[next].x - corners[i].x, corners[next].y - corners[i].y};
                float side_length = sqrtf(side_vec.x * side_vec.x + side_vec.y * side_vec.y);
                
                if (side_length > 0) {
                    // Normalize side vector
                    side_vec.x /= side_length;
                    side_vec.y /= side_length;
                    
                    // Calculate perpendicular direction (pointing outward from center)
                    ImVec2 perp_dir = {-side_vec.y, side_vec.x};
                    
                    // Check if perpendicular points away from center
                    ImVec2 side_center = {(corners[i].x + corners[next].x) / 2.0f, (corners[i].y + corners[next].y) / 2.0f};
                    ImVec2 to_center = {obb.center.x - side_center.x, obb.center.y - side_center.y};
                    if (perp_dir.x * to_center.x + perp_dir.y * to_center.y > 0) {
                        perp_dir.x = -perp_dir.x;
                        perp_dir.y = -perp_dir.y;
                    }
                    
                    // Project mouse movement onto perpendicular direction
                    float perp_movement = mouse_delta.x * perp_dir.x + mouse_delta.y * perp_dir.y;
                    
                    // Update width or height based on which side is being dragged
                    if (i == 0 || i == 2) {  // Top/bottom sides (height)
                        obb.height = fmaxf(5.0f, obb.height + 2.0f * perp_movement);
                    } else {  // Left/right sides (width)
                        obb.width = fmaxf(5.0f, obb.width + 2.0f * perp_movement);
                    }
                }
                break;
            }
        }
        
        last_mouse_pos = mouse_pos;
    } else {
        // Reset drag state when not actively dragging
        static bool reset_needed = true;
        if (reset_needed) {
            static bool first_drag = true;
            first_drag = true;
            reset_needed = false;
        }
    }
}

#endif
