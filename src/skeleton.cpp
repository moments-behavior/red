#include "skeleton.h"
#include "global.h"
#include "render.h"

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
        {"Table3Corners", Table3Corners},
        {"Rat22", Rat22},
        {"Rat20", Rat20},
        {"Rat24", Rat24},
        {"Rat20Target", Rat20Target},
        {"Rat24Target", Rat24Target},
        {"Fly50", Fly50}};
    return skeleton_all;
}

void load_skeleton_json(std::string file_name, SkeletonContext *skeleton) {
    std::ifstream f(file_name);
    nlohmann::json s_config = nlohmann::json::parse(f);
    skeleton->name = file_name;
    skeleton->has_skeleton = s_config.contains("has_skeleton")
                                 ? s_config["has_skeleton"].get<bool>()
                                 : true;
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
    for (int i = 0; i < skeleton->num_nodes; i++) {
        ImVec4 color =
            (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 1.0f, 1.0f);
        skeleton->node_colors.push_back(color);
    }
}

void skeleton_initialize(std::string name, SkeletonContext *skeleton,
                         SkeletonPrimitive skeleton_type) {
    skeleton->has_skeleton = true;

    switch (skeleton_type) {
    case Table3Corners:
        skeleton->name = name;
        skeleton->num_nodes = 3;
        skeleton->num_edges = 2;
        skeleton->node_names = {"BottomLeft", "BottomRight", "TopLeft"};
        for (int i = 0; i < skeleton->num_nodes; i++)
            skeleton->node_colors.push_back(
                (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 0.8f, 0.8f));
        skeleton->edges = {{0, 1}, {0, 2}};
        break;

    case Target:
        skeleton->name = name;
        skeleton->num_nodes = 1;
        skeleton->num_edges = 0;
        skeleton->node_names = {"Target"};
        for (int i = 0; i < skeleton->num_nodes; i++)
            skeleton->node_colors.push_back(
                (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 1.0f, 1.0f));
        break;

    case Rat7Target:
        skeleton->name = name;
        skeleton->num_nodes = 9;
        skeleton->num_edges = 8;
        skeleton->node_names = {"Snout",  "EarL",     "EarR",
                                "Neck",   "SpineF",   "SpineM",
                                "SpineL", "TailBase", "Target"};
        for (int i = 0; i < skeleton->num_nodes; i++)
            skeleton->node_colors.push_back(
                (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 1.0f, 1.0f));
        skeleton->edges = {{0, 1}, {0, 2}, {1, 3}, {2, 3},
                           {3, 4}, {4, 5}, {5, 6}, {6, 7}};
        break;

    case RatTarget:
        skeleton->name = name;
        skeleton->num_nodes = 2;
        skeleton->num_edges = 1;
        skeleton->node_names = {"Snout", "Target"};
        for (int i = 0; i < skeleton->num_nodes; i++)
            skeleton->node_colors.push_back(
                (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 1.0f, 1.0f));
        skeleton->edges = {{0, 1}};
        break;

    case Rat3Target:
        skeleton->name = name;
        skeleton->num_nodes = 4;
        skeleton->num_edges = 2;
        skeleton->node_names = {"Snout", "EarL", "EarR", "Target"};
        for (int i = 0; i < skeleton->num_nodes; i++)
            skeleton->node_colors.push_back(
                (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 1.0f, 1.0f));
        skeleton->edges = {{0, 1}, {0, 2}};
        break;

    case Rat4Target:
        skeleton->name = name;
        skeleton->num_nodes = 5;
        skeleton->num_edges = 4;
        skeleton->node_names = {"Snout", "EarL", "EarR", "Tail", "Target"};
        for (int i = 0; i < skeleton->num_nodes; i++)
            skeleton->node_colors.push_back(
                (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 1.0f, 1.0f));
        skeleton->edges = {{0, 1}, {0, 2}, {1, 3}, {2, 3}};
        break;

    case Rat4:
        skeleton->name = name;
        skeleton->num_nodes = 4;
        skeleton->num_edges = 4;
        skeleton->node_names = {"Snout", "EarL", "EarR", "Tail"};
        for (int i = 0; i < skeleton->num_nodes; i++)
            skeleton->node_colors.push_back(
                (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 1.0f, 1.0f));
        skeleton->edges = {{0, 1}, {0, 2}, {1, 3}, {2, 3}};
        break;

    case Rat6Target:
        skeleton->name = name;
        skeleton->num_nodes = 7;
        skeleton->num_edges = 6;
        skeleton->node_names = {"Snout",  "EarL",     "EarR",  "Neck",
                                "SpineL", "TailBase", "Target"};
        for (int i = 0; i < skeleton->num_nodes; i++)
            skeleton->node_colors.push_back(
                (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 1.0f, 1.0f));
        skeleton->edges = {{0, 1}, {0, 2}, {1, 3}, {2, 3}, {3, 4}, {4, 5}};
        break;

    case Rat6Target2:
        skeleton->name = name;
        skeleton->num_nodes = 8;
        skeleton->num_edges = 6;
        skeleton->node_names = {"Snout",  "EarL",     "EarR",    "Neck",
                                "SpineL", "TailBase", "Target1", "Target2"};
        for (int i = 0; i < skeleton->num_nodes; i++)
            skeleton->node_colors.push_back(
                (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 1.0f, 1.0f));
        skeleton->edges = {{0, 1}, {0, 2}, {1, 3}, {2, 3}, {3, 4}, {4, 5}};
        break;

    case Rat6:
        skeleton->name = name;
        skeleton->num_nodes = 6;
        skeleton->num_edges = 6;
        skeleton->node_names = {"Snout", "EarL",   "EarR",
                                "Neck",  "SpineL", "TailBase"};
        for (int i = 0; i < skeleton->num_nodes; i++)
            skeleton->node_colors.push_back(
                (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 1.0f, 1.0f));
        skeleton->edges = {{0, 1}, {0, 2}, {1, 3}, {2, 3}, {3, 4}, {4, 5}};
        break;

    case Rat10Target2:
        skeleton->name = name;
        skeleton->num_nodes = 12;
        skeleton->num_edges = 10;
        skeleton->node_names = {"Snout",  "EarL",     "EarR",    "Neck",
                                "SpineL", "TailBase", "HandL",   "HandR",
                                "FootL",  "FootR",    "Target1", "Target2"};
        for (int i = 0; i < skeleton->num_nodes; i++)
            skeleton->node_colors.push_back(
                (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 1.0f, 1.0f));
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
        for (int i = 0; i < skeleton->num_nodes; i++)
            skeleton->node_colors.push_back(
                (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 1.0f, 1.0f));
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
        for (int i = 0; i < skeleton->num_nodes; i++)
            skeleton->node_colors.push_back(
                (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 1.0f, 1.0f));
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
        for (int i = 0; i < skeleton->num_nodes; i++)
            skeleton->node_colors.push_back(
                (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 1.0f, 1.0f));
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
        for (int i = 0; i < skeleton->num_nodes; i++)
            skeleton->node_colors.push_back(
                (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 1.0f, 1.0f));
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
        for (int i = 0; i < skeleton->num_nodes; i++)
            skeleton->node_colors.push_back(
                (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 1.0f, 1.0f));
        skeleton->edges = {{0, 1},   {0, 2},   {1, 3},   {2, 3},   {3, 4},
                           {4, 5},   {3, 6},   {6, 7},   {7, 8},   {8, 9},
                           {3, 10},  {10, 11}, {11, 12}, {12, 13}, {4, 14},
                           {14, 15}, {15, 16}, {16, 17}, {4, 18},  {18, 19},
                           {19, 20}, {20, 21}};
        break;

    case Fly50:
        skeleton->name = name;
        skeleton->num_nodes = 50;
        skeleton->num_edges = 44;
        skeleton->node_names = {
            "Antenna_Base", "EyeL",       "EyeR",       "Scutellum",
            "Abd_A4",       "Abd_tip",    "WingL_base", "WingL_V12",
            "WingL_V13",    "T1L_ThxCx",  "T1L_Tro",    "T1L_FeTi",
            "T1L_TiTa",     "T1L_TaT1",   "T1L_TaT3",   "T1L_TaTip",
            "T2L_Tro",      "T2L_FeTi",   "T2L_TiTa",   "T2L_TaT1",
            "T2L_TaT3",     "T2L_TaTip",  "T3L_Tro",    "T3L_FeTi",
            "T3L_TiTa",     "T3L_TaT1",   "T3L_TaT3",   "T3L_TaTip",
            "WingR_base",   "WingR_V12",  "WingR_V13",  "T1R_ThxCx",
            "T1R_Tro",      "T1R_FeTi",   "T1R_TiTa",   "T1R_TaT1",
            "T1R_TaT3",     "T1R_TaTip",  "T2R_Tro",    "T2R_FeTi",
            "T2R_TiTa",     "T2R_TaT1",   "T2R_TaT3",   "T2R_TaTip",
            "T3R_Tro",      "T3R_FeTi",   "T3R_TiTa",   "T3R_TaT1",
            "T3R_TaT3",     "T3R_TaTip"};
        for (int i = 0; i < skeleton->num_nodes; i++)
            skeleton->node_colors.push_back(
                (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 1.0f, 1.0f));
        skeleton->edges = {
            {0, 1},   {0, 2},   {1, 3},   {2, 3},   {3, 4},   {4, 5},
            {6, 7},   {7, 8},   {8, 6},   {9, 10},  {10, 11}, {11, 12},
            {12, 13}, {13, 14}, {14, 15}, {16, 17}, {17, 18}, {18, 19},
            {19, 20}, {20, 21}, {22, 23}, {23, 24}, {24, 25}, {25, 26},
            {26, 27}, {28, 29}, {29, 30}, {30, 28}, {31, 32}, {32, 33},
            {33, 34}, {34, 35}, {35, 36}, {36, 37}, {38, 39}, {39, 40},
            {40, 41}, {41, 42}, {42, 43}, {44, 45}, {45, 46}, {46, 47},
            {47, 48}, {48, 49}};
        break;
    }
}

bool has_labeled_frames(const std::map<u32, KeyPoints *> &keypoints_map,
                        SkeletonContext *skeleton) {
    for (const auto &[frame_num, keypoints] : keypoints_map) {
        if (!keypoints) continue;
        if (skeleton->has_skeleton) {
            for (int cam_id = 0; cam_id < MAX_VIEWS; cam_id++) {
                for (int kp_id = 0; kp_id < skeleton->num_nodes; kp_id++) {
                    if (keypoints->kp2d[cam_id][kp_id].is_labeled)
                        return true;
                }
            }
        }
    }
    return false;
}

void allocate_keypoints(KeyPoints *keypoints, RenderScene *scene,
                        SkeletonContext *skeleton) {
    keypoints->active_id = (u32 *)malloc(sizeof(u32) * scene->num_cams);

    if (skeleton->has_skeleton) {
        keypoints->kp3d =
            (KeyPoints3D *)malloc(sizeof(KeyPoints3D) * skeleton->num_nodes);
        keypoints->kp2d =
            (KeyPoints2D **)malloc(sizeof(KeyPoints2D *) * scene->num_cams);
        for (u32 j = 0; j < scene->num_cams; j++) {
            keypoints->kp2d[j] = (KeyPoints2D *)malloc(sizeof(KeyPoints2D) *
                                                       skeleton->num_nodes);
        }

        for (u32 j = 0; j < scene->num_cams; j++) {
            keypoints->active_id[j] = 0;
            for (u32 k = 0; k < skeleton->num_nodes; k++) {
                keypoints->kp2d[j][k].is_labeled = false;
                keypoints->kp2d[j][k].position.x = 1E7;
                keypoints->kp2d[j][k].position.y = 1E7;
                keypoints->kp2d[j][k].last_position =
                    keypoints->kp2d[j][k].position;
                keypoints->kp2d[j][k].last_is_labeled = false;
            }
        }

        for (u32 k = 0; k < skeleton->num_nodes; k++) {
            keypoints->kp3d[k].position.x = 1E7;
            keypoints->kp3d[k].position.y = 1E7;
            keypoints->kp3d[k].position.z = 1E7;
            keypoints->kp3d[k].is_triangulated = false;
        }
    } else {
        keypoints->kp3d = nullptr;
        keypoints->kp2d = nullptr;
        for (u32 j = 0; j < scene->num_cams; j++)
            keypoints->active_id[j] = 0;
    }
}

void free_keypoints(KeyPoints *keypoints, RenderScene *scene) {
    if (!keypoints) return;

    if (keypoints->kp2d) {
        for (u32 j = 0; j < scene->num_cams; ++j)
            free(keypoints->kp2d[j]);
        free(keypoints->kp2d);
    }

    free(keypoints->active_id);
    free(keypoints->kp3d);
    free(keypoints);
}

void free_all_keypoints(std::map<u32, KeyPoints *> &keypoints_map,
                        RenderScene *scene) {
    for (auto &[frame, kp] : keypoints_map)
        free_keypoints(kp, scene);
    keypoints_map.clear();
}

void cleanup_skeleton_data(std::map<u32, KeyPoints *> &keypoints_map,
                           RenderScene *scene) {
    if (!keypoints_map.empty()) {
        free_all_keypoints(keypoints_map, scene);
        keypoints_map.clear();
    }
}

bool has_any_labels(const KeyPoints *keypoints, const SkeletonContext &skeleton,
                    const RenderScene *scene, float yolo_thresh) {
    if (!keypoints) return false;

    if (skeleton.has_skeleton) {
        for (int cam_id = 0; cam_id < scene->num_cams && cam_id < MAX_VIEWS;
             ++cam_id) {
            for (int kp_id = 0; kp_id < skeleton.num_nodes; ++kp_id) {
                if (keypoints->kp2d[cam_id][kp_id].is_labeled)
                    return true;
            }
        }
    }

    return false;
}

void copy_keypoints(KeyPoints *dst, const KeyPoints *src,
                    const RenderScene *scene, const SkeletonContext *skeleton) {
    const u32 num_cams = scene->num_cams;
    const u32 num_nodes = skeleton->num_nodes;

    if (src->active_id && dst->active_id)
        memcpy(dst->active_id, src->active_id, sizeof(u32) * num_cams);

    if (skeleton->has_skeleton) {
        if (src->kp3d && dst->kp3d)
            memcpy(dst->kp3d, src->kp3d, sizeof(KeyPoints3D) * num_nodes);

        if (src->kp2d && dst->kp2d) {
            for (u32 cam = 0; cam < num_cams; ++cam) {
                if (src->kp2d[cam] && dst->kp2d[cam])
                    memcpy(dst->kp2d[cam], src->kp2d[cam],
                           sizeof(KeyPoints2D) * num_nodes);
            }
        }
    }
}
