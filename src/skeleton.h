#ifndef RED_SKELETON
#define RED_SKELETON
#include <vector>
#include <string>
#include "types.h"
#include <map>

struct SkeletonContext {
    int num_nodes;
    int num_edges;
    std::vector<triple_f> node_colors; 
    std::vector<tuple_i> edges;
    std::vector<std::string> node_names;
};

enum skeleton_primitive { 
    CalibrationFourCorners=0,
    Rat10Target2=1

};


std::map<std::string, skeleton_primitive> skeleton_get_all()
{
    std::map<std::string, skeleton_primitive> skeleton_all = {
        {"CalibrationFourCorners", CalibrationFourCorners},
        {"Rat10Target2", Rat10Target2}
    };
    return skeleton_all;
}


void skeleton_initialize(SkeletonContext* skeleton, skeleton_primitive skeleton_type)
{
    switch (skeleton_type){
        case CalibrationFourCorners:
            skeleton->num_nodes = 4;
            skeleton->num_edges = 4;
             
            skeleton->node_names = {"TopLeft", "TopRight", "BottomRight", "BottomLeft"};

            skeleton->node_colors = {
                {1.0f, 0.0f, 1.0f},
                {1.0f, 0.0f, 1.0f},
                {1.0f, 0.0f, 1.0f},
                {1.0f, 0.0f, 1.0f}};
            
            skeleton->edges ={
                {0, 1},
                {1, 2},
                {2, 3},
                {3, 0}};
            break;

        case Rat10Target2:
                skeleton->num_nodes = 12;
                skeleton->num_edges = 10;
                skeleton->node_names = {"Snout", "EarL", "EarR", "Neck", "SpineL", "TailBase", "HandL", "HandR", "FootL", "FootR", "Target1", "Target2"};

                skeleton->node_colors = {
                    {0.8f, 0.8f, 1.0f},
                    {0.8f, 0.8f, 1.0f},
                    {0.8f, 0.8f, 1.0f},
                    {0.5f, 0.5f, 0.5f},
                    {0.5f, 0.5f, 0.5f},
                    {1.0f, 1.0f, 0.0f},
                    {1.0f, 0.0f, 0.0f},
                    {0.0f, 1.0f, 0.0f},
                    {1.0f, 0.0f, 1.0f},
                    {0.0f, 1.0f, 1.0f},
                    {0.3f, 0.3f, 0.1f},
                    {0.3f, 0.3f, 0.1f}
                    };
                
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

#endif
