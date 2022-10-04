#ifndef SKELWORLD_H
#define SKELWORLD_H

#include <iostream>
#include <vector>
#include <tuple>
#include "FrameData2D.h"

struct worldKeyPoints{
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;
    std::vector<float*> px;
    std::vector<float*> py;
    std::vector<float*> pz;
    std::vector<bool> isLabeled;
};

class SkelWorld
{
    private:

    public:
        SkelWorld(FrameData2D *frameData);
        SkelWorld(SkelEnum skelEnum);
        void genericSkelWorldSetup();
        int nNodes;
        int nEdges;
        worldKeyPoints g;
        int activeKeyPoint;
};

#endif