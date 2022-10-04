#ifndef FRAMEDATA2D_H
#define FRAMEDATA2D_H

#include <iostream>
#include <vector>
#include <tuple>
#include "util.h"

using namespace util;

struct keyPoints{
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double*> px;
    std::vector<double*> py;
    std::vector<bool> isLabeled;
};

class FrameData2D
{
    private:

    public:
        FrameData2D(FrameData2D *frameData);
        FrameData2D(SkelEnum skelEnum);
        void genericSkelSetup();
        int nNodes;
        int nEdges;
        vector<vector<float>> nodeColors;
        float nodeSize;
        vector<int> nodeColorIdx;
        std::vector<std::string> nodeNames;
        std::vector<std::tuple<int,int>> edges;

        keyPoints g;
        int activeKeyPoint;

};

#endif