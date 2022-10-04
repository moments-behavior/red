#ifndef LABELMANAGER_H
#define LABELMANAGER_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "Camera.h"
#include "SkelWorld.h"

#include <iostream>
#include <vector>

class LabelManager
{
    private:

    public:
        LabelManager(std::vector<Camera*> cameras);
        std::vector<Camera*> cameras;
        int nCams;
        int nNodes;
        int nEdges;
        std::vector<std::vector<bool>> isLabeled;
        std::vector<int> numViewsLabeled;
        SkelEnum skelEnum;
        SkelWorld* skelWorld;
        SkelWorld* activeSkelWorld;

        std::map<uint, SkelWorld*> skelWorldMap;
        std::map<uint, SkelWorld*>::iterator skelWorldMapIter;

        string labeled_data_dir;
        bool SaveWorldKeyPoints();
        bool LoadWorldSkels();
        bool LoadCameraSkels();

};

#endif