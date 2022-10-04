#include "SkelWorld.h"


SkelWorld::SkelWorld(FrameData2D* frameData)
{
    nNodes = frameData->nNodes;
    nEdges = frameData->nEdges;
    genericSkelWorldSetup();
}

SkelWorld::SkelWorld(SkelEnum skelEnum)
{
switch (skelEnum) {
        case Rat13:
            nNodes = 11;
            nEdges = 10;

            std::cout << "Rat13 SkelWorld created" << std::endl;
            break;

        case Rat22:
            std::cout << "Rat22 SkelWorld not yet supported" << std::endl;
            break;

        case Rat22Target2:
            nNodes = 24;
            nEdges = 22;
            std::cout << "Rat22Target2 SkelWorld created" << std::endl;
            break;

        case Rat23:
            nNodes = 23;
            nEdges = 23;
            std::cout << "Rat23 SkelWorld created" << std::endl;
            break;

        case Rat23Target2:
            nNodes = 25;
            nEdges = 23;
            std::cout << "Rat23Target2 SkelWorld created" << std::endl;
            break;
        
        case Rat10AndTarget:
            nNodes = 11;
            nEdges = 9;
            std::cout << "Rat10AndTarget SkelWorld created" << std::endl;
            break;

        case TargetOnly:
            nNodes = 1;
            nEdges = 0; 
            std::cout << "TargetOnly SkelWorld created" << std::endl;
            break;

        case RobotBolts:
            nNodes = 6;
            nEdges = 0; 
            std::cout << "RobotBolts SkelWorld created" << std::endl;
            break;

        case Rat10Target2:
            nNodes = 12;
            nEdges = 10; 
            std::cout << "Rat10Target2 SkelWorld created" << std::endl;
            break;

        case CalibrationFourCorners:
            nNodes = 4;
            nEdges = 4;
            std::cout << "CalibrationFourCorners SkelWorld Created" << std::endl;
            break;

        default:
            std::cout << "SkelEnum not recognized" << std::endl;
    }
    genericSkelWorldSetup();
}

void SkelWorld::genericSkelWorldSetup()
{
    g.x = std::vector<float> (nNodes, 0.0);
    g.y = std::vector<float> (nNodes, 0.0);
    g.z = std::vector<float> (nNodes, 0.0);

    g.px = std::vector<float*> (nNodes, nullptr);
    g.py = std::vector<float*> (nNodes, nullptr);
    g.pz = std::vector<float*> (nNodes, nullptr);

    for (int i=0; i<nNodes; i++)
    {
        g.px[i] = &g.x[i];
        g.py[i] = &g.y[i];
        g.pz[i] = &g.z[i];
    }

    activeKeyPoint = 0;
    g.isLabeled = std::vector<bool> (nNodes, false);
}

