#include "FrameData2D.h"

FrameData2D::FrameData2D(FrameData2D *frameData)
{
    nNodes = frameData->nNodes;
    nEdges = frameData->nEdges;

    g.x = frameData->g.x;
    g.y = frameData->g.y;

    g.px = std::vector<double*> (nNodes, nullptr);
    g.py = std::vector<double*> (nNodes, nullptr);

    for (int i=0; i<nNodes; i++)
    {
        g.px[i] = &g.x[i];
        g.py[i] = &g.y[i];
    }

    edges = frameData->edges;
    nodeNames = frameData->nodeNames;

    activeKeyPoint = frameData->activeKeyPoint;
    g.isLabeled = frameData->g.isLabeled;
}


FrameData2D::FrameData2D(SkelEnum skelEnum)
{
    cout << "FrameData2D Constructor called: skelEnum = " << skelEnum << endl;
    switch (skelEnum) {
        case Rat13:
            nNodes = 11;
            nEdges = 10;

            edges = std::vector<std::tuple<int, int>> (nEdges);
            edges[0] = {0, 2};
            edges[1] = {1, 2};
            edges[2] = {2, 3};
            edges[3] = {3, 4};
            edges[4] = {4, 5};
            edges[5] = {5, 6};
            edges[6] = {3, 7};
            edges[7] = {3, 9};
            edges[8] = {4, 8};
            edges[9] = {4, 10};

            nodeNames.push_back("EarL");
            nodeNames.push_back("EarR");
            nodeNames.push_back("Nose");
            nodeNames.push_back("Neck");
            nodeNames.push_back("TailBase");
            nodeNames.push_back("TailMid");
            nodeNames.push_back("TailTip");
            nodeNames.push_back("FrontPawL");
            nodeNames.push_back("RearPawL");
            nodeNames.push_back("FrontPawR");
            nodeNames.push_back("RearPawR");

            std::cout << "FrameData2D constructor: Rat13 Skel created" << std::endl;
            break;

        case Rat22:
            std::cout << "FrameData2D constructor: Rat22 Skel not yet supported" << std::endl;
            break;

        case Rat22Target2:
            nNodes = 24;
            nEdges = 22;

            edges = std::vector<std::tuple<int, int>> (nEdges);
            edges[0] = {0, 1};
            edges[1] = {0, 2};
            edges[2] = {0, 3};
            edges[3] = {1, 2};
            edges[4] = {3, 4};
            edges[5] = {3, 6};
            edges[6] = {3, 10};
            edges[7] = {4, 5};
            edges[8] = {4, 14};
            edges[9] = {4, 18};
            edges[10] = {6, 7};
            edges[11] = {7, 8};
            edges[12] = {8, 9};
            edges[13] = {10, 11};
            edges[14] = {11, 12};
            edges[15] = {12, 13};
            edges[16] = {14, 15};
            edges[17] = {15, 16};
            edges[18] = {16, 17};
            edges[19] = {18, 19};
            edges[20] = {19, 20};
            edges[21] = {20, 21};

            nodeNames.push_back("Snout");
            nodeNames.push_back("EarL");
            nodeNames.push_back("EarR");
            nodeNames.push_back("SpineF");
            nodeNames.push_back("SpineL");
            nodeNames.push_back("TaleBase");
            nodeNames.push_back("ShoulderL");
            nodeNames.push_back("ElbowL");
            nodeNames.push_back("WristL");
            nodeNames.push_back("HandL");
            nodeNames.push_back("ShoulderR");
            nodeNames.push_back("ElbowR");
            nodeNames.push_back("WristR");
            nodeNames.push_back("HandR");
            nodeNames.push_back("HipL");
            nodeNames.push_back("KneeL");
            nodeNames.push_back("AnkleL");
            nodeNames.push_back("FootL");
            nodeNames.push_back("HipR");
            nodeNames.push_back("KneeR");
            nodeNames.push_back("AnkleR");
            nodeNames.push_back("FootR");
            nodeNames.push_back("Target1");
            nodeNames.push_back("Target2");

            nodeColors.push_back(vector<float>{0.0f, 0.0f, 1.0f});   // blue
            nodeColors.push_back(vector<float>{0.5f, 0.5f, 0.5f});   // gray
            nodeColors.push_back(vector<float>{1.0f, 0.0f, 0.0f});   // red
            nodeColors.push_back(vector<float>{0.0f, 1.0f, 0.0f});   // green
            nodeColors.push_back(vector<float>{1.0f, 0.0f, 1.0f});   // magenta
            nodeColors.push_back(vector<float>{0.0f, 1.0f, 1.0f});   // cyan
            nodeColors.push_back(vector<float>{1.0f, 1.0f, 0.0f});   // yellow
            nodeColors.push_back(vector<float>{0.9f, 0.9f, 0.9f});   // white

            nodeColorIdx = vector<int> (nNodes, 0);
            nodeColorIdx[0] = 0;
            nodeColorIdx[1] = 0;
            nodeColorIdx[2] = 0;
            nodeColorIdx[3] = 1;
            nodeColorIdx[4] = 1;
            nodeColorIdx[5] = 6;
            nodeColorIdx[6] = 2;
            nodeColorIdx[7] = 2;
            nodeColorIdx[8] = 2;
            nodeColorIdx[9] = 2;
            nodeColorIdx[10] = 3;
            nodeColorIdx[11] = 3;
            nodeColorIdx[12] = 3;
            nodeColorIdx[13] = 3;
            nodeColorIdx[14] = 4;
            nodeColorIdx[15] = 4;
            nodeColorIdx[16] = 4;
            nodeColorIdx[17] = 4;
            nodeColorIdx[18] = 5;
            nodeColorIdx[19] = 5;
            nodeColorIdx[20] = 5;
            nodeColorIdx[21] = 5;
            nodeColorIdx[22] = 7;
            nodeColorIdx[23] = 7;

            std::cout << "FrameData2D constructor: Rat22Target2 Skel created" << std::endl;
            break;

        case Rat23:
            nNodes = 23;
            nEdges = 23;

            edges = std::vector<std::tuple<int, int>> (nEdges);
            edges[0] = {0, 1};
            edges[1] = {0, 2};
            edges[2] = {0, 3};
            edges[3] = {1, 2};
            edges[4] = {3, 4};
            edges[5] = {3, 7};
            edges[6] = {3, 11};
            edges[7] = {4, 5};
            edges[8] = {5, 6};
            edges[9] = {5, 15};
            edges[10] = {5, 19};
            edges[11] = {7, 8};
            edges[12] = {8, 9};
            edges[13] = {9, 10};
            edges[14] = {11, 12};
            edges[15] = {12, 13};
            edges[16] = {13, 14};
            edges[17] = {15, 16};
            edges[18] = {16, 17};
            edges[19] = {17, 18};
            edges[20] = {19, 20};
            edges[21] = {20, 21};
            edges[22] = {21, 22};

            nodeNames.push_back("Snout");
            nodeNames.push_back("EarL");
            nodeNames.push_back("EarR");
            nodeNames.push_back("SpineF");
            nodeNames.push_back("SpineM");
            nodeNames.push_back("SpineL");
            nodeNames.push_back("TaleBase");
            nodeNames.push_back("ShoulderL");
            nodeNames.push_back("ElbowL");
            nodeNames.push_back("WristL");
            nodeNames.push_back("HandL");
            nodeNames.push_back("ShoulderR");
            nodeNames.push_back("ElbowR");
            nodeNames.push_back("WristR");
            nodeNames.push_back("HandR");
            nodeNames.push_back("HipL");
            nodeNames.push_back("KneeL");
            nodeNames.push_back("AnkleL");
            nodeNames.push_back("FootL");
            nodeNames.push_back("HipR");
            nodeNames.push_back("KneeR");
            nodeNames.push_back("AnkleR");
            nodeNames.push_back("FootR");

            nodeColors.push_back(vector<float>{0.0f, 0.0f, 1.0f});   // blue
            nodeColors.push_back(vector<float>{0.5f, 0.5f, 0.5f});   // gray
            nodeColors.push_back(vector<float>{1.0f, 0.0f, 0.0f});   // red
            nodeColors.push_back(vector<float>{0.0f, 1.0f, 0.0f});   // green
            nodeColors.push_back(vector<float>{1.0f, 0.0f, 1.0f});   // magenta
            nodeColors.push_back(vector<float>{0.0f, 1.0f, 1.0f});   // cyan
            nodeColors.push_back(vector<float>{1.0f, 1.0f, 0.0f});   // yellow

            nodeColorIdx = vector<int> (nNodes, 0);
            nodeColorIdx[0] = 0;
            nodeColorIdx[1] = 0;
            nodeColorIdx[2] = 0;
            nodeColorIdx[3] = 1;
            nodeColorIdx[4] = 1;
            nodeColorIdx[5] = 1;
            nodeColorIdx[6] = 6;
            nodeColorIdx[7] = 2;
            nodeColorIdx[8] = 2;
            nodeColorIdx[9] = 2;
            nodeColorIdx[10] = 2;
            nodeColorIdx[11] = 3;
            nodeColorIdx[12] = 3;
            nodeColorIdx[13] = 3;
            nodeColorIdx[14] = 3;
            nodeColorIdx[15] = 4;
            nodeColorIdx[16] = 4;
            nodeColorIdx[17] = 4;
            nodeColorIdx[18] = 4;
            nodeColorIdx[19] = 5;
            nodeColorIdx[20] = 5;
            nodeColorIdx[21] = 5;
            nodeColorIdx[22] = 5;

            std::cout << "FrameData2D constructor: Rat23 Skel created" << std::endl;
            break;

        case Rat23Target2:
            nNodes = 25;
            nEdges = 23;

            edges = std::vector<std::tuple<int, int>> (nEdges);
            edges[0] = {0, 1};
            edges[1] = {0, 2};
            edges[2] = {0, 3};
            edges[3] = {1, 2};
            edges[4] = {3, 4};
            edges[5] = {3, 7};
            edges[6] = {3, 11};
            edges[7] = {4, 5};
            edges[8] = {5, 6};
            edges[9] = {5, 15};
            edges[10] = {5, 19};
            edges[11] = {7, 8};
            edges[12] = {8, 9};
            edges[13] = {9, 10};
            edges[14] = {11, 12};
            edges[15] = {12, 13};
            edges[16] = {13, 14};
            edges[17] = {15, 16};
            edges[18] = {16, 17};
            edges[19] = {17, 18};
            edges[20] = {19, 20};
            edges[21] = {20, 21};
            edges[22] = {21, 22};

            nodeNames.push_back("Snout");
            nodeNames.push_back("EarL");
            nodeNames.push_back("EarR");
            nodeNames.push_back("SpineF");
            nodeNames.push_back("SpineM");
            nodeNames.push_back("SpineL");
            nodeNames.push_back("TaleBase");
            nodeNames.push_back("ShoulderL");
            nodeNames.push_back("ElbowL");
            nodeNames.push_back("WristL");
            nodeNames.push_back("HandL");
            nodeNames.push_back("ShoulderR");
            nodeNames.push_back("ElbowR");
            nodeNames.push_back("WristR");
            nodeNames.push_back("HandR");
            nodeNames.push_back("HipL");
            nodeNames.push_back("KneeL");
            nodeNames.push_back("AnkleL");
            nodeNames.push_back("FootL");
            nodeNames.push_back("HipR");
            nodeNames.push_back("KneeR");
            nodeNames.push_back("AnkleR");
            nodeNames.push_back("FootR");
            nodeNames.push_back("Target1");
            nodeNames.push_back("Target2");

            nodeColors.push_back(vector<float>{0.0f, 0.0f, 1.0f});   // blue
            nodeColors.push_back(vector<float>{0.5f, 0.5f, 0.5f});   // gray
            nodeColors.push_back(vector<float>{1.0f, 0.0f, 0.0f});   // red
            nodeColors.push_back(vector<float>{0.0f, 1.0f, 0.0f});   // green
            nodeColors.push_back(vector<float>{1.0f, 0.0f, 1.0f});   // magenta
            nodeColors.push_back(vector<float>{0.0f, 1.0f, 1.0f});   // cyan
            nodeColors.push_back(vector<float>{1.0f, 1.0f, 0.0f});   // yellow
            nodeColors.push_back(vector<float>{0.0f, 0.0f, 0.0f});   // black

            nodeColorIdx = vector<int> (nNodes, 0);
            nodeColorIdx[0] = 0;
            nodeColorIdx[1] = 0;
            nodeColorIdx[2] = 0;
            nodeColorIdx[3] = 1;
            nodeColorIdx[4] = 1;
            nodeColorIdx[5] = 1;
            nodeColorIdx[6] = 6;
            nodeColorIdx[7] = 2;
            nodeColorIdx[8] = 2;
            nodeColorIdx[9] = 2;
            nodeColorIdx[10] = 2;
            nodeColorIdx[11] = 3;
            nodeColorIdx[12] = 3;
            nodeColorIdx[13] = 3;
            nodeColorIdx[14] = 3;
            nodeColorIdx[15] = 4;
            nodeColorIdx[16] = 4;
            nodeColorIdx[17] = 4;
            nodeColorIdx[18] = 4;
            nodeColorIdx[19] = 5;
            nodeColorIdx[20] = 5;
            nodeColorIdx[21] = 5;
            nodeColorIdx[22] = 5;
            nodeColorIdx[23] = 7;
            nodeColorIdx[24] = 7;

            std::cout << "FrameData2D constructor: Rat23Target2 Skel created" << std::endl;
            break;

        case Rat10AndTarget:
            nNodes = 11;
            nEdges = 9;

            nodeNames.push_back("Snout");
            nodeNames.push_back("EarL");
            nodeNames.push_back("EarR");
            nodeNames.push_back("SpineF");
            nodeNames.push_back("ShoulderL");
            nodeNames.push_back("ShoulderR");
            nodeNames.push_back("SpineL");
            nodeNames.push_back("HipL");
            nodeNames.push_back("HipR");
            nodeNames.push_back("TaleBase");
            nodeNames.push_back("Target");

            edges = std::vector<std::tuple<int, int>> (nEdges);
            edges[0] = {0, 1};
            edges[1] = {0, 2};
            edges[2] = {0, 3};
            edges[3] = {3, 4};
            edges[4] = {3, 5};
            edges[5] = {3, 6};
            edges[6] = {6, 7};
            edges[7] = {6, 8};
            edges[8] = {6, 9};

            nodeColors.push_back(vector<float>{1.0f, 1.0f, 0.5f});   // blue
            nodeColors.push_back(vector<float>{0.5f, 0.5f, 0.5f});   // gray
            nodeColors.push_back(vector<float>{1.0f, 0.0f, 0.0f});   // red
            nodeColors.push_back(vector<float>{0.0f, 1.0f, 0.0f});   // green
            nodeColors.push_back(vector<float>{1.0f, 0.0f, 1.0f});   // magenta
            nodeColors.push_back(vector<float>{0.0f, 1.0f, 1.0f});   // cyan
            nodeColors.push_back(vector<float>{1.0f, 1.0f, 0.0f});   // yellow
            nodeColors.push_back(vector<float>{0.0f, 0.0f, 0.0f});   // black

            nodeColorIdx = vector<int> (nNodes, 0);
            nodeColorIdx[0] = 0;
            nodeColorIdx[1] = 0;
            nodeColorIdx[2] = 0;
            nodeColorIdx[3] = 1;
            nodeColorIdx[4] = 2;
            nodeColorIdx[5] = 3;
            nodeColorIdx[6] = 1;
            nodeColorIdx[7] = 4;
            nodeColorIdx[8] = 5;
            nodeColorIdx[9] = 6;
            nodeColorIdx[10] = 7;

            std::cout << "FrameData2D constructor: Rat10AndTarget Skel created" << std::endl;
            break;

        case Rat10Target2:
            nNodes = 12;
            nEdges = 10;

            nodeNames.push_back("Snout");
            nodeNames.push_back("EarL");
            nodeNames.push_back("EarR");
            nodeNames.push_back("Neck");
            nodeNames.push_back("SpineL");
            nodeNames.push_back("TailBase");
            nodeNames.push_back("HandL");
            nodeNames.push_back("HandR");
            nodeNames.push_back("FootL");
            nodeNames.push_back("FootR");
            nodeNames.push_back("Target1");
            nodeNames.push_back("Target2");

            edges = std::vector<std::tuple<int, int>> (nEdges);
            edges[0] = {0, 1};
            edges[1] = {0, 2};
            edges[2] = {0, 3};
            edges[3] = {1, 2};
            edges[4] = {3, 4};
            edges[5] = {4, 5};
            edges[6] = {3, 6};
            edges[7] = {3, 7};
            edges[8] = {4, 8};
            edges[9] = {4, 9};


            nodeColors.push_back(vector<float>{0.8f, 0.8f, 1.0f});   // blue
            nodeColors.push_back(vector<float>{0.5f, 0.5f, 0.5f});   // gray
            nodeColors.push_back(vector<float>{1.0f, 0.0f, 0.0f});   // red
            nodeColors.push_back(vector<float>{0.0f, 1.0f, 0.0f});   // green
            nodeColors.push_back(vector<float>{1.0f, 0.0f, 1.0f});   // magenta
            nodeColors.push_back(vector<float>{0.0f, 1.0f, 1.0f});   // cyan
            nodeColors.push_back(vector<float>{1.0f, 1.0f, 0.0f});   // yellow
            nodeColors.push_back(vector<float>{0.3f, 0.3f, 0.1f});   // black

            nodeColorIdx = vector<int> (nNodes, 0);
            nodeColorIdx[0] = 0;
            nodeColorIdx[1] = 0;
            nodeColorIdx[2] = 0;
            nodeColorIdx[3] = 1;
            nodeColorIdx[4] = 1;
            nodeColorIdx[5] = 6;
            nodeColorIdx[6] = 2;
            nodeColorIdx[7] = 3;
            nodeColorIdx[8] = 4;
            nodeColorIdx[9] = 5;
            nodeColorIdx[10] = 7;
            nodeColorIdx[11] = 7;

            std::cout << "FrameData2D constructor: Rat10Target2 Skel created" << std::endl;
            break;

        case TargetOnly:
            nNodes = 1;
            nEdges = 0;

            nodeNames.push_back("Target");

            nodeColors.push_back(vector<float>{1.0f, 0.0f, 1.0f});   // magenta

            nodeColorIdx = vector<int> (nNodes, 0);
            nodeColorIdx[0] = 0;
            std::cout << "FrameData2D constructor: TargetOnly Skel created" << std::endl;
            break;

        case RobotBolts:
            nNodes = 6;
            nEdges = 0;

            nodeNames.push_back("Bumblebee1");
            nodeNames.push_back("Bumblebee2");
            nodeNames.push_back("Grimlock1");
            nodeNames.push_back("Grimlock2");
            nodeNames.push_back("Optimus1");
            nodeNames.push_back("Optimus2");


            nodeColors.push_back(vector<float>{1.0f, 0.0f, 1.0f});   // magenta
            nodeColors.push_back(vector<float>{1.0f, 0.0f, 1.0f});   // magenta
            nodeColors.push_back(vector<float>{1.0f, 0.0f, 1.0f});   // magenta
            nodeColors.push_back(vector<float>{1.0f, 0.0f, 1.0f});   // magenta
            nodeColors.push_back(vector<float>{1.0f, 0.0f, 1.0f});   // magenta
            nodeColors.push_back(vector<float>{1.0f, 0.0f, 1.0f});   // magenta

            nodeColorIdx = vector<int> (nNodes, 0);

            std::cout << "FrameData2D constructor: RobotBolts Skel created" << std::endl;
            break;

        case CalibrationFourCorners:
            nNodes = 4;
            nEdges = 4;

            nodeNames.push_back("TopLeft");
            nodeNames.push_back("TopRight");
            nodeNames.push_back("BottomRight");
            nodeNames.push_back("BottomLeft");

            nodeColors.push_back(vector<float>{1.0f, 0.0f, 1.0f});   // magenta
            nodeColors.push_back(vector<float>{1.0f, 0.0f, 1.0f});   // magenta
            nodeColors.push_back(vector<float>{1.0f, 0.0f, 1.0f});   // magenta
            nodeColors.push_back(vector<float>{1.0f, 0.0f, 1.0f});   // magenta

            nodeColorIdx = vector<int> (nNodes, 0);

            edges = std::vector<std::tuple<int, int>> (nEdges);
            edges[0] = {0, 1};
            edges[1] = {1, 2};
            edges[2] = {2, 3};
            edges[3] = {3, 0};

            std::cout << "FrameData2D constructor: CalibrationFourCorners Skel created" << std::endl;
            break;


        default:
            std::cout << "FrameData2D constructor: SkelEnum not recognized" << std::endl;
    }

    genericSkelSetup();
}

void FrameData2D::genericSkelSetup()
{
    g.x = std::vector<double> (nNodes, 0.0);
    g.y = std::vector<double> (nNodes, 0.0);

    g.px = std::vector<double*> (nNodes, nullptr);
    g.py = std::vector<double*> (nNodes, nullptr);

    for (int i=0; i<nNodes; i++)
    {
        g.px[i] = &g.x[i];
        g.py[i] = &g.y[i];
    }

    activeKeyPoint = 0;
    g.isLabeled = std::vector<bool> (nNodes, false);
}

