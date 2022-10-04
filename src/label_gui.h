#ifndef labeling_gui
#define labeling_gui

#include <stdlib.h>
#include <stdio.h>
#include "LabelManager.h"
#include "implot.h"


static void plot_keypoints(LabelManager* labelMgr, int cam_idx, uint64_t current_frame, int* draw_id)
{
    if (labelMgr->activeFrameData2DList[cam_idx])
    {
        int this_draw_id = *draw_id;
        for (int node=0; node<labelMgr->nNodes; node++)
        {
            // plot node if it is labeled
            if (labelMgr->activeFrameData2DList[cam_idx]->g.isLabeled[node])
            {
                
                ImVec4 nodeColor;
                nodeColor.w = 1.0f;
                nodeColor.x = labelMgr->frameData2DList[cam_idx]->nodeColors.at(labelMgr->frameData2DList[cam_idx]->nodeColorIdx[node])[0];
                nodeColor.y = labelMgr->frameData2DList[cam_idx]->nodeColors.at(labelMgr->frameData2DList[cam_idx]->nodeColorIdx[node])[1];
                nodeColor.z = labelMgr->frameData2DList[cam_idx]->nodeColors.at(labelMgr->frameData2DList[cam_idx]->nodeColorIdx[node])[2];

                ImPlot::DragPoint(this_draw_id, labelMgr->activeFrameData2DList[cam_idx]->g.px[node], labelMgr->activeFrameData2DList[cam_idx]->g.py[node], nodeColor);
                this_draw_id++;
            }
        }

        for (int edge=0; edge<labelMgr->nEdges; edge++)
        {
            auto[a,b] = labelMgr->frameData2DList[cam_idx]->edges[edge];

            if (labelMgr->activeFrameData2DList[cam_idx]->g.isLabeled[a] && labelMgr->activeFrameData2DList[cam_idx]->g.isLabeled[b])
            {
                double xs[2] {labelMgr->activeFrameData2DList[cam_idx]->g.x[a], labelMgr->activeFrameData2DList[cam_idx]->g.x[b]};
                double ys[2] {labelMgr->activeFrameData2DList[cam_idx]->g.y[a], labelMgr->activeFrameData2DList[cam_idx]->g.y[b]};
                ImPlot::PlotLine("##line", xs, ys, 2);
            }
        }

        (*draw_id) = this_draw_id;
    }
}

#endif
