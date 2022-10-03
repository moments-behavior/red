

static void plot_keypoints(LabelManager* labelMgr, int cam_idx, uint64_t current_frame, int* draw_id)
{
    if (labelMgr->activeFrameData[cam_idx])
    {
        int this_draw_id = *draw_id;
        for (int node=0; node<labelMgr->nNodes; node++)
        {
            // plot node if it is labeled
            if (cam->activeFrameData->g.isLabeled[node])
            {
                
                ImVec4 nodeColor;
                nodeColor.w = 1.0f;
                nodeColor.x = cam->frameData->nodeColors.at(cam->frameData->nodeColorIdx[node])[0];
                nodeColor.y = cam->frameData->nodeColors.at(cam->frameData->nodeColorIdx[node])[1];
                nodeColor.z = cam->frameData->nodeColors.at(cam->frameData->nodeColorIdx[node])[2];

                ImPlot::DragPoint(this_draw_id, cam->activeFrameData->g.px[node], cam->activeFrameData->g.py[node], nodeColor);
                this_draw_id++;

                // TODO: test if current point is grabbed

            }
        }

        for (int edge=0; edge<cam->frameData->nEdges; edge++)
        {
            auto[a,b] = cam->frameData->edges[edge];

            if (cam->activeFrameData->g.isLabeled[a] && cam->activeFrameData->g.isLabeled[b])
            {
                double xs[2] {cam->activeFrameData->g.x[a], cam->activeFrameData->g.x[b]};
                double ys[2] {cam->activeFrameData->g.y[a], cam->activeFrameData->g.y[b]};
                ImPlot::PlotLine("##line", xs, ys, 2);
            }
        }

        (*draw_id) = this_draw_id;
    }
}