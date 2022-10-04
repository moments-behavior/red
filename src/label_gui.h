#ifndef labeling_gui
#define labeling_gui

#include "Camera.h"
#include "implot.h"
#include "LabelManager.h"

#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <time.h>
#include <ctime>
#include <numeric>


struct point2F{
    float x;
    float y;
};

struct point3F{
    float x;
    float y;
    float z;
};



bool MySliderU64(const char *label, uint64_t* value, uint64_t min, uint64_t max, const char* format = "%lu")
{
    return ImGui::SliderScalar(label, ImGuiDataType_U64, value, &min, &max, format);
}

static void plot_perimeter(Camera* cam)
{
    ImVec4 fill_color = ImVec4(1.0, 1.0, 1.0,1.0);
    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 6.0, fill_color);
    ImPlot::SetNextLineStyle(fill_color, 3.0);

    std::string name = "perimeter " + cam->name;
    ImPlot::PlotLine(name.c_str(), &cam->arena_x[0], &cam->arena_y[0], cam->arena_x.size());
}

static void set_active_skel3D(LabelManager* labelMgr, uint64_t current_frame)
{
    labelMgr->skelWorldMapIter = std::map<uint, SkelWorld*>::iterator(labelMgr->skelWorldMap.lower_bound(current_frame));
    if (labelMgr->skelWorldMapIter == labelMgr->skelWorldMap.end() || current_frame < labelMgr->skelWorldMapIter->first) { labelMgr->activeSkelWorld = nullptr; }
    else { labelMgr->activeSkelWorld = labelMgr->skelWorldMapIter->second; }
}

static void set_active_skel2D(Camera* cam, uint64_t current_frame)
{
    cam->frameDataMapIter = std::map<uint, FrameData2D*>::iterator(cam->frameDataMap.lower_bound(current_frame));  // get iterator to check if skel exists
    if (cam->frameDataMapIter == cam->frameDataMap.end() || current_frame < cam->frameDataMapIter->first) { cam->activeFrameData = nullptr; }
    else { cam->activeFrameData = cam->frameDataMapIter->second; }
}

static void plot_keypoints(LabelManager* labelMgr, Camera* cam, uint64_t current_frame, int* draw_id)
{
    if (cam->activeFrameData)
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


int get_consensus_count(const int a[], int n)
{
    const int a0 = a[0];
    for (int i=1; i<n; i++)
    {
        if (a[i] != a0)
            return 0;
    }
    return a0;
}

int get_consensus_count(std::vector<int> a)
{
    const int a0 = a[0];
    const int n = a.size();
    for (int i=1; i<n; i++)
    {
        if (a[i] != a0)
            return 0;
    }
    return a0;
}

int get_consensus_count(std::vector<int> a, int len)
{
    const int a0 = a[0];
    const int n = len;
    for (int i=1; i<n; i++)
    {
        if (a[i] != a0)
            return 0;
    }
    return a0;
}



int factorial(int n)
{
    int val = n;
    for (int i=1; i<n; i++)
    {
        val = val * i;
    }
    return val;
}



static void clear_cam_correspondence(int frame, int nBalls, std::vector<std::vector<float>> x, std::vector<std::vector<float>> y, std::vector<Camera*> cams)
{
    const int CLEARVIEW_CAMS[] = {0, 2, 4, 6};
    const int NUM_CLEARVIEW_CAMS = 4;

    if (nBalls == 1)
    {
        std::vector<cv::Mat> sfmPoints2d;
        std::vector<cv::Mat> projection_matrices;
        cv::Mat output;

        for (int i=0; i<NUM_CLEARVIEW_CAMS; i++)
        {
            int cam_i = CLEARVIEW_CAMS[i];
            cv::Mat point = (cv::Mat_<float>(2, 1) << x[i][0], y[i][0]);
            cv::Mat pointUndistort;
            cv::undistortPoints(point, pointUndistort, cams[cam_i]->cvp.K, cams[cam_i]->cvp.distCoeffs, cv::noArray(), cams[cam_i]->cvp.K);
            sfmPoints2d.push_back(pointUndistort.reshape(1, 2));
            projection_matrices.push_back(cams[cam_i]->cvp.projectionMat);
        }

        cv::sfm::triangulatePoints(sfmPoints2d, projection_matrices, output);
        output.convertTo(output, CV_32F);
        float x3d, y3d, z3d;
        x3d = output.at<float>(0);
        y3d = output.at<float>(1);
        z3d = output.at<float>(2);

        std::cout << "frame: " << frame << "   x3d: " << x3d << "   y3d: " << y3d << "   z3d: " << z3d << "\n";
    }
    else if (nBalls == 2)
    {
        std::vector<std::vector<short>> ball_id;
        for (int i=0; i<NUM_CLEARVIEW_CAMS; i++)
        {
            std::vector<short> b(nBalls);
            std::iota (std::begin(b), std::end(b), 0);
            ball_id.push_back(b);
        }


    }
    else
    {
        std::cout << "frame: " << frame << "   nBalls: " << nBalls << " not yet supported" << "\n";
    }

}





static void triangulate_and_reproject_with_error(std::vector<float> x, std::vector<float> y, std::vector<Camera*> cams)
{
    int nCams = cams.size();

    std::vector<cv::Mat> sfmPoints2d;
    std::vector<cv::Mat> projection_matrices;
    cv::Mat output;



    for (int i=0; i<nCams; i++)
    {
        cv::Mat point = (cv::Mat_<float>(2, 1) << x[i], y[i]);
        cv::Mat pointUndistort;
        cv::undistortPoints(point, pointUndistort, cams[i]->cvp.K, cams[i]->cvp.distCoeffs, cv::noArray(), cams[i]->cvp.K);
        sfmPoints2d.push_back(pointUndistort.reshape(1, 2));
        projection_matrices.push_back(cams[i]->cvp.projectionMat);
    }

    cv::sfm::triangulatePoints(sfmPoints2d, projection_matrices, output);
    output.convertTo(output, CV_32F);

    float x3d, y3d, z3d;
    x3d = output.at<float>(0);
    y3d = output.at<float>(1);
    z3d = output.at<float>(2);

    std::vector<float> x_out;
    std::vector<float> y_out;

    for (int i=0; i<nCams; i++)
    {
        cv::Mat imagePts;
        cv::projectPoints(output, cams[i]->cvp.rvec, cams[i]->cvp.tvec, cams[i]->cvp.K, cams[i]->cvp.distCoeffs, imagePts);
        float x2d = imagePts.at<float>(0, 0);
        float y2d = imagePts.at<float>(0, 1);

        x_out.push_back(x2d);
        y_out.push_back(y2d);

    }

    std::vector<float> reprojection_error;
    for (int i=0; i<nCams; i++)
    {
        std::cout << cams[i]->name << "   x: " << x[i] << "   x_out: " << x_out[i] << "\n";
        float error = std::pow(std::pow(x[i] - x_out[i], 2) + std::pow(y[i] - y_out[i], 2), 0.5);

        reprojection_error.push_back(error);
    }

    for (int i=0; i<nCams; i++)
    {
        std::cout << cams[i]->name << " reprojection error: " << reprojection_error[i] << "\n";
    }
}

float euclideanDist(cv::Point2f& a, cv::Point2f& b)
{
    cv::Point2f diff = a - b;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

float average(std::vector<float> const& v){
    if(v.empty()){
        return 0;
    }

    auto const count = static_cast<float>(v.size());
    return std::reduce(v.begin(), v.end()) / count;
}

void print(const std::vector<int>& v)
{
    for (int e : v) {
        std::cout << " " << e;
    }
    std::cout << std::endl;
}

void print(const std::vector<float>& v)
{
    for (float e : v) {
        std::cout << " " << e;
    }
    std::cout << std::endl;
}


static cv::Point3f triangulate_and_reproject_with_error(std::vector<cv::Point2f> inPts2d, std::vector<Camera*> cams)
{
    std::vector<cv::Mat> sfmPoints2d;
    std::vector<cv::Mat> projection_matrices;
    cv::Mat output;

    int nCams = cams.size();
    for (int i=0; i<nCams; i++)
    {
        cv::Mat point = (cv::Mat_<float>(2, 1) << inPts2d[i].x, inPts2d[i].y);
        cv::Mat pointUndistort;
        cv::undistortPoints(point, pointUndistort, cams[i]->cvp.K, cams[i]->cvp.distCoeffs, cv::noArray(), cams[i]->cvp.K);
        sfmPoints2d.push_back(pointUndistort.reshape(1, 2));
        projection_matrices.push_back(cams[i]->cvp.projectionMat);
    }

    cv::sfm::triangulatePoints(sfmPoints2d, projection_matrices, output);
    output.convertTo(output, CV_32F);
    cv::Point3f pt3f;
    pt3f.x = output.at<float>(0);
    pt3f.y = output.at<float>(1);
    pt3f.z = output.at<float>(2);

    std::vector<cv::Point2f> outPts2d;
    std::vector<float> error;

    for (int i=0; i<nCams; i++)
    {
        std::vector<cv::Point2f> p;
        cv::projectPoints(output, cams[i]->cvp.rvec, cams[i]->cvp.tvec, cams[i]->cvp.K, cams[i]->cvp.distCoeffs, p);
        outPts2d.push_back(p[0]);
    }

    for (int i=0; i<nCams; i++)
    {
        error.push_back(euclideanDist(inPts2d[i], outPts2d[i]));
    }

    return pt3f;
}

static std::vector<float> triangulate_and_reproject_with_error(std::vector<cv::Point2f> inPts2d, std::vector<cv::Point2f> &outPts2d,
                                                               cv::Point3f &pt3d, std::vector<Camera*> cams)
{
    std::vector<cv::Mat> sfmPoints2d;
    std::vector<cv::Mat> projection_matrices;
    cv::Mat output;

    int nCams = cams.size();
    for (int i=0; i<nCams; i++)
    {
        cv::Mat point = (cv::Mat_<float>(2, 1) << inPts2d[i].x, inPts2d[i].y);
        cv::Mat pointUndistort;
        cv::undistortPoints(point, pointUndistort, cams[i]->cvp.K, cams[i]->cvp.distCoeffs, cv::noArray(), cams[i]->cvp.K);
        sfmPoints2d.push_back(pointUndistort.reshape(1, 2));
        projection_matrices.push_back(cams[i]->cvp.projectionMat);
    }

    cv::sfm::triangulatePoints(sfmPoints2d, projection_matrices, output);
    output.convertTo(output, CV_32F);
    cv::Point3f pt3f;
    pt3f.x = output.at<float>(0);
    pt3f.y = output.at<float>(1);
    pt3f.z = output.at<float>(2);

    std::vector<float> error;

    for (int i=0; i<nCams; i++)
    {
        std::vector<cv::Point2f> p;
        cv::projectPoints(output, cams[i]->cvp.rvec, cams[i]->cvp.tvec, cams[i]->cvp.K, cams[i]->cvp.distCoeffs, p);
        outPts2d.push_back(p[0]);
    }

    for (int i=0; i<nCams; i++)
    {
        error.push_back(euclideanDist(inPts2d[i], outPts2d[i]));
    }

    return error;
}

static std::vector<float> triangulate_and_reproject_with_error(std::vector<cv::Point2f> inPts2d, std::vector<cv::Point2f> &outPts2d, std::vector<Camera*> cams)
{
    std::vector<cv::Mat> sfmPoints2d;
    std::vector<cv::Mat> projection_matrices;
    cv::Mat output;

    int nCams = cams.size();
    for (int i=0; i<nCams; i++)
    {
        cv::Mat point = (cv::Mat_<float>(2, 1) << inPts2d[i].x, inPts2d[i].y);
        cv::Mat pointUndistort;
        cv::undistortPoints(point, pointUndistort, cams[i]->cvp.K, cams[i]->cvp.distCoeffs, cv::noArray(), cams[i]->cvp.K);
        sfmPoints2d.push_back(pointUndistort.reshape(1, 2));
        projection_matrices.push_back(cams[i]->cvp.projectionMat);
    }

    cv::sfm::triangulatePoints(sfmPoints2d, projection_matrices, output);
    output.convertTo(output, CV_32F);

    std::vector<float> error;

    for (int i=0; i<nCams; i++)
    {
        std::vector<cv::Point2f> p;
        cv::projectPoints(output, cams[i]->cvp.rvec, cams[i]->cvp.tvec, cams[i]->cvp.K, cams[i]->cvp.distCoeffs, p);
        outPts2d.push_back(p[0]);
    }

    for (int i=0; i<nCams; i++)
    {
        error.push_back(euclideanDist(inPts2d[i], outPts2d[i]));
    }

    return error;
}

static cv::Point3f triangulate_pt(std::vector<cv::Point2f> inPts2d, std::vector<Camera*> cams)
{
    std::vector<cv::Mat> sfmPoints2d;
    std::vector<cv::Mat> projection_matrices;
    cv::Mat output;

    int nCams = cams.size();
    for (int i=0; i<nCams; i++)
    {
        cv::Mat point = (cv::Mat_<float>(2, 1) << inPts2d[i].x, inPts2d[i].y);
        cv::Mat pointUndistort;
        cv::undistortPoints(point, pointUndistort, cams[i]->cvp.K, cams[i]->cvp.distCoeffs, cv::noArray(), cams[i]->cvp.K);
        sfmPoints2d.push_back(pointUndistort.reshape(1, 2));
        projection_matrices.push_back(cams[i]->cvp.projectionMat);
    }

    cv::sfm::triangulatePoints(sfmPoints2d, projection_matrices, output);
    output.convertTo(output, CV_32F);

    cv::Point3f pt3f;
    pt3f.x = output.at<float>(0);
    pt3f.y = output.at<float>(1);
    pt3f.z = output.at<float>(2);

    return pt3f;
}



static void triangulate_to_correspond(std::vector<std::vector<cv::Point2f>> xy_in, std::vector<cv::Point3f> &xyz_out, std::vector<Camera*> cams,
                                      std::vector<std::vector<int>> &ball_id, bool test_existing_ball_id=false)
{
    int nCams = xy_in.size();
    int nPts  = xy_in[0].size();
    

    if (test_existing_ball_id)
    {
        if (!ball_id.size() == 0)
        {
            std::vector<cv::Point3f> pts3d;
            std::vector<float> perm_errors;
            for (int j=0; j<nPts; j++)
            {
                std::vector<cv::Point2f> xy_set_in;
                std::vector<cv::Point2f> xy_set_out;
                for (int k=0; k<nCams; k++)
                {
                    xy_set_in.push_back(xy_in[k][ball_id[k][j]]);
                }
                cv::Point3f xyz;
                std::vector<float> errors = triangulate_and_reproject_with_error(xy_set_in, xy_set_out, xyz, cams);
                perm_errors.push_back(average(errors));
                pts3d.push_back(xyz);
            }

            float error = average(perm_errors);

            if (error < 5.0)
            {
                std::cout << "error from suggested ball indices: " << error << "\n";
                xyz_out = pts3d;
                return;
            }
            else
            {
                std::cout << "error from suggested ball indices: " << error << " -- too high -- going to exhaustive search" << "\n";
            }
        }
    }

    std::vector<int> v(nPts);
    std::vector<std::vector<int>> pt_perms;
    std::iota(std::begin(v), std::end(v), 0);

    // vector should be sorted at the beginning.
    do {
        pt_perms.push_back(v);
    } while (std::next_permutation(v.begin(), v.end()));

    int n_perms = pt_perms.size();

    std::vector<std::vector<int>> perm_idx;

    int c = nCams - 1;
    int n = std::pow(factorial(nPts), c);
    
    vector<int> idx(c);
    int curr_idx = 0;

    bool go = true;
    while (go)
    {
        perm_idx.push_back(idx);
        if (get_consensus_count(idx) == n_perms-1) { break; }

        for (int i=0; i<c; i++)
        {
            if (idx[i] == (n_perms-1))
            {
                idx[i] = 0;
            }
            else
            {
                idx[i] += 1;
                break;
            }
        }
    }

    std::vector<float> errors;
    for (int i=0; i<perm_idx.size(); i++)
    {

        std::vector<std::vector<int>> perms;
        for (int j=0; j<nCams; j++)
        {
            if (j==0)
            {
                perms.push_back(pt_perms[0]);
            }
            else{
                perms.push_back(pt_perms[perm_idx[i][j-1]]);
            }
        }

        std::vector<float> perm_errors;
        for (int j=0; j<nPts; j++)
        {
            std::vector<cv::Point2f> xy_set_in;
            std::vector<cv::Point2f> xy_set_out;
            for (int k=0; k<nCams; k++)
            {
                xy_set_in.push_back(xy_in[k][perms[k][j]]);
            }
            std::vector<float> errors = triangulate_and_reproject_with_error(xy_set_in, xy_set_out, cams);
            perm_errors.push_back(average(errors));
        }

        errors.push_back(average(perm_errors));

    }

    float min = *min_element(errors.begin(), errors.end());
    int i = std::min_element(errors.begin(),errors.end()) - errors.begin();

    // recover correct permutation
    std::vector<std::vector<int>> perms;
    for (int j=0; j<nCams; j++)
    {
        if (j==0)
        {
            perms.push_back(pt_perms[0]);
        }
        else{
            perms.push_back(pt_perms[perm_idx[i][j-1]]);
        }
    }
    ball_id = perms;

    // recover triangulation from this example

    std::vector<cv::Point3f> pts3d;
    for (int j=0; j<nPts; j++)
    {
        std::vector<cv::Point2f> xy_set_in;
        std::vector<cv::Point2f> xy_set_out;
        for (int k=0; k<nCams; k++)
        {
            xy_set_in.push_back(xy_in[k][perms[k][j]]);
        }
        cv::Point3f xyz;
        std::vector<float> errors = triangulate_and_reproject_with_error(xy_set_in, xy_set_out, xyz, cams);
        pts3d.push_back(xyz);
    }

    std::cout << "min error from exhaustive search of ball id combinations: " << min << "\n";
    xyz_out = pts3d;
    return;
}




static void project_one_point(cv::Mat output2, Camera* cam, uint64_t current_frame, std::map<uint, FrameData2D*>::iterator iter, ImVec4 color, int cam_idx){
    cv::Mat imagePts2;
    cv::Mat distCoeffs;
    
    cv::projectPoints(output2, cam->cvp.rvec, cam->cvp.tvec, cam->cvp.K, distCoeffs, imagePts2);
    double x1 = imagePts2.at<float>(0, 0);
    double y1 = (float)2200 - imagePts2.at<float>(0, 1);
    if (current_frame == 100) {
        std::cout << "x1: " << x1 << std::endl;
        std::cout << "y1: " << y1 << std::endl;
    }
    ImPlot::DragPoint(cam_idx+200, &x1, &y1, ImVec4(1.0, 0.1, 1.1, 1.0));
}

static void labeling_one_view(Camera* cam, uint64_t current_frame)
{

    if (ImPlot::IsPlotHovered())
    {
        cam->isImgHovered = true;
        // get pointer to active keypoint on activeFrameData (if it exists), otherwise get pointer to active keypoint on default frameData
        int *kp = (cam->activeFrameData) ? &cam->activeFrameData->activeKeyPoint : &cam->frameData->activeKeyPoint;

        // Use "Q" and "E" keys to scroll through and set active keypoint to label
        if (ImGui::IsKeyPressed(ImGuiKey_Q, false))
        {
            if (*kp <= 0) { *kp = 0; }
            else (*kp)--;
        }
        else if (ImGui::IsKeyPressed(ImGuiKey_E, false))
        {
            if (*kp >= cam->frameData->nNodes-1) { *kp = cam->frameData->nNodes-1; }
            else (*kp)++;
        }
        else if (ImGui::IsKeyPressed(ImGuiKey_T, false))   // skip to the last keypoint
        {
            *kp = cam->frameData->nNodes-1;
        }
        else if (ImGui::IsKeyPressed(ImGuiKey_Delete, false))  // delete the active keypoint
        {
            cam->activeFrameData->g.x[*kp] = 0.0;
            cam->activeFrameData->g.y[*kp] = 0.0;
            cam->activeFrameData->g.isLabeled[*kp] = false;
            std::cout << cam->frameData->nodeNames.at(*kp) << " deleted on " << cam->name << std:: endl;
        }

        else if (ImGui::IsKeyPressed(ImGuiKey_W, false))
        {
            ImPlotPoint mouse = ImPlot::GetPlotMousePos();

            if(!cam->activeFrameData)
            {
                FrameData2D* frameData_ptr = new FrameData2D(cam->skelEnum);
                cam->frameDataMap.insert(cam->frameDataMapIter, std::make_pair(current_frame, frameData_ptr));
                std::cout << "frameData created for " << cam->name << " on image " << current_frame << std::endl;
                cam->activeFrameData = frameData_ptr;

                // set value of activeKeyPoint on activeFrameData to currently selected keypoint value
                cam->activeFrameData->activeKeyPoint = *kp;
                // switch *kp to activeFrameData instead of default frameData
                kp = &cam->activeFrameData->activeKeyPoint;
            }

            cam->activeFrameData->g.x[*kp] = (double)mouse.x;
            cam->activeFrameData->g.y[*kp] = (double)mouse.y;
            cam->activeFrameData->g.isLabeled[*kp] = true;

            if(*kp < (cam->frameData->nNodes - 1)) { (*kp)++; }
        }
    }
    else { cam->isImgHovered = false; }
}



static void keypoint_button(Camera* cam, int i, uint64_t current_frame, std::map<uint, FrameData2D*>::iterator iter, LabelManager* labelMgr, int* draw_id)
{
    // Show a button for each keypoint -- only one button can be active per window at a time (this button will be colored)
    int *kp = (cam->activeFrameData) ? &cam->activeFrameData->activeKeyPoint : &cam->frameData->activeKeyPoint;
    static int buttonsPerLine = 12;

    int this_draw_id = *draw_id;
    for (int j = 0; j < cam->frameData->nNodes; j++)
    {
        if (j > 0) ImGui::SameLine();
        
        ImGui::PushID(this_draw_id);

        if (*kp == j)
        {
            ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(j / (float)cam->frameData->nNodes, 0.5f, 0.5f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(j / (float)cam->frameData->nNodes, 0.7f, 0.7f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(j / (float)cam->frameData->nNodes, 0.9f, 0.9f));
        }
        else
        {
            ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0.2f, 0.2f, 0.2f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0.2f, 0.2f, 0.2f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(0.2f, 0.2f, 0.2f));
        }
        
        ImGui::Button(cam->frameData->nodeNames.at(j).c_str());

        if (ImGui::IsItemClicked())
        {
            *kp = j;
        }

        ImGui::PopStyleColor(3);
        ImGui::PopID();

        if (j % buttonsPerLine == (buttonsPerLine - 1))
        {
            ImGui::NewLine();
        }

        this_draw_id++;
    }

    // Button to erase frameDataeton in current frame

    ImGui::SameLine();
    ImGui::PushID(this_draw_id);
    this_draw_id++;
    ImGui::Button("Clear");
    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(.2f, .2f, .2f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(.5f, .2f, .2f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(.8f, .2f, .2f));

    if (ImGui::IsItemClicked())
    {
        if (cam->activeFrameData)
        {
            delete cam->activeFrameData;
            cam->activeFrameData = nullptr;
            cam->frameDataMap.erase(iter);
        }
    }
    ImGui::PopStyleColor(3);
    ImGui::PopID();

    // Button to copy frameDataeton from previous frame
    ImGui::SameLine();
    ImGui::PushID(this_draw_id);
    this_draw_id++;
    ImGui::Button("Copy Previous");
    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(.2f, .2f, .2f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(.2f, .5f, .2f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(.2f, .8f, .2f));

    if (ImGui::IsItemClicked() && current_frame > 0 && cam->frameDataMap.count(current_frame - 1) > 0)
    {
        if (cam->activeFrameData)
        {
            delete cam->activeFrameData;
            FrameData2D* previous_frameData_ptr = cam->frameDataMap.at(current_frame - 1);
            FrameData2D* frameData_ptr = new FrameData2D(previous_frameData_ptr);
            iter->second = frameData_ptr;
        }
        else
        {
            FrameData2D* previous_frameData_ptr = cam->frameDataMap.at(current_frame - 1);
            FrameData2D* frameData_ptr = new FrameData2D(previous_frameData_ptr);
            cam->frameDataMap.insert(iter, std::make_pair(current_frame, frameData_ptr));
        }
        cam->activeFrameData = cam->frameDataMap.at(current_frame);

    }
    ImGui::PopStyleColor(3);
    ImGui::PopID();

    // update label manager
    if (i == 0)
    {
        labelMgr->numViewsLabeled.assign(labelMgr->nNodes, 0);
    }
    for (int j=0; j<labelMgr->nNodes; j++)
    {
        if (cam->activeFrameData && cam->activeFrameData->g.isLabeled[j])
        {
            labelMgr->isLabeled[i][j] = true;
            labelMgr->numViewsLabeled[j]++;
        }
        else
        {
            labelMgr->isLabeled[i][j] = false;
        }
    }

    (*draw_id) = this_draw_id;
}




static void reprojection(LabelManager* labelMgr, std::vector<Camera*> cams, uint64_t current_frame){

    // check if any nodes qualify for reprojection
    bool isReadyForReproject = false;
    for (int j=0; j<labelMgr->nNodes; j++)
    {
        if (labelMgr->numViewsLabeled[j] >= 2)
        {
            isReadyForReproject = true;
            break;
        }
    }
    // check that frameDatas exist for all views for this image frame. If not, create them
    for (int i=0; i<labelMgr->nCams; i++)
    {
        if (!cams[i]->activeFrameData)
        {
            FrameData2D* frameData_ptr = new FrameData2D(cams[i]->skelEnum);
            std::map<uint, FrameData2D*>::iterator iter(cams[i]->frameDataMap.lower_bound(current_frame));
            cams[i]->frameDataMap.insert(iter, std::make_pair(current_frame, frameData_ptr));
            cams[i]->activeFrameData = frameData_ptr;
        }
    }

    if (isReadyForReproject)
    {
        if(!labelMgr->activeSkelWorld)
            {
                SkelWorld* skelWorld_ptr = new SkelWorld(cams[0]->activeFrameData);
                labelMgr->skelWorldMap.insert(labelMgr->skelWorldMapIter, std::make_pair(current_frame, skelWorld_ptr));
                labelMgr->activeSkelWorld = skelWorld_ptr;
            }
    }

    for (int j=0; j<labelMgr->nNodes; j++)
    {
        if (labelMgr->numViewsLabeled[j] >= 2)
        {
            std::vector<cv::Mat> sfmPoints2d;
            std::vector<cv::Mat> projection_matrices;
            cv::Mat output;
            for (int i=0; i<labelMgr->nCams; i++)
            {
                if (labelMgr->isLabeled[i][j])
                {
                    cv::Mat point = (cv::Mat_<float>(2, 1) << (float)cams[i]->activeFrameData->g.x[j], (float)2200 - (float)cams[i]->activeFrameData->g.y[j]);
                    cv::Mat pointUndistort;
                    cv::undistortPoints(point, pointUndistort, cams[i]->cvp.K, cams[i]->cvp.distCoeffs, cv::noArray(), cams[i]->cvp.K);
                    sfmPoints2d.push_back(pointUndistort.reshape(1, 2));
                    projection_matrices.push_back(cams[i]->cvp.projectionMat);
                }
            }

            cv::sfm::triangulatePoints(sfmPoints2d, projection_matrices, output);
            output.convertTo(output, CV_32F);

            float x, y, z;
            x = output.at<float>(0);
            y = output.at<float>(1);
            z = output.at<float>(2);
            labelMgr->activeSkelWorld->g.x[j] = x;
            labelMgr->activeSkelWorld->g.y[j] = y;
            labelMgr->activeSkelWorld->g.z[j] = z;
            labelMgr->activeSkelWorld->g.isLabeled[j] = true;

            for (int i=0; i<labelMgr->nCams; i++)
            {
                cv::Mat imagePts;
                cv::projectPoints(output, cams[i]->cvp.rvec, cams[i]->cvp.tvec, cams[i]->cvp.K, cams[i]->cvp.distCoeffs, imagePts);
                double x = imagePts.at<float>(0, 0);
                double y = float(2200) - imagePts.at<float>(0, 1);
                cams[i]->activeFrameData->g.x[j] = x;
                cams[i]->activeFrameData->g.y[j] = y;
                cams[i]->activeFrameData->g.isLabeled[j] = true;
            }
        }
    }
}

static void reprojection_overweight_activeKP(LabelManager* labelMgr, std::vector<Camera*> cams, uint64_t current_frame){

    // get hovered img
    int hoveredCam = -1;
    int activeKP = -1;
    for (int i=0; i<labelMgr->nCams; i++)
    {
        if (labelMgr->cameras[i]->isImgHovered)
        {
            hoveredCam = i;
            activeKP = labelMgr->cameras[i]->activeFrameData->activeKeyPoint;
            break;
        }
    }

    if (hoveredCam == -1)
    {
        std::cout << "No image hovered. Must hover an image to use reprojection_overweight_activeKP function" << std::endl;
        return;
    }

    if (activeKP == -1)
    {
        std::cout << "No active keypoint on hovered image. Must have active keypoint on hovered image to use reprojection_overweight_activeKP function" << std::endl;
        return;
    }

    // check if any nodes qualify for reprojection
    bool isReadyForReproject = false;
    for (int j=0; j<labelMgr->nNodes; j++)
    {
        if (labelMgr->numViewsLabeled[j] >= 2)
        {
            isReadyForReproject = true;
            break;
        }
    }
    // check that frameDatas exist for all views for this image frame. If not, create them
    for (int i=0; i<labelMgr->nCams; i++)
    {
        if (!cams[i]->activeFrameData)
        {
            FrameData2D* frameData_ptr = new FrameData2D(cams[i]->skelEnum);
            std::map<uint, FrameData2D*>::iterator iter(cams[i]->frameDataMap.lower_bound(current_frame));
            cams[i]->frameDataMap.insert(iter, std::make_pair(current_frame, frameData_ptr));
            cams[i]->activeFrameData = frameData_ptr;
        }
    }

    if (isReadyForReproject)
    {
        if(!labelMgr->activeSkelWorld)
            {
                SkelWorld* skelWorld_ptr = new SkelWorld(cams[0]->activeFrameData);
                labelMgr->skelWorldMap.insert(labelMgr->skelWorldMapIter, std::make_pair(current_frame, skelWorld_ptr));
                labelMgr->activeSkelWorld = skelWorld_ptr;
            }
    }

    for (int j=0; j<labelMgr->nNodes; j++)
    {
        if (labelMgr->numViewsLabeled[j] >= 2)
        {
            std::vector<cv::Mat> sfmPoints2d;
            std::vector<cv::Mat> projection_matrices;
            cv::Mat output;
            for (int i=0; i<labelMgr->nCams; i++)
            {
                if (labelMgr->isLabeled[i][j])
                {
                    cv::Mat point = (cv::Mat_<float>(2, 1) << (float)cams[i]->activeFrameData->g.x[j], (float)2200 - (float)cams[i]->activeFrameData->g.y[j]);
                    cv::Mat pointUndistort;
                    cv::undistortPoints(point, pointUndistort, cams[i]->cvp.K, cams[i]->cvp.distCoeffs, cv::noArray(), cams[i]->cvp.K);
                    sfmPoints2d.push_back(pointUndistort.reshape(1, 2));
                    projection_matrices.push_back(cams[i]->cvp.projectionMat);

                    // overweight the active keypoint on the active image
                    if (j == activeKP && i == hoveredCam)
                    {
                        for (int k = 0; k < 100; k++)
                        {
                            sfmPoints2d.push_back(pointUndistort.reshape(1, 2));
                            projection_matrices.push_back(cams[i]->cvp.projectionMat);
                            
                        }
                        std::cout << "overweighted kp" << j << "on cam" << i << std::endl;
                    }
                }
            }

            std::cout << "sfmPoints2d size: " << sfmPoints2d.size() << std::endl;

            cv::sfm::triangulatePoints(sfmPoints2d, projection_matrices, output);
            output.convertTo(output, CV_32F);

            float x, y, z;
            x = output.at<float>(0);
            y = output.at<float>(1);
            z = output.at<float>(2);
            labelMgr->activeSkelWorld->g.x[j] = x;
            labelMgr->activeSkelWorld->g.y[j] = y;
            labelMgr->activeSkelWorld->g.z[j] = z;
            labelMgr->activeSkelWorld->g.isLabeled[j] = true;

            for (int i=0; i<labelMgr->nCams; i++)
            {
                cv::Mat imagePts;
                cv::projectPoints(output, cams[i]->cvp.rvec, cams[i]->cvp.tvec, cams[i]->cvp.K, cams[i]->cvp.distCoeffs, imagePts);
                double x = imagePts.at<float>(0, 0);
                double y = float(2200) - imagePts.at<float>(0, 1);
                cams[i]->activeFrameData->g.x[j] = x;
                cams[i]->activeFrameData->g.y[j] = y;
                cams[i]->activeFrameData->g.isLabeled[j] = true;
            }
        }
    }
}

#endif
