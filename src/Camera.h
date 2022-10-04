#ifndef CAMERA_H
#define CAMERA_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/sfm.hpp>
#include <opencv2/calib3d.hpp>

#include "imgui.h"
#include "FrameData2D.h"

#include <iostream>
#include <vector>
#include <opencv2/core/mat.hpp>


using namespace std;

struct cvCamParams {
    cv::Mat K;
    cv::Mat distCoeffs;
    cv::Mat r;
    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat projectionMat;
};



class Camera
{
    private:

    public:
        Camera(int index, int camNum, std::string rootDir, SkelEnum skelEnum);
        std::string root_dir;
        std::string name;
        std::string label_dir;
        std::string movie_dir;
        std::string calibration_dir;
        static bool LoadTextureFromFile(const char* filename, GLuint* out_texture, int* out_width, int* out_height);
        bool LoadCameraParamsFromCSV();
        string type2str(int type);
        int index;
        int camNum;
        SkelEnum skelEnum;
        int nImages;
        std::string path;
        std::vector<GLuint> textures;
        FrameData2D *frameData;
        FrameData2D *activeFrameData;
        cvCamParams cvp;
        std::map<uint, FrameData2D*> frameDataMap;
        std::map<uint, FrameData2D*>::iterator frameDataMapIter;
        bool isImgHovered;
        bool isWindowHovered;
        std::vector<float> csvCamValues;
        void PrintCameraParams();
        vector<float> k;    // 9
        vector<float> r_m;  // 9
        vector<float> t;    // 3
        vector<float> d;    // 4
        bool SaveSkelMap();
        bool LoadSkelMap();

        void get_circle_pts();
        std::vector<double> arena_x;
        std::vector<double> arena_y;
        cv::Mat arena_mask;




};

#endif