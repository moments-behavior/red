#pragma once
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include "FrameData2D.h"


class LabelManager
{
    public:
        typedef struct CameraIntrinics {
            cv::Mat intrinsicMatrix;
            cv::Mat distortionCoefficients;
        } CameraIntrinics;

        typedef struct CameraExtrinsics {
            cv::Mat rotationMatrix;
            cv::Mat translationVector;
            cv::Mat locationMatrix;	//combination of rotation and Translation of secondary
            cv::Mat essentialMatrix;
            cv::Mat fundamentalMatrix;
        } CameraExtrinsics;

        LabelManager(std::vector<std::string> cameraParamsPaths);
        void setSkel(std::string skelName);
        void set_active_skel2D(int cam_idx, uint64_t current_frame);


    private: 
        void readIntrinsics(const std::string& path, CameraIntrinics& cameraIntrinics);
        void readExtrinsincs(const std::string& path, CameraExtrinsics& cameraExtrinsics);

        std::vector<CameraIntrinics> m_cameraIntrinsicsList;
        std::vector<CameraExtrinsics> m_cameraExtrinsicsList;
        std::vector<FrameData2D*> m_frameData2DList;
        std::vector<FrameData2D*> m_activeFrameData2DList;


        std::vector<std::map<uint, FrameData2D*>> frameDataMapList;
        std::vector<std::map<uint, FrameData2D*>::iterator> frameDataMapIterList;

        SkelEnum skelEnum;
        int numCams;
};