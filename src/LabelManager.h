#pragma once
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

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
    
    private: 
        void readIntrinsics(const std::string& path, CameraIntrinics& cameraIntrinics);
        void readExtrinsincs(const std::string& path, CameraExtrinsics& cameraExtrinsics);

        std::vector<CameraIntrinics> m_cameraIntrinsicsList;
        std::vector<CameraExtrinsics> m_cameraExtrinsicsList;
};