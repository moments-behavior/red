#include "LabelManager.h"


LabelManager::LabelManager(std::vector<std::string> intrinsicsPaths,
            std::vector<std::string>  extrinsicsPaths)
{
    for (const auto& path : intrinsicsPaths) {
        CameraIntrinics cameraIntrinics;
        readIntrinsics(path, cameraIntrinics);
        m_cameraIntrinsicsList.push_back(cameraIntrinics);
        CameraExtrinsics cameraExtrinsics;
        readExtrinsincs(path, cameraExtrinsics);
        m_cameraExtrinsicsList.push_back(cameraExtrinsics);
    }
}


void LabelManager::readIntrinsics(const std::string& path,
            CameraIntrinics& cameraIntrinics) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    fs["intrinsicMatrix"] >> cameraIntrinics.intrinsicMatrix;
    fs["distortionCoefficients"] >> cameraIntrinics.distortionCoefficients;
}


void LabelManager::readExtrinsincs(const std::string& path,
            CameraExtrinsics& cameraExtrinsics) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    fs["R"] >> cameraExtrinsics.rotationMatrix;
    fs["T"] >> cameraExtrinsics.translationVector;
    cv::vconcat(cameraExtrinsics.rotationMatrix,
                            cameraExtrinsics.translationVector.t(),
                            cameraExtrinsics.locationMatrix);
}


