#include "LabelManager.h"


LabelManager::LabelManager(std::vector<std::string> cameraParamsPaths)
{
    this->numCams = cameraParamsPaths.size();
    for (const auto& path : cameraParamsPaths) {
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


void LabelManager::setSkel(std::string skelName)
{
    int idx = getIndex(SkelEnumNames, skelName);
    this->skelEnum = (SkelEnum)idx;
    // create framedata
    for(int cam=0; cam<this->numCams; cam++)
    {
        FrameData2D* framedata2d = new FrameData2D(skelEnum);
        m_frameData2DList.push_back(framedata2d);
    }

}


void LabelManager::set_active_skel2D(Camera* cam, uint64_t current_frame)
{
    cam->frameDataMapIter = std::map<uint, FrameData2D*>::iterator(cam->frameDataMap.lower_bound(current_frame));  // get iterator to check if skel exists
    if (cam->frameDataMapIter == cam->frameDataMap.end() || current_frame < cam->frameDataMapIter->first) { cam->activeFrameData = nullptr; }
    else { cam->activeFrameData = cam->frameDataMapIter->second; }
}
