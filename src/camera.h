#ifndef RED_CAMERA
#define RED_CAMERA
#include <string>
#include <opencv2/core.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/sfm.hpp>
#include <opencv2/calib3d.hpp>


struct CameraParams {
    cv::Mat K;
    cv::Mat distCoeffs;
    cv::Mat r;
    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat projectionMat;
};

void camera_print_parameters(CameraParams* cvp){
    std::cout << "K = " << std::endl << cv::format(cvp->K, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
    std::cout << " distCoeffs  = " << std::endl << cv::format(cvp->distCoeffs, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
    std::cout << "r = " << std::endl << cv::format(cvp->r, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
    std::cout << "tvec = " << std::endl << cv::format(cvp->tvec, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
    std::cout << "rvec = " << std::endl << cv::format(cvp->rvec, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
    std::cout << "projectionMat = " << std::endl << cv::format(cvp->projectionMat, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
}

void camera_arena_projection_points(CameraParams* cvp, float* arena_x, float* arena_y, int n)
{
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;

    float radius = 1473.0f;
    std::vector<cv::Point3f> inPts;

    for (int i=0; i<=n; i++)
    {
        float angle = (3.14159265358979323846 * 2) * (float(i) / float(n-1));
        x.push_back(sin(angle) * radius);
        y.push_back(cos(angle) * radius);
        z.push_back(0.0f);
    }

    for (int i=0; i<n; i++)
    {
        cv::Point3f p;
        p.x = x[i];
        p.y = y[i];
        p.z = z[i];
        inPts.push_back(p);
    }

    std::vector<cv::Point2f> img_pts;
    cv::projectPoints(inPts, cvp->rvec, cvp->tvec, cvp->K, cvp->distCoeffs, img_pts);

    for (int i = 0; i < n; i++){
        arena_x[i] = img_pts.at(i).x;
        arena_y[i] = 2200 - img_pts.at(i).y;
    }
}

CameraParams camera_load_params_from_csv(std::string csv_filename, int cam_idx)
{
    std::cout << csv_filename << std::endl;
    CameraParams cvp;

    std::ifstream fin;
    fin.open(csv_filename);
    if (fin.fail()) throw csv_filename;  

    std::string line;
    std::string delimeter = ",";
    size_t pos = 0;
    std::string token;

    // read csv file with cam parameters and tokenize line for this camera
    int lineNum = 0;
    std::vector<float> csvCamValues;

    while(!fin.eof()){
        fin >> line;

        while ((pos = line.find(delimeter)) != std::string::npos)
        {
            token = line.substr(0, pos);
            if (lineNum == cam_idx)
            {
                csvCamValues.push_back(stof(token));
            }
            line.erase(0, pos + delimeter.length());
        }
        lineNum++;
    }

    std::vector<float> k;    // 9
    std::vector<float> r_m;  // 9
    std::vector<float> t;    // 3
    std::vector<float> d;    // 4

    for (int i=0; i<9; i++)
    {
        k.push_back(csvCamValues[i]);
    }
    for (int i=9; i<18; i++)
    {
        r_m.push_back(csvCamValues[i]);
    }
    for (int i=18; i<21; i++)
    {
        t.push_back(csvCamValues[i]);
    }
    for (int i=21; i<25; i++)
    {
        d.push_back(csvCamValues[i]);
    }

    cvp.K = cv::Mat_<float>(k, true).reshape(0, 3);
    cvp.distCoeffs = cv::Mat_<float>(d, true);
    cvp.r = cv::Mat_<float>(r_m, true).reshape(0, 3);
    cvp.tvec = cv::Mat_<float>(t, true);
    cv::Rodrigues(cvp.r, cvp.rvec);
    cv::sfm::projectionFromKRt(cvp.K, cvp.r, cvp.tvec, cvp.projectionMat);
    return cvp;
}

#endif
