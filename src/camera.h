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
    cv::Mat k;
    cv::Mat dist_coeffs;
    cv::Mat r;
    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat projection_mat;
};

void camera_print_parameters(CameraParams* cvp){
    std::cout << "k = " << std::endl << cv::format(cvp->k, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
    std::cout << "dist_coeffs  = " << std::endl << cv::format(cvp->dist_coeffs, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
    std::cout << "r = " << std::endl << cv::format(cvp->r, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
    std::cout << "tvec = " << std::endl << cv::format(cvp->tvec, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
    std::cout << "rvec = " << std::endl << cv::format(cvp->rvec, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
    std::cout << "projection_mat = " << std::endl << cv::format(cvp->projection_mat, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
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

    cvp.k = cv::Mat_<float>(k, true).reshape(0, 3);
    cvp.dist_coeffs = cv::Mat_<float>(d, true);
    cvp.r = cv::Mat_<float>(r_m, true).reshape(0, 3);
    cvp.tvec = cv::Mat_<float>(t, true);
    cv::Rodrigues(cvp.r, cvp.rvec);
    cv::sfm::projectionFromKRt(cvp.k, cvp.r, cvp.tvec, cvp.projection_mat);
    return cvp;
}

#endif
