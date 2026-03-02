#ifndef RED_CAMERA
#define RED_CAMERA
#include "opencv_yaml_io.h"
#include "red_math.h"
#include <Eigen/Core>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct CameraParams {
    Eigen::Matrix3d k = Eigen::Matrix3d::Zero();
    Eigen::Matrix<double, 5, 1> dist_coeffs = Eigen::Matrix<double, 5, 1>::Zero();
    Eigen::Matrix3d r = Eigen::Matrix3d::Identity();
    Eigen::Vector3d rvec = Eigen::Vector3d::Zero();
    Eigen::Vector3d tvec = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 3, 4> projection_mat = Eigen::Matrix<double, 3, 4>::Zero();
};

void camera_print_parameters(CameraParams *cvp) {
    Eigen::IOFormat fmt(Eigen::StreamPrecision, 0, ", ", ",\n", "[", "]", "[", "]");
    std::cout << "k = " << std::endl
              << cvp->k.format(fmt) << std::endl
              << std::endl;
    std::cout << "dist_coeffs  = " << std::endl
              << cvp->dist_coeffs.transpose().format(fmt) << std::endl
              << std::endl;
    std::cout << "r = " << std::endl
              << cvp->r.format(fmt) << std::endl
              << std::endl;
    std::cout << "tvec = " << std::endl
              << cvp->tvec.transpose().format(fmt) << std::endl
              << std::endl;
    std::cout << "rvec = " << std::endl
              << cvp->rvec.transpose().format(fmt) << std::endl
              << std::endl;
    std::cout << "projection_mat = " << std::endl
              << cvp->projection_mat.format(fmt) << std::endl
              << std::endl;
}

bool camera_load_params_from_yaml(const std::string &calibration_file,
                                  CameraParams &camera_params,
                                  std::string &error_message) {
    error_message.clear();

    if (!std::filesystem::exists(calibration_file)) {
        error_message = "File does not exist: " + calibration_file;
        return false;
    }

    try {
        opencv_yaml::YamlFile yaml = opencv_yaml::read(calibration_file);

        if (!yaml.hasKey("camera_matrix") || !yaml.hasKey("rc_ext") ||
            !yaml.hasKey("tc_ext")) {
            error_message = "Missing fields in: " + calibration_file;
            return false;
        }

        Eigen::MatrixXd K_raw = yaml.getMatrix("camera_matrix");
        Eigen::MatrixXd R_raw = yaml.getMatrix("rc_ext");
        Eigen::MatrixXd T_raw = yaml.getMatrix("tc_ext");

        camera_params.k = K_raw;
        camera_params.r = R_raw;

        // tvec: could be 3x1 or 1x3
        if (T_raw.rows() == 1 && T_raw.cols() == 3)
            camera_params.tvec = T_raw.transpose();
        else
            camera_params.tvec = Eigen::Vector3d(T_raw(0), T_raw(1), T_raw(2));

        // dist_coeffs: optional, could be various shapes
        if (yaml.hasKey("distortion_coefficients")) {
            Eigen::MatrixXd D_raw = yaml.getMatrix("distortion_coefficients");
            int n = std::min((int)(D_raw.rows() * D_raw.cols()), 5);
            camera_params.dist_coeffs = Eigen::Matrix<double, 5, 1>::Zero();
            for (int i = 0; i < n; i++)
                camera_params.dist_coeffs(i) = D_raw.data()[i];
        }

    } catch (const std::exception &e) {
        error_message = "Error reading " + calibration_file + ": " + e.what();
        return false;
    }

    camera_params.rvec = red_math::rotationMatrixToVector(camera_params.r);
    camera_params.projection_mat = red_math::projectionFromKRt(
        camera_params.k, camera_params.r, camera_params.tvec);
    return true;
}

CameraParams camera_load_params_from_csv(std::string csv_filename,
                                         int cam_idx) {
    std::cout << csv_filename << std::endl;
    CameraParams cvp;

    std::ifstream fin;
    fin.open(csv_filename);
    if (fin.fail())
        throw csv_filename;

    std::string line;
    std::string delimeter = ",";
    size_t pos = 0;
    std::string token;

    // read csv file with cam parameters and tokenize line for this camera
    int lineNum = 0;
    std::vector<float> csvCamValues;

    while (!fin.eof()) {
        fin >> line;

        while ((pos = line.find(delimeter)) != std::string::npos) {
            token = line.substr(0, pos);
            if (lineNum == cam_idx) {
                csvCamValues.push_back(stof(token));
            }
            line.erase(0, pos + delimeter.length());
        }
        lineNum++;
    }

    // k: 9 values (row-major 3x3)
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            cvp.k(i, j) = csvCamValues[i * 3 + j];

    // r: 9 values (row-major 3x3)
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            cvp.r(i, j) = csvCamValues[9 + i * 3 + j];

    // tvec: 3 values
    cvp.tvec = Eigen::Vector3d(csvCamValues[18], csvCamValues[19], csvCamValues[20]);

    // dist_coeffs: 4 values (pad 5th with 0)
    cvp.dist_coeffs = Eigen::Matrix<double, 5, 1>::Zero();
    for (int i = 0; i < 4; i++)
        cvp.dist_coeffs(i) = csvCamValues[21 + i];

    cvp.rvec = red_math::rotationMatrixToVector(cvp.r);
    cvp.projection_mat = red_math::projectionFromKRt(cvp.k, cvp.r, cvp.tvec);
    return cvp;
}

#endif
