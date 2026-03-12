#ifndef RED_CAMERA
#define RED_CAMERA
#include "opencv_yaml_io.h"
#include "red_math.h"
#include <Eigen/Core>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct CameraParams {
    Eigen::Matrix3d k = Eigen::Matrix3d::Zero();
    Eigen::Matrix<double, 5, 1> dist_coeffs = Eigen::Matrix<double, 5, 1>::Zero();
    Eigen::Matrix3d r = Eigen::Matrix3d::Identity();
    Eigen::Vector3d rvec = Eigen::Vector3d::Zero();
    Eigen::Vector3d tvec = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 3, 4> projection_mat = Eigen::Matrix<double, 3, 4>::Zero();
    bool telecentric = false;
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

// Load telecentric camera parameters from DLT coefficient CSV + optional distortion CSV.
// The DLT CSV has 11 values (one per line): P(0,0)..P(0,3), P(1,0)..P(1,3), 0, 0, 0.
// The distortion CSV (optional) has header "k1,k2,sx,sy,skew" then one data row.
bool camera_load_params_from_dlt_csv(const std::string &dlt_csv_path,
                                      CameraParams &camera_params,
                                      std::string &error_message) {
    error_message.clear();
    namespace fs = std::filesystem;

    if (!fs::exists(dlt_csv_path)) {
        error_message = "File does not exist: " + dlt_csv_path;
        return false;
    }

    try {
        // Read 11 DLT coefficients (one per line)
        std::ifstream f(dlt_csv_path);
        if (!f.is_open()) {
            error_message = "Cannot open: " + dlt_csv_path;
            return false;
        }
        double coeff[11];
        for (int i = 0; i < 11; i++) {
            std::string line;
            if (!std::getline(f, line) || line.empty()) {
                error_message = "Expected 11 coefficients in: " + dlt_csv_path;
                return false;
            }
            coeff[i] = std::stod(line);
        }

        // Build 3x4 P matrix: [A t; 0 0 0 1]
        camera_params.projection_mat << coeff[0], coeff[1], coeff[2], coeff[3],
                                         coeff[4], coeff[5], coeff[6], coeff[7],
                                         0, 0, 0, 1;

        // Decompose A to extract R and K2 for undistortion
        Eigen::Matrix<double, 2, 3> A;
        A << coeff[0], coeff[1], coeff[2],
             coeff[4], coeff[5], coeff[6];
        Eigen::Vector2d t_affine(coeff[3], coeff[7]);

        double sx = A.row(0).norm();
        double sy = A.row(1).norm();

        // Store telecentric intrinsics in K as [sx skew tx; 0 sy ty; 0 0 1]
        // For now, compute skew from A decomposition
        // A = K2 * R(1:2,:), where K2 = [sx k; 0 sy]
        // Normalize rows to get approximate R
        Eigen::RowVector3d r1_hat = A.row(0) / sx;
        Eigen::RowVector3d r2_hat = A.row(1) / sy;
        double skew = 0; // default; will be overridden by distortion CSV if present

        camera_params.k = Eigen::Matrix3d::Identity();
        camera_params.k(0, 0) = sx;
        camera_params.k(0, 1) = skew;
        camera_params.k(0, 2) = t_affine.x(); // tx
        camera_params.k(1, 1) = sy;
        camera_params.k(1, 2) = t_affine.y(); // ty

        // Extract rotation matrix from A (orthographic decomposition)
        Eigen::Matrix<double, 2, 3> Anorm;
        Anorm.row(0) = A.row(0) / sx;
        Anorm.row(1) = A.row(1) / sy;
        Eigen::JacobiSVD<Eigen::Matrix<double, 2, 3>> svd(
            Anorm, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix<double, 2, 3> S23 = Eigen::Matrix<double, 2, 3>::Zero();
        S23(0, 0) = 1.0; S23(1, 1) = 1.0;
        Eigen::Matrix<double, 2, 3> Aorth = svd.matrixU() * S23 * svd.matrixV().transpose();
        Eigen::Vector3d c0 = Aorth.row(0).transpose();
        Eigen::Vector3d c1 = Aorth.row(1).transpose();
        Eigen::Vector3d c2 = c0.cross(c1);
        camera_params.r.col(0) = c0;
        camera_params.r.col(1) = c1;
        camera_params.r.col(2) = c2;
        if (camera_params.r.determinant() < 0)
            camera_params.r.col(2) = -c2;

        camera_params.rvec = red_math::rotationMatrixToVector(camera_params.r);
        camera_params.tvec = Eigen::Vector3d(t_affine.x(), t_affine.y(), 0);

        // Default: no distortion
        camera_params.dist_coeffs = Eigen::Matrix<double, 5, 1>::Zero();

        // Try to load distortion CSV (same folder, same base name with _distortion suffix)
        // e.g., CamXXXX_dlt.csv → CamXXXX_distortion.csv
        std::string dist_path = dlt_csv_path;
        auto pos = dist_path.rfind("_dlt.csv");
        if (pos != std::string::npos) {
            dist_path.replace(pos, 8, "_distortion.csv");
            if (fs::exists(dist_path)) {
                std::ifstream df(dist_path);
                std::string header;
                std::getline(df, header); // skip "k1,k2,sx,sy,skew"
                std::string data_line;
                if (std::getline(df, data_line) && !data_line.empty()) {
                    std::istringstream ss(data_line);
                    std::string tok;
                    double dk1 = 0, dk2 = 0, dsx = 0, dsy = 0, dskew = 0;
                    std::getline(ss, tok, ','); dk1 = std::stod(tok);
                    std::getline(ss, tok, ','); dk2 = std::stod(tok);
                    std::getline(ss, tok, ','); dsx = std::stod(tok);
                    std::getline(ss, tok, ','); dsy = std::stod(tok);
                    std::getline(ss, tok, ','); dskew = std::stod(tok);

                    camera_params.dist_coeffs(0) = dk1;
                    camera_params.dist_coeffs(1) = dk2;
                    // Update K2 with refined values from distortion CSV
                    camera_params.k(0, 0) = dsx;
                    camera_params.k(0, 1) = dskew;
                    camera_params.k(1, 1) = dsy;
                }
            }
        }

        camera_params.telecentric = true;

    } catch (const std::exception &e) {
        error_message = "Error reading " + dlt_csv_path + ": " + e.what();
        return false;
    }

    return true;
}

#endif
