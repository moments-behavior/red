#pragma once
// red_math.h — Eigen-based replacements for OpenCV camera math functions.
// Replaces: cv::sfm::projectionFromKRt, cv::Rodrigues, cv::undistortPoints,
//           cv::sfm::triangulatePoints, cv::projectPoints

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <cmath>
#include <vector>

namespace red_math {

// ---- projectionFromKRt ----
// P = K * [R | t]  (3x4 projection matrix)
inline Eigen::Matrix<double, 3, 4>
projectionFromKRt(const Eigen::Matrix3d &K, const Eigen::Matrix3d &R,
                  const Eigen::Vector3d &t) {
    Eigen::Matrix<double, 3, 4> Rt;
    Rt.block<3, 3>(0, 0) = R;
    Rt.col(3) = t;
    return K * Rt;
}

// ---- Rodrigues: rotation matrix → rotation vector ----
inline Eigen::Vector3d rotationMatrixToVector(const Eigen::Matrix3d &R) {
    Eigen::AngleAxisd aa(R);
    return aa.axis() * aa.angle();
}

// ---- Rodrigues: rotation vector → rotation matrix ----
inline Eigen::Matrix3d rotationVectorToMatrix(const Eigen::Vector3d &rvec) {
    double angle = rvec.norm();
    if (angle < 1e-12)
        return Eigen::Matrix3d::Identity();
    Eigen::Vector3d axis = rvec / angle;
    return Eigen::AngleAxisd(angle, axis).toRotationMatrix();
}

// ---- undistortPoints ----
// Iterative Brown-Conrady undistortion: given a distorted pixel coordinate,
// produce the undistorted pixel coordinate (still in pixel space via K).
// dist_coeffs: [k1, k2, p1, p2, k3]
inline Eigen::Vector2d
undistortPoint(const Eigen::Vector2d &pt, const Eigen::Matrix3d &K,
               const Eigen::Matrix<double, 5, 1> &dist) {
    // Normalize to camera coords
    double fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2);
    double x0 = (pt(0) - cx) / fx;
    double y0 = (pt(1) - cy) / fy;

    double k1 = dist(0), k2 = dist(1), p1 = dist(2), p2 = dist(3), k3 = dist(4);

    // Iterative undistortion (10 iterations)
    double x = x0, y = y0;
    for (int i = 0; i < 10; i++) {
        double r2 = x * x + y * y;
        double r4 = r2 * r2;
        double r6 = r4 * r2;
        double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        double dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        double dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
        x = (x0 - dx) / radial;
        y = (y0 - dy) / radial;
    }

    // Re-project to pixel coords using K
    return Eigen::Vector2d(x * fx + cx, y * fy + cy);
}

// ---- triangulatePoints (DLT) ----
// pts2d: vector of 2D points (one per view, in pixel coords after undistortion)
// Ps: corresponding 3x4 projection matrices
// Returns homogeneous 3D point (x, y, z).
inline Eigen::Vector3d
triangulatePoints(const std::vector<Eigen::Vector2d> &pts2d,
                  const std::vector<Eigen::Matrix<double, 3, 4>> &Ps) {
    int n = static_cast<int>(pts2d.size());
    Eigen::MatrixXd A(2 * n, 4);
    for (int i = 0; i < n; i++) {
        double x = pts2d[i](0);
        double y = pts2d[i](1);
        A.row(2 * i + 0) = x * Ps[i].row(2) - Ps[i].row(0);
        A.row(2 * i + 1) = y * Ps[i].row(2) - Ps[i].row(1);
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d X = svd.matrixV().col(3);
    return X.head<3>() / X(3);
}

// ---- projectPoints ----
// Project 3D points to 2D image coordinates with distortion.
// Each 3D point is transformed by [R|t], then distorted, then mapped to pixels.
// dist_coeffs: [k1, k2, p1, p2, k3]
inline std::vector<Eigen::Vector2d>
projectPoints(const std::vector<Eigen::Vector3d> &pts3d,
              const Eigen::Vector3d &rvec, const Eigen::Vector3d &tvec,
              const Eigen::Matrix3d &K,
              const Eigen::Matrix<double, 5, 1> &dist) {
    Eigen::Matrix3d R = rotationVectorToMatrix(rvec);
    double fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2);
    double k1 = dist(0), k2 = dist(1), p1 = dist(2), p2 = dist(3), k3 = dist(4);

    std::vector<Eigen::Vector2d> result;
    result.reserve(pts3d.size());

    for (const auto &pt : pts3d) {
        Eigen::Vector3d cam = R * pt + tvec;
        double xp = cam(0) / cam(2);
        double yp = cam(1) / cam(2);

        double r2 = xp * xp + yp * yp;
        double r4 = r2 * r2;
        double r6 = r4 * r2;
        double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        double xpp = xp * radial + 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp);
        double ypp = yp * radial + p1 * (r2 + 2.0 * yp * yp) + 2.0 * p2 * xp * yp;

        result.push_back(Eigen::Vector2d(xpp * fx + cx, ypp * fy + cy));
    }
    return result;
}

// Convenience: project a single 3D point
inline Eigen::Vector2d
projectPoint(const Eigen::Vector3d &pt3d, const Eigen::Vector3d &rvec,
             const Eigen::Vector3d &tvec, const Eigen::Matrix3d &K,
             const Eigen::Matrix<double, 5, 1> &dist) {
    return projectPoints({pt3d}, rvec, tvec, K, dist)[0];
}

// Convenience: project a single 3D point with zero distortion
inline Eigen::Vector2d
projectPointNoDist(const Eigen::Vector3d &pt3d, const Eigen::Vector3d &rvec,
                   const Eigen::Vector3d &tvec, const Eigen::Matrix3d &K) {
    Eigen::Matrix<double, 5, 1> zero_dist = Eigen::Matrix<double, 5, 1>::Zero();
    return projectPoints({pt3d}, rvec, tvec, K, zero_dist)[0];
}

} // namespace red_math
