#pragma once
// intrinsic_calibration.h — Zhang's camera calibration method (Eigen + Ceres).
// Replaces cv::calibrateCamera for planar calibration targets (ChArUco boards).
//
// Algorithm:
//   1. DLT homography estimation (planar object → image)
//   2. Zhang's closed-form intrinsic extraction from homographies
//   3. Per-image extrinsic estimation from K⁻¹H
//   4. Ceres bundle adjustment refinement
//
// Reference: Z. Zhang, "A Flexible New Technique for Camera Calibration",
// IEEE TPAMI, 2000.

#include "red_math.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <cmath>
#include <vector>

namespace intrinsic_calib {

// ─────────────────────────────────────────────────────────────────────────────
// Step 1: Estimate homography from planar object points to image points
// ─────────────────────────────────────────────────────────────────────────────

inline Eigen::Matrix3d
estimateHomography(const std::vector<Eigen::Vector2d> &obj_pts_2d,
                   const std::vector<Eigen::Vector2d> &img_pts) {
    int n = (int)obj_pts_2d.size();

    std::vector<Eigen::Vector2d> obj_norm, img_norm;
    Eigen::Matrix3d T_obj = red_math::hartleyNormalize(obj_pts_2d, obj_norm);
    Eigen::Matrix3d T_img = red_math::hartleyNormalize(img_pts, img_norm);

    Eigen::MatrixXd A(2 * n, 9);
    for (int i = 0; i < n; i++) {
        double X = obj_norm[i].x(), Y = obj_norm[i].y();
        double u = img_norm[i].x(), v = img_norm[i].y();
        A(2*i,   0) = X;     A(2*i,   1) = Y;     A(2*i,   2) = 1.0;
        A(2*i,   3) = 0.0;   A(2*i,   4) = 0.0;   A(2*i,   5) = 0.0;
        A(2*i,   6) = -u*X;  A(2*i,   7) = -u*Y;  A(2*i,   8) = -u;
        A(2*i+1, 0) = 0.0;   A(2*i+1, 1) = 0.0;   A(2*i+1, 2) = 0.0;
        A(2*i+1, 3) = X;     A(2*i+1, 4) = Y;     A(2*i+1, 5) = 1.0;
        A(2*i+1, 6) = -v*X;  A(2*i+1, 7) = -v*Y;  A(2*i+1, 8) = -v;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd h = svd.matrixV().col(8);
    Eigen::Matrix3d H_norm;
    H_norm << h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7), h(8);

    Eigen::Matrix3d H = T_img.inverse() * H_norm * T_obj;
    H /= H(2, 2);
    return H;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 2: Extract intrinsics from homographies (Zhang's method)
// ─────────────────────────────────────────────────────────────────────────────

inline Eigen::Matrix<double, 1, 6>
vij(const Eigen::Matrix3d &H, int i, int j) {
    Eigen::Matrix<double, 1, 6> v;
    v(0) = H(0,i)*H(0,j);
    v(1) = H(0,i)*H(1,j) + H(1,i)*H(0,j);
    v(2) = H(1,i)*H(1,j);
    v(3) = H(2,i)*H(0,j) + H(0,i)*H(2,j);
    v(4) = H(2,i)*H(1,j) + H(1,i)*H(2,j);
    v(5) = H(2,i)*H(2,j);
    return v;
}

inline bool
extractIntrinsicsFromHomographies(
    const std::vector<Eigen::Matrix3d> &homographies,
    int image_width, int image_height,
    bool fix_aspect_ratio,
    Eigen::Matrix3d &K) {

    int num_views = (int)homographies.size();
    if (num_views < 3) return false;

    // Build constraint matrix V*b = 0
    Eigen::MatrixXd V(2 * num_views, 6);
    for (int k = 0; k < num_views; k++) {
        V.row(2*k)   = vij(homographies[k], 0, 1);
        V.row(2*k+1) = vij(homographies[k], 0, 0) - vij(homographies[k], 1, 1);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(V, Eigen::ComputeFullV);
    Eigen::VectorXd b = svd.matrixV().col(5);

    double B11 = b(0), B12 = b(1), B22 = b(2);
    double B13 = b(3), B23 = b(4), B33 = b(5);

    double d = B11 * B22 - B12 * B12;
    if (std::abs(d) < 1e-15) return false;

    double v0 = (B12 * B13 - B11 * B23) / d;
    double lambda = B33 - (B13 * B13 + v0 * (B12 * B13 - B11 * B23)) / B11;

    if (lambda / B11 < 0 || lambda / d < 0) return false;

    double alpha = std::sqrt(std::abs(lambda / B11));
    double beta = std::sqrt(std::abs(lambda * B11 / d));
    double gamma = -B12 * alpha * alpha * beta / lambda;
    double u0 = gamma * v0 / beta - B13 * alpha * alpha / lambda;

    if (fix_aspect_ratio) {
        double f = (alpha + beta) * 0.5;
        alpha = beta = f;
        gamma = 0;
    }

    // Validate: focal length and principal point should be reasonable
    if (alpha < image_width * 0.3 || alpha > image_width * 5.0 ||
        u0 < -image_width || u0 > 2 * image_width ||
        v0 < -image_height || v0 > 2 * image_height) {
        return false; // Signal caller to use fallback
    }

    K = Eigen::Matrix3d::Zero();
    K(0, 0) = alpha;
    K(0, 1) = gamma;
    K(0, 2) = u0;
    K(1, 1) = beta;
    K(1, 2) = v0;
    K(2, 2) = 1.0;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 3: Estimate per-image extrinsics from K and H
// ─────────────────────────────────────────────────────────────────────────────

inline void
estimateExtrinsics(const Eigen::Matrix3d &K, const Eigen::Matrix3d &H,
                   Eigen::Matrix3d &R, Eigen::Vector3d &t) {
    Eigen::Matrix3d Kinv = K.inverse();
    Eigen::Matrix3d M = Kinv * H;

    // λ = average of 1/||m1|| and 1/||m2|| for robustness
    double lambda = 0.5 * (1.0 / M.col(0).norm() + 1.0 / M.col(1).norm());
    Eigen::Vector3d r1 = lambda * M.col(0);
    Eigen::Vector3d r2 = lambda * M.col(1);
    t = lambda * M.col(2);

    Eigen::Vector3d r3 = r1.cross(r2);

    Eigen::Matrix3d R_approx;
    R_approx.col(0) = r1;
    R_approx.col(1) = r2;
    R_approx.col(2) = r3;

    // Project to nearest proper rotation via SVD
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        R_approx, Eigen::ComputeFullU | Eigen::ComputeFullV);
    R = svd.matrixU() * svd.matrixV().transpose();
    if (R.determinant() < 0) R = -R;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 4: Ceres cost functors
// ─────────────────────────────────────────────────────────────────────────────

// Cost functor with shared focal length (fix_aspect_ratio = true).
// intrinsics[0] = f, [1] = cx, [2] = cy, [3..7] = k1,k2,p1,p2,k3
struct IntrinsicReprojCostFixedAR {
    double obs_x, obs_y, obj_x, obj_y;

    IntrinsicReprojCostFixedAR(double ox, double oy, double px, double py)
        : obs_x(ox), obs_y(oy), obj_x(px), obj_y(py) {}

    template <typename T>
    bool operator()(const T *intr, const T *ext, T *res) const {
        T pt[3] = {T(obj_x), T(obj_y), T(0)};
        T p[3];
        ceres::AngleAxisRotatePoint(ext, pt, p);
        p[0] += ext[3]; p[1] += ext[4]; p[2] += ext[5];

        T xp = p[0] / p[2], yp = p[1] / p[2];
        T r2 = xp*xp + yp*yp, r4 = r2*r2, r6 = r4*r2;
        T k1 = intr[3], k2 = intr[4], p1 = intr[5], p2 = intr[6], k3 = intr[7];
        T radial = T(1) + k1*r2 + k2*r4 + k3*r6;
        T xpp = xp*radial + T(2)*p1*xp*yp + p2*(r2 + T(2)*xp*xp);
        T ypp = yp*radial + p1*(r2 + T(2)*yp*yp) + T(2)*p2*xp*yp;

        T f = intr[0], cx = intr[1], cy = intr[2];
        res[0] = f*xpp + cx - T(obs_x);
        res[1] = f*ypp + cy - T(obs_y);
        return true;
    }

    static ceres::CostFunction *
    Create(double ox, double oy, double px, double py) {
        return new ceres::AutoDiffCostFunction<IntrinsicReprojCostFixedAR, 2, 8, 6>(
            new IntrinsicReprojCostFixedAR(ox, oy, px, py));
    }
};

// Cost functor with separate fx, fy (fix_aspect_ratio = false).
// intrinsics[0] = fx, [1] = fy, [2] = cx, [3] = cy, [4..8] = k1,k2,p1,p2,k3
struct IntrinsicReprojCost {
    double obs_x, obs_y, obj_x, obj_y;

    IntrinsicReprojCost(double ox, double oy, double px, double py)
        : obs_x(ox), obs_y(oy), obj_x(px), obj_y(py) {}

    template <typename T>
    bool operator()(const T *intr, const T *ext, T *res) const {
        T pt[3] = {T(obj_x), T(obj_y), T(0)};
        T p[3];
        ceres::AngleAxisRotatePoint(ext, pt, p);
        p[0] += ext[3]; p[1] += ext[4]; p[2] += ext[5];

        T xp = p[0] / p[2], yp = p[1] / p[2];
        T r2 = xp*xp + yp*yp, r4 = r2*r2, r6 = r4*r2;
        T k1 = intr[4], k2 = intr[5], p1 = intr[6], p2 = intr[7], k3 = intr[8];
        T radial = T(1) + k1*r2 + k2*r4 + k3*r6;
        T xpp = xp*radial + T(2)*p1*xp*yp + p2*(r2 + T(2)*xp*xp);
        T ypp = yp*radial + p1*(r2 + T(2)*yp*yp) + T(2)*p2*xp*yp;

        T fx = intr[0], fy = intr[1], cx = intr[2], cy = intr[3];
        res[0] = fx*xpp + cx - T(obs_x);
        res[1] = fy*ypp + cy - T(obs_y);
        return true;
    }

    static ceres::CostFunction *
    Create(double ox, double oy, double px, double py) {
        return new ceres::AutoDiffCostFunction<IntrinsicReprojCost, 2, 9, 6>(
            new IntrinsicReprojCost(ox, oy, px, py));
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Top-level: calibrate a single camera (replaces cv::calibrateCamera)
// ─────────────────────────────────────────────────────────────────────────────

struct CalibResult {
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 5, 1> dist = Eigen::Matrix<double, 5, 1>::Zero();
    double reproj_error = 0.0;
    bool success = false;
};

inline CalibResult
calibrateCamera(
    const std::vector<std::vector<Eigen::Vector3f>> &obj_points_per_view,
    const std::vector<std::vector<Eigen::Vector2f>> &img_points_per_view,
    int image_width, int image_height,
    bool fix_aspect_ratio = true) {

    CalibResult result;
    int num_views = (int)obj_points_per_view.size();
    if (num_views < 3) return result;

    // ── Step 1: Estimate homographies ──
    std::vector<Eigen::Matrix3d> homographies(num_views);
    for (int v = 0; v < num_views; v++) {
        int n = (int)obj_points_per_view[v].size();
        std::vector<Eigen::Vector2d> obj_2d(n), img_2d(n);
        for (int i = 0; i < n; i++) {
            obj_2d[i] = Eigen::Vector2d(obj_points_per_view[v][i].x(),
                                        obj_points_per_view[v][i].y());
            img_2d[i] = Eigen::Vector2d(img_points_per_view[v][i].x(),
                                        img_points_per_view[v][i].y());
        }
        homographies[v] = estimateHomography(obj_2d, img_2d);
    }

    // ── Step 2: Extract initial K from homographies ──
    Eigen::Matrix3d K;
    bool closed_form_ok = extractIntrinsicsFromHomographies(
        homographies, image_width, image_height, fix_aspect_ratio, K);

    if (!closed_form_ok) {
        // Fallback: reasonable initial guess
        K = Eigen::Matrix3d::Identity();
        double f_init = image_width; // typical for most cameras
        K(0, 0) = f_init;
        K(1, 1) = f_init;
        K(0, 2) = image_width / 2.0;
        K(1, 2) = image_height / 2.0;
    }

    // ── Step 3: Estimate per-image extrinsics from K and H ──
    std::vector<Eigen::Vector3d> rvecs(num_views), tvecs(num_views);
    for (int v = 0; v < num_views; v++) {
        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        estimateExtrinsics(K, homographies[v], R, t);
        rvecs[v] = red_math::rotationMatrixToVector(R);
        tvecs[v] = t;
    }

    // ── Step 4: Ceres refinement ──
    // Pack per-view extrinsics: [rvec(3), tvec(3)]
    std::vector<std::array<double, 6>> extrinsics(num_views);
    for (int v = 0; v < num_views; v++) {
        extrinsics[v] = {rvecs[v].x(), rvecs[v].y(), rvecs[v].z(),
                         tvecs[v].x(), tvecs[v].y(), tvecs[v].z()};
    }

    // Intrinsic parameter layout depends on fix_aspect_ratio
    int intr_size = fix_aspect_ratio ? 8 : 9;
    std::vector<double> intrinsics(intr_size, 0.0);

    if (fix_aspect_ratio) {
        // [f, cx, cy, k1, k2, p1, p2, k3]
        intrinsics[0] = K(0, 0);        // f
        intrinsics[1] = K(0, 2);        // cx
        intrinsics[2] = K(1, 2);        // cy
        // dist = 0 initially
    } else {
        // [fx, fy, cx, cy, k1, k2, p1, p2, k3]
        intrinsics[0] = K(0, 0);        // fx
        intrinsics[1] = K(1, 1);        // fy
        intrinsics[2] = K(0, 2);        // cx
        intrinsics[3] = K(1, 2);        // cy
    }

    ceres::Problem problem;
    auto *loss = new ceres::CauchyLoss(2.0); // Robust loss for outliers
    int total_points = 0;
    for (int v = 0; v < num_views; v++) {
        int n = (int)obj_points_per_view[v].size();
        for (int i = 0; i < n; i++) {
            ceres::CostFunction *cost;
            if (fix_aspect_ratio) {
                cost = IntrinsicReprojCostFixedAR::Create(
                    img_points_per_view[v][i].x(),
                    img_points_per_view[v][i].y(),
                    obj_points_per_view[v][i].x(),
                    obj_points_per_view[v][i].y());
            } else {
                cost = IntrinsicReprojCost::Create(
                    img_points_per_view[v][i].x(),
                    img_points_per_view[v][i].y(),
                    obj_points_per_view[v][i].x(),
                    obj_points_per_view[v][i].y());
            }
            problem.AddResidualBlock(cost, loss,
                                     intrinsics.data(), extrinsics[v].data());
            total_points++;
        }
    }

    // Add bounds on intrinsic parameters to prevent divergence.
    // Focal length should be between 0.3x and 5x the image width.
    // Principal point should be within the image bounds (with margin).
    if (fix_aspect_ratio) {
        // [f, cx, cy, k1, k2, p1, p2, k3]
        problem.SetParameterLowerBound(intrinsics.data(), 0,
                                        image_width * 0.3);
        problem.SetParameterUpperBound(intrinsics.data(), 0,
                                        image_width * 5.0);
        problem.SetParameterLowerBound(intrinsics.data(), 1, -image_width * 0.5);
        problem.SetParameterUpperBound(intrinsics.data(), 1, image_width * 1.5);
        problem.SetParameterLowerBound(intrinsics.data(), 2, -image_height * 0.5);
        problem.SetParameterUpperBound(intrinsics.data(), 2, image_height * 1.5);
    } else {
        // [fx, fy, cx, cy, k1, k2, p1, p2, k3]
        problem.SetParameterLowerBound(intrinsics.data(), 0,
                                        image_width * 0.3);
        problem.SetParameterUpperBound(intrinsics.data(), 0,
                                        image_width * 5.0);
        problem.SetParameterLowerBound(intrinsics.data(), 1,
                                        image_width * 0.3);
        problem.SetParameterUpperBound(intrinsics.data(), 1,
                                        image_width * 5.0);
        problem.SetParameterLowerBound(intrinsics.data(), 2, -image_width * 0.5);
        problem.SetParameterUpperBound(intrinsics.data(), 2, image_width * 1.5);
        problem.SetParameterLowerBound(intrinsics.data(), 3, -image_height * 0.5);
        problem.SetParameterUpperBound(intrinsics.data(), 3, image_height * 1.5);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 100;
    options.function_tolerance = 1e-10;
    options.parameter_tolerance = 1e-10;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Unpack results
    result.K = Eigen::Matrix3d::Identity();
    if (fix_aspect_ratio) {
        result.K(0, 0) = intrinsics[0]; // f
        result.K(1, 1) = intrinsics[0]; // f
        result.K(0, 2) = intrinsics[1]; // cx
        result.K(1, 2) = intrinsics[2]; // cy
        result.dist << intrinsics[3], intrinsics[4], intrinsics[5],
                       intrinsics[6], intrinsics[7];
    } else {
        result.K(0, 0) = intrinsics[0]; // fx
        result.K(1, 1) = intrinsics[1]; // fy
        result.K(0, 2) = intrinsics[2]; // cx
        result.K(1, 2) = intrinsics[3]; // cy
        result.dist << intrinsics[4], intrinsics[5], intrinsics[6],
                       intrinsics[7], intrinsics[8];
    }

    // Compute mean reprojection error
    double total_err = 0.0;
    for (int v = 0; v < num_views; v++) {
        int n = (int)obj_points_per_view[v].size();
        for (int i = 0; i < n; i++) {
            Eigen::Vector3d pt3d(obj_points_per_view[v][i].x(),
                                 obj_points_per_view[v][i].y(), 0.0);
            Eigen::Vector3d rv(extrinsics[v][0], extrinsics[v][1],
                               extrinsics[v][2]);
            Eigen::Vector3d tv(extrinsics[v][3], extrinsics[v][4],
                               extrinsics[v][5]);
            Eigen::Vector2d proj = red_math::projectPoint(
                pt3d, rv, tv, result.K, result.dist);
            double dx = proj.x() - img_points_per_view[v][i].x();
            double dy = proj.y() - img_points_per_view[v][i].y();
            total_err += std::sqrt(dx * dx + dy * dy);
        }
    }
    result.reproj_error = (total_points > 0) ? (total_err / total_points) : 0.0;
    result.success = summary.IsSolutionUsable();
    return result;
}

} // namespace intrinsic_calib
