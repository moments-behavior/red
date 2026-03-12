#pragma once
// Telecentric DLT Calibration — affine camera model for telecentric lenses.
// Ports the lab's MATLAB procedure (fitTelecentricDLT + calibrateTelecentricMulti
// + refineTelecentricBA) to C++/Eigen.

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace TelecentricDLT {

constexpr double DLT_UNLABELED = 1e7;

// ── Calibration Method ──────────────────────────────────────────────────────

enum class Method {
    LinearDLT = 0,       // Pure affine (8 params/cam)
    DLT_k1 = 1,          // Affine + radial k1 (9 params/cam)
    DLT_k1k2 = 2,        // Affine + radial k1,k2 (10 params/cam)
};

inline const char *method_name(Method m) {
    switch (m) {
    case Method::LinearDLT: return "Linear DLT";
    case Method::DLT_k1:    return "DLT + k1";
    case Method::DLT_k1k2:  return "DLT + k1,k2";
    }
    return "Unknown";
}

// ── Configuration ───────────────────────────────────────────────────────────

struct DLTConfig {
    std::vector<std::string> camera_names; // serial numbers
    std::string landmark_labels_folder;    // folder with Cam<serial>.csv 2D labels
    std::string landmarks_3d_file;         // CSV with 3D landmark coordinates
    std::string output_folder;             // where to write DLT CSVs
    int image_width = 0;
    int image_height = 0;
    bool flip_y = true;        // MATLAB convention: y = image_height - y
    bool square_pixels = false; // enforce sx == sy in K2
    bool zero_skew = false;     // enforce skew k == 0 in K2
    bool do_ba = true;          // run bundle adjustment refinement
    Method method = Method::LinearDLT;
};

// ── Results ─────────────────────────────────────────────────────────────────

struct PerCameraResult {
    std::string serial;
    Eigen::Matrix<double, 3, 4> P; // [A t; 0 0 0 1]
    Eigen::Matrix<double, 2, 3> A;
    Eigen::Vector2d t;
    Eigen::Matrix3d R;
    double sx = 0, sy = 0;
    double skew = 0;
    double rmse_init = 0;
    double rmse_ba = 0;
    int num_points = 0;
    // Radial distortion (relative to normalized coords centered on principal point)
    double k1 = 0, k2 = 0;
    Eigen::Vector2d dist_center = Eigen::Vector2d::Zero(); // principal point in pixels

    // Return the best available RMSE (post-BA if available, otherwise initial).
    double final_rmse() const { return (rmse_ba > 0) ? rmse_ba : rmse_init; }
};

struct CrossValidationResult {
    double loo_rmse = 0;
    std::vector<double> point_errors;
    std::vector<int> point_indices;
    std::vector<int> outlier_indices;
};

struct DLTResult {
    bool success = false;
    std::string error;
    std::vector<PerCameraResult> cameras;
    double mean_rmse = 0;
    std::string output_folder;
    std::vector<CrossValidationResult> cv_results;
    Method method = Method::LinearDLT;
};

// ── CSV Parsing ─────────────────────────────────────────────────────────────

// Parse 3D landmarks CSV: header "x,y,z", then rows of x,y,z
inline std::vector<Eigen::Vector3d> parse_3d_landmarks(const std::string &path) {
    std::vector<Eigen::Vector3d> pts;
    std::ifstream f(path);
    if (!f.is_open()) return pts;
    std::string line;
    std::getline(f, line); // skip header
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tok;
        double x, y, z;
        std::getline(ss, tok, ','); x = std::stod(tok);
        std::getline(ss, tok, ','); y = std::stod(tok);
        std::getline(ss, tok, ','); z = std::stod(tok);
        pts.emplace_back(x, y, z);
    }
    return pts;
}

// Parse 2D labels CSV: header "Target", rows of frame_id,kp_id,x,y
// Returns vector indexed by landmark_id (frame_id). Sentinel DLT_UNLABELED = unlabeled.
inline std::vector<Eigen::Vector2d> parse_2d_labels(
    const std::string &path, int image_height, bool flip_y) {
    std::vector<Eigen::Vector2d> pts;
    std::ifstream f(path);
    if (!f.is_open()) return pts;
    std::string line;
    std::getline(f, line); // skip header ("Target")
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tok;
        int frame_id, kp_id;
        double x, y;
        std::getline(ss, tok, ','); frame_id = std::stoi(tok);
        std::getline(ss, tok, ','); kp_id = std::stoi(tok);
        std::getline(ss, tok, ','); x = std::stod(tok);
        std::getline(ss, tok, ','); y = std::stod(tok);
        // Ensure vector is large enough
        if (frame_id >= (int)pts.size())
            pts.resize(frame_id + 1, Eigen::Vector2d(DLT_UNLABELED, DLT_UNLABELED));
        if (x > DLT_UNLABELED * 0.9 || y > DLT_UNLABELED * 0.9) {
            pts[frame_id] = Eigen::Vector2d(DLT_UNLABELED, DLT_UNLABELED); // sentinel
        } else {
            if (flip_y) y = image_height - y;
            pts[frame_id] = Eigen::Vector2d(x, y);
        }
    }
    return pts;
}

// ── Normalization ───────────────────────────────────────────────────────────

// 2D normalization: translate centroid to origin, scale RMS distance to sqrt(2)
inline Eigen::Matrix3d normalize_2d(const std::vector<Eigen::Vector2d> &pts,
                                     const std::vector<int> &idx) {
    Eigen::Vector2d mean = Eigen::Vector2d::Zero();
    for (int i : idx) mean += pts[i];
    mean /= idx.size();

    double rms = 0;
    for (int i : idx) rms += (pts[i] - mean).squaredNorm();
    rms = std::sqrt(rms / idx.size());
    double s = (rms > 1e-12) ? std::sqrt(2.0) / rms : 1.0;

    Eigen::Matrix3d T = Eigen::Matrix3d::Identity();
    T(0, 0) = s; T(1, 1) = s;
    T(0, 2) = -s * mean.x();
    T(1, 2) = -s * mean.y();
    return T;
}

// 3D normalization: translate centroid to origin, scale RMS distance to sqrt(3)
inline Eigen::Matrix4d normalize_3d(const std::vector<Eigen::Vector3d> &pts,
                                     const std::vector<int> &idx) {
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (int i : idx) mean += pts[i];
    mean /= idx.size();

    double rms = 0;
    for (int i : idx) rms += (pts[i] - mean).squaredNorm();
    rms = std::sqrt(rms / idx.size());
    double s = (rms > 1e-12) ? std::sqrt(3.0) / rms : 1.0;

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T(0, 0) = s; T(1, 1) = s; T(2, 2) = s;
    T(0, 3) = -s * mean.x();
    T(1, 3) = -s * mean.y();
    T(2, 3) = -s * mean.z();
    return T;
}

// ── Core DLT Fit (single camera) ───────────────────────────────────────────

// Fit affine projection: [u; v] = A * [X; Y; Z] + t (no perspective divide)
// Returns PerCameraResult with P, A, t, R, sx, sy, rmse_init
inline PerCameraResult fit_telecentric_dlt(
    const std::vector<Eigen::Vector3d> &pts3d,
    const std::vector<Eigen::Vector2d> &pts2d,
    const std::vector<int> &valid_idx) {
    PerCameraResult res;
    int N = (int)valid_idx.size();
    res.num_points = N;

    if (N < 4) {
        res.rmse_init = -1;
        return res;
    }

    // Normalization
    Eigen::Matrix3d T2 = normalize_2d(pts2d, valid_idx);
    Eigen::Matrix4d T3 = normalize_3d(pts3d, valid_idx);

    // Build 2N x 8 system: [XN 1 0 0 0 0; 0 0 0 0 XN 1] * p = [uN; vN]
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(2 * N, 8);
    Eigen::VectorXd b(2 * N);

    for (int i = 0; i < N; i++) {
        int idx = valid_idx[i];
        // Normalize 3D point
        Eigen::Vector4d Xh(pts3d[idx].x(), pts3d[idx].y(), pts3d[idx].z(), 1.0);
        Eigen::Vector4d XN = T3 * Xh;
        // Normalize 2D point
        Eigen::Vector3d xh(pts2d[idx].x(), pts2d[idx].y(), 1.0);
        Eigen::Vector3d xN = T2 * xh;

        // u equation (row 2*i)
        M(2 * i, 0) = XN(0);
        M(2 * i, 1) = XN(1);
        M(2 * i, 2) = XN(2);
        M(2 * i, 3) = 1.0;
        b(2 * i) = xN(0);

        // v equation (row 2*i+1)
        M(2 * i + 1, 4) = XN(0);
        M(2 * i + 1, 5) = XN(1);
        M(2 * i + 1, 6) = XN(2);
        M(2 * i + 1, 7) = 1.0;
        b(2 * i + 1) = xN(1);
    }

    // Solve via QR
    Eigen::VectorXd p = M.colPivHouseholderQr().solve(b);

    // Assemble normalized P: [An tn; 0 0 0 1]
    Eigen::Matrix<double, 3, 4> Pn;
    Pn << p(0), p(1), p(2), p(3),
          p(4), p(5), p(6), p(7),
          0,    0,    0,    1;

    // Denormalize: P = T2^{-1} * Pn * T3
    Eigen::Matrix3d T2inv = T2.inverse();
    Eigen::Matrix<double, 3, 4> P = T2inv * Pn * T3;

    res.P = P;
    res.A = P.block<2, 3>(0, 0);
    res.t = P.block<2, 1>(0, 3);

    // Orthographic decomposition: A = K2 * R(1:2,:)
    res.sx = res.A.row(0).norm();
    res.sy = res.A.row(1).norm();

    Eigen::Matrix<double, 2, 3> Anorm;
    Anorm.row(0) = res.A.row(0) / res.sx;
    Anorm.row(1) = res.A.row(1) / res.sy;

    // SVD to find closest matrix with orthonormal rows
    Eigen::JacobiSVD<Eigen::Matrix<double, 2, 3>> svd(
        Anorm, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    // Closest orthonormal: U * [I_2 0] * V^T
    Eigen::Matrix<double, 2, 3> S23 = Eigen::Matrix<double, 2, 3>::Zero();
    S23(0, 0) = 1.0;
    S23(1, 1) = 1.0;
    Eigen::Matrix<double, 2, 3> Aorth = U * S23 * V.transpose();

    Eigen::Vector3d r1 = Aorth.row(0).transpose();
    Eigen::Vector3d r2 = Aorth.row(1).transpose();
    Eigen::Vector3d r3 = r1.cross(r2);
    res.R.col(0) = r1;
    res.R.col(1) = r2;
    res.R.col(2) = r3;
    // Ensure proper rotation (det > 0)
    if (res.R.determinant() < 0)
        res.R.col(2) = -res.R.col(2);

    // Compute reprojection RMSE
    double sse = 0;
    for (int i = 0; i < N; i++) {
        int idx = valid_idx[i];
        Eigen::Vector2d proj = res.A * pts3d[idx] + res.t;
        sse += (proj - pts2d[idx]).squaredNorm();
    }
    res.rmse_init = std::sqrt(sse / N);

    return res;
}

// ── K2 Constraints ──────────────────────────────────────────────────────────

// Constrain K2 and rebuild A, t.
// square_pixels: enforce sx == sy. zero_skew: enforce k == 0.
inline void constrain_k2(PerCameraResult &res, bool square_pixels, bool zero_skew,
                          const std::vector<Eigen::Vector3d> &pts3d,
                          const std::vector<Eigen::Vector2d> &pts2d,
                          const std::vector<int> &valid_idx) {
    if (!square_pixels && !zero_skew) return;

    double sx = res.sx, sy = res.sy, k = res.skew;

    if (square_pixels) {
        double s = (sx + sy) / 2.0;
        sx = s;
        sy = s;
    }
    if (zero_skew) {
        k = 0;
    }

    // Build K2 (2x2): [sx k; 0 sy]
    Eigen::Matrix2d K2;
    K2 << sx, k,
          0,  sy;

    // Rebuild A from constrained K2 and R
    // B = top two rows of R: R columns are r1,r2,r3
    Eigen::Matrix<double, 2, 3> B;
    B.row(0) = res.R.col(0).transpose();
    B.row(1) = res.R.col(1).transpose();

    res.A = K2 * B;
    res.sx = sx;
    res.sy = sy;
    res.skew = k;

    // Refit translation: t = mean(x - A*X)
    int N = (int)valid_idx.size();
    Eigen::Vector2d t_sum = Eigen::Vector2d::Zero();
    for (int i = 0; i < N; i++) {
        int idx = valid_idx[i];
        t_sum += pts2d[idx] - res.A * pts3d[idx];
    }
    res.t = t_sum / N;

    // Rebuild P
    res.P.block<2, 3>(0, 0) = res.A;
    res.P.block<2, 1>(0, 3) = res.t;
    res.P.row(2) << 0, 0, 0, 1;

    // Recompute RMSE
    double sse = 0;
    for (int i = 0; i < N; i++) {
        int idx = valid_idx[i];
        Eigen::Vector2d proj = res.A * pts3d[idx] + res.t;
        sse += (proj - pts2d[idx]).squaredNorm();
    }
    res.rmse_init = std::sqrt(sse / N);
}

// ── Bundle Adjustment ───────────────────────────────────────────────────────

// Unified Levenberg-Marquardt bundle adjustment across all cameras.
// Parameters per camera: rotation vector (3) + translation (2) + K2 params (1-3)
//   + optional radial distortion coefficients (n_dist_coeffs: 0, 1, or 2).
//
// Distortion model (telecentric, when n_dist_coeffs > 0):
//   1. Orthographic: x_n = R(0,:)*X, y_n = R(1,:)*X  (normalized coords)
//   2. Distort:      r2 = x_n^2 + y_n^2
//                    x_d = x_n * (1 + k1*r2 + k2*r4)
//                    y_d = y_n * (1 + k1*r2 + k2*r4)
//   3. Pixel:        u = sx*x_d + skew*y_d + tx
//                    v = sy*y_d + ty
// When n_dist_coeffs == 0, the distortion factor is 1.0 (pure affine).
// The 3x4 output P uses the *undistorted* projection (for DLT triangulation),
// and distortion coefficients are saved separately.
// At triangulation time: undistort observed 2D -> then use P.

inline void refine_ba(
    std::vector<PerCameraResult> &cameras,
    const std::vector<Eigen::Vector3d> &pts3d,
    const std::vector<std::vector<Eigen::Vector2d>> &pts2d_all,
    const std::vector<std::vector<int>> &valid_idx_all,
    bool square_pixels, bool zero_skew, int n_dist_coeffs = 0) {

    int M = (int)cameras.size();
    int nK = 3; // sx, k, sy
    if (square_pixels && zero_skew) nK = 1;
    else if (square_pixels || zero_skew) nK = 2;

    // rot(3) + t(2) + K2(nK) + distortion(n_dist_coeffs)
    int params_per_cam = 3 + 2 + nK + n_dist_coeffs;
    int total_params = M * params_per_cam;

    int total_res = 0;
    for (int m = 0; m < M; m++)
        total_res += 2 * (int)valid_idx_all[m].size();

    if (total_res <= total_params) return;

    // Pack parameters
    Eigen::VectorXd x(total_params);
    for (int m = 0; m < M; m++) {
        int off = m * params_per_cam;
        Eigen::AngleAxisd aa(cameras[m].R);
        x.segment<3>(off) = aa.axis() * aa.angle();
        x.segment<2>(off + 3) = cameras[m].t;
        // K2 params
        int ko = off + 5;
        if (square_pixels && zero_skew) {
            x(ko) = (cameras[m].sx + cameras[m].sy) / 2.0;
        } else if (square_pixels) {
            x(ko) = (cameras[m].sx + cameras[m].sy) / 2.0;
            x(ko + 1) = cameras[m].skew;
        } else if (zero_skew) {
            x(ko) = cameras[m].sx;
            x(ko + 1) = cameras[m].sy;
        } else {
            x(ko) = cameras[m].sx;
            x(ko + 1) = cameras[m].skew;
            x(ko + 2) = cameras[m].sy;
        }
        // Distortion (initialize to 0)
        int do_ = off + 5 + nK;
        for (int d = 0; d < n_dist_coeffs; d++)
            x(do_ + d) = cameras[m].k1 * (d == 0) + cameras[m].k2 * (d == 1);
    }

    auto unpack_K2 = [&](const Eigen::VectorXd &p, int off) -> Eigen::Matrix2d {
        Eigen::Matrix2d K2;
        if (square_pixels && zero_skew) {
            double s = p(off); K2 << s, 0, 0, s;
        } else if (square_pixels) {
            double s = p(off); K2 << s, p(off + 1), 0, s;
        } else if (zero_skew) {
            K2 << p(off), 0, 0, p(off + 1);
        } else {
            K2 << p(off), p(off + 1), 0, p(off + 2);
        }
        return K2;
    };

    auto compute_residuals = [&](const Eigen::VectorXd &params) -> Eigen::VectorXd {
        Eigen::VectorXd r(total_res);
        int ri = 0;
        for (int m = 0; m < M; m++) {
            int off = m * params_per_cam;
            Eigen::Vector3d rv = params.segment<3>(off);
            double angle = rv.norm();
            Eigen::Matrix3d Rm = (angle > 1e-12)
                ? Eigen::AngleAxisd(angle, rv / angle).toRotationMatrix()
                : Eigen::Matrix3d::Identity();
            Eigen::Vector2d tm = params.segment<2>(off + 3);
            Eigen::Matrix2d K2 = unpack_K2(params, off + 5);

            int do_ = off + 5 + nK;
            double k1 = (n_dist_coeffs >= 1) ? params(do_) : 0;
            double k2 = (n_dist_coeffs >= 2) ? params(do_ + 1) : 0;

            Eigen::Vector3d r1 = Rm.col(0);
            Eigen::Vector3d r2 = Rm.col(1);

            for (int idx : valid_idx_all[m]) {
                // Normalized coords (orthographic projection)
                double xn = r1.dot(pts3d[idx]);
                double yn = r2.dot(pts3d[idx]);

                // Radial distortion
                double r2d = xn * xn + yn * yn;
                double r4d = r2d * r2d;
                double dist = 1.0 + k1 * r2d + k2 * r4d;
                double xd = xn * dist;
                double yd = yn * dist;

                // Pixel coords: K2 * [xd; yd] + t
                double u = K2(0, 0) * xd + K2(0, 1) * yd + tm.x();
                double v = K2(1, 1) * yd + tm.y();

                r(ri++) = u - pts2d_all[m][idx].x();
                r(ri++) = v - pts2d_all[m][idx].y();
            }
        }
        return r;
    };

    // Levenberg-Marquardt
    const int max_iter = 100;
    double lambda = 1e-3;
    Eigen::VectorXd r = compute_residuals(x);
    double cost = r.squaredNorm();

    for (int iter = 0; iter < max_iter; iter++) {
        const double eps = 1e-7;
        Eigen::MatrixXd J(total_res, total_params);
        for (int j = 0; j < total_params; j++) {
            Eigen::VectorXd x_plus = x;
            x_plus(j) += eps;
            J.col(j) = (compute_residuals(x_plus) - r) / eps;
        }

        Eigen::MatrixXd JtJ = J.transpose() * J;
        Eigen::VectorXd Jtr = J.transpose() * r;
        Eigen::VectorXd diag = JtJ.diagonal();
        for (int j = 0; j < total_params; j++)
            diag(j) = std::max(diag(j), 1e-6);
        JtJ.diagonal() += lambda * diag;

        Eigen::VectorXd dx = JtJ.ldlt().solve(-Jtr);
        Eigen::VectorXd x_new = x + dx;
        Eigen::VectorXd r_new = compute_residuals(x_new);
        double cost_new = r_new.squaredNorm();

        if (cost_new < cost) {
            double improvement = (cost - cost_new) / cost;
            x = x_new;
            r = r_new;
            cost = cost_new;
            lambda = std::max(lambda * 0.5, 1e-8);
            if (improvement < 1e-10) break;
        } else {
            lambda *= 5.0;
            if (lambda > 1e8) break;
        }
    }

    // Unpack results
    for (int m = 0; m < M; m++) {
        int off = m * params_per_cam;
        Eigen::Vector3d rv = x.segment<3>(off);
        double angle = rv.norm();
        cameras[m].R = (angle > 1e-12)
            ? Eigen::AngleAxisd(angle, rv / angle).toRotationMatrix()
            : Eigen::Matrix3d::Identity();
        cameras[m].t = x.segment<2>(off + 3);
        Eigen::Matrix2d K2 = unpack_K2(x, off + 5);
        cameras[m].sx = K2(0, 0);
        cameras[m].skew = K2(0, 1);
        cameras[m].sy = K2(1, 1);

        int do_ = off + 5 + nK;
        cameras[m].k1 = (n_dist_coeffs >= 1) ? x(do_) : 0;
        cameras[m].k2 = (n_dist_coeffs >= 2) ? x(do_ + 1) : 0;

        // Rebuild A and P from undistorted model (for DLT triangulation)
        Eigen::Matrix<double, 2, 3> B;
        B.row(0) = cameras[m].R.col(0).transpose();
        B.row(1) = cameras[m].R.col(1).transpose();
        cameras[m].A = K2 * B;
        cameras[m].P.block<2, 3>(0, 0) = cameras[m].A;
        cameras[m].P.block<2, 1>(0, 3) = cameras[m].t;
        cameras[m].P.row(2) << 0, 0, 0, 1;

        // Compute RMSE (with distortion model — the actual fit quality)
        double sse = 0;
        int N = (int)valid_idx_all[m].size();
        Eigen::Vector3d r1 = cameras[m].R.col(0);
        Eigen::Vector3d r2 = cameras[m].R.col(1);
        for (int idx : valid_idx_all[m]) {
            double xn = r1.dot(pts3d[idx]);
            double yn = r2.dot(pts3d[idx]);
            double r2d = xn * xn + yn * yn;
            double r4d = r2d * r2d;
            double dist = 1.0 + cameras[m].k1 * r2d + cameras[m].k2 * r4d;
            double u = K2(0, 0) * xn * dist + K2(0, 1) * yn * dist + cameras[m].t.x();
            double v = K2(1, 1) * yn * dist + cameras[m].t.y();
            Eigen::Vector2d obs = pts2d_all[m][idx];
            sse += (u - obs.x()) * (u - obs.x()) + (v - obs.y()) * (v - obs.y());
        }
        cameras[m].rmse_ba = std::sqrt(sse / N);
    }
}

// ── Cross-Validation ────────────────────────────────────────────────────────

// Leave-one-out cross-validation for a single camera's linear DLT fit.
// Returns LOO RMSE and flags outlier points (error > 3x median).
inline CrossValidationResult cross_validate_camera(
    const std::vector<Eigen::Vector3d> &pts3d,
    const std::vector<Eigen::Vector2d> &pts2d,
    const std::vector<int> &valid_idx) {
    CrossValidationResult cv;
    int N = (int)valid_idx.size();
    if (N < 5) return cv; // need at least 5 for LOO with 4+ remaining

    cv.point_indices = valid_idx;
    cv.point_errors.resize(N);

    double sse = 0;
    for (int i = 0; i < N; i++) {
        // Build leave-one-out index set
        std::vector<int> loo_idx;
        loo_idx.reserve(N - 1);
        for (int j = 0; j < N; j++) {
            if (j != i) loo_idx.push_back(valid_idx[j]);
        }

        // Refit without point i
        auto res = fit_telecentric_dlt(pts3d, pts2d, loo_idx);

        // Predict held-out point
        int idx = valid_idx[i];
        Eigen::Vector2d proj = res.A * pts3d[idx] + res.t;
        double err = (proj - pts2d[idx]).norm();
        cv.point_errors[i] = err;
        sse += err * err;
    }

    cv.loo_rmse = std::sqrt(sse / N);

    // Flag outliers: error > 3x median
    std::vector<double> sorted_errors = cv.point_errors;
    std::sort(sorted_errors.begin(), sorted_errors.end());
    double median = sorted_errors[N / 2];
    double threshold = 3.0 * median;
    for (int i = 0; i < N; i++) {
        if (cv.point_errors[i] > threshold) {
            cv.outlier_indices.push_back(valid_idx[i]);
        }
    }

    return cv;
}

// ── Label Import (DLT CSV → AnnotationMap) ──────────────────────────────────

// Import DLT-format 2D label CSVs into an AnnotationMap at a given frame.
// Reads Cam<name>.csv files from the labels folder and populates keypoints.
// Returns the number of total labeled points imported.
template <typename AnnotationMap>
inline int import_dlt_labels(
    AnnotationMap &annotations,
    int frame_num,
    int num_nodes,
    int num_cameras,
    const std::vector<std::string> &camera_names,
    const std::string &labels_folder) {

    namespace fs = std::filesystem;
    if (labels_folder.empty() || !fs::is_directory(labels_folder))
        return 0;

    int total = 0;
    auto &fa = annotations[frame_num];
    // Ensure cameras array is large enough
    if ((int)fa.cameras.size() < num_cameras)
        fa.cameras.resize(num_cameras);

    for (int c = 0; c < num_cameras; c++) {
        std::string cam_name = (c < (int)camera_names.size())
                                   ? camera_names[c]
                                   : "camera" + std::to_string(c);
        std::string path = labels_folder + "/" + cam_name + ".csv";

        auto pts = parse_2d_labels(path, 0, false); // no flip — raw coords

        // Ensure keypoints array is large enough
        if ((int)fa.cameras[c].keypoints.size() < num_nodes)
            fa.cameras[c].keypoints.resize(num_nodes);

        for (int n = 0; n < num_nodes && n < (int)pts.size(); n++) {
            if (pts[n].x() < DLT_UNLABELED * 0.9 && pts[n].y() < DLT_UNLABELED * 0.9) {
                fa.cameras[c].keypoints[n].x = pts[n].x();
                fa.cameras[c].keypoints[n].y = pts[n].y();
                fa.cameras[c].keypoints[n].labeled = true;
                total++;
            }
        }
    }

    // Initialize 3D keypoints array
    if ((int)fa.kp3d.size() < num_nodes)
        fa.kp3d.resize(num_nodes);

    return total;
}

// ── Label Export ─────────────────────────────────────────────────────────────

// Export annotation keypoints as DLT-format 2D label CSVs.
// Reads keypoints from a single frame (frame_num) across all cameras.
// Each keypoint index becomes a landmark index in the output.
// Output: one Cam<serial>.csv per camera with format: "Target\nidx,0,x,y\n..."
// Unlabeled keypoints are written with sentinel value 1e7.
struct ExportLabelsResult {
    bool success = false;
    std::string error;
    std::string output_folder;
    int num_cameras = 0;
    int num_landmarks = 0;
    int total_labeled = 0;
};

template <typename AnnotationMap>
inline ExportLabelsResult export_labels_for_dlt(
    const AnnotationMap &annotations,
    int frame_num,
    int num_nodes,
    int num_cameras,
    const std::vector<std::string> &camera_names,
    const std::string &output_folder,
    const std::string &skeleton_name = "Target") {

    ExportLabelsResult result;
    namespace fs = std::filesystem;

    std::error_code ec;
    fs::create_directories(output_folder, ec);
    if (ec) {
        result.error = "Cannot create folder: " + output_folder;
        return result;
    }

    if (annotations.find(frame_num) == annotations.end()) {
        result.error = "Frame " + std::to_string(frame_num) +
                       " has no annotations";
        return result;
    }

    const auto &fa = annotations.at(frame_num);
    result.num_cameras = num_cameras;
    result.num_landmarks = num_nodes;
    result.output_folder = output_folder;

    for (int c = 0; c < num_cameras; c++) {
        std::string cam_name = (c < (int)camera_names.size())
                                   ? camera_names[c]
                                   : "camera" + std::to_string(c);
        std::string path = output_folder + "/" + cam_name + ".csv";
        std::ofstream f(path);
        if (!f.is_open()) {
            result.error = "Cannot write: " + path;
            return result;
        }
        f << skeleton_name << "\n";

        if (c < (int)fa.cameras.size()) {
            for (int n = 0; n < num_nodes; n++) {
                if (n < (int)fa.cameras[c].keypoints.size() &&
                    fa.cameras[c].keypoints[n].labeled) {
                    f << n << ",0," << fa.cameras[c].keypoints[n].x << ","
                      << fa.cameras[c].keypoints[n].y << "\n";
                    result.total_labeled++;
                } else {
                    f << n << ",0,1e+07,1e+07\n";
                }
            }
        } else {
            for (int n = 0; n < num_nodes; n++)
                f << n << ",0,1e+07,1e+07\n";
        }
    }

    result.success = true;
    return result;
}

// ── Output ──────────────────────────────────────────────────────────────────

// Save 11-element DLT coefficients per camera (matching MATLAB format)
inline void save_dlt_coefficients(const DLTResult &result,
                                   const std::string &output_folder) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(output_folder, ec);

    for (const auto &cam : result.cameras) {
        std::string path = output_folder + "/Cam" + cam.serial + "_dlt.csv";
        std::ofstream f(path);
        if (!f.is_open()) continue;
        f << cam.P(0, 0) << "\n"
          << cam.P(0, 1) << "\n"
          << cam.P(0, 2) << "\n"
          << cam.P(0, 3) << "\n"
          << cam.P(1, 0) << "\n"
          << cam.P(1, 1) << "\n"
          << cam.P(1, 2) << "\n"
          << cam.P(1, 3) << "\n"
          << "0\n0\n0\n";
    }

    // If distortion model, also save distortion params per camera
    bool has_dist = false;
    for (const auto &cam : result.cameras)
        if (cam.k1 != 0 || cam.k2 != 0) { has_dist = true; break; }

    if (has_dist) {
        for (const auto &cam : result.cameras) {
            std::string path = output_folder + "/Cam" + cam.serial + "_distortion.csv";
            std::ofstream f(path);
            if (!f.is_open()) continue;
            f << "k1,k2,sx,sy,skew\n"
              << cam.k1 << "," << cam.k2 << ","
              << cam.sx << "," << cam.sy << "," << cam.skew << "\n";
        }
    }
}

// ── Main Entry Point ────────────────────────────────────────────────────────

inline DLTResult run_dlt_calibration(const DLTConfig &config,
                                      std::string *status_ptr = nullptr) {
    DLTResult result;
    namespace fs = std::filesystem;

    auto set_status = [&](const std::string &s) {
        if (status_ptr) *status_ptr = s;
    };

    // Parse 3D landmarks
    set_status("Parsing 3D landmarks...");
    auto pts3d = parse_3d_landmarks(config.landmarks_3d_file);
    if (pts3d.empty()) {
        result.error = "Failed to parse 3D landmarks: " + config.landmarks_3d_file;
        return result;
    }

    int M = (int)config.camera_names.size();
    std::vector<std::vector<Eigen::Vector2d>> pts2d_all(M);
    std::vector<std::vector<int>> valid_idx_all(M);

    // Parse 2D labels and build valid index lists
    for (int m = 0; m < M; m++) {
        std::string label_path = config.landmark_labels_folder + "/Cam" +
                                 config.camera_names[m] + ".csv";
        set_status("Parsing labels: Cam" + config.camera_names[m] + "...");
        pts2d_all[m] = parse_2d_labels(label_path, config.image_height, config.flip_y);

        if (pts2d_all[m].empty()) {
            result.error = "Failed to parse 2D labels: " + label_path;
            return result;
        }

        // Build valid indices (where both 2D and 3D are available)
        int N = std::min((int)pts3d.size(), (int)pts2d_all[m].size());
        for (int i = 0; i < N; i++) {
            if (pts2d_all[m][i].x() < DLT_UNLABELED * 0.9 && pts2d_all[m][i].y() < DLT_UNLABELED * 0.9) {
                valid_idx_all[m].push_back(i);
            }
        }

        if ((int)valid_idx_all[m].size() < 4) {
            result.error = "Cam" + config.camera_names[m] +
                           ": only " + std::to_string(valid_idx_all[m].size()) +
                           " valid correspondences (need >= 4)";
            return result;
        }
    }

    // Step 1: Per-camera linear DLT
    result.cameras.resize(M);
    for (int m = 0; m < M; m++) {
        set_status("DLT fit: Cam" + config.camera_names[m] +
                   " (" + std::to_string(m + 1) + "/" + std::to_string(M) + ")...");
        result.cameras[m] = fit_telecentric_dlt(pts3d, pts2d_all[m], valid_idx_all[m]);
        result.cameras[m].serial = config.camera_names[m];
    }

    // Cross-validation (LOO)
    set_status("Running cross-validation...");
    result.cv_results.resize(M);
    for (int m = 0; m < M; m++) {
        result.cv_results[m] = cross_validate_camera(
            pts3d, pts2d_all[m], valid_idx_all[m]);
    }

    // Step 2: K2 constraints
    if (config.square_pixels || config.zero_skew) {
        set_status("Applying K2 constraints...");
        for (int m = 0; m < M; m++) {
            constrain_k2(result.cameras[m], config.square_pixels, config.zero_skew,
                         pts3d, pts2d_all[m], valid_idx_all[m]);
        }
    }

    // Step 3: Bundle adjustment (with or without distortion)
    if (config.method == Method::LinearDLT) {
        if (config.do_ba) {
            set_status("Running bundle adjustment (" + std::to_string(M) + " cameras)...");
            refine_ba(result.cameras, pts3d, pts2d_all, valid_idx_all,
                      config.square_pixels, config.zero_skew, 0);
        }
    } else {
        // Distortion methods always use BA (that's where distortion is estimated)
        int n_dist = (config.method == Method::DLT_k1) ? 1 : 2;
        set_status("Running BA with " + std::to_string(n_dist) +
                   " distortion coeff(s) (" + std::to_string(M) + " cameras)...");
        refine_ba(result.cameras, pts3d, pts2d_all, valid_idx_all,
                  config.square_pixels, config.zero_skew, n_dist);
    }

    result.method = config.method;

    // Compute mean RMSE
    double sum_rmse = 0;
    for (const auto &cam : result.cameras) {
        sum_rmse += cam.final_rmse();
    }
    result.mean_rmse = sum_rmse / M;

    // Save output
    result.output_folder = config.output_folder;
    set_status("Saving DLT coefficients to " + result.output_folder + "...");
    save_dlt_coefficients(result, result.output_folder);

    result.success = true;
    set_status("DLT calibration complete. Mean RMSE: " +
               std::to_string(result.mean_rmse) + " px");
    return result;
}

} // namespace TelecentricDLT
