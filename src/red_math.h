#pragma once
// red_math.h — Eigen-based replacements for OpenCV camera math functions.
// Replaces: cv::sfm::projectionFromKRt, cv::Rodrigues, cv::undistortPoints,
//           cv::sfm::triangulatePoints, cv::projectPoints

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <cmath>
#include <random>
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
        double x_prev = x, y_prev = y;
        double r2 = x * x + y * y;
        double r4 = r2 * r2;
        double r6 = r4 * r2;
        double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        double dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        double dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
        x = (x0 - dx) / radial;
        y = (y0 - dy) / radial;
        if (std::abs(x - x_prev) + std::abs(y - y_prev) < 1e-14) break;
    }

    // Re-project to pixel coords using K
    return Eigen::Vector2d(x * fx + cx, y * fy + cy);
}

// ---- Telecentric undistortion ----
// For telecentric cameras with radial distortion (k1, k2).
// K stores telecentric intrinsics: K = [sx skew tx; 0 sy ty; 0 0 1]
// dist_coeffs: [k1, k2, 0, 0, 0] (only first two used)
// Distortion model: x_d = x_n * (1 + k1*r2 + k2*r4) where r2 = x_n^2 + y_n^2
// Given distorted pixel, returns undistorted pixel.
inline Eigen::Vector2d
undistortPointTelecentric(const Eigen::Vector2d &pt, const Eigen::Matrix3d &K,
                          const Eigen::Matrix<double, 5, 1> &dist) {
    double k1 = dist(0), k2 = dist(1);

    // No distortion — early out
    if (std::abs(k1) < 1e-15 && std::abs(k2) < 1e-15)
        return pt;

    // Unpack K2 and translation
    double sx = K(0, 0), skew = K(0, 1), tx = K(0, 2);
    double sy = K(1, 1), ty = K(1, 2);

    // Pixel → normalized distorted coords: K2^{-1} * (pt - t)
    // K2 = [sx skew; 0 sy], K2^{-1} = [1/sx -skew/(sx*sy); 0 1/sy]
    double xd = (pt(0) - tx) / sx - skew / (sx * sy) * (pt(1) - ty);
    double yd = (pt(1) - ty) / sy;

    // Iterative undistortion: find (xn, yn) such that
    //   xd = xn * (1 + k1*r2 + k2*r4)
    //   yd = yn * (1 + k1*r2 + k2*r4)
    double xn = xd, yn = yd;
    for (int i = 0; i < 15; i++) {
        double xn_prev = xn, yn_prev = yn;
        double r2 = xn * xn + yn * yn;
        double r4 = r2 * r2;
        double d = 1.0 + k1 * r2 + k2 * r4;
        if (std::abs(d) < 1e-15) break;
        xn = xd / d;
        yn = yd / d;
        if (std::abs(xn - xn_prev) + std::abs(yn - yn_prev) < 1e-14) break;
    }

    // Normalized undistorted → pixel: K2 * (xn, yn) + t
    double u = sx * xn + skew * yn + tx;
    double v = sy * yn + ty;
    return Eigen::Vector2d(u, v);
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

// Convenience: project a single 3D point (rotation vector)
inline Eigen::Vector2d
projectPoint(const Eigen::Vector3d &pt3d, const Eigen::Vector3d &rvec,
             const Eigen::Vector3d &tvec, const Eigen::Matrix3d &K,
             const Eigen::Matrix<double, 5, 1> &dist) {
    return projectPoints({pt3d}, rvec, tvec, K, dist)[0];
}

// Project a single 3D point using rotation matrix directly.
// Safe for improper rotations (det=-1, e.g., after Z-flip) since it
// bypasses Rodrigues conversion which requires det=+1.
inline Eigen::Vector2d
projectPointR(const Eigen::Vector3d &pt3d, const Eigen::Matrix3d &R,
              const Eigen::Vector3d &tvec, const Eigen::Matrix3d &K,
              const Eigen::Matrix<double, 5, 1> &dist) {
    double fx = K(0,0), fy = K(1,1), cx = K(0,2), cy = K(1,2);
    double k1 = dist(0), k2 = dist(1), p1 = dist(2), p2 = dist(3), k3 = dist(4);
    Eigen::Vector3d cam = R * pt3d + tvec;
    double xp = cam(0) / cam(2), yp = cam(1) / cam(2);
    double r2 = xp*xp + yp*yp, r4 = r2*r2, r6 = r4*r2;
    double radial = 1.0 + k1*r2 + k2*r4 + k3*r6;
    double xpp = xp*radial + 2.0*p1*xp*yp + p2*(r2 + 2.0*xp*xp);
    double ypp = yp*radial + p1*(r2 + 2.0*yp*yp) + 2.0*p2*xp*yp;
    return Eigen::Vector2d(xpp*fx + cx, ypp*fy + cy);
}

// Convenience: project a single 3D point with zero distortion
inline Eigen::Vector2d
projectPointNoDist(const Eigen::Vector3d &pt3d, const Eigen::Vector3d &rvec,
                   const Eigen::Vector3d &tvec, const Eigen::Matrix3d &K) {
    Eigen::Matrix<double, 5, 1> zero_dist = Eigen::Matrix<double, 5, 1>::Zero();
    return projectPoints({pt3d}, rvec, tvec, K, zero_dist)[0];
}

// ---- Telecentric projection ----
// Project a 3D point using telecentric affine model (with optional radial distortion).
// P is 3x4 affine: [A t; 0 0 0 1]. K stores [sx skew tx; 0 sy ty; 0 0 1].
// dist_coeffs: [k1, k2, 0, 0, 0]
inline Eigen::Vector2d
projectPointTelecentric(const Eigen::Vector3d &pt3d,
                        const Eigen::Matrix<double, 3, 4> &P,
                        const Eigen::Matrix3d &K,
                        const Eigen::Matrix<double, 5, 1> &dist) {
    double k1 = dist(0), k2 = dist(1);

    // Affine projection (no distortion): u = P(0,:)*[X;1], v = P(1,:)*[X;1]
    Eigen::Vector4d Xh(pt3d.x(), pt3d.y(), pt3d.z(), 1.0);
    double u_lin = P.row(0).dot(Xh);
    double v_lin = P.row(1).dot(Xh);

    if (std::abs(k1) < 1e-15 && std::abs(k2) < 1e-15)
        return Eigen::Vector2d(u_lin, v_lin);

    // Apply distortion: pixel → normalized → distort → pixel
    double sx = K(0, 0), skew = K(0, 1), tx = K(0, 2);
    double sy = K(1, 1), ty = K(1, 2);

    // Undistorted normalized coords
    double xn = (u_lin - tx) / sx - skew / (sx * sy) * (v_lin - ty);
    double yn = (v_lin - ty) / sy;

    double r2 = xn * xn + yn * yn;
    double r4 = r2 * r2;
    double d = 1.0 + k1 * r2 + k2 * r4;
    double xd = xn * d;
    double yd = yn * d;

    return Eigen::Vector2d(sx * xd + skew * yd + tx, sy * yd + ty);
}

// ─────────────────────────────────────────────────────────────────────────────
// Essential / Fundamental matrix estimation (replaces OpenCV calib3d)
// ─────────────────────────────────────────────────────────────────────────────

// Sampson distance for an epipolar constraint x'^T F x.
// Returns the squared Sampson distance (used for RANSAC scoring).
inline double sampsonDistanceSq(const Eigen::Vector2d &x1,
                                const Eigen::Vector2d &x2,
                                const Eigen::Matrix3d &F) {
    Eigen::Vector3d h1(x1.x(), x1.y(), 1.0);
    Eigen::Vector3d h2(x2.x(), x2.y(), 1.0);
    double xFx = h2.dot(F * h1);
    Eigen::Vector3d Fx1 = F * h1;
    Eigen::Vector3d Ftx2 = F.transpose() * h2;
    double denom = Fx1(0) * Fx1(0) + Fx1(1) * Fx1(1) +
                   Ftx2(0) * Ftx2(0) + Ftx2(1) * Ftx2(1);
    if (denom < 1e-30) return 1e30;
    return (xFx * xFx) / denom;
}

// 8-point algorithm for essential/fundamental matrix from N point pairs.
// pts_a, pts_b: matched 2D points (normalized or pixel coords).
// Returns the 3×3 matrix (E or F depending on input normalization).
inline Eigen::Matrix3d
eightPointAlgorithm(const std::vector<Eigen::Vector2d> &pts_a,
                    const std::vector<Eigen::Vector2d> &pts_b,
                    const std::vector<int> &indices,
                    bool enforce_essential) {
    int n = (int)indices.size();
    Eigen::MatrixXd A(n, 9);
    for (int i = 0; i < n; i++) {
        const auto &pa = pts_a[indices[i]];
        const auto &pb = pts_b[indices[i]];
        double u1 = pa.x(), v1 = pa.y();
        double u2 = pb.x(), v2 = pb.y();
        A(i, 0) = u2 * u1;
        A(i, 1) = u2 * v1;
        A(i, 2) = u2;
        A(i, 3) = v2 * u1;
        A(i, 4) = v2 * v1;
        A(i, 5) = v2;
        A(i, 6) = u1;
        A(i, 7) = v1;
        A(i, 8) = 1.0;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd f = svd.matrixV().col(8);
    Eigen::Matrix3d M;
    M << f(0), f(1), f(2),
         f(3), f(4), f(5),
         f(6), f(7), f(8);

    // Enforce rank-2 constraint
    Eigen::JacobiSVD<Eigen::Matrix3d> svd2(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d sv = svd2.singularValues();

    if (enforce_essential) {
        // Essential matrix: two equal singular values, one zero
        double s = (sv(0) + sv(1)) * 0.5;
        M = svd2.matrixU() * Eigen::Vector3d(s, s, 0).asDiagonal() *
            svd2.matrixV().transpose();
    } else {
        // Fundamental matrix: zero out smallest singular value
        sv(2) = 0.0;
        M = svd2.matrixU() * sv.asDiagonal() * svd2.matrixV().transpose();
    }
    return M;
}

// Hartley normalization: translate+scale points so mean=0, avg_dist=sqrt(2).
// Returns the 3×3 normalization matrix T such that x_norm = T * [x; 1].
inline Eigen::Matrix3d
hartleyNormalize(const std::vector<Eigen::Vector2d> &pts,
                 std::vector<Eigen::Vector2d> &pts_out) {
    int n = (int)pts.size();
    Eigen::Vector2d mean = Eigen::Vector2d::Zero();
    for (const auto &p : pts) mean += p;
    mean /= n;

    double avg_dist = 0.0;
    for (const auto &p : pts) avg_dist += (p - mean).norm();
    avg_dist /= n;

    double scale = (avg_dist > 1e-12) ? (std::sqrt(2.0) / avg_dist) : 1.0;

    pts_out.resize(n);
    for (int i = 0; i < n; i++)
        pts_out[i] = scale * (pts[i] - mean);

    Eigen::Matrix3d T = Eigen::Matrix3d::Identity();
    T(0, 0) = scale;
    T(1, 1) = scale;
    T(0, 2) = -scale * mean.x();
    T(1, 2) = -scale * mean.y();
    return T;
}

struct EssentialMatResult {
    Eigen::Matrix3d E = Eigen::Matrix3d::Zero();
    std::vector<bool> inlier_mask;
    int num_inliers = 0;
    bool success = false;
};

// Find essential matrix from normalized point correspondences using RANSAC
// with the 8-point algorithm + Hartley normalization for conditioning.
// Points must be in normalized camera coordinates (i.e. K^{-1} * [pixel; 1]).
// threshold: Sampson distance threshold for inlier classification.
// confidence: RANSAC confidence (typically 0.999).
inline EssentialMatResult
findEssentialMatRANSAC(const std::vector<Eigen::Vector2d> &pts_a,
                       const std::vector<Eigen::Vector2d> &pts_b,
                       double confidence = 0.999,
                       double threshold = 0.001,
                       int max_iterations = 2000) {
    EssentialMatResult result;
    int n = (int)pts_a.size();
    if (n < 8) return result;

    // Hartley normalize for numerical conditioning (critical for 8-point)
    std::vector<Eigen::Vector2d> pts_a_h, pts_b_h;
    Eigen::Matrix3d Ta = hartleyNormalize(pts_a, pts_a_h);
    Eigen::Matrix3d Tb = hartleyNormalize(pts_b, pts_b_h);

    double thresh_sq = threshold * threshold;
    std::mt19937 rng(42); // deterministic seed for reproducibility

    int best_inlier_count = 0;
    Eigen::Matrix3d best_E = Eigen::Matrix3d::Zero();
    std::vector<bool> best_mask(n, false);

    // Adaptive RANSAC iteration count
    int adaptive_iters = max_iterations;

    for (int iter = 0; iter < adaptive_iters; iter++) {
        // Select 8 random distinct indices
        std::vector<int> sample;
        sample.reserve(8);
        while ((int)sample.size() < 8) {
            int idx = std::uniform_int_distribution<int>(0, n - 1)(rng);
            bool dup = false;
            for (int s : sample) if (s == idx) { dup = true; break; }
            if (!dup) sample.push_back(idx);
        }

        // Compute candidate E in Hartley-normalized space, then denormalize
        Eigen::Matrix3d E_h = eightPointAlgorithm(pts_a_h, pts_b_h, sample, false);
        Eigen::Matrix3d E_cand = Tb.transpose() * E_h * Ta;

        // Enforce essential matrix constraint on denormalized E
        Eigen::JacobiSVD<Eigen::Matrix3d> svd_e(
            E_cand, Eigen::ComputeFullU | Eigen::ComputeFullV);
        double s = (svd_e.singularValues()(0) + svd_e.singularValues()(1)) * 0.5;
        E_cand = svd_e.matrixU() * Eigen::Vector3d(s, s, 0).asDiagonal() *
                 svd_e.matrixV().transpose();

        // Score: count inliers by Sampson distance (in original coords)
        int inlier_count = 0;
        std::vector<bool> mask(n, false);
        for (int i = 0; i < n; i++) {
            double d2 = sampsonDistanceSq(pts_a[i], pts_b[i], E_cand);
            if (d2 < thresh_sq) {
                mask[i] = true;
                inlier_count++;
            }
        }

        if (inlier_count > best_inlier_count) {
            best_inlier_count = inlier_count;
            best_E = E_cand;
            best_mask = mask;

            // Update adaptive iteration count
            double inlier_ratio = (double)inlier_count / n;
            if (inlier_ratio > 0.0 && inlier_ratio < 1.0) {
                double p_fail = 1.0 - std::pow(inlier_ratio, 8);
                if (p_fail > 0.0 && p_fail < 1.0) {
                    int new_iters = (int)std::ceil(
                        std::log(1.0 - confidence) / std::log(p_fail));
                    adaptive_iters = std::min(max_iterations,
                                              std::max(new_iters, iter + 1));
                }
            }
        }
    }

    if (best_inlier_count < 8) return result;

    // Iterative refinement: refine E from inliers, update inlier set
    for (int refine_iter = 0; refine_iter < 3; refine_iter++) {
        std::vector<int> inlier_indices;
        for (int i = 0; i < n; i++)
            if (best_mask[i]) inlier_indices.push_back(i);

        Eigen::Matrix3d E_h = eightPointAlgorithm(
            pts_a_h, pts_b_h, inlier_indices, false);
        Eigen::Matrix3d E_ref = Tb.transpose() * E_h * Ta;

        // Enforce essential constraint
        Eigen::JacobiSVD<Eigen::Matrix3d> svd_r(
            E_ref, Eigen::ComputeFullU | Eigen::ComputeFullV);
        double s = (svd_r.singularValues()(0) + svd_r.singularValues()(1)) * 0.5;
        E_ref = svd_r.matrixU() * Eigen::Vector3d(s, s, 0).asDiagonal() *
                svd_r.matrixV().transpose();

        // Re-classify inliers
        int new_count = 0;
        std::vector<bool> new_mask(n, false);
        for (int i = 0; i < n; i++) {
            double d2 = sampsonDistanceSq(pts_a[i], pts_b[i], E_ref);
            if (d2 < thresh_sq) {
                new_mask[i] = true;
                new_count++;
            }
        }

        if (new_count >= best_inlier_count) {
            best_E = E_ref;
            best_mask = new_mask;
            best_inlier_count = new_count;
        }
    }

    result.E = best_E;
    result.inlier_mask = best_mask;
    result.num_inliers = best_inlier_count;
    result.success = true;
    return result;
}

// Find fundamental matrix from pixel-coordinate correspondences using RANSAC.
// Uses Hartley normalization + 8-point algorithm.
inline EssentialMatResult
findFundamentalMatRANSAC(const std::vector<Eigen::Vector2d> &pts_a,
                         const std::vector<Eigen::Vector2d> &pts_b,
                         double confidence = 0.999,
                         double threshold = 3.0,
                         int max_iterations = 1000) {
    EssentialMatResult result;
    int n = (int)pts_a.size();
    if (n < 8) return result;

    // Hartley normalize
    std::vector<Eigen::Vector2d> pts_a_norm, pts_b_norm;
    Eigen::Matrix3d Ta = hartleyNormalize(pts_a, pts_a_norm);
    Eigen::Matrix3d Tb = hartleyNormalize(pts_b, pts_b_norm);

    double thresh_sq = threshold * threshold;
    std::mt19937 rng(42);

    int best_inlier_count = 0;
    Eigen::Matrix3d best_F = Eigen::Matrix3d::Zero();
    std::vector<bool> best_mask(n, false);

    int adaptive_iters = max_iterations;

    for (int iter = 0; iter < adaptive_iters; iter++) {
        std::vector<int> sample;
        sample.reserve(8);
        while ((int)sample.size() < 8) {
            int idx = std::uniform_int_distribution<int>(0, n - 1)(rng);
            bool dup = false;
            for (int s : sample) if (s == idx) { dup = true; break; }
            if (!dup) sample.push_back(idx);
        }

        Eigen::Matrix3d F_norm = eightPointAlgorithm(
            pts_a_norm, pts_b_norm, sample, false);
        // Denormalize: F = Tb^T * F_norm * Ta
        Eigen::Matrix3d F_cand = Tb.transpose() * F_norm * Ta;

        // Score using Sampson distance in pixel coords
        int inlier_count = 0;
        std::vector<bool> mask(n, false);
        for (int i = 0; i < n; i++) {
            double d2 = sampsonDistanceSq(pts_a[i], pts_b[i], F_cand);
            if (d2 < thresh_sq) {
                mask[i] = true;
                inlier_count++;
            }
        }

        if (inlier_count > best_inlier_count) {
            best_inlier_count = inlier_count;
            best_F = F_cand;
            best_mask = mask;

            double inlier_ratio = (double)inlier_count / n;
            if (inlier_ratio > 0.0 && inlier_ratio < 1.0) {
                double p_fail = 1.0 - std::pow(inlier_ratio, 8);
                if (p_fail > 0.0 && p_fail < 1.0) {
                    int new_iters = (int)std::ceil(
                        std::log(1.0 - confidence) / std::log(p_fail));
                    adaptive_iters = std::min(max_iterations,
                                              std::max(new_iters, iter + 1));
                }
            }
        }
    }

    if (best_inlier_count < 8) return result;

    // Refine with all inliers (in normalized coords, then denormalize)
    std::vector<int> inlier_indices;
    for (int i = 0; i < n; i++)
        if (best_mask[i]) inlier_indices.push_back(i);

    Eigen::Matrix3d F_norm_ref = eightPointAlgorithm(
        pts_a_norm, pts_b_norm, inlier_indices, false);
    Eigen::Matrix3d F_refined = Tb.transpose() * F_norm_ref * Ta;

    // Re-score
    result.inlier_mask.resize(n, false);
    result.num_inliers = 0;
    for (int i = 0; i < n; i++) {
        double d2 = sampsonDistanceSq(pts_a[i], pts_b[i], F_refined);
        if (d2 < thresh_sq) {
            result.inlier_mask[i] = true;
            result.num_inliers++;
        }
    }

    result.E = F_refined; // stored in E field but is actually F
    result.success = true;
    return result;
}

// Decompose an essential matrix into two candidate rotations and a translation
// direction. E = U * diag(1,1,0) * V^T (after enforcing rank-2).
// Returns R1, R2, t (unit vector). The four candidate poses are:
// (R1, +t), (R1, -t), (R2, +t), (R2, -t).
struct EssentialDecomposition {
    Eigen::Matrix3d R1, R2;
    Eigen::Vector3d t;
};

inline EssentialDecomposition
decomposeEssentialMatrix(const Eigen::Matrix3d &E) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d Vt = svd.matrixV().transpose();

    // W rotation matrix
    Eigen::Matrix3d W;
    W << 0, -1, 0,
         1,  0, 0,
         0,  0, 1;

    EssentialDecomposition dec;
    dec.R1 = U * W * Vt;
    dec.R2 = U * W.transpose() * Vt;
    dec.t = U.col(2);

    // Ensure proper rotations (det = +1)
    if (dec.R1.determinant() < 0) dec.R1 = -dec.R1;
    if (dec.R2.determinant() < 0) dec.R2 = -dec.R2;
    dec.t.normalize();

    return dec;
}

} // namespace red_math
