#pragma once
// arena_alignment.h — Rigid transform alignment between calibration world
// frame and MuJoCo world frame, computed from 4 arena corner correspondences.
//
// Uses Procrustes/SVD to find the optimal rotation, translation, and scale
// mapping calibration coordinates (mm) to MuJoCo coordinates (meters).
// Tries all 24 permutations of the 4 corners to find the best match.

#include <Eigen/Core>
#include <Eigen/SVD>
#include <array>
#include <cmath>
#include <algorithm>
#include <numeric>

struct ArenaAlignment {
    bool valid = false;

    // Rigid transform: p_mujoco = scale * R * p_calib + t
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    double scale = 0.001; // default mm → m

    // Diagnostics
    double residual_mm = 0.0;  // RMS corner alignment error in mm
    int best_permutation = -1;

    // The 4 user-provided corner positions in calibration frame (mm)
    std::array<Eigen::Vector3d, 4> calib_corners;
    bool corners_set = false;

    // Apply transform to a single point (calib mm → MuJoCo meters)
    Eigen::Vector3d transform(const Eigen::Vector3d &p_calib) const {
        return scale * R * p_calib + t;
    }

    // Apply transform to keypoint data in-place (mm → MuJoCo meters)
    void transform_keypoints(double *xyz, int n) const {
        for (int i = 0; i < n; i++) {
            Eigen::Vector3d p(xyz[3*i], xyz[3*i+1], xyz[3*i+2]);
            Eigen::Vector3d q = transform(p);
            xyz[3*i] = q.x(); xyz[3*i+1] = q.y(); xyz[3*i+2] = q.z();
        }
    }
};

// MuJoCo arena corners: rectangular, centered at origin, Z=0
// Default: 1828mm x 1828mm (rodent). Fly: 24mm x 5.6mm (0.024m x 0.0056m).
inline std::array<Eigen::Vector3d, 4> mujoco_arena_corners(
        double width = 1.828, double depth = 1.828,
        double ox = 0, double oy = 0, double oz = 0) {
    double hw = width * 0.5, hd = depth * 0.5;
    return {{
        {ox+hw, oy+hd, oz},
        {ox-hw, oy+hd, oz},
        {ox-hw, oy-hd, oz},
        {ox+hw, oy-hd, oz},
    }};
}

// Compute optimal rigid transform (rotation + translation + uniform scale)
// from source points to target points using Procrustes/SVD.
// Returns RMS residual. Sets R, t, scale.
inline double procrustes_align(const Eigen::Vector3d *src, const Eigen::Vector3d *tgt,
                                int n, Eigen::Matrix3d &R, Eigen::Vector3d &t,
                                double &scale) {
    // Centroids
    Eigen::Vector3d src_c = Eigen::Vector3d::Zero();
    Eigen::Vector3d tgt_c = Eigen::Vector3d::Zero();
    for (int i = 0; i < n; i++) { src_c += src[i]; tgt_c += tgt[i]; }
    src_c /= n; tgt_c /= n;

    // Center
    Eigen::MatrixXd S(n, 3), T(n, 3);
    for (int i = 0; i < n; i++) {
        S.row(i) = (src[i] - src_c).transpose();
        T.row(i) = (tgt[i] - tgt_c).transpose();
    }

    // Scale: ratio of RMS distances from centroid
    double src_rms = std::sqrt(S.squaredNorm() / n);
    double tgt_rms = std::sqrt(T.squaredNorm() / n);
    scale = (src_rms > 1e-12) ? tgt_rms / src_rms : 0.001;

    // Scale source to match target magnitude
    Eigen::MatrixXd S_scaled = S * scale;

    // Rotation via SVD: H = S_scaled^T * T
    Eigen::Matrix3d H = S_scaled.transpose() * T;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    // Ensure proper rotation (det = +1, not reflection)
    Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
    if ((V * U.transpose()).determinant() < 0)
        D(2, 2) = -1.0;
    R = V * D * U.transpose();

    // Translation
    t = tgt_c - scale * R * src_c;

    // RMS residual (in target units)
    double err = 0.0;
    for (int i = 0; i < n; i++) {
        Eigen::Vector3d diff = (scale * R * src[i] + t) - tgt[i];
        err += diff.squaredNorm();
    }
    return std::sqrt(err / n);
}

// Compute arena alignment: tries all 24 permutations of the 4 calibration
// corners against the MuJoCo corners, picks the one with lowest residual.
inline void compute_arena_alignment(ArenaAlignment &align,
                                     double arena_width = 1.828,
                                     double arena_depth = 1.828,
                                     double arena_ox = 0, double arena_oy = 0,
                                     double arena_oz = 0) {
    if (!align.corners_set) { align.valid = false; return; }

    auto mj_corners = mujoco_arena_corners(arena_width, arena_depth,
                                              arena_ox, arena_oy, arena_oz);

    // All 24 permutations of {0,1,2,3}
    int perm[4] = {0, 1, 2, 3};
    double best_residual = 1e12;
    Eigen::Matrix3d best_R;
    Eigen::Vector3d best_t;
    double best_scale = 0.001;
    int best_perm_idx = 0;
    int perm_idx = 0;

    do {
        // Build permuted source array
        Eigen::Vector3d src[4], tgt[4];
        for (int i = 0; i < 4; i++) {
            src[i] = align.calib_corners[perm[i]];
            tgt[i] = mj_corners[i];
        }

        Eigen::Matrix3d R; Eigen::Vector3d t; double s;
        double res = procrustes_align(src, tgt, 4, R, t, s);

        if (res < best_residual) {
            best_residual = res;
            best_R = R; best_t = t; best_scale = s;
            best_perm_idx = perm_idx;
        }
        perm_idx++;
    } while (std::next_permutation(perm, perm + 4));

    align.R = best_R;
    align.t = best_t;
    align.scale = best_scale;
    align.residual_mm = best_residual * 1000.0; // convert m to mm
    align.best_permutation = best_perm_idx;
    align.valid = true;

    // Auto-correct Z: shift so transformed corners have mean Z = 0
    // (compensates for labeling corners slightly above/below the arena surface)
    double mean_z = 0.0;
    for (int i = 0; i < 4; i++) {
        Eigen::Vector3d p = align.transform(align.calib_corners[i]);
        mean_z += p.z();
    }
    mean_z /= 4.0;
    align.t.z() -= mean_z;

    // Print diagnostics
    std::cout << "[Arena] Alignment: scale=" << align.scale
              << " residual=" << align.residual_mm << " mm"
              << " z_correction=" << (mean_z * 1000.0) << " mm"
              << " (perm #" << best_perm_idx << ")" << std::endl;
    std::cout << "[Arena] R:\n" << align.R << std::endl;
    std::cout << "[Arena] t: " << align.t.transpose() << std::endl;
}
