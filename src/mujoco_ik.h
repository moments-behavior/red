#pragma once
// mujoco_ik.h — Damped least-squares IK solver using MuJoCo C API
//
// Uses analytical Jacobians (mj_jacSite) for maximum performance.
// Warm-starts from previous frame for temporal consistency.
// Pre-allocates all workspace to avoid per-frame heap allocation.

#include "mujoco_context.h"
#include "annotation.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>

struct MujocoIKState {
    // Solver configuration
    int   max_iterations = 50;
    double damping       = 1e-3;     // lambda for damped least-squares
    double tolerance     = 1e-4;     // convergence threshold (meters)
    double max_update    = 2.0;      // clamp step size

    // Warm-start state
    std::vector<double> prev_qpos;
    bool has_warm_start = false;
    int  prev_frame     = -1;

    // Solver output (last call)
    double final_residual   = 0.0;
    int    iterations_used  = 0;
    bool   converged        = false;
    double solve_time_ms    = 0.0;
    int    active_sites     = 0;

    // Pre-allocated workspace (sized on first use)
    std::vector<double> jacp_buf;  // 3*N_active * nv (stacked Jacobian)
    std::vector<double> residual;  // 3*N_active
    std::vector<double> dq_buf;    // nv
};

#ifdef RED_HAS_MUJOCO

// Solve IK: pose the MuJoCo model to match triangulated 3D keypoints.
// kp3d: array indexed by skeleton node index (length = num_nodes).
// Only nodes where kp3d[i].triangulated==true are used as targets.
// Returns true if converged within tolerance.
inline bool mujoco_ik_solve(MujocoContext &mj, MujocoIKState &state,
                            const Keypoint3D *kp3d, int num_nodes,
                            int current_frame) {
    if (!mj.loaded || !mj.model || !mj.data) return false;

    auto t0 = std::chrono::high_resolution_clock::now();

    const int nv = (int)mj.model->nv;
    const int nq = (int)mj.model->nq;

    // --- Initialize qpos ---
    // Warm-start from previous frame if recent (within 5 frames)
    bool cold_start = true;
    if (state.has_warm_start && state.prev_frame >= 0 &&
        std::abs(current_frame - state.prev_frame) <= 5 &&
        (int)state.prev_qpos.size() == nq) {
        std::copy(state.prev_qpos.begin(), state.prev_qpos.end(), mj.data->qpos);
        cold_start = false;
    } else {
        // Cold start: use default pose
        std::copy(mj.model->qpos0, mj.model->qpos0 + nq, mj.data->qpos);
    }
    mj_forward(mj.model, mj.data);

    // --- Build active site list ---
    // Collect (mj_site_idx, target_xyz) for all triangulated keypoints
    struct SiteTarget {
        int site_idx;
        double target[3];
    };
    std::vector<SiteTarget> targets;
    targets.reserve(num_nodes);

    // Determine scale factor. If scale_factor > 0, use it directly.
    // If scale_factor == 0 (auto), detect the unit system from data magnitude.
    // MuJoCo models are in meters (typical body ~0.2m). If keypoint coordinates
    // are much larger, they're likely in mm (common for camera calibration).
    double sf = (double)mj.scale_factor;
    if (sf <= 0.0) {
        // Compute centroid magnitude of keypoints
        double centroid[3] = {0, 0, 0};
        int n_active = 0;
        for (int n = 0; n < num_nodes; n++) {
            if (!kp3d[n].triangulated) continue;
            int si = (n < (int)mj.skeleton_to_site.size()) ? mj.skeleton_to_site[n] : -1;
            if (si < 0) continue;
            centroid[0] += kp3d[n].x; centroid[1] += kp3d[n].y; centroid[2] += kp3d[n].z;
            n_active++;
        }
        if (n_active > 0) {
            double mag = std::sqrt(centroid[0]*centroid[0] + centroid[1]*centroid[1] +
                                   centroid[2]*centroid[2]) / n_active;
            // Heuristic: if centroid magnitude > 10, data is likely in mm
            // Model centroid is typically < 1m, so 10 is a safe threshold
            if (mag > 10.0)
                sf = 0.001; // mm → meters
            else
                sf = 1.0;   // already in meters
        } else {
            sf = 1.0;
        }
    }

    for (int n = 0; n < num_nodes; n++) {
        if (!kp3d[n].triangulated) continue;
        int site_idx = (n < (int)mj.skeleton_to_site.size())
                           ? mj.skeleton_to_site[n] : -1;
        if (site_idx < 0) continue;
        targets.push_back({site_idx, {kp3d[n].x * sf, kp3d[n].y * sf, kp3d[n].z * sf}});
    }

    int N = (int)targets.size();
    state.active_sites = N;
    if (N == 0) {
        state.converged = false;
        state.final_residual = 0.0;
        state.iterations_used = 0;
        state.solve_time_ms = 0.0;
        return false;
    }

    // --- Cold-start root alignment ---
    // When the model has a free joint (added by mujoco_context.h), set the
    // root position to align the model centroid with the target centroid.
    // This handles arbitrary world coordinate systems.
    if (cold_start && mj.has_free_joint) {
        double tc[3] = {0, 0, 0}, mc[3] = {0, 0, 0};
        for (int k = 0; k < N; k++) {
            for (int c = 0; c < 3; c++) {
                tc[c] += targets[k].target[c];
                mc[c] += mj.data->site_xpos[3 * targets[k].site_idx + c];
            }
        }
        // Find the free joint's qpos address (first joint with type FREE)
        for (int j = 0; j < (int)mj.model->njnt; j++) {
            if (mj.model->jnt_type[j] == mjJNT_FREE) {
                int qa = (int)mj.model->jnt_qposadr[j];
                mj.data->qpos[qa + 0] += tc[0] / N - mc[0] / N;
                mj.data->qpos[qa + 1] += tc[1] / N - mc[1] / N;
                mj.data->qpos[qa + 2] += tc[2] / N - mc[2] / N;
                break;
            }
        }
        mj_forward(mj.model, mj.data);
    }

    // --- Ensure workspace is sized ---
    int jacp_size = 3 * N * nv;
    if ((int)state.jacp_buf.size() != jacp_size)
        state.jacp_buf.resize(jacp_size);
    if ((int)state.residual.size() != 3 * N)
        state.residual.resize(3 * N);
    if ((int)state.dq_buf.size() != nv)
        state.dq_buf.resize(nv);

    // --- Iterative solve ---
    using RowMajorMatXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    state.converged = false;
    state.iterations_used = 0;
    state.final_residual = 0.0;

    for (int iter = 0; iter < state.max_iterations; iter++) {
        // Compute residual: target - current site position
        double err_sq = 0.0;
        for (int k = 0; k < N; k++) {
            const double *sp = mj.data->site_xpos + 3 * targets[k].site_idx;
            for (int c = 0; c < 3; c++) {
                double r = targets[k].target[c] - sp[c];
                state.residual[3 * k + c] = r;
                err_sq += r * r;
            }
        }

        double err_norm = std::sqrt(err_sq / N);
        state.final_residual = err_norm;
        state.iterations_used = iter + 1;

        if (err_norm < state.tolerance) {
            state.converged = true;
            break;
        }

        // Compute stacked Jacobian [3N x nv] (row-major)
        std::fill(state.jacp_buf.begin(), state.jacp_buf.begin() + jacp_size, 0.0);
        for (int k = 0; k < N; k++) {
            mj_jacSite(mj.model, mj.data,
                       state.jacp_buf.data() + 3 * k * nv,
                       nullptr, // no rotation Jacobian
                       targets[k].site_idx);
        }

        // Map to Eigen (no copy)
        Eigen::Map<RowMajorMatXd> J(state.jacp_buf.data(), 3 * N, nv);
        Eigen::Map<Eigen::VectorXd> r(state.residual.data(), 3 * N);
        Eigen::Map<Eigen::VectorXd> dq(state.dq_buf.data(), nv);

        // Damped least-squares: dq = J^T (J J^T + lambda*I)^{-1} r
        // For N<nv (underdetermined), this form is more efficient.
        Eigen::MatrixXd JJt = J * J.transpose();
        JJt.diagonal().array() += state.damping;
        Eigen::VectorXd y = JJt.ldlt().solve(r);
        dq = J.transpose() * y;

        // Clamp update norm
        double dq_norm = dq.norm();
        if (dq_norm > state.max_update)
            dq *= state.max_update / dq_norm;

        // Integrate qpos (handles quaternion joints correctly)
        mj_integratePos(mj.model, mj.data->qpos, state.dq_buf.data(), 1.0);

        // Project hinge and slide joints onto their limits
        for (int j = 0; j < (int)mj.model->njnt; j++) {
            if (!mj.model->jnt_limited[j]) continue;
            int type = mj.model->jnt_type[j];
            if (type != mjJNT_HINGE && type != mjJNT_SLIDE) continue;
            int qa = mj.model->jnt_qposadr[j];
            double lo = mj.model->jnt_range[2 * j];
            double hi = mj.model->jnt_range[2 * j + 1];
            if (mj.data->qpos[qa] < lo) mj.data->qpos[qa] = lo;
            if (mj.data->qpos[qa] > hi) mj.data->qpos[qa] = hi;
        }

        // Forward kinematics to update site positions
        mj_forward(mj.model, mj.data);
    }

    // Store warm-start for next frame
    state.prev_qpos.assign(mj.data->qpos, mj.data->qpos + nq);
    state.has_warm_start = true;
    state.prev_frame = current_frame;

    auto t1 = std::chrono::high_resolution_clock::now();
    state.solve_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return state.converged;
}

#else // !RED_HAS_MUJOCO

inline bool mujoco_ik_solve(MujocoContext &, MujocoIKState &,
                            const Keypoint3D *, int, int) {
    return false;
}

#endif // RED_HAS_MUJOCO

// Reset warm-start (e.g., on project switch or large frame jump)
inline void mujoco_ik_reset(MujocoIKState &state) {
    state.prev_qpos.clear();
    state.has_warm_start = false;
    state.prev_frame = -1;
    state.converged = false;
    state.final_residual = 0.0;
    state.iterations_used = 0;
    state.solve_time_ms = 0.0;
    state.active_sites = 0;
}
