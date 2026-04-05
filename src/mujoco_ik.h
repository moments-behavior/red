#pragma once
// mujoco_ik.h — Gradient descent IK solver with momentum (matches Python pipeline)
//
// Algorithm from inverse_kinematics.py (IK_resources):
//   objective = ||s(q) - s*||² + reg_strength * ||q_hinge||²
//   gradient  = 2 * J^T * (s(q) - s*) + 2 * reg_strength * q_hinge
//   update    = beta * update + gradient
//   qpos     -= lr * update
//
// Uses analytical Jacobians (mj_jacSite) for maximum performance.
// Warm-starts from previous frame for temporal consistency.
// Pre-allocates all workspace to avoid per-frame heap allocation.

#include "mujoco_context.h"
#include "annotation.h"
#include <Eigen/Core>
#include <chrono>
#include <algorithm>

struct MujocoIKState {
    // Solver configuration
    int    max_iterations   = 5000;
    double lr               = 0.01;    // learning rate (translation DOFs)
    double lr_joint         = 0.0;     // joint/rotation lr (0 = same as lr)
    double beta             = 0.99;    // momentum coefficient
    double reg_strength     = 1e-4;    // L2 regularization on hinge joints
    double progress_thresh  = 0.01;    // convergence: lr*||update||/err < thresh
    int    check_every      = 100;     // convergence check interval
    bool   cosine_annealing = false;   // decay lr with cosine schedule

    // Time budget: 0 = unlimited, >0 = stop after this many ms (for auto-solve)
    double time_budget_ms   = 0.0;

    // Warm-start state
    std::vector<double> prev_qpos;
    bool has_warm_start = false;
    int  prev_frame     = -1;

    // Solver output (last call)
    double final_residual   = 0.0;   // position-only error (no regularization)
    double final_objective  = 0.0;   // full objective (with regularization)
    int    iterations_used  = 0;
    bool   converged        = false;
    bool   time_limited     = false; // true if stopped by time budget
    double solve_time_ms    = 0.0;
    int    active_sites     = 0;

    // Pre-allocated workspace (sized on first use)
    std::vector<double> jacp_buf;    // 3*N_active * nv (stacked Jacobian)
    std::vector<double> grad_buf;    // nv (gradient)
    std::vector<double> update_buf;  // nv (momentum accumulator)
    std::vector<double> nv_step;     // nv (full step vector)
};

#ifdef RED_HAS_MUJOCO

// Solve IK: pose the MuJoCo model to match triangulated 3D keypoints.
// Matches the algorithm from inverse_kinematics.py: gradient descent with
// momentum and L2 regularization on hinge joint angles.
inline bool mujoco_ik_solve(MujocoContext &mj, MujocoIKState &state,
                            const Keypoint3D *kp3d, int num_nodes,
                            int current_frame) {
    if (!mj.loaded || !mj.model || !mj.data) return false;

    auto t0 = std::chrono::high_resolution_clock::now();

    const int nv = (int)mj.model->nv;
    const int nq = (int)mj.model->nq;

    // --- Initialize qpos ---
    bool cold_start = true;
    if (state.has_warm_start && state.prev_frame >= 0 &&
        std::abs(current_frame - state.prev_frame) <= 5 &&
        (int)state.prev_qpos.size() == nq) {
        std::copy(state.prev_qpos.begin(), state.prev_qpos.end(), mj.data->qpos);
        cold_start = false;
    } else {
        std::copy(mj.model->qpos0, mj.model->qpos0 + nq, mj.data->qpos);
    }
    mj_forward(mj.model, mj.data);

    // --- Build active site list ---
    struct SiteTarget { int site_idx; double target[3]; };
    std::vector<SiteTarget> targets;
    targets.reserve(num_nodes);

    // Auto-detect unit system
    double sf = (double)mj.scale_factor;
    if (sf <= 0.0) {
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
            double cx = centroid[0]/n_active, cy = centroid[1]/n_active, cz = centroid[2]/n_active;
            double mag = std::sqrt(cx*cx + cy*cy + cz*cz);
            sf = (mag > 10.0) ? 0.001 : 1.0;
        } else {
            sf = 1.0;
        }
    }

    for (int n = 0; n < num_nodes; n++) {
        if (!kp3d[n].triangulated) continue;
        int site_idx = (n < (int)mj.skeleton_to_site.size()) ? mj.skeleton_to_site[n] : -1;
        if (site_idx < 0) continue;
        targets.push_back({site_idx, {kp3d[n].x * sf, kp3d[n].y * sf, kp3d[n].z * sf}});
    }

    int N = (int)targets.size();
    state.active_sites = N;
    if (N == 0) {
        state.converged = false;
        state.final_residual = 0.0;
        state.final_objective = 0.0;
        state.iterations_used = 0;
        state.solve_time_ms = 0.0;
        return false;
    }

    // --- Cold-start root alignment ---
    if (cold_start && mj.has_free_joint) {
        double tc[3] = {0, 0, 0}, mc[3] = {0, 0, 0};
        for (int k = 0; k < N; k++) {
            for (int c = 0; c < 3; c++) {
                tc[c] += targets[k].target[c];
                mc[c] += mj.data->site_xpos[3 * targets[k].site_idx + c];
            }
        }
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

    // --- Pre-identify hinge joints and their DOF indices ---
    // Matches Python: regularization only applies to hinge joints
    std::vector<int> hinge_dof_indices;
    hinge_dof_indices.reserve(nv);
    for (int j = 0; j < (int)mj.model->njnt; j++) {
        if (mj.model->jnt_type[j] == mjJNT_HINGE) {
            int dof_adr = (int)mj.model->jnt_dofadr[j];
            hinge_dof_indices.push_back(dof_adr);
        }
    }

    // --- Ensure workspace is sized ---
    int jacp_size = 3 * N * nv;
    if ((int)state.jacp_buf.size() < jacp_size)
        state.jacp_buf.resize(jacp_size);
    if ((int)state.grad_buf.size() < nv)
        state.grad_buf.resize(nv);
    if ((int)state.update_buf.size() < nv) {
        state.update_buf.resize(nv);
        std::fill(state.update_buf.begin(), state.update_buf.end(), 0.0);
    }
    if ((int)state.nv_step.size() < nv)
        state.nv_step.resize(nv);

    // Reset momentum on cold start
    if (cold_start)
        std::fill(state.update_buf.begin(), state.update_buf.begin() + nv, 0.0);

    // --- Iterative gradient descent with momentum ---
    using RowMajorMatXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    state.converged = false;
    state.time_limited = false;
    state.iterations_used = 0;
    state.final_residual = 0.0;
    state.final_objective = 0.0;

    // Pre-allocate diff vector outside the loop
    std::vector<double> diff(3 * N);

    for (int iter = 0; iter < state.max_iterations; iter++) {
        state.iterations_used = iter + 1;

        // Compute stacked Jacobian [3N x nv] (row-major)
        std::fill(state.jacp_buf.begin(), state.jacp_buf.begin() + jacp_size, 0.0);
        for (int k = 0; k < N; k++) {
            mj_jacSite(mj.model, mj.data,
                       state.jacp_buf.data() + 3 * k * nv,
                       nullptr, targets[k].site_idx);
        }

        // Compute residual: site_xpos - target (note: Python uses this sign)
        // and position error squared
        double err_sq = 0.0;
        for (int k = 0; k < N; k++) {
            const double *sp = mj.data->site_xpos + 3 * targets[k].site_idx;
            for (int c = 0; c < 3; c++) {
                double d = sp[c] - targets[k].target[c];
                diff[3 * k + c] = d;
                err_sq += d * d;
            }
        }

        // Compute gradient: 2 * J^T * diff + 2 * reg * hinge_qpos
        // J is [3N x nv], diff is [3N], grad is [nv]
        Eigen::Map<RowMajorMatXd> J(state.jacp_buf.data(), 3 * N, nv);
        Eigen::Map<Eigen::VectorXd> d_vec(diff.data(), 3 * N);
        Eigen::Map<Eigen::VectorXd> grad(state.grad_buf.data(), nv);

        grad = 2.0 * J.transpose() * d_vec;

        // Add regularization gradient: 2 * reg_strength * q_hinge
        if (state.reg_strength > 0.0) {
            for (int dof : hinge_dof_indices) {
                // qpos index for a hinge joint == its dof address (1:1 mapping)
                int qa = (int)mj.model->jnt_qposadr[
                    mj.model->dof_jntid[dof]];
                grad[dof] += 2.0 * state.reg_strength * mj.data->qpos[qa];
            }
        }

        // Momentum update: update = beta * update + grad
        Eigen::Map<Eigen::VectorXd> update(state.update_buf.data(), nv);
        update = state.beta * update + grad;

        // Learning rate (with optional cosine annealing)
        double lr_t = state.lr;
        double lr_j = (state.lr_joint > 0) ? state.lr_joint : state.lr;
        if (state.cosine_annealing && state.max_iterations > 0) {
            double t = (double)iter / state.max_iterations;
            double decay = 0.5 * (1.0 + cos(M_PI * t));
            lr_t *= decay;
            lr_j *= decay;
        }

        // Build step with separate learning rates for translation vs joint DOFs
        Eigen::Map<Eigen::VectorXd> step(state.nv_step.data(), nv);
        if (std::abs(lr_t - lr_j) > 1e-12) {
            // Per-DOF learning rate: free joint trans (first 3) use lr_t,
            // free joint rot (next 3) + all hinge joints use lr_j
            for (int v = 0; v < nv; v++) {
                bool is_trans = false;
                if (mj.has_free_joint) {
                    // Free joint DOFs are typically the first 6 (3 trans + 3 rot)
                    int jnt_id = mj.model->dof_jntid[v];
                    if (mj.model->jnt_type[jnt_id] == mjJNT_FREE) {
                        int dof_in_jnt = v - mj.model->jnt_dofadr[jnt_id];
                        is_trans = (dof_in_jnt < 3); // first 3 = translation
                    }
                }
                step[v] = -(is_trans ? lr_t : lr_j) * update[v];
            }
        } else {
            step = -lr_t * update;
        }

        // Integrate qpos (handles quaternion joints correctly)
        mj_integratePos(mj.model, mj.data->qpos, state.nv_step.data(), 1.0);

        // Clamp joints to their defined ranges (prevents spine collapse)
        for (int j = 0; j < (int)mj.model->njnt; j++) {
            if (!mj.model->jnt_limited[j]) continue;
            int qa = (int)mj.model->jnt_qposadr[j];
            if (mj.model->jnt_type[j] == mjJNT_HINGE ||
                mj.model->jnt_type[j] == mjJNT_SLIDE) {
                double lo = mj.model->jnt_range[2 * j];
                double hi = mj.model->jnt_range[2 * j + 1];
                if (mj.data->qpos[qa] < lo) mj.data->qpos[qa] = lo;
                if (mj.data->qpos[qa] > hi) mj.data->qpos[qa] = hi;
            }
        }

        // Forward kinematics to update site positions
        mj_fwdPosition(mj.model, mj.data);

        // Periodic checks (every check_every iterations, skip iter 0)
        if (state.check_every > 0 && iter > 0 && iter % state.check_every == 0) {
            // Recompute error after update (err_sq above was pre-update)
            double post_err_sq = 0.0;
            for (int k = 0; k < N; k++) {
                const double *sp = mj.data->site_xpos + 3 * targets[k].site_idx;
                for (int c = 0; c < 3; c++) {
                    double d = sp[c] - targets[k].target[c];
                    post_err_sq += d * d;
                }
            }
            double reg_sq = 0.0;
            for (int dof : hinge_dof_indices) {
                int qa = (int)mj.model->jnt_qposadr[mj.model->dof_jntid[dof]];
                reg_sq += mj.data->qpos[qa] * mj.data->qpos[qa];
            }
            double obj = post_err_sq + state.reg_strength * reg_sq;
            state.final_objective = obj;
            state.final_residual = std::sqrt(post_err_sq / N);

            // Convergence check: use base LR (not decayed) so cosine
            // annealing doesn't trigger early convergence
            {
                double base_lr = (state.lr_joint > 0) ? std::max(state.lr, state.lr_joint) : state.lr;
                double update_norm = base_lr * update.norm();
                if (obj > 1e-12 && update_norm / obj < state.progress_thresh) {
                    state.converged = true;
                    break;
                }
            }

            // Time budget check
            if (state.time_budget_ms > 0.0) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double, std::milli>(now - t0).count();
                if (elapsed >= state.time_budget_ms) {
                    state.time_limited = true;
                    break;
                }
            }
        }
    }

    // Final residual (position-only, for display)
    {
        double err_sq = 0.0;
        for (int k = 0; k < N; k++) {
            const double *sp = mj.data->site_xpos + 3 * targets[k].site_idx;
            for (int c = 0; c < 3; c++) {
                double d = sp[c] - targets[k].target[c];
                err_sq += d * d;
            }
        }
        state.final_residual = std::sqrt(err_sq / N);
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

// Continue solving from the current pose — runs additional iterations without
// resetting qpos or momentum. Useful for incremental refinement.
inline bool mujoco_ik_continue(MujocoContext &mj, MujocoIKState &state,
                               const Keypoint3D *kp3d, int num_nodes,
                               int current_frame, int extra_iterations) {
    if (!mj.loaded || !mj.model || !mj.data) return false;
    // Temporarily increase max_iterations, solve (warm-started), restore
    int saved_max = state.max_iterations;
    state.max_iterations = extra_iterations;
    // Force warm-start: the model is already in the current pose
    state.prev_qpos.assign(mj.data->qpos, mj.data->qpos + mj.model->nq);
    state.has_warm_start = true;
    state.prev_frame = current_frame;
    bool result = mujoco_ik_solve(mj, state, kp3d, num_nodes, current_frame);
    state.max_iterations = saved_max;
    return result;
}

// Reset warm-start and momentum (e.g., on project switch or large frame jump).
// Does NOT reset solver configuration (lr, beta, reg_strength, etc.)
inline void mujoco_ik_reset(MujocoIKState &state) {
    state.prev_qpos.clear();
    state.has_warm_start = false;
    state.prev_frame = -1;
    state.converged = false;
    state.time_limited = false;
    state.final_residual = 0.0;
    state.final_objective = 0.0;
    state.iterations_used = 0;
    state.solve_time_ms = 0.0;
    state.active_sites = 0;
    // Reset momentum accumulator
    std::fill(state.update_buf.begin(), state.update_buf.end(), 0.0);
}
