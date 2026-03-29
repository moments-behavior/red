#pragma once
// mujoco_stac.h — STAC: Simultaneous Tracking and Calibration
//
// Implements site offset calibration following Wu et al. 2013 and
// talmolab/stac-mjx. Alternates between:
//   Q-phase: IK solve per frame (reuses mujoco_ik_solve / IK_dm_control)
//   M-phase: SGD on site offsets to minimize aggregate residual
//
// The key gradient: d(site_xpos)/d(site_offset) = body_xmat (rotation matrix
// of the parent body), which MuJoCo already computes during mj_fwdPosition.
// No autodiff or MJX needed.

#include "mujoco_context.h"
#include "mujoco_ik.h"
#include "annotation.h"
#include "arena_alignment.h"
#include <Eigen/Core>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <random>

// Symmetry constraint: maps skeleton node pairs for bilateral symmetry.
// Y is the bilateral axis in the rodent MuJoCo model.
struct SymmetryTable {
    // Midline site indices (Y offset forced to 0)
    std::vector<int> midline;
    // L/R paired site indices
    struct Pair { int left; int right; };
    std::vector<Pair> pairs;
};

struct StacState {
    // Calibration parameters
    int    n_iters         = 3;      // alternating Q/M rounds
    int    n_sample_frames = 100;    // frames for M-phase (0 = use all)
    double m_lr            = 5e-4;   // offset SGD learning rate
    double m_momentum      = 0.9;    // offset SGD momentum
    double m_reg_coef      = 0.1;    // L2 regularization on offsets
    int    m_max_iters     = 500;    // M-phase SGD iterations per round
    int    q_max_iters     = 300;    // IK iterations during calibration
    bool   symmetric       = true;   // enforce bilateral symmetry on offsets

    // State
    bool   calibrated      = false;
    int    frames_used     = 0;
    double pre_residual    = 0.0;    // mean residual before calibration (mm)
    double post_residual   = 0.0;    // mean residual after calibration (mm)
    double calibration_time_s = 0.0;

    // Per-site calibrated offsets (delta from original, in local body frame)
    std::vector<double> site_offsets;     // 3 * nsite
    std::vector<double> original_site_pos; // backup of model->site_pos

    // M-phase momentum buffer
    std::vector<double> m_velocity;       // 3 * nsite

    // Live progress (atomic for thread-safe UI reading)
    std::atomic<int> current_round{0};   // 1-based alternating round
    std::atomic<int> total_rounds{0};
    std::atomic<int> round_frames_done{0};
    std::atomic<int> round_frames_total{0};
    std::atomic<int> phase{0};           // 0=idle, 1=Q-phase, 2=M-phase, 3=final Q
    std::atomic<int> m_iter_done{0};
};

#ifdef RED_HAS_MUJOCO

// Apply current offsets to the model's site positions.
// Call after modifying stac.site_offsets.
inline void stac_apply_offsets(MujocoContext &mj, const StacState &stac) {
    if (stac.original_site_pos.empty()) return;
    int nsite = (int)mj.model->nsite;
    for (int i = 0; i < nsite * 3; i++)
        mj.model->site_pos[i] = stac.original_site_pos[i] + stac.site_offsets[i];
}

// Reset offsets to zero and restore original site positions.
// Build symmetry table from skeleton node names.
// Identifies midline sites (Y offset = 0) and L/R pairs (mirrored Y).
inline SymmetryTable build_symmetry_table(const MujocoContext &mj,
                                           const std::vector<std::string> *node_names,
                                           int num_nodes) {
    SymmetryTable sym;
    if (!node_names || num_nodes == 0) return sym;

    // Known L/R skeleton name pairs
    static const std::pair<std::string, std::string> kPairs[] = {
        {"EarL",      "EarR"},
        {"ShoulderL", "ShoulderR"},
        {"ElbowL",    "ElbowR"},
        {"WristL",    "WristR"},
        {"HandL",     "HandR"},
        {"KneeL",     "KneeR"},
        {"AnkleL",    "AnkleR"},
        {"FootL",     "FootR"},
    };
    // Known midline skeleton names
    static const std::string kMidline[] = {
        "Snout", "Neck", "SpineL", "TailBase",
        "TailTip", "TailMid", "Tail1Q", "Tail3Q",
    };

    // Build node name → site index lookup
    auto node_to_site = [&](const std::string &name) -> int {
        for (int n = 0; n < num_nodes; n++) {
            if ((*node_names)[n] == name) {
                return (n < (int)mj.skeleton_to_site.size()) ? mj.skeleton_to_site[n] : -1;
            }
        }
        return -1;
    };

    for (const auto &m : kMidline) {
        int si = node_to_site(m);
        if (si >= 0) sym.midline.push_back(si);
    }
    for (const auto &[l, r] : kPairs) {
        int si_l = node_to_site(l), si_r = node_to_site(r);
        if (si_l >= 0 && si_r >= 0) sym.pairs.push_back({si_l, si_r});
    }
    return sym;
}

// Enforce symmetry on gradient vector (in-place).
// Midline: zero Y gradient. L/R pairs: average X/Z, mirror Y.
inline void enforce_symmetry_gradient(std::vector<double> &grad,
                                       const SymmetryTable &sym) {
    for (int si : sym.midline)
        grad[3*si + 1] = 0.0; // Y gradient = 0

    for (const auto &p : sym.pairs) {
        int L = p.left, R = p.right;
        // Average X and Z gradients from both sides
        double avg_x = (grad[3*L + 0] + grad[3*R + 0]) * 0.5;
        double avg_z = (grad[3*L + 2] + grad[3*R + 2]) * 0.5;
        // Mirror Y: L and R Y gradients should be opposite
        double avg_y = (grad[3*L + 1] - grad[3*R + 1]) * 0.5;

        grad[3*L + 0] = avg_x;  grad[3*L + 1] = avg_y;   grad[3*L + 2] = avg_z;
        grad[3*R + 0] = avg_x;  grad[3*R + 1] = -avg_y;  grad[3*R + 2] = avg_z;
    }
}

// Enforce symmetry on offset vector (in-place).
inline void enforce_symmetry_offsets(std::vector<double> &offsets,
                                      const SymmetryTable &sym) {
    for (int si : sym.midline)
        offsets[3*si + 1] = 0.0; // Y offset = 0

    for (const auto &p : sym.pairs) {
        int L = p.left, R = p.right;
        double avg_x = (offsets[3*L + 0] + offsets[3*R + 0]) * 0.5;
        double avg_z = (offsets[3*L + 2] + offsets[3*R + 2]) * 0.5;
        double avg_y = (offsets[3*L + 1] - offsets[3*R + 1]) * 0.5;

        offsets[3*L + 0] = avg_x;  offsets[3*L + 1] = avg_y;   offsets[3*L + 2] = avg_z;
        offsets[3*R + 0] = avg_x;  offsets[3*R + 1] = -avg_y;  offsets[3*R + 2] = avg_z;
    }
}

inline void stac_reset(MujocoContext &mj, StacState &stac) {
    if (!stac.original_site_pos.empty()) {
        int nsite = (int)mj.model->nsite;
        for (int i = 0; i < nsite * 3; i++)
            mj.model->site_pos[i] = stac.original_site_pos[i];
    }
    stac.site_offsets.clear();
    stac.m_velocity.clear();
    stac.calibrated = false;
    stac.frames_used = 0;
    stac.pre_residual = 0.0;
    stac.post_residual = 0.0;
}

// Compute mean per-site residual (mm) across a set of frames.
// Assumes qpos for each frame is stored in all_qpos.
inline double stac_mean_residual(MujocoContext &mj,
                                 const std::vector<std::vector<double>> &all_qpos,
                                 const std::vector<const Keypoint3D *> &all_kp3d,
                                 const std::vector<int> &frame_indices,
                                 int num_nodes, double scale_factor) {
    double total_err = 0.0;
    int total_sites = 0;

    for (int fi : frame_indices) {
        std::copy(all_qpos[fi].begin(), all_qpos[fi].end(), mj.data->qpos);
        mj_fwdPosition(mj.model, mj.data);

        const Keypoint3D *kp = all_kp3d[fi];
        for (int n = 0; n < num_nodes; n++) {
            if (!kp[n].triangulated) continue;
            int si = (n < (int)mj.skeleton_to_site.size()) ? mj.skeleton_to_site[n] : -1;
            if (si < 0) continue;
            double tx = kp[n].x * scale_factor;
            double ty = kp[n].y * scale_factor;
            double tz = kp[n].z * scale_factor;
            const double *sp = mj.data->site_xpos + 3 * si;
            double dx = sp[0] - tx, dy = sp[1] - ty, dz = sp[2] - tz;
            total_err += std::sqrt(dx*dx + dy*dy + dz*dz);
            total_sites++;
        }
    }
    // Return in mm
    return (total_sites > 0) ? (total_err / total_sites) * 1000.0 : 0.0;
}

// Run STAC calibration: alternating Q-phase (IK) and M-phase (offset SGD).
// Collects frames from the annotation map and runs the full pipeline.
inline bool stac_calibrate(MujocoContext &mj, StacState &stac, MujocoIKState &ik,
                           const AnnotationMap &annotations, int num_nodes,
                           const std::vector<std::string> *node_names = nullptr,
                           const ArenaAlignment *arena_align = nullptr) {
    if (!mj.loaded || !mj.model || !mj.data) return false;

    auto t_start = std::chrono::high_resolution_clock::now();
    int nsite = (int)mj.model->nsite;
    int nq = (int)mj.model->nq;

    // Auto-detect scale factor
    double sf = (double)mj.scale_factor;
    if (sf <= 0.0) sf = 0.001; // mm -> m

    // --- Collect frames with 3D keypoints ---
    // Deep-copy and transform keypoints if arena alignment is active.
    // When aligned, sf should be 1.0 (transform handles scaling).
    bool use_arena = (arena_align && arena_align->valid);
    if (use_arena) sf = 1.0;

    struct FrameData {
        int frame_num;
        std::vector<Keypoint3D> kp3d_owned; // transformed copy
        const Keypoint3D *kp3d = nullptr;   // set after collection (avoids dangling)
    };
    std::vector<FrameData> frames;
    for (const auto &[fnum, fa] : annotations) {
        if (!fa.kp3d.empty()) {
            int active = 0;
            for (int n = 0; n < num_nodes && n < (int)fa.kp3d.size(); n++)
                if (fa.kp3d[n].triangulated) active++;
            if (active >= 4) {
                FrameData fd;
                fd.frame_num = (int)fnum;
                fd.kp3d_owned = fa.kp3d; // deep copy
                if (use_arena) {
                    for (auto &kp : fd.kp3d_owned) {
                        if (!kp.triangulated) continue;
                        Eigen::Vector3d p = arena_align->transform(
                            Eigen::Vector3d(kp.x, kp.y, kp.z));
                        kp.x = p.x(); kp.y = p.y(); kp.z = p.z();
                    }
                }
                frames.push_back(std::move(fd));
            }
        }
    }
    if (frames.empty()) return false;
    // Set kp3d pointers after vector is fully built (no more reallocs)
    for (auto &fd : frames) fd.kp3d = fd.kp3d_owned.data();

    // Sort by frame number for warm-starting
    std::sort(frames.begin(), frames.end(),
              [](const FrameData &a, const FrameData &b) { return a.frame_num < b.frame_num; });

    // Subsample if needed
    int N = (int)frames.size();
    std::vector<int> sample_indices(N);
    std::iota(sample_indices.begin(), sample_indices.end(), 0);
    if (stac.n_sample_frames > 0 && stac.n_sample_frames < N) {
        // Uniform subsample for diversity
        std::mt19937 rng(42);
        std::shuffle(sample_indices.begin(), sample_indices.end(), rng);
        sample_indices.resize(stac.n_sample_frames);
        std::sort(sample_indices.begin(), sample_indices.end()); // re-sort for warm-start
        N = stac.n_sample_frames;
    }

    // --- Initialize ---
    stac.original_site_pos.assign(mj.model->site_pos, mj.model->site_pos + 3 * nsite);
    stac.site_offsets.assign(3 * nsite, 0.0);
    stac.m_velocity.assign(3 * nsite, 0.0);
    stac.frames_used = N;

    // Pointers for residual computation
    std::vector<const Keypoint3D *> all_kp3d(N);
    for (int i = 0; i < N; i++)
        all_kp3d[i] = frames[sample_indices[i]].kp3d;

    // Storage for solved qpos per frame
    std::vector<std::vector<double>> all_qpos(N, std::vector<double>(nq));

    // Save IK settings, use faster settings for calibration
    int saved_max_iters = ik.max_iterations;
    ik.max_iterations = stac.q_max_iters;

    // Build symmetry table if enabled
    SymmetryTable sym;
    if (stac.symmetric && node_names)
        sym = build_symmetry_table(mj, node_names, num_nodes);

    // Frame index list for residual computation
    std::vector<int> all_idx(N);
    std::iota(all_idx.begin(), all_idx.end(), 0);

    // Initialize progress
    stac.total_rounds = stac.n_iters;
    stac.round_frames_total = N;

    // --- Alternating optimization ---
    for (int iter = 0; iter < stac.n_iters; iter++) {
        stac.current_round = iter + 1;

        // === Q-phase: IK solve for each sampled frame ===
        stac.phase = 1;
        stac.round_frames_done = 0;
        mujoco_ik_reset(ik); // cold start for first frame of each round
        for (int i = 0; i < N; i++) {
            int fi = sample_indices[i];
            mujoco_ik_solve(mj, ik, frames[fi].kp3d, num_nodes, frames[fi].frame_num);
            all_qpos[i].assign(mj.data->qpos, mj.data->qpos + nq);
            stac.round_frames_done.fetch_add(1, std::memory_order_relaxed);
        }

        // Compute residual before M-phase on first iteration
        if (iter == 0)
            stac.pre_residual = stac_mean_residual(mj, all_qpos, all_kp3d, all_idx,
                                                    num_nodes, sf);

        // === M-phase: SGD on site offsets ===
        // Pre-cache body transforms (xpos, xmat) for all frames — these don't
        // change during M-phase since qpos is fixed. This avoids O(N * m_iters)
        // calls to mj_fwdPosition, reducing to O(N) + O(m_iters * N_sites).
        int nbody = (int)mj.model->nbody;
        std::vector<std::vector<double>> cached_xpos(N, std::vector<double>(3 * nbody));
        std::vector<std::vector<double>> cached_xmat(N, std::vector<double>(9 * nbody));
        // Also cache per-frame targets for mapped sites
        struct SiteTarget { int site_idx; double target[3]; };
        std::vector<std::vector<SiteTarget>> cached_targets(N);

        for (int i = 0; i < N; i++) {
            std::copy(all_qpos[i].begin(), all_qpos[i].end(), mj.data->qpos);
            mj_fwdPosition(mj.model, mj.data);
            std::copy(mj.data->xpos, mj.data->xpos + 3 * nbody, cached_xpos[i].begin());
            std::copy(mj.data->xmat, mj.data->xmat + 9 * nbody, cached_xmat[i].begin());

            const Keypoint3D *kp = all_kp3d[i];
            for (int n = 0; n < num_nodes; n++) {
                if (!kp[n].triangulated) continue;
                int si = (n < (int)mj.skeleton_to_site.size()) ? mj.skeleton_to_site[n] : -1;
                if (si < 0) continue;
                cached_targets[i].push_back({si,
                    {kp[n].x * sf, kp[n].y * sf, kp[n].z * sf}});
            }
        }

        stac.phase = 2;
        stac.m_iter_done = 0;
        std::vector<double> grad(3 * nsite);
        for (int m_iter = 0; m_iter < stac.m_max_iters; m_iter++) {
            std::fill(grad.begin(), grad.end(), 0.0);

            for (int i = 0; i < N; i++) {
                const double *xpos = cached_xpos[i].data();
                const double *xmat = cached_xmat[i].data();

                for (const auto &st : cached_targets[i]) {
                    int si = st.site_idx;
                    int bodyid = mj.model->site_bodyid[si];
                    const double *bp = xpos + 3 * bodyid;
                    const double *R  = xmat + 9 * bodyid; // row-major

                    // Compute site_xpos = body_xpos + R * site_pos (current offset)
                    const double *sp = mj.model->site_pos + 3 * si;
                    double sx = bp[0] + R[0]*sp[0] + R[1]*sp[1] + R[2]*sp[2];
                    double sy = bp[1] + R[3]*sp[0] + R[4]*sp[1] + R[5]*sp[2];
                    double sz = bp[2] + R[6]*sp[0] + R[7]*sp[1] + R[8]*sp[2];

                    double res[3] = {sx - st.target[0], sy - st.target[1], sz - st.target[2]};

                    // R^T * res
                    for (int c = 0; c < 3; c++) {
                        double g = 2.0 * (R[0*3+c]*res[0] + R[1*3+c]*res[1] + R[2*3+c]*res[2]);
                        grad[3*si + c] += g;
                    }
                }
            }

            // Average gradient and add L2 regularization
            double inv_N = 1.0 / N;
            for (int i = 0; i < 3 * nsite; i++) {
                grad[i] = grad[i] * inv_N + 2.0 * stac.m_reg_coef * stac.site_offsets[i];
            }

            // Enforce bilateral symmetry on gradients
            if (stac.symmetric && !sym.midline.empty())
                enforce_symmetry_gradient(grad, sym);

            // SGD with momentum
            for (int i = 0; i < 3 * nsite; i++) {
                stac.m_velocity[i] = stac.m_momentum * stac.m_velocity[i] + grad[i];
                stac.site_offsets[i] -= stac.m_lr * stac.m_velocity[i];
            }

            // Enforce bilateral symmetry on offsets (belt + suspenders)
            if (stac.symmetric && !sym.midline.empty())
                enforce_symmetry_offsets(stac.site_offsets, sym);

            // Apply offsets to model
            stac_apply_offsets(mj, stac);
            stac.m_iter_done.store(m_iter + 1, std::memory_order_relaxed);
        }
    }

    // --- Final Q-phase with calibrated offsets ---
    stac.phase = 3;
    stac.round_frames_done = 0;
    mujoco_ik_reset(ik);
    for (int i = 0; i < N; i++) {
        int fi = sample_indices[i];
        mujoco_ik_solve(mj, ik, frames[fi].kp3d, num_nodes, frames[fi].frame_num);
        all_qpos[i].assign(mj.data->qpos, mj.data->qpos + nq);
        stac.round_frames_done.fetch_add(1, std::memory_order_relaxed);
    }

    // Compute final residual
    stac.post_residual = stac_mean_residual(mj, all_qpos, all_kp3d,
                                             all_idx, num_nodes, sf);

    // Restore IK settings
    ik.max_iterations = saved_max_iters;

    stac.calibrated = true;
    stac.phase = 0;
    auto t_end = std::chrono::high_resolution_clock::now();
    stac.calibration_time_s = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "[STAC] Calibration complete: " << N << " frames, "
              << stac.n_iters << " rounds, "
              << stac.calibration_time_s << "s" << std::endl;
    std::cout << "[STAC] Residual: " << stac.pre_residual << " mm -> "
              << stac.post_residual << " mm" << std::endl;

    return true;
}

#else // !RED_HAS_MUJOCO

inline void stac_apply_offsets(MujocoContext &, const StacState &) {}
inline void stac_reset(MujocoContext &, StacState &) {}
inline bool stac_calibrate(MujocoContext &, StacState &, MujocoIKState &,
                           const AnnotationMap &, int,
                           const std::vector<std::string> * = nullptr,
                           const ArenaAlignment * = nullptr) { return false; }

#endif // RED_HAS_MUJOCO
