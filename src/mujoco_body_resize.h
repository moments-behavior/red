#pragma once
// mujoco_body_resize.h — Per-segment body model resizing
//
// Optimizes per-segment scale factors to fit the MuJoCo body model to
// 3D keypoint data. Uses the same alternating Q/M optimization as STAC:
//   Q-phase: IK solve per frame (reuses mujoco_ik_solve)
//   M-phase: SGD on segment scale factors to minimize aggregate residual
//
// The key gradient: d(site_xpos)/d(scale_g) for a site on body b in
// segment group g is: body_xmat * original_body_pos_b (the original
// offset from parent, rotated into world frame). MuJoCo computes
// body_xmat during mj_fwdPosition — no autodiff needed.
//
// L/R symmetry: paired segment groups (upper_leg, lower_arm, etc.)
// share a single scale factor by construction.

#include "mujoco_context.h"
#include "mujoco_ik.h"
#include "annotation.h"
#include "arena_alignment.h"
#include <Eigen/Core>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>
#include <map>

// A segment group: a named set of bodies that scale together.
struct SegmentGroup {
    std::string name;
    std::vector<int> body_ids;    // MuJoCo body indices
    bool is_paired = false;       // L/R pair (same scale enforced)
    double scale = 1.0;           // current scale factor
};

struct BodyResizeState {
    // Segment groups (built from model on load)
    std::vector<SegmentGroup> segments;

    // Optimization parameters
    int    n_iters         = 6;      // alternating Q/M rounds
    int    n_sample_frames = 200;    // frames for optimization
    double m_lr            = 0.01;   // SGD learning rate for scale factors
    double m_momentum      = 0.9;    // SGD momentum
    double m_reg_coef      = 1.0;    // L2 penalty on (scale - 1)^2
    int    m_max_iters     = 200;    // M-phase SGD iterations per round
    int    q_max_iters     = 300;    // IK iterations during Q-phase

    // State
    bool   calibrated = false;
    double pre_residual  = 0.0;     // mm
    double post_residual = 0.0;     // mm
    double calibration_time_s = 0.0;

    // Backup of original model arrays (before any resizing)
    std::vector<double> original_body_pos;     // 3 * nbody
    std::vector<double> original_body_ipos;    // 3 * nbody
    std::vector<double> original_body_mass;    // nbody
    std::vector<double> original_body_inertia; // 3 * nbody
    std::vector<double> original_geom_pos;     // 3 * ngeom
    std::vector<double> original_geom_size;    // 3 * ngeom
    std::vector<double> original_site_pos;     // 3 * nsite
    std::vector<double> original_jnt_pos;      // 3 * njnt

    // SGD velocity per segment
    std::vector<double> m_velocity;  // n_segments

    // Loaded from session (applied after segments are built)
    std::map<std::string, double> loaded_scales;

    // Live progress
    std::atomic<int> current_round{0};
    std::atomic<int> total_rounds{0};
    std::atomic<int> phase{0};       // 0=idle, 1=Q-phase, 2=M-phase, 3=done
    std::atomic<int> round_frames_done{0};
    std::atomic<int> round_frames_total{0};
};

#ifdef RED_HAS_MUJOCO

// Build segment groups for the rodent model.
inline std::vector<SegmentGroup> build_rodent_segments(const mjModel *model) {
    auto id = [&](const char *name) -> int {
        return mj_name2id(model, mjOBJ_BODY, name);
    };

    auto add = [&](std::vector<SegmentGroup> &segs,
                    const char *name, std::initializer_list<const char *> bodies,
                    bool paired) {
        SegmentGroup g;
        g.name = name;
        g.is_paired = paired;
        for (auto *bname : bodies) {
            int bid = id(bname);
            if (bid >= 0) g.body_ids.push_back(bid);
        }
        if (!g.body_ids.empty()) segs.push_back(std::move(g));
    };

    std::vector<SegmentGroup> segs;

    add(segs, "head", {"skull", "jaw"}, false);

    // Neck (cervical chain)
    {
        SegmentGroup g;
        g.name = "neck";
        for (auto *n : {"vertebra_cervical_5", "vertebra_cervical_4",
                         "vertebra_cervical_3", "vertebra_cervical_2",
                         "vertebra_cervical_1", "vertebra_axis",
                         "vertebra_atlant"}) {
            int bid = id(n);
            if (bid >= 0) g.body_ids.push_back(bid);
        }
        if (!g.body_ids.empty()) segs.push_back(std::move(g));
    }

    // Lumbar spine
    {
        SegmentGroup g;
        g.name = "spine";
        for (int i = 1; i <= 6; i++) {
            std::string name = "vertebra_" + std::to_string(i);
            int bid = id(name.c_str());
            if (bid >= 0) g.body_ids.push_back(bid);
        }
        if (!g.body_ids.empty()) segs.push_back(std::move(g));
    }

    add(segs, "pelvis", {"pelvis"}, false);

    // Tail (all caudal vertebrae)
    {
        SegmentGroup g;
        g.name = "tail";
        for (int i = 1; i <= 30; i++) {
            std::string name = "vertebra_C" + std::to_string(i);
            int bid = id(name.c_str());
            if (bid >= 0) g.body_ids.push_back(bid);
        }
        if (!g.body_ids.empty()) segs.push_back(std::move(g));
    }

    // Limbs (L/R paired)
    add(segs, "scapula",   {"scapula_L", "scapula_R"}, true);
    add(segs, "upper_arm", {"upper_arm_L", "upper_arm_R"}, true);
    add(segs, "lower_arm", {"lower_arm_L", "lower_arm_R"}, true);
    add(segs, "hand",      {"hand_L", "hand_R", "finger_L", "finger_R"}, true);
    add(segs, "upper_leg", {"upper_leg_L", "upper_leg_R"}, true);
    add(segs, "lower_leg", {"lower_leg_L", "lower_leg_R"}, true);
    add(segs, "foot",      {"foot_L", "foot_R", "toe_L", "toe_R"}, true);

    return segs;
}

// Backup original model arrays (call once after model load, before any resizing).
inline void body_resize_backup(const MujocoContext &mj, BodyResizeState &state) {
    const mjModel *m = mj.model;
    state.original_body_pos.assign(m->body_pos, m->body_pos + 3 * m->nbody);
    state.original_body_ipos.assign(m->body_ipos, m->body_ipos + 3 * m->nbody);
    state.original_body_mass.assign(m->body_mass, m->body_mass + m->nbody);
    state.original_body_inertia.assign(m->body_inertia, m->body_inertia + 3 * m->nbody);
    state.original_geom_pos.assign(m->geom_pos, m->geom_pos + 3 * m->ngeom);
    state.original_geom_size.assign(m->geom_size, m->geom_size + 3 * m->ngeom);
    state.original_site_pos.assign(m->site_pos, m->site_pos + 3 * m->nsite);
    state.original_jnt_pos.assign(m->jnt_pos, m->jnt_pos + 3 * m->njnt);
}

// Restore model to original proportions (undo all resizing).
inline void body_resize_restore(MujocoContext &mj, const BodyResizeState &state) {
    if (state.original_body_pos.empty()) return;
    mjModel *m = mj.model;
    std::copy(state.original_body_pos.begin(), state.original_body_pos.end(), m->body_pos);
    std::copy(state.original_body_ipos.begin(), state.original_body_ipos.end(), m->body_ipos);
    std::copy(state.original_body_mass.begin(), state.original_body_mass.end(), m->body_mass);
    std::copy(state.original_body_inertia.begin(), state.original_body_inertia.end(), m->body_inertia);
    std::copy(state.original_geom_pos.begin(), state.original_geom_pos.end(), m->geom_pos);
    std::copy(state.original_geom_size.begin(), state.original_geom_size.end(), m->geom_size);
    std::copy(state.original_site_pos.begin(), state.original_site_pos.end(), m->site_pos);
    std::copy(state.original_jnt_pos.begin(), state.original_jnt_pos.end(), m->jnt_pos);
}

// Apply current segment scale factors to the model.
// Restores to originals first, then scales each segment group.
inline void body_resize_apply(MujocoContext &mj, const BodyResizeState &state) {
    if (state.original_body_pos.empty()) return;
    body_resize_restore(mj, state);

    mjModel *m = mj.model;
    for (const auto &seg : state.segments) {
        double s = seg.scale;
        if (std::abs(s - 1.0) < 1e-8) continue;

        double s3 = s * s * s, s5 = s3 * s * s;

        for (int bid : seg.body_ids) {
            // Scale body_pos (offset from parent — this is the "bone length")
            for (int c = 0; c < 3; c++) {
                m->body_pos[3*bid + c] *= s;
                m->body_ipos[3*bid + c] *= s;
            }
            m->body_mass[bid] *= s3;
            for (int c = 0; c < 3; c++)
                m->body_inertia[3*bid + c] *= s5;

            // Scale geoms attached to this body
            for (int g = 0; g < m->ngeom; g++) {
                if (m->geom_bodyid[g] == bid) {
                    for (int c = 0; c < 3; c++) {
                        m->geom_pos[3*g + c] *= s;
                        m->geom_size[3*g + c] *= s;
                    }
                }
            }

            // Scale sites attached to this body
            for (int si = 0; si < m->nsite; si++) {
                if (m->site_bodyid[si] == bid) {
                    for (int c = 0; c < 3; c++)
                        m->site_pos[3*si + c] *= s;
                }
            }

            // Scale joints attached to this body
            for (int j = 0; j < m->njnt; j++) {
                if (m->jnt_bodyid[j] == bid) {
                    for (int c = 0; c < 3; c++)
                        m->jnt_pos[3*j + c] *= s;
                }
            }
        }
    }

    // Recompute derived quantities
    mj_setConst(m, mj.data);
    mj_forward(m, mj.data);
}

// Reset all scales to 1.0 and restore model.
inline void body_resize_reset(MujocoContext &mj, BodyResizeState &state) {
    for (auto &seg : state.segments) seg.scale = 1.0;
    body_resize_restore(mj, state);
    mj_setConst(mj.model, mj.data);
    mj_forward(mj.model, mj.data);
    state.calibrated = false;
    state.pre_residual = 0.0;
    state.post_residual = 0.0;
}

// Helper: apply a uniform scale to ALL segments and recompute model.
inline void body_resize_apply_global(MujocoContext &mj, BodyResizeState &state, double s) {
    for (auto &seg : state.segments) seg.scale = s;
    body_resize_apply(mj, state);
}

// Helper: compute mean IK residual across sample frames.
// Sites are at default XML positions (no STAC). Returns residual in mm.
inline double body_resize_eval(MujocoContext &mj, MujocoIKState &ik,
                                const std::vector<std::pair<int, const Keypoint3D *>> &sample_frames,
                                int num_nodes, double sf,
                                std::vector<std::vector<double>> &all_qpos,
                                BodyResizeState &state) {
    int N = (int)sample_frames.size();
    int nq = (int)mj.model->nq;
    if ((int)all_qpos.size() != N) all_qpos.resize(N, std::vector<double>(nq));

    mujoco_ik_reset(ik);
    for (int i = 0; i < N; i++) {
        mujoco_ik_solve(mj, ik, sample_frames[i].second, num_nodes,
                        sample_frames[i].first);
        all_qpos[i].assign(mj.data->qpos, mj.data->qpos + nq);
        state.round_frames_done.fetch_add(1, std::memory_order_relaxed);
    }

    // Measure residual
    double total_err = 0.0;
    int total_sites = 0;
    for (int i = 0; i < N; i++) {
        std::copy(all_qpos[i].begin(), all_qpos[i].end(), mj.data->qpos);
        mj_fwdPosition(mj.model, mj.data);
        const Keypoint3D *kp = sample_frames[i].second;
        for (int n = 0; n < num_nodes; n++) {
            if (!kp[n].triangulated) continue;
            int si = (n < (int)mj.skeleton_to_site.size()) ? mj.skeleton_to_site[n] : -1;
            if (si < 0) continue;
            double dx = mj.data->site_xpos[3*si] - kp[n].x * sf;
            double dy = mj.data->site_xpos[3*si+1] - kp[n].y * sf;
            double dz = mj.data->site_xpos[3*si+2] - kp[n].z * sf;
            total_err += std::sqrt(dx*dx + dy*dy + dz*dz);
            total_sites++;
        }
    }
    return (total_sites > 0) ? (total_err / total_sites) * 1000.0 : 0.0;
}

// Run body resize: Phase 1 (global scale sweep) + Phase 2 (per-segment refinement).
// Sites stay at XML defaults throughout — NO STAC. STAC runs after this.
inline bool body_resize_calibrate(MujocoContext &mj, BodyResizeState &state,
                                   MujocoIKState &ik,
                                   const AnnotationMap &annotations, int num_nodes,
                                   const ArenaAlignment *arena_align = nullptr) {
    if (!mj.loaded || !mj.model || !mj.data) return false;
    if (state.segments.empty()) return false;
    if (state.original_body_pos.empty()) body_resize_backup(mj, state);

    // CRITICAL: Reset sites to default XML positions. Body resize fits the
    // body geometry itself — STAC site offsets come later.
    {
        int nsite_total = (int)mj.model->nsite;
        for (int i = 0; i < nsite_total * 3; i++)
            mj.model->site_pos[i] = state.original_site_pos[i];
        mj_fwdPosition(mj.model, mj.data);
        std::cout << "[BodyResize] Reset sites to XML defaults (no STAC)" << std::endl;
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    int nq = (int)mj.model->nq;
    int n_seg = (int)state.segments.size();

    // Scale factor
    double sf = (double)mj.scale_factor;
    if (sf <= 0.0) sf = 0.001;
    bool use_arena = (arena_align && arena_align->valid);
    if (use_arena) sf = 1.0;

    // --- Collect and transform frames ---
    struct FrameData {
        int frame_num;
        std::vector<Keypoint3D> kp3d_owned;
        const Keypoint3D *kp3d = nullptr;
    };
    std::vector<FrameData> frames;
    for (const auto &[fnum, fa] : annotations) {
        if (fa.kp3d.empty()) continue;
        int active = 0;
        for (int n = 0; n < num_nodes && n < (int)fa.kp3d.size(); n++)
            if (fa.kp3d[n].triangulated) active++;
        if (active < 4) continue;
        FrameData fd;
        fd.frame_num = (int)fnum;
        fd.kp3d_owned = fa.kp3d;
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
    if (frames.empty()) return false;
    for (auto &fd : frames) fd.kp3d = fd.kp3d_owned.data();
    std::sort(frames.begin(), frames.end(),
              [](const FrameData &a, const FrameData &b) { return a.frame_num < b.frame_num; });

    // Subsample
    int N = (int)frames.size();
    std::vector<int> sample_indices(N);
    std::iota(sample_indices.begin(), sample_indices.end(), 0);
    if (state.n_sample_frames > 0 && state.n_sample_frames < N) {
        std::mt19937 rng(42);
        std::shuffle(sample_indices.begin(), sample_indices.end(), rng);
        sample_indices.resize(state.n_sample_frames);
        std::sort(sample_indices.begin(), sample_indices.end());
        N = state.n_sample_frames;
    }

    // Build sample frame list for eval helper
    std::vector<std::pair<int, const Keypoint3D *>> sample_frames(N);
    for (int i = 0; i < N; i++) {
        int fi = sample_indices[i];
        sample_frames[i] = {frames[fi].frame_num, frames[fi].kp3d};
    }

    // Save IK settings, use calibration settings
    int saved_max_iters = ik.max_iterations;
    ik.max_iterations = state.q_max_iters;

    state.round_frames_total = N;
    std::vector<std::vector<double>> all_qpos;

    // =====================================================================
    // PHASE 1: Global scale sweep
    // Find the single best uniform scale factor for the entire body.
    // =====================================================================
    std::cout << "[BodyResize] Phase 1: Global scale sweep (" << N << " frames)" << std::endl;
    state.total_rounds = 2; // phase 1 + phase 2
    state.current_round = 1;
    state.phase = 1;

    double best_global = 1.0;
    double best_global_res = 1e9;

    // Coarse sweep: 0.80 to 1.30 in steps of 0.05
    for (double s = 0.80; s <= 1.301; s += 0.05) {
        state.round_frames_done = 0;
        body_resize_apply_global(mj, state, s);
        double res = body_resize_eval(mj, ik, sample_frames, num_nodes, sf, all_qpos, state);
        std::cout << "[BodyResize]   global=" << std::fixed << std::setprecision(3) << s
                  << " residual=" << std::setprecision(2) << res << " mm" << std::endl;
        if (res < best_global_res) {
            best_global_res = res;
            best_global = s;
        }
    }

    // Fine sweep around best: ±0.05 in steps of 0.01
    for (double s = best_global - 0.04; s <= best_global + 0.041; s += 0.01) {
        if (std::abs(s - best_global) < 0.001) continue; // skip already tested
        state.round_frames_done = 0;
        body_resize_apply_global(mj, state, s);
        double res = body_resize_eval(mj, ik, sample_frames, num_nodes, sf, all_qpos, state);
        if (res < best_global_res) {
            best_global_res = res;
            best_global = s;
        }
    }

    // Apply best global scale
    body_resize_apply_global(mj, state, best_global);
    state.pre_residual = best_global_res;
    std::cout << "[BodyResize] Best global scale: " << std::fixed << std::setprecision(3)
              << best_global << " (residual: " << std::setprecision(2)
              << best_global_res << " mm)" << std::endl;

    // =====================================================================
    // PHASE 2: Per-segment coordinate descent
    // Starting from the best global scale, optimize each segment individually.
    // =====================================================================
    std::cout << "[BodyResize] Phase 2: Per-segment refinement" << std::endl;
    state.current_round = 2;

    for (int outer = 0; outer < state.n_iters; outer++) {
        bool any_improved = false;

        for (int g = 0; g < n_seg; g++) {
            double base_scale = state.segments[g].scale;
            double best_scale = base_scale;
            double best_res = 1e9;

            // Test a wide range of multipliers on this segment independently.
            // Other segments stay at their current (possibly already adjusted) scales.
            double multipliers[] = {0.70, 0.80, 0.85, 0.90, 0.95, 1.00,
                                     1.05, 1.10, 1.15, 1.20, 1.30, 1.50};
            for (double mult : multipliers) {
                double trial = best_global * mult;
                if (trial < 0.5 || trial > 2.0) continue;

                state.segments[g].scale = trial;
                state.round_frames_done = 0;
                body_resize_apply(mj, state);
                double res = body_resize_eval(mj, ik, sample_frames, num_nodes, sf, all_qpos, state);

                // Accept if better (with regularization toward global optimum)
                double penalized = res + state.m_reg_coef * std::abs(mult - 1.0) * 1000.0;
                double penalized_best = best_res + state.m_reg_coef * std::abs(best_scale / best_global - 1.0) * 1000.0;
                if (penalized < penalized_best || best_res > 1e8) {
                    best_scale = trial;
                    best_res = res;
                }
            }

            if (std::abs(best_scale - base_scale) > 0.001) {
                any_improved = true;
                std::cout << "[BodyResize]   " << state.segments[g].name
                          << ": " << std::fixed << std::setprecision(3) << base_scale
                          << " -> " << best_scale
                          << " (" << std::setprecision(2) << (best_scale/best_global)
                          << "x relative)" << std::endl;
            }
            state.segments[g].scale = best_scale;
        }

        body_resize_apply(mj, state);

        if (!any_improved) {
            std::cout << "[BodyResize] No improvement in round " << outer+1 << ", stopping" << std::endl;
            break;
        }
    }

    // =====================================================================
    // Final evaluation
    // =====================================================================
    state.phase = 3;
    state.round_frames_done = 0;
    body_resize_apply(mj, state);
    state.post_residual = body_resize_eval(mj, ik, sample_frames, num_nodes, sf, all_qpos, state);

    // Restore IK settings
    ik.max_iterations = saved_max_iters;

    state.calibrated = true;
    state.phase = 0;
    auto t_end = std::chrono::high_resolution_clock::now();
    state.calibration_time_s = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "[BodyResize] Complete: " << state.calibration_time_s << "s" << std::endl;
    std::cout << "[BodyResize] Residual: " << state.pre_residual << " mm -> "
              << state.post_residual << " mm" << std::endl;
    std::cout << "[BodyResize] Scales:";
    for (const auto &seg : state.segments)
        std::cout << " " << seg.name << "=" << std::fixed << std::setprecision(3) << seg.scale;
    std::cout << std::endl;

    return true;
}

#else // !RED_HAS_MUJOCO
inline void body_resize_apply(MujocoContext &, const BodyResizeState &) {}
inline void body_resize_reset(MujocoContext &, BodyResizeState &) {}
inline bool body_resize_calibrate(MujocoContext &, BodyResizeState &, MujocoIKState &,
                                   const AnnotationMap &, int, const ArenaAlignment *) { return false; }
#endif
