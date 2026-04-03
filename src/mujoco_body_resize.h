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
#include <thread>
#include <mutex>

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
    int    q_max_iters     = 1000;   // IK iterations during Q-phase (needs enough to converge)

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

// Evaluate IK residual on a PRIVATE model/data copy (thread-safe).
// model must be a per-thread copy. Returns mean residual in mm.
inline double body_resize_eval_private(
        mjModel *model, mjData *data_thread, MujocoIKState &ik,
        const std::vector<int> &skeleton_to_site,
        const std::vector<std::pair<int, const Keypoint3D *>> &sample_frames,
        int num_nodes, double sf, bool has_free_joint) {

    int N = (int)sample_frames.size();
    int nq = model->nq;

    // Build a temporary MujocoContext for ik_solve
    MujocoContext thread_mj;
    thread_mj.model = model;
    thread_mj.data = data_thread;
    thread_mj.loaded = true;
    thread_mj.scale_factor = 0.0f; // arena handles it
    thread_mj.skeleton_to_site = skeleton_to_site;
    thread_mj.mapped_count = (int)skeleton_to_site.size();
    thread_mj.has_free_joint = has_free_joint;

    mujoco_ik_reset(ik);
    std::vector<std::vector<double>> all_qpos(N, std::vector<double>(nq));

    for (int i = 0; i < N; i++) {
        mujoco_ik_solve(thread_mj, ik, sample_frames[i].second, num_nodes,
                        sample_frames[i].first);
        all_qpos[i].assign(data_thread->qpos, data_thread->qpos + nq);
    }

    // Measure residual
    double total_err = 0.0;
    int total_sites = 0;
    for (int i = 0; i < N; i++) {
        std::copy(all_qpos[i].begin(), all_qpos[i].end(), data_thread->qpos);
        mj_fwdPosition(model, data_thread);
        const Keypoint3D *kp = sample_frames[i].second;
        for (int n = 0; n < num_nodes; n++) {
            if (!kp[n].triangulated) continue;
            int si = (n < (int)skeleton_to_site.size()) ? skeleton_to_site[n] : -1;
            if (si < 0) continue;
            double dx = data_thread->site_xpos[3*si] - kp[n].x * sf;
            double dy = data_thread->site_xpos[3*si+1] - kp[n].y * sf;
            double dz = data_thread->site_xpos[3*si+2] - kp[n].z * sf;
            total_err += std::sqrt(dx*dx + dy*dy + dz*dz);
            total_sites++;
        }
    }

    // Don't let destructor free the model (shared)
    thread_mj.model = nullptr;
    thread_mj.data = nullptr;
    thread_mj.loaded = false;

    return (total_sites > 0) ? (total_err / total_sites) * 1000.0 : 0.0;
}

// Evaluate multiple scale candidates in parallel. Returns {scale, residual} for each.
inline std::vector<std::pair<double, double>> body_resize_eval_parallel(
        MujocoContext &mj, BodyResizeState &state,
        const std::vector<double> &candidates, int seg_idx,
        const std::vector<std::pair<int, const Keypoint3D *>> &sample_frames,
        int num_nodes, double sf, int q_max_iters) {

    int n_cand = (int)candidates.size();
    int n_threads = std::min(n_cand, std::max(1, (int)std::thread::hardware_concurrency()));
    n_threads = std::min(n_threads, 16);

    std::vector<std::pair<double, double>> results(n_cand);

    // Detect free joint
    bool has_free = false;
    for (int j = 0; j < mj.model->njnt; j++) {
        if (mj.model->jnt_type[j] == mjJNT_FREE) { has_free = true; break; }
    }

    // Create model copies — one per thread
    std::vector<mjModel *> models(n_threads);
    std::vector<mjData *> datas(n_threads);
    for (int t = 0; t < n_threads; t++) {
        models[t] = mj_copyModel(nullptr, mj.model);
        datas[t] = mj_makeData(models[t]);
    }

    std::atomic<int> next_job{0};

    auto worker = [&](int tid) {
        while (true) {
            int job = next_job.fetch_add(1, std::memory_order_relaxed);
            if (job >= n_cand) break;

            double trial_scale = candidates[job];
            mjModel *m = models[tid];

            // Apply: restore originals, then set all segments, overriding seg_idx
            int nsite = m->nsite, nbody = m->nbody, ngeom = m->ngeom, njnt = m->njnt;
            std::copy(state.original_body_pos.begin(), state.original_body_pos.end(), m->body_pos);
            std::copy(state.original_body_ipos.begin(), state.original_body_ipos.end(), m->body_ipos);
            std::copy(state.original_body_mass.begin(), state.original_body_mass.end(), m->body_mass);
            std::copy(state.original_body_inertia.begin(), state.original_body_inertia.end(), m->body_inertia);
            std::copy(state.original_geom_pos.begin(), state.original_geom_pos.end(), m->geom_pos);
            std::copy(state.original_geom_size.begin(), state.original_geom_size.end(), m->geom_size);
            std::copy(state.original_site_pos.begin(), state.original_site_pos.end(), m->site_pos);
            std::copy(state.original_jnt_pos.begin(), state.original_jnt_pos.end(), m->jnt_pos);

            // Apply current scales for all segments
            for (int g = 0; g < (int)state.segments.size(); g++) {
                double s = (g == seg_idx) ? trial_scale : state.segments[g].scale;
                if (std::abs(s - 1.0) < 1e-8) continue;
                double s3 = s*s*s, s5 = s3*s*s;
                for (int bid : state.segments[g].body_ids) {
                    for (int c = 0; c < 3; c++) { m->body_pos[3*bid+c] *= s; m->body_ipos[3*bid+c] *= s; }
                    m->body_mass[bid] *= s3;
                    for (int c = 0; c < 3; c++) m->body_inertia[3*bid+c] *= s5;
                    for (int gi = 0; gi < ngeom; gi++) {
                        if (m->geom_bodyid[gi] == bid) {
                            for (int c = 0; c < 3; c++) { m->geom_pos[3*gi+c] *= s; m->geom_size[3*gi+c] *= s; }
                        }
                    }
                    for (int si = 0; si < nsite; si++) {
                        if (m->site_bodyid[si] == bid) { for (int c = 0; c < 3; c++) m->site_pos[3*si+c] *= s; }
                    }
                    for (int ji = 0; ji < njnt; ji++) {
                        if (m->jnt_bodyid[ji] == bid) { for (int c = 0; c < 3; c++) m->jnt_pos[3*ji+c] *= s; }
                    }
                }
            }
            mj_setConst(m, datas[tid]);

            // Run IK
            MujocoIKState ik_thread;
            ik_thread.max_iterations = q_max_iters;
            ik_thread.lr = 0.001;
            ik_thread.beta = 0.99;
            ik_thread.reg_strength = 1e-4;
            ik_thread.cosine_annealing = true;

            double res = body_resize_eval_private(m, datas[tid], ik_thread,
                                                   mj.skeleton_to_site, sample_frames,
                                                   num_nodes, sf, has_free);
            results[job] = {trial_scale, res};
            state.round_frames_done.fetch_add((int)sample_frames.size(), std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; t++)
        threads.emplace_back(worker, t);
    for (auto &t : threads) t.join();

    // Cleanup
    for (int t = 0; t < n_threads; t++) {
        mj_deleteData(datas[t]);
        mj_deleteModel(models[t]);
    }

    return results;
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
    // PHASE 1: Global scale sweep (PARALLEL)
    // Find the single best uniform scale factor for the entire body.
    // =====================================================================
    std::cout << "[BodyResize] Phase 1: Global scale sweep (" << N << " frames, parallel)" << std::endl;
    state.total_rounds = 2;
    state.current_round = 1;
    state.phase = 1;
    state.round_frames_done = 0;

    double best_global = 1.0;
    double best_global_res = 1e9;

    // Coarse sweep: 0.80 to 1.30 in steps of 0.05 — all in parallel
    // Use seg_idx = -1 trick: set ALL segments to candidate scale
    {
        std::vector<double> coarse;
        for (double s = 0.80; s <= 1.301; s += 0.05) coarse.push_back(s);

        // For global sweep, temporarily set all segments to 1.0 so the
        // parallel eval applies the candidate as the sole scale.
        // We abuse seg_idx=0 but override all segments to the same scale.
        // Actually, simpler: for each candidate, set all segment scales
        // to that value before copying to threads. Do it sequentially
        // but run the IK evaluations in parallel.

        // Since eval_parallel modifies one segment at a time, for global
        // sweep we need a different approach: evaluate each candidate by
        // setting ALL segments to that scale. We'll run them in parallel.
        int n_cand = (int)coarse.size();
        int n_threads = std::min(n_cand, std::max(1, (int)std::thread::hardware_concurrency()));
        n_threads = std::min(n_threads, 16);

        bool has_free = false;
        for (int j = 0; j < mj.model->njnt; j++)
            if (mj.model->jnt_type[j] == mjJNT_FREE) { has_free = true; break; }

        std::vector<mjModel *> models(n_threads);
        std::vector<mjData *> datas(n_threads);
        for (int t = 0; t < n_threads; t++) {
            models[t] = mj_copyModel(nullptr, mj.model);
            datas[t] = mj_makeData(models[t]);
        }

        std::vector<std::pair<double, double>> results(n_cand);
        std::atomic<int> next_job{0};

        auto worker = [&](int tid) {
            while (true) {
                int job = next_job.fetch_add(1);
                if (job >= n_cand) break;
                double s = coarse[job];
                mjModel *m = models[tid];

                // Restore + apply uniform scale
                std::copy(state.original_body_pos.begin(), state.original_body_pos.end(), m->body_pos);
                std::copy(state.original_body_ipos.begin(), state.original_body_ipos.end(), m->body_ipos);
                std::copy(state.original_body_mass.begin(), state.original_body_mass.end(), m->body_mass);
                std::copy(state.original_body_inertia.begin(), state.original_body_inertia.end(), m->body_inertia);
                std::copy(state.original_geom_pos.begin(), state.original_geom_pos.end(), m->geom_pos);
                std::copy(state.original_geom_size.begin(), state.original_geom_size.end(), m->geom_size);
                std::copy(state.original_site_pos.begin(), state.original_site_pos.end(), m->site_pos);
                std::copy(state.original_jnt_pos.begin(), state.original_jnt_pos.end(), m->jnt_pos);

                double s3 = s*s*s, s5 = s3*s*s;
                for (int i = 1; i < m->nbody; i++) {  // skip world body
                    for (int c = 0; c < 3; c++) { m->body_pos[3*i+c] *= s; m->body_ipos[3*i+c] *= s; }
                    m->body_mass[i] *= s3;
                    for (int c = 0; c < 3; c++) m->body_inertia[3*i+c] *= s5;
                }
                for (int i = 0; i < m->ngeom; i++)
                    for (int c = 0; c < 3; c++) { m->geom_pos[3*i+c] *= s; m->geom_size[3*i+c] *= s; }
                for (int i = 0; i < m->nsite; i++)
                    for (int c = 0; c < 3; c++) m->site_pos[3*i+c] *= s;
                for (int i = 0; i < m->njnt; i++)
                    for (int c = 0; c < 3; c++) m->jnt_pos[3*i+c] *= s;
                mj_setConst(m, datas[tid]);

                MujocoIKState ik_t;
                ik_t.max_iterations = state.q_max_iters;
                ik_t.lr = 0.001; ik_t.beta = 0.99;
                ik_t.reg_strength = 1e-4; ik_t.cosine_annealing = true;

                double res = body_resize_eval_private(m, datas[tid], ik_t,
                    mj.skeleton_to_site, sample_frames, num_nodes, sf, has_free);
                results[job] = {s, res};
                state.round_frames_done.fetch_add(N);
            }
        };

        std::vector<std::thread> threads;
        for (int t = 0; t < n_threads; t++) threads.emplace_back(worker, t);
        for (auto &t : threads) t.join();
        for (int t = 0; t < n_threads; t++) { mj_deleteData(datas[t]); mj_deleteModel(models[t]); }

        for (auto &[s, res] : results) {
            std::cout << "[BodyResize]   global=" << std::fixed << std::setprecision(3) << s
                      << " residual=" << std::setprecision(2) << res << " mm" << std::endl;
            if (res < best_global_res) { best_global_res = res; best_global = s; }
        }
    }

    // Fine sweep (also parallel)
    {
        std::vector<double> fine;
        for (double s = best_global - 0.04; s <= best_global + 0.041; s += 0.01) {
            if (std::abs(s - best_global) < 0.001) continue;
            fine.push_back(s);
        }
        // Set all segments to best_global for the fine sweep
        for (auto &seg : state.segments) seg.scale = best_global;
        auto fine_results = body_resize_eval_parallel(mj, state, fine, -1,
                                                       sample_frames, num_nodes, sf, state.q_max_iters);
        // seg_idx=-1 won't work with eval_parallel, do it inline
        // Actually let's just do fine sweep sequentially — only 8 evaluations
        for (double s : fine) {
            state.round_frames_done = 0;
            body_resize_apply_global(mj, state, s);

            // Quick eval using main model
            MujocoIKState ik_fine;
            ik_fine.max_iterations = state.q_max_iters;
            ik_fine.lr = 0.001; ik_fine.beta = 0.99;
            ik_fine.reg_strength = 1e-4; ik_fine.cosine_annealing = true;
            std::vector<std::vector<double>> qpos_tmp;
            double res = body_resize_eval_private(mj.model, mj.data, ik_fine,
                mj.skeleton_to_site, sample_frames, num_nodes, sf, mj.has_free_joint);
            if (res < best_global_res) { best_global_res = res; best_global = s; }
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

            // Test multipliers in PARALLEL — each gets its own model copy.
            double multipliers[] = {0.80, 0.85, 0.90, 0.95, 1.00,
                                     1.05, 1.10, 1.15, 1.20, 1.30, 1.40, 1.50};
            std::vector<double> candidates;
            for (double mult : multipliers) {
                double trial = best_global * mult;
                if (trial >= 0.5 && trial <= 2.0)
                    candidates.push_back(trial);
            }

            auto seg_results = body_resize_eval_parallel(
                mj, state, candidates, g, sample_frames, num_nodes, sf, state.q_max_iters);

            best_res = 1e9;
            for (auto &[s, r] : seg_results) {
                if (r < best_res) { best_scale = s; best_res = r; }
            }

            std::cout << "[BodyResize]   " << state.segments[g].name
                      << " sweep: base=" << std::fixed << std::setprecision(3)
                      << base_scale << " -> best=" << best_scale
                      << " (" << std::setprecision(2) << best_res << "mm)"
                      << std::endl;

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
    {
        MujocoIKState ik_final;
        ik_final.max_iterations = state.q_max_iters;
        ik_final.lr = 0.001; ik_final.beta = 0.99;
        ik_final.reg_strength = 1e-4; ik_final.cosine_annealing = true;
        state.post_residual = body_resize_eval_private(mj.model, mj.data, ik_final,
            mj.skeleton_to_site, sample_frames, num_nodes, sf, mj.has_free_joint);
    }

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
