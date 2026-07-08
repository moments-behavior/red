// jarvis_hn_metal.h — GPU (Metal) port of hn_reproject (jarvis_hn_reproject.h).
//
// Runs the HybridNet host reprojection (project 50³ grid per camera →
// trilinear-upsample to 100³ → int-truncated heatmap gather → mean over cams)
// on the Apple GPU. fp32, fast-math disabled, so it matches the CPU reference
// (hn_reproject) to fp noise. Replaces the single-threaded host loop that
// dominates the Mac HybridNet frame time (~450ms at -O3 → target <5ms on GPU).
//
// Usage:
//   HNMetalReproject m;
//   m.init(P);                       // compile kernels + allocate persistent buffers
//   // per frame:
//   memcpy(m.heatmaps_ptr(), ...);   // NC*NJ*hs*hs, cam-major (or write directly)
//   m.reproject(camMats, intrMats, distC, center3D, cHM);
//   const float* h3d = m.h3d_ptr();  // NJ*gf³, valid after reproject() returns
//
// The heatmap and h3d buffers live in shared (unified) memory — the CPU pointers
// alias the GPU buffers, so callers can fill heatmaps in place and read h3d out
// with no extra copy. Pure geometry, ZERO learned params.
#pragma once
#include <string>
#include "jarvis_hn_reproject.h"   // HNReproParams

class HNMetalReproject {
public:
    HNMetalReproject() = default;
    ~HNMetalReproject();
    HNMetalReproject(const HNMetalReproject &) = delete;
    HNMetalReproject &operator=(const HNMetalReproject &) = delete;

    // Compile the compute kernels and allocate persistent buffers for the given
    // (fixed) geometry. Returns false (and sets err) on failure — caller should
    // fall back to the CPU hn_reproject. Safe to call once per model load.
    bool init(const HNReproParams &P, std::string *err = nullptr);
    bool ready() const { return impl_ != nullptr; }

    // Shared-memory buffer the caller fills each frame with the padded, cam-major
    // keypoint heatmaps (NC*NJ*hs*hs floats). Writing here directly avoids a
    // staging copy. fp32 — bit-identical to the validated CPU hn_reproject path.
    float *heatmaps_ptr();
    // Output voxel volume (NJ*gf³ floats). Valid after reproject() returns.
    const float *h3d_ptr();

    // Run the reprojection over the heatmaps currently in heatmaps_ptr(), using
    // the per-frame camera geometry. Blocks until the GPU finishes; on success
    // h3d_ptr() holds the mean-over-cameras voxel volume. Layouts match
    // hn_reproject exactly (camMats=P^T NC*4*3, intrMats=K^T NC*3*3,
    // distC NC*5, center3D 3, cHM NC*2).
    bool reproject(const float *camMats, const float *intrMats,
                   const float *distC, const float *center3D, const float *cHM);

    // GPU execution time (ms) of the last reproject(), from Metal timestamps.
    // last_project_ms/last_gather_ms populated only when HN_DIAG env is set.
    float last_gpu_ms = 0.0f;
    float last_project_ms = 0.0f;
    float last_gather_ms = 0.0f;

private:
    void *impl_ = nullptr;   // Obj-C++ state (device/queue/pipelines/buffers)
};
