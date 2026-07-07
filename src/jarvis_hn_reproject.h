// jarvis_hn_reproject.h — host reprojection + soft-argmax for the CoreML HybridNet
// pipeline. Pure C++ (no Obj-C / CUDA), a faithful port of JARVIS's torch
// ReprojectionLayer + HybridNetBackbone soft-argmax, validated to ~0 vs the Python
// reference (hybridnet_reference.py). Replaces the TensorRT hybrid3d engine.
//
// Conventions (verified — see HYBRIDNET_MAC_NOTES.md):
//   camMats  = P^T, P = K[R|t]   layout (num_cameras,4,3) row-major
//   intrMats = K^T               layout (num_cameras,3,3)  cx=[2,0] cy=[2,1] fx=[0,0] fy=[1,1]
//   distC    = (num_cameras,5)   uses k1=[0], k2=[1] (radial only, matches ReprojectionLayer)
//   heatmaps = (num_cameras,num_joints,hs,hs) row-major, hs = bbox/2+2 (padded, e.g. 354)
//   grid: 50^3 sampling grid, projected coords trilinear-upsampled to 100^3 before indexing.
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdint>

struct HNReproParams {
    int num_cameras;      // 16
    int num_joints;       // 24
    int grid_full;        // 100  (= roi/spacing)
    float grid_spacing;   // 2.0
    float roi_cube;       // 200.0
    int heatmap_size;     // 354  (= bbox/2 + 2, padded)
};

// Trilinear sample of a (g,g,g) float volume at fractional (fi,fj,fk), border-clamped.
static inline float hn_trilinear(const float *vol, int g, float fi, float fj, float fk) {
    fi = std::min(std::max(fi, 0.0f), (float)(g - 1));
    fj = std::min(std::max(fj, 0.0f), (float)(g - 1));
    fk = std::min(std::max(fk, 0.0f), (float)(g - 1));
    int i0 = (int)fi, j0 = (int)fj, k0 = (int)fk;
    int i1 = std::min(i0 + 1, g - 1), j1 = std::min(j0 + 1, g - 1), k1 = std::min(k0 + 1, g - 1);
    float di = fi - i0, dj = fj - j0, dk = fk - k0;
    auto V = [&](int i, int j, int k) { return vol[(i * g + j) * g + k]; };
    float c00 = V(i0, j0, k0) * (1 - dk) + V(i0, j0, k1) * dk;
    float c01 = V(i0, j1, k0) * (1 - dk) + V(i0, j1, k1) * dk;
    float c10 = V(i1, j0, k0) * (1 - dk) + V(i1, j0, k1) * dk;
    float c11 = V(i1, j1, k0) * (1 - dk) + V(i1, j1, k1) * dk;
    float c0 = c00 * (1 - dj) + c01 * dj;
    float c1 = c10 * (1 - dj) + c11 * dj;
    return c0 * (1 - di) + c1 * di;
}

// Build the reprojected voxel volume h3d (num_joints * grid_full^3), mean over cameras.
// heatmaps: num_cameras*num_joints*hs*hs. h3d must be pre-sized to num_joints*gf^3.
static inline void hn_reproject(const HNReproParams &P,
        const float *camMats, const float *intrMats, const float *distC,
        const float *center3D, const float *cHM, const float *heatmaps,
        float *h3d) {
    const int NC = P.num_cameras, NJ = P.num_joints;
    const int gf = P.grid_full, gh = gf / 2;          // 100, 50
    const int hs = P.heatmap_size;
    const float step = P.grid_spacing * 2.0f;         // 4mm between 50^3 samples
    const int half = gh / 2;                          // 25
    const float hsm1 = (float)(hs - 1), hsm2 = (float)(hs - 2);
    const size_t vf = (size_t)gf * gf * gf;
    const size_t vh = (size_t)gh * gh * gh;

    // --- Stage 1: project the 50^3 grid per camera -> val1/val2 (clamped, shifted) ---
    std::vector<float> val1(NC * vh), val2(NC * vh);
    for (int c = 0; c < NC; ++c) {
        const float *M = camMats + (size_t)c * 12;     // 4x3
        const float *K = intrMats + (size_t)c * 9;     // 3x3 (K^T)
        float cx = K[2 * 3 + 0], cy = K[2 * 3 + 1], fx = K[0 * 3 + 0], fy = K[1 * 3 + 1];
        float k1 = distC[(size_t)c * 5 + 0], k2 = distC[(size_t)c * 5 + 1];
        float chm0 = cHM[(size_t)c * 2 + 0], chm1 = cHM[(size_t)c * 2 + 1];
        float *v1 = val1.data() + (size_t)c * vh;
        float *v2 = val2.data() + (size_t)c * vh;
        size_t idx = 0;
        for (int i = 0; i < gh; ++i)
        for (int j = 0; j < gh; ++j)
        for (int k = 0; k < gh; ++k, ++idx) {
            float wx = (i - half) * step + center3D[0];
            float wy = (j - half) * step + center3D[1];
            float wz = (k - half) * step + center3D[2];
            // partial = [wx,wy,wz,1] @ M (4x3)
            float pu = wx * M[0] + wy * M[3] + wz * M[6] + M[9];
            float pv = wx * M[1] + wy * M[4] + wz * M[7] + M[10];
            float pw = wx * M[2] + wy * M[5] + wz * M[8] + M[11];
            float a = pu / pw - cx;
            float b = pv / pw - cy;
            float r2 = (a / fx) * (a / fx) + (b / fy) * (b / fy);
            float distort = 1.0f + (k1 + k2 * r2) * r2;
            a = a * distort + cx;
            b = b * distort + cy;
            a = std::min(std::max(a, chm0 - hsm1), chm0 + hsm2) - chm0 + hsm1;  // -> [0, 2*hs-3]
            b = std::min(std::max(b, chm1 - hsm1), chm1 + hsm2) - chm1 + hsm1;
            v1[idx] = a;
            v2[idx] = b;
        }
    }

    // --- Stage 2: trilinear-upsample val1/val2 to 100^3, index heatmaps, mean over cams ---
    for (size_t t = 0; t < (size_t)NJ * vf; ++t) h3d[t] = 0.0f;
    const float inv_nc = 1.0f / (float)NC;
    for (int c = 0; c < NC; ++c) {
        const float *v1 = val1.data() + (size_t)c * vh;
        const float *v2 = val2.data() + (size_t)c * vh;
        const float *hm_c = heatmaps + (size_t)c * NJ * hs * hs;
        size_t vidx = 0;
        for (int i = 0; i < gf; ++i) {
            float fi = i * 0.5f - 0.25f;                 // align_corners=False, scale 0.5
            for (int j = 0; j < gf; ++j) {
                float fj = j * 0.5f - 0.25f;
                for (int k = 0; k < gf; ++k, ++vidx) {
                    float fk = k * 0.5f - 0.25f;
                    float a = hn_trilinear(v1, gh, fi, fj, fk);
                    float b = hn_trilinear(v2, gh, fi, fj, fk);
                    int col = (int)(a * 0.5f);           // truncation toward zero (a>=0)
                    int row = (int)(b * 0.5f);
                    int hidx = row * hs + col;
                    for (int jt = 0; jt < NJ; ++jt)
                        h3d[(size_t)jt * vf + vidx] += hm_c[(size_t)jt * hs * hs + hidx] * inv_nc;
                }
            }
        }
    }
}

// Numerically-stable softplus matching torch (beta=1, threshold=20).
static inline float hn_softplus(float x) {
    return x > 20.0f ? x : std::log1p(std::exp(x));
}

// Soft-argmax over a (num_joints * g^3) volume -> world-mm points (num_joints*3) + conf.
// pts and conf must be pre-sized. Mirrors HybridNetBackbone: axis0->X, axis1->Y, axis2->Z.
static inline void hn_soft_argmax(const HNReproParams &P, const float *vol, int g,
        const float *center3D, float *pts, float *conf) {
    const int NJ = P.num_joints;
    const size_t v = (size_t)g * g * g;
    const float scale = P.grid_spacing * 2.0f;          // *4
    const float offset = P.roi_cube * 0.5f;             // -100
    for (int jt = 0; jt < NJ; ++jt) {
        const float *vp = vol + (size_t)jt * v;
        double norm = 0, sx = 0, sy = 0, sz = 0, mx = -1e30;
        size_t idx = 0;
        for (int i = 0; i < g; ++i)
        for (int j = 0; j < g; ++j)
        for (int k = 0; k < g; ++k, ++idx) {
            float h = hn_softplus(vp[idx]);
            norm += h; sx += h * i; sy += h * j; sz += h * k;
            if (h > mx) mx = h;
        }
        float x = (float)(sx / norm), y = (float)(sy / norm), z = (float)(sz / norm);
        pts[jt * 3 + 0] = x * scale - offset + center3D[0];
        pts[jt * 3 + 1] = y * scale - offset + center3D[1];
        pts[jt * 3 + 2] = z * scale - offset + center3D[2];
        conf[jt] = std::min((float)mx, 255.0f) / 255.0f;
    }
}
