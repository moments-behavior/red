// tests/test_jarvis_hn_parity.mm
//
// Parity test: the CPU host reprojection (hn_reproject, jarvis_hn_reproject.h)
// and the Metal GPU reprojection (HNMetalReproject, jarvis_hn_metal.mm) implement
// the SAME projection + trilinear + int-truncated gather + camera-mean math in two
// languages, and the headers claim they are "bit-exact". Nothing enforced that.
// This feeds both identical randomized (but deterministic) geometry + heatmaps and
// asserts the two output voxel volumes agree to fp noise.
//
// Apple-only (Metal). Registered as an Apple test target in CMakeLists.txt.
//
// Standalone build (no full app):
//   clang++ -std=c++17 -ObjC++ -O2 -Isrc \
//     tests/test_jarvis_hn_parity.mm src/jarvis_hn_metal.mm \
//     -framework Metal -framework Foundation -o /tmp/hn_parity && /tmp/hn_parity

#include "jarvis_hn_reproject.h"
#include "jarvis_hn_metal.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    // Fly-scale geometry: affine telecentric DLT (P row2 = [0,0,0,1] -> pw==1),
    // zero distortion, ~82 px/mm, small ROI. This is the real use case and it
    // also exercises the gather inside the heatmap rather than all-border-clamp.
    HNReproParams P;
    P.num_cameras  = 7;
    P.num_joints   = 8;      // fewer joints than the model to keep it quick
    P.grid_full    = 48;     // grid_in
    P.grid_spacing = 0.1f;   // mm per step (fly)
    P.roi_cube     = 4.8f;   // mm cube
    P.heatmap_size = 226;    // keypoint_bbox/2 + 2 = 448/2 + 2

    const int NC = P.num_cameras, NJ = P.num_joints, hs = P.heatmap_size;
    const int gf = P.grid_full;
    const size_t vf = (size_t)NJ * gf * gf * gf;
    const size_t hm_n = (size_t)NC * NJ * hs * hs;

    std::mt19937 rng(12345);                       // deterministic
    std::uniform_real_distribution<float> jit(-0.3f, 0.3f);

    // Center of the animal in world mm.
    const float center3D[3] = {11.0f, 2.0f, 1.3f};

    // Per-camera affine DLT (P^T layout, 4x3 row-major) + K^T + zero distortion.
    std::vector<float> camMats((size_t)NC * 12, 0.0f);
    std::vector<float> intrMats((size_t)NC * 9, 0.0f);
    std::vector<float> distC((size_t)NC * 5, 0.0f);   // zero distortion (telecentric)
    std::vector<float> cHM((size_t)NC * 2, 0.0f);

    const float PXMM = 82.0f;                       // ~pixels per mm
    for (int c = 0; c < NC; ++c) {
        // A small per-camera in-plane rotation so views differ.
        float ang = 0.25f * c + jit(rng);
        float ca = std::cos(ang), sa = std::sin(ang);
        // Affine 3x4 projection Pm (row-major); row2 = [0,0,0,1].
        // u = PXMM*( ca*X - sa*Y) + tx ;  v = PXMM*( sa*X + ca*Y) + ty
        float tx = 224.0f, ty = 224.0f;             // put the animal near crop centre
        float Pm[3][4] = {
            { PXMM * ca, -PXMM * sa, 0.0f, tx - PXMM * (ca * center3D[0] - sa * center3D[1]) },
            { PXMM * sa,  PXMM * ca, 0.0f, ty - PXMM * (sa * center3D[0] + ca * center3D[1]) },
            { 0.0f, 0.0f, 0.0f, 1.0f },
        };
        // camMats = P^T : camMats[c*12 + r*3 + col] = Pm[col][r]
        float *M = camMats.data() + (size_t)c * 12;
        for (int r = 0; r < 4; ++r)
            for (int col = 0; col < 3; ++col)
                M[r * 3 + col] = Pm[col][r];
        // intrMats = K^T with K = [sx 0 tx; 0 sy ty; 0 0 1]; only fx,fy,cx,cy read.
        float *K = intrMats.data() + (size_t)c * 9;
        K[0] = PXMM;            // K^T[0,0] = fx
        K[4] = PXMM;            // K^T[1,1] = fy
        K[6] = tx;              // K^T[2,0] = cx
        K[7] = ty;              // K^T[2,1] = cy
        K[8] = 1.0f;
        // centerHM = projection of center3D (pw==1 so it's just the affine part).
        cHM[(size_t)c * 2 + 0] = Pm[0][0] * center3D[0] + Pm[0][1] * center3D[1] +
                                 Pm[0][2] * center3D[2] + Pm[0][3];
        cHM[(size_t)c * 2 + 1] = Pm[1][0] * center3D[0] + Pm[1][1] * center3D[1] +
                                 Pm[1][2] * center3D[2] + Pm[1][3];
    }

    // Smooth heatmaps: a Gaussian bump per (cam,joint) so a 1-px truncation flip
    // between CPU and GPU produces a SMALL value difference (realistic; strict
    // bit-equality on a random field would be dominated by boundary flips).
    std::vector<float> heatmaps(hm_n, 0.0f);
    std::uniform_real_distribution<float> pk(60.0f, 180.0f);
    std::uniform_real_distribution<float> ctr(90.0f, 140.0f);
    for (int c = 0; c < NC; ++c)
        for (int j = 0; j < NJ; ++j) {
            float amp = pk(rng), mu_x = ctr(rng), mu_y = ctr(rng), sig = 18.0f;
            float *h = heatmaps.data() + ((size_t)c * NJ + j) * hs * hs;
            for (int y = 0; y < hs; ++y)
                for (int x = 0; x < hs; ++x) {
                    float dx = x - mu_x, dy = y - mu_y;
                    h[y * hs + x] = amp * std::exp(-(dx * dx + dy * dy) / (2 * sig * sig));
                }
        }

    // --- CPU reference ---
    std::vector<float> h3d_cpu(vf, 0.0f);
    hn_reproject(P, camMats.data(), intrMats.data(), distC.data(),
                 center3D, cHM.data(), heatmaps.data(), h3d_cpu.data());

    // --- GPU ---
    HNMetalReproject m;
    std::string err;
    if (!m.init(P, &err)) {
        std::fprintf(stderr, "SKIP: Metal init failed (%s)\n", err.c_str());
        return 0;   // no GPU -> nothing to compare; not a failure of the CPU path
    }
    std::memcpy(m.heatmaps_ptr(), heatmaps.data(), hm_n * sizeof(float));
    if (!m.reproject(camMats.data(), intrMats.data(), distC.data(),
                     center3D, cHM.data())) {
        std::fprintf(stderr, "FAIL: Metal reproject() returned false\n");
        return 1;
    }
    const float *h3d_gpu = m.h3d_ptr();

    // --- Compare ---
    double max_abs = 0.0, sum_abs = 0.0, max_val = 0.0;
    size_t n_big = 0;
    for (size_t i = 0; i < vf; ++i) {
        double a = h3d_cpu[i], b = h3d_gpu[i];
        double d = std::fabs(a - b);
        max_abs = std::max(max_abs, d);
        sum_abs += d;
        max_val = std::max(max_val, std::max(std::fabs(a), std::fabs(b)));
        if (d > 1.0) ++n_big;                       // > 1/255 of full scale
    }
    double mean_abs = sum_abs / (double)vf;
    double frac_big = (double)n_big / (double)vf;

    std::printf("CPU/GPU reproject parity over %zu voxels:\n", vf);
    std::printf("  max|Δ|      = %.6g  (peak value ~%.1f)\n", max_abs, max_val);
    std::printf("  mean|Δ|     = %.6g\n", mean_abs);
    std::printf("  frac(Δ>1)   = %.4g%% (%zu voxels)\n", frac_big * 100.0, n_big);

    // Thresholds: the projection/interpolation is fp32 on both sides; the only
    // source of divergence is a rare 1-px flip in the int-truncated gather at a
    // voxel that projects exactly on a heatmap-pixel boundary. With smooth heatmaps
    // that is a small local value difference. Require agreement to fp noise on the
    // mean, a bounded worst case, and essentially no gross disagreements.
    bool ok = (mean_abs < 1e-2) && (max_abs < 2.0) && (frac_big < 1e-3);
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
