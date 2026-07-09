// jarvis_hn_metal.mm — Metal compute implementation of HNMetalReproject.
//
// Two kernels mirror hn_reproject (jarvis_hn_reproject.h) exactly:
//   hn_project : one thread per (camera, 50³ grid point) → clamped/shifted pixel
//                coords val1/val2 (perspective divide + radial k1,k2 distortion).
//   hn_gather  : one thread per 100³ voxel → trilinear-upsample val1/val2 from
//                50³, int-truncate to a heatmap pixel, gather all joints, mean
//                over cameras → h3d[joint][voxel].
//
// fp32, fast-math DISABLED so the perspective divide and int-truncation
// boundaries match the CPU reference to fp noise. Geometry ints (NC/NJ/gf/gh/hs
// /half) are baked into the shader source as #defines so the joint/camera loops
// unroll and the per-voxel accumulator stays in registers.
#ifdef __APPLE__

#import <Metal/Metal.h>
#include "jarvis_hn_metal.h"
#include <cstdio>
#include <cstring>

// ── Shader source (geometry baked in via a generated #define preamble) ──
static NSString *hn_metal_source() {
    return @R"METAL(
#include <metal_stdlib>
using namespace metal;

struct HNConst { float step; float hsm1; float hsm2; float inv_nc; int halfg; };

// Project the 50³ sampling grid for one camera → clamped/shifted pixel coords.
kernel void hn_project(
    device const float *camMats  [[buffer(0)]],   // NC*12 (P^T)
    device const float *intrMats [[buffer(1)]],   // NC*9  (K^T)
    device const float *distC    [[buffer(2)]],   // NC*5
    device const float *center3D [[buffer(3)]],   // 3
    device const float *cHM      [[buffer(4)]],   // NC*2
    device float       *val1     [[buffer(5)]],   // NC*GH³
    device float       *val2     [[buffer(6)]],
    constant HNConst   &C        [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    const uint vh = (uint)HN_GH * HN_GH * HN_GH;
    if (gid >= (uint)HN_NC * vh) return;
    int c   = (int)(gid / vh);
    uint r  = gid - (uint)c * vh;
    int i   = (int)(r / (HN_GH * HN_GH));
    int j   = (int)((r / HN_GH) % HN_GH);
    int k   = (int)(r % HN_GH);

    device const float *M = camMats  + c * 12;
    device const float *K = intrMats + c * 9;
    float cx = K[6], cy = K[7], fx = K[0], fy = K[4];
    float k1 = distC[c * 5 + 0], k2 = distC[c * 5 + 1];
    float chm0 = cHM[c * 2 + 0], chm1 = cHM[c * 2 + 1];

    float wx = (i - C.halfg) * C.step + center3D[0];
    float wy = (j - C.halfg) * C.step + center3D[1];
    float wz = (k - C.halfg) * C.step + center3D[2];
    float pu = wx * M[0] + wy * M[3] + wz * M[6] + M[9];
    float pv = wx * M[1] + wy * M[4] + wz * M[7] + M[10];
    float pw = wx * M[2] + wy * M[5] + wz * M[8] + M[11];
    float a = pu / pw - cx;
    float b = pv / pw - cy;
    float r2 = (a / fx) * (a / fx) + (b / fy) * (b / fy);
    float distort = 1.0f + (k1 + k2 * r2) * r2;
    a = a * distort + cx;
    b = b * distort + cy;
    a = clamp(a, chm0 - C.hsm1, chm0 + C.hsm2) - chm0 + C.hsm1;
    b = clamp(b, chm1 - C.hsm1, chm1 + C.hsm2) - chm1 + C.hsm1;
    val1[gid] = a;
    val2[gid] = b;
}

// Trilinear sample of a (GH,GH,GH) volume at fractional (fi,fj,fk), clamped.
static inline float hn_trilin(device const float *vol, float fi, float fj, float fk) {
    fi = clamp(fi, 0.0f, (float)(HN_GH - 1));
    fj = clamp(fj, 0.0f, (float)(HN_GH - 1));
    fk = clamp(fk, 0.0f, (float)(HN_GH - 1));
    int i0 = (int)fi, j0 = (int)fj, k0 = (int)fk;
    int i1 = min(i0 + 1, HN_GH - 1), j1 = min(j0 + 1, HN_GH - 1), k1 = min(k0 + 1, HN_GH - 1);
    float di = fi - i0, dj = fj - j0, dk = fk - k0;
#define HN_V(i,j,k) vol[((i) * HN_GH + (j)) * HN_GH + (k)]
    float c00 = HN_V(i0,j0,k0) * (1 - dk) + HN_V(i0,j0,k1) * dk;
    float c01 = HN_V(i0,j1,k0) * (1 - dk) + HN_V(i0,j1,k1) * dk;
    float c10 = HN_V(i1,j0,k0) * (1 - dk) + HN_V(i1,j0,k1) * dk;
    float c11 = HN_V(i1,j1,k0) * (1 - dk) + HN_V(i1,j1,k1) * dk;
#undef HN_V
    float c0 = c00 * (1 - dj) + c01 * dj;
    float c1 = c10 * (1 - dj) + c11 * dj;
    return c0 * (1 - di) + c1 * di;
}

// One thread per (joint, voxel) over NJ*GF³ (gid = jt*vf + voxel). A warp shares
// one joint over consecutive voxels → consecutive projected pixels → coalesced
// heatmap reads. No per-thread accumulator array, so occupancy stays high and
// memory latency is hidden. The projected pixel is recomputed per joint via
// trilinear (cheap ALU, fully hidden under the latency-bound gather). fp32
// heatmaps — bit-identical to the validated CPU path regardless of the CoreML
// output dtype (fp16 storage measured no faster: the gather is latency-bound).
kernel void hn_gather(
    device const float *val1     [[buffer(0)]],   // NC*GH³
    device const float *val2     [[buffer(1)]],
    device const float *heatmaps [[buffer(2)]],   // NC*NJ*HS*HS
    device float       *h3d      [[buffer(3)]],   // NJ*GF³
    constant HNConst   &C        [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const uint vf = (uint)HN_GF * HN_GF * HN_GF;
    if (gid >= (uint)HN_NJ * vf) return;
    uint jt    = gid / vf;
    uint voxel = gid - jt * vf;
    int i = (int)(voxel / (HN_GF * HN_GF));
    int j = (int)((voxel / HN_GF) % HN_GF);
    int k = (int)(voxel % HN_GF);
    float fi = i * 0.5f - 0.25f;
    float fj = j * 0.5f - 0.25f;
    float fk = k * 0.5f - 0.25f;

    const uint vh = (uint)HN_GH * HN_GH * HN_GH;
    const uint hh = (uint)HN_HS * HN_HS;
    float acc = 0.0f;
    for (int c = 0; c < HN_NC; ++c) {
        device const float *v1 = val1 + (uint)c * vh;
        device const float *v2 = val2 + (uint)c * vh;
        float a = hn_trilin(v1, fi, fj, fk);
        float b = hn_trilin(v2, fi, fj, fk);
        int col = (int)(a * 0.5f);
        int row = (int)(b * 0.5f);
        uint hidx = (uint)row * HN_HS + (uint)col;
        acc += heatmaps[((uint)c * HN_NJ + jt) * hh + hidx] * C.inv_nc;
    }
    h3d[gid] = acc;
}
)METAL";
}

struct HNConstGPU { float step; float hsm1; float hsm2; float inv_nc; int halfg; };

@interface HNMetalImpl : NSObject
@property (nonatomic, strong) id<MTLDevice>              device;
@property (nonatomic, strong) id<MTLCommandQueue>        queue;
@property (nonatomic, strong) id<MTLComputePipelineState> projectPSO;
@property (nonatomic, strong) id<MTLComputePipelineState> gatherPSO;
@property (nonatomic, strong) id<MTLBuffer> camMats;    // NC*12
@property (nonatomic, strong) id<MTLBuffer> intrMats;   // NC*9
@property (nonatomic, strong) id<MTLBuffer> distC;      // NC*5
@property (nonatomic, strong) id<MTLBuffer> center3D;   // 3
@property (nonatomic, strong) id<MTLBuffer> cHM;        // NC*2
@property (nonatomic, strong) id<MTLBuffer> val1;       // NC*GH³
@property (nonatomic, strong) id<MTLBuffer> val2;       // NC*GH³
@property (nonatomic, strong) id<MTLBuffer> heatmaps;   // NC*NJ*HS*HS (fp32)
@property (nonatomic, strong) id<MTLBuffer> h3d;        // NJ*GF³
@property (nonatomic, assign) HNConstGPU consts;
@property (nonatomic, assign) HNReproParams params;
@end

@implementation HNMetalImpl
@end

// ── C++ wrapper ──

static id<MTLBuffer> mk_shared(id<MTLDevice> dev, size_t n_floats) {
    return [dev newBufferWithLength:n_floats * sizeof(float)
                            options:MTLResourceStorageModeShared];
}

bool HNMetalReproject::init(const HNReproParams &P, std::string *err) {
    @autoreleasepool {
        auto fail = [&](const std::string &m) { if (err) *err = m; return false; };

        // MTLCreateSystemDefaultDevice() can return nil in headless/daemon
        // contexts (no window-server session); MTLCopyAllDevices() still
        // enumerates the GPU. Prefer a low-power/unified device if present.
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            NSArray<id<MTLDevice>> *all = MTLCopyAllDevices();
            if (all.count) dev = all[0];
        }
        if (!dev) return fail("no Metal device");

        const int NC = P.num_cameras, NJ = P.num_joints;
        const int gf = P.grid_full, gh = gf / 2, hs = P.heatmap_size;
        const int half = gh / 2;

        // Bake geometry into the shader as #defines so loops unroll and the
        // per-voxel accumulator (acc[NJ]) is a compile-time-sized register array.
        char preamble[256];
        std::snprintf(preamble, sizeof(preamble),
            "#define HN_NC %d\n#define HN_NJ %d\n#define HN_GF %d\n"
            "#define HN_GH %d\n#define HN_HS %d\n#define HN_HALF %d\n",
            NC, NJ, gf, gh, hs, half);
        NSString *src = [NSString stringWithUTF8String:preamble];
        src = [src stringByAppendingString:hn_metal_source()];

        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        opts.fastMathEnabled = NO;   // correctly-rounded divide + no fma reorder
        NSError *e = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&e];
        if (!lib) return fail(std::string("kernel compile: ") +
                              (e ? e.localizedDescription.UTF8String : "?"));

        id<MTLFunction> fp = [lib newFunctionWithName:@"hn_project"];
        id<MTLFunction> fg = [lib newFunctionWithName:@"hn_gather"];
        if (!fp || !fg) return fail("kernel functions missing");
        id<MTLComputePipelineState> pProject = [dev newComputePipelineStateWithFunction:fp error:&e];
        if (!pProject) return fail(std::string("project PSO: ") +
                                   (e ? e.localizedDescription.UTF8String : "?"));
        id<MTLComputePipelineState> pGather = [dev newComputePipelineStateWithFunction:fg error:&e];
        if (!pGather) return fail(std::string("gather PSO: ") +
                                  (e ? e.localizedDescription.UTF8String : "?"));

        HNMetalImpl *o = [[HNMetalImpl alloc] init];
        o.device = dev;
        o.queue = [dev newCommandQueue];
        o.projectPSO = pProject;
        o.gatherPSO = pGather;
        o.params = P;

        const size_t vh = (size_t)gh * gh * gh, vf = (size_t)gf * gf * gf;
        o.camMats  = mk_shared(dev, (size_t)NC * 12);
        o.intrMats = mk_shared(dev, (size_t)NC * 9);
        o.distC    = mk_shared(dev, (size_t)NC * 5);
        o.center3D = mk_shared(dev, 3);
        o.cHM      = mk_shared(dev, (size_t)NC * 2);
        o.val1     = mk_shared(dev, (size_t)NC * vh);
        o.val2     = mk_shared(dev, (size_t)NC * vh);
        o.heatmaps = mk_shared(dev, (size_t)NC * NJ * hs * hs);
        o.h3d      = mk_shared(dev, (size_t)NJ * vf);
        if (!o.queue || !o.camMats || !o.intrMats || !o.distC || !o.center3D ||
            !o.cHM || !o.val1 || !o.val2 || !o.heatmaps || !o.h3d)
            return fail("buffer/queue alloc failed");

        HNConstGPU C;
        C.step   = P.grid_spacing * 2.0f;
        C.hsm1   = (float)(hs - 1);
        C.hsm2   = (float)(hs - 2);
        C.inv_nc = 1.0f / (float)NC;
        C.halfg  = half;
        o.consts = C;

        impl_ = (__bridge_retained void *)o;
        return true;
    }
}

HNMetalReproject::~HNMetalReproject() {
    if (impl_) {
        HNMetalImpl *o = (__bridge_transfer HNMetalImpl *)impl_;
        (void)o;   // ARC releases
        impl_ = nullptr;
    }
}

float *HNMetalReproject::heatmaps_ptr() {
    if (!impl_) return nullptr;
    HNMetalImpl *o = (__bridge HNMetalImpl *)impl_;
    return (float *)o.heatmaps.contents;
}

const float *HNMetalReproject::h3d_ptr() {
    if (!impl_) return nullptr;
    HNMetalImpl *o = (__bridge HNMetalImpl *)impl_;
    return (const float *)o.h3d.contents;
}

bool HNMetalReproject::reproject(const float *camMats, const float *intrMats,
                                 const float *distC, const float *center3D,
                                 const float *cHM) {
    if (!impl_) return false;
    @autoreleasepool {
        HNMetalImpl *o = (__bridge HNMetalImpl *)impl_;
        const HNReproParams &P = o.params;
        const int NC = P.num_cameras, NJ = P.num_joints;
        const int gf = P.grid_full, gh = gf / 2;
        const size_t vh = (size_t)gh * gh * gh, vf = (size_t)gf * gf * gf;

        // Upload per-frame geometry into the shared buffers.
        memcpy(o.camMats.contents,  camMats,  (size_t)NC * 12 * sizeof(float));
        memcpy(o.intrMats.contents, intrMats, (size_t)NC * 9  * sizeof(float));
        memcpy(o.distC.contents,    distC,    (size_t)NC * 5  * sizeof(float));
        memcpy(o.center3D.contents, center3D, 3 * sizeof(float));
        memcpy(o.cHM.contents,      cHM,      (size_t)NC * 2  * sizeof(float));
        HNConstGPU C = o.consts;

        static const bool split = getenv("HN_DIAG") != nullptr;
        id<MTLCommandBuffer> cmd = [o.queue commandBuffer];

        // Kernel 1: project 50³ grid per camera.
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:o.projectPSO];
            [enc setBuffer:o.camMats  offset:0 atIndex:0];
            [enc setBuffer:o.intrMats offset:0 atIndex:1];
            [enc setBuffer:o.distC    offset:0 atIndex:2];
            [enc setBuffer:o.center3D offset:0 atIndex:3];
            [enc setBuffer:o.cHM      offset:0 atIndex:4];
            [enc setBuffer:o.val1     offset:0 atIndex:5];
            [enc setBuffer:o.val2     offset:0 atIndex:6];
            [enc setBytes:&C length:sizeof(C) atIndex:7];
            NSUInteger total = (NSUInteger)NC * vh;
            NSUInteger tg = o.projectPSO.maxTotalThreadsPerThreadgroup;
            if (tg > 256) tg = 256;
            [enc dispatchThreads:MTLSizeMake(total, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
        }
        if (split) {
            [cmd commit]; [cmd waitUntilCompleted];
            last_project_ms = (float)((cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0);
            cmd = [o.queue commandBuffer];
        }
        // Kernel 2: gather into h3d (mean over cams).
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:o.gatherPSO];
            [enc setBuffer:o.val1     offset:0 atIndex:0];
            [enc setBuffer:o.val2     offset:0 atIndex:1];
            [enc setBuffer:o.heatmaps offset:0 atIndex:2];
            [enc setBuffer:o.h3d      offset:0 atIndex:3];
            [enc setBytes:&C length:sizeof(C) atIndex:4];
            NSUInteger tg = o.gatherPSO.maxTotalThreadsPerThreadgroup;
            if (tg > 256) tg = 256;
            [enc dispatchThreads:MTLSizeMake((NSUInteger)NJ * vf, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
        }

        [cmd commit];
        [cmd waitUntilCompleted];
        (void)NJ;
        // Check status before reading GPU timestamps: on error they are undefined.
        if (cmd.status == MTLCommandBufferStatusError) {
            NSLog(@"[HNMetal] GPU command buffer error: %@",
                  cmd.error ? cmd.error.localizedDescription : @"unknown");
            return false;
        }
        float t = (float)((cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0);
        if (split) { last_gather_ms = t; last_gpu_ms = last_project_ms + t; }
        else last_gpu_ms = t;
        return true;
    }
}

#endif // __APPLE__
