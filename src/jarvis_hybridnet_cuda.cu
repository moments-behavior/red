// jarvis_hybridnet_cuda.cu — CUDA kernels for the HN TRT path.

#include "jarvis_hybridnet_cuda.h"

namespace {

// ───────────────────────────────────────────────────────────────────────────
// pad_heatmaps_kernel: zero-pad (N*J, H, W) → (N*J, H+2, W+2). Border = 0,
// interior copy from src[..., y-1, x-1]. One thread per output element.
// ───────────────────────────────────────────────────────────────────────────
__global__ void pad_heatmaps_kernel(
    const float * __restrict__ src,
    float * __restrict__ dst,
    int NJ, int H, int W, int Hp, int Wp)
{
    int x  = blockIdx.x * blockDim.x + threadIdx.x;
    int y  = blockIdx.y * blockDim.y + threadIdx.y;
    int nj = blockIdx.z;

    if (x >= Wp || y >= Hp || nj >= NJ) return;

    float v;
    if (y == 0 || y == Hp - 1 || x == 0 || x == Wp - 1) {
        v = 0.0f;
    } else {
        v = src[((size_t)nj * H + (y - 1)) * W + (x - 1)];
    }
    dst[((size_t)nj * Hp + y) * Wp + x] = v;
}

// ───────────────────────────────────────────────────────────────────────────
// resize_normalize_kernel: bilinear resize + ImageNet normalize, RGBA→NCHW.
// One thread per output pixel. Matches CPU `jarvis_resize_normalize`:
//   src_pix_x = (dst_x + 0.5) * (src_w / dst_size) - 0.5
//   x0 = floor(src_pix_x); wx = src_pix_x - x0
//   x0/x1 then clamped to [0, src_w-1]
// Alpha channel (offset +3 in RGBA) is read but ignored.
// ───────────────────────────────────────────────────────────────────────────
__global__ void resize_normalize_kernel(
    const uint8_t * const * __restrict__ d_rgba_ptrs,
    const int * __restrict__ d_widths,
    const int * __restrict__ d_heights,
    float * __restrict__ d_dst,
    int dst,
    float mean_r, float mean_g, float mean_b,
    float inv_std_r, float inv_std_g, float inv_std_b)
{
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    int cam   = blockIdx.z;
    if (dst_x >= dst || dst_y >= dst) return;

    const int src_w = d_widths[cam];
    const int src_h = d_heights[cam];
    const uint8_t * __restrict__ src = d_rgba_ptrs[cam];

    const float sx = (float)src_w / (float)dst;
    const float sy = (float)src_h / (float)dst;
    const float fx = ((float)dst_x + 0.5f) * sx - 0.5f;
    const float fy = ((float)dst_y + 0.5f) * sy - 0.5f;
    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    const float wx = fx - (float)x0;
    const float wy = fy - (float)y0;
    if (x0 < 0)         x0 = 0;
    if (x0 > src_w - 1) x0 = src_w - 1;
    if (x1 < 0)         x1 = 0;
    if (x1 > src_w - 1) x1 = src_w - 1;
    if (y0 < 0)         y0 = 0;
    if (y0 > src_h - 1) y0 = src_h - 1;
    if (y1 < 0)         y1 = 0;
    if (y1 > src_h - 1) y1 = src_h - 1;

    // RGBA32 row-major: each pixel is 4 bytes (R, G, B, A).
    const uint8_t *p00 = src + ((size_t)y0 * src_w + x0) * 4;
    const uint8_t *p01 = src + ((size_t)y0 * src_w + x1) * 4;
    const uint8_t *p10 = src + ((size_t)y1 * src_w + x0) * 4;
    const uint8_t *p11 = src + ((size_t)y1 * src_w + x1) * 4;

    const float w00 = (1.0f - wx) * (1.0f - wy);
    const float w01 = wx * (1.0f - wy);
    const float w10 = (1.0f - wx) * wy;
    const float w11 = wx * wy;

    float r = w00 * (float)p00[0] + w01 * (float)p01[0]
            + w10 * (float)p10[0] + w11 * (float)p11[0];
    float g = w00 * (float)p00[1] + w01 * (float)p01[1]
            + w10 * (float)p10[1] + w11 * (float)p11[1];
    float b = w00 * (float)p00[2] + w01 * (float)p01[2]
            + w10 * (float)p10[2] + w11 * (float)p11[2];

    r = (r * (1.0f / 255.0f) - mean_r) * inv_std_r;
    g = (g * (1.0f / 255.0f) - mean_g) * inv_std_g;
    b = (b * (1.0f / 255.0f) - mean_b) * inv_std_b;

    // NCHW layout: d_dst[cam][channel][dst_y][dst_x]
    const size_t plane = (size_t)dst * dst;
    const size_t base  = (size_t)cam * 3 * plane + (size_t)dst_y * dst + dst_x;
    d_dst[base + 0 * plane] = r;
    d_dst[base + 1 * plane] = g;
    d_dst[base + 2 * plane] = b;
}

// ───────────────────────────────────────────────────────────────────────────
// crop_normalize_kernel: BxB crop centered at (cx, cy), edge-clamped, then
// ImageNet-normalized. Matches CPU `jarvis_crop_normalize`:
//   src_x = clamp(cx - B/2 + x, 0, src_w-1)   (integer indexing, no interp)
// ───────────────────────────────────────────────────────────────────────────
__global__ void crop_normalize_kernel(
    const uint8_t * const * __restrict__ d_rgba_ptrs,
    const int * __restrict__ d_widths,
    const int * __restrict__ d_heights,
    const int * __restrict__ d_cx,
    const int * __restrict__ d_cy,
    float * __restrict__ d_dst,
    int B,
    float mean_r, float mean_g, float mean_b,
    float inv_std_r, float inv_std_g, float inv_std_b)
{
    int dx  = blockIdx.x * blockDim.x + threadIdx.x;
    int dy  = blockIdx.y * blockDim.y + threadIdx.y;
    int cam = blockIdx.z;
    if (dx >= B || dy >= B) return;

    const int src_w = d_widths[cam];
    const int src_h = d_heights[cam];
    const uint8_t * __restrict__ src = d_rgba_ptrs[cam];
    const int cx = d_cx[cam];
    const int cy = d_cy[cam];

    const int x0 = cx - B / 2;
    const int y0 = cy - B / 2;
    int sx = x0 + dx;
    int sy = y0 + dy;
    if (sx < 0)         sx = 0;
    if (sx > src_w - 1) sx = src_w - 1;
    if (sy < 0)         sy = 0;
    if (sy > src_h - 1) sy = src_h - 1;

    const uint8_t *p = src + ((size_t)sy * src_w + sx) * 4;
    const float r = ((float)p[0] * (1.0f / 255.0f) - mean_r) * inv_std_r;
    const float g = ((float)p[1] * (1.0f / 255.0f) - mean_g) * inv_std_g;
    const float b = ((float)p[2] * (1.0f / 255.0f) - mean_b) * inv_std_b;

    const size_t plane = (size_t)B * B;
    const size_t base  = (size_t)cam * 3 * plane + (size_t)dy * B + dx;
    d_dst[base + 0 * plane] = r;
    d_dst[base + 1 * plane] = g;
    d_dst[base + 2 * plane] = b;
}

} // namespace

// ── Host wrappers ──────────────────────────────────────────────────────────

cudaError_t jarvis_hn_pad_heatmaps_device(
    const float *d_src, float *d_dst,
    int N, int J, int H, int W,
    cudaStream_t stream)
{
    if (!d_src || !d_dst || N <= 0 || J <= 0 || H <= 0 || W <= 0)
        return cudaErrorInvalidValue;
    const int Hp = H + 2, Wp = W + 2, NJ = N * J;
    dim3 block(16, 16, 1);
    dim3 grid((Wp + block.x - 1) / block.x,
              (Hp + block.y - 1) / block.y,
              NJ);
    pad_heatmaps_kernel<<<grid, block, 0, stream>>>(d_src, d_dst, NJ, H, W, Hp, Wp);
    return cudaGetLastError();
}

cudaError_t jarvis_hn_resize_normalize_device(
    const uint8_t *const *d_rgba_ptrs,
    const int *d_widths,
    const int *d_heights,
    float *d_dst,
    int N, int dst,
    const float mean[3], const float inv_std[3],
    cudaStream_t stream)
{
    if (!d_rgba_ptrs || !d_widths || !d_heights || !d_dst ||
        N <= 0 || dst <= 0 || !mean || !inv_std)
        return cudaErrorInvalidValue;
    dim3 block(16, 16, 1);
    dim3 grid((dst + block.x - 1) / block.x,
              (dst + block.y - 1) / block.y,
              N);
    resize_normalize_kernel<<<grid, block, 0, stream>>>(
        d_rgba_ptrs, d_widths, d_heights, d_dst, dst,
        mean[0], mean[1], mean[2], inv_std[0], inv_std[1], inv_std[2]);
    return cudaGetLastError();
}

cudaError_t jarvis_hn_crop_normalize_device(
    const uint8_t *const *d_rgba_ptrs,
    const int *d_widths,
    const int *d_heights,
    const int *d_cx,
    const int *d_cy,
    float *d_dst,
    int N, int B,
    const float mean[3], const float inv_std[3],
    cudaStream_t stream)
{
    if (!d_rgba_ptrs || !d_widths || !d_heights || !d_cx || !d_cy || !d_dst ||
        N <= 0 || B <= 0 || !mean || !inv_std)
        return cudaErrorInvalidValue;
    dim3 block(16, 16, 1);
    dim3 grid((B + block.x - 1) / block.x,
              (B + block.y - 1) / block.y,
              N);
    crop_normalize_kernel<<<grid, block, 0, stream>>>(
        d_rgba_ptrs, d_widths, d_heights, d_cx, d_cy, d_dst, B,
        mean[0], mean[1], mean[2], inv_std[0], inv_std[1], inv_std[2]);
    return cudaGetLastError();
}
