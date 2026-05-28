// jarvis_hybridnet_cuda.cu — CUDA kernels for the HN TRT path.

#include "jarvis_hybridnet_cuda.h"

namespace {

// One thread per output element. Threads in border rows/cols write 0;
// interior threads copy from d_src at the (y-1, x-1) offset.
//
// Layout (row-major, contiguous):
//   d_src: [N*J, H, W]
//   d_dst: [N*J, H+2, W+2]   (the leading "1" axis is collapsed — Hybrid3D
//                              accepts (1, N, J, Hp, Wp), which is identical
//                              in memory to (N*J, Hp, Wp))
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

} // namespace

cudaError_t jarvis_hn_pad_heatmaps_device(
    const float *d_src, float *d_dst,
    int N, int J, int H, int W,
    cudaStream_t stream)
{
    if (!d_src || !d_dst || N <= 0 || J <= 0 || H <= 0 || W <= 0)
        return cudaErrorInvalidValue;

    const int Hp = H + 2;
    const int Wp = W + 2;
    const int NJ = N * J;

    // 16x16 = 256 threads per block. With H=W=352, Hp=Wp=354 → 23x23x(N*J)
    // blocks. Plenty of occupancy on Ada; each thread does a single load+store.
    dim3 block(16, 16, 1);
    dim3 grid((Wp + block.x - 1) / block.x,
              (Hp + block.y - 1) / block.y,
              NJ);

    pad_heatmaps_kernel<<<grid, block, 0, stream>>>(
        d_src, d_dst, NJ, H, W, Hp, Wp);

    return cudaGetLastError();
}
