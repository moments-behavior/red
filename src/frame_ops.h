#pragma once
// frame_ops.h -- CPU equivalents of the display-side CUDA kernels, used by the
// software backend where no CUDA context exists to run the real ones.
//
// Kept platform-neutral and header-only: macOS has its own Metal versions, but
// nothing here depends on a backend, and a plain function is far easier to
// check against the kernel it mirrors.
#include <algorithm>
#include <cstddef>
#include <cstdint>

// CPU twin of apply_contrast_brightness_rgba (kernel.cu). Same formula,
// same clamp, same rounding, and likewise touches only the first three
// channels so the alpha byte survives.
//
// Built as a 256-entry table because the transform depends on nothing but the
// input byte. Per-pixel it would be a float multiply, an add and two clamps
// across every camera every frame -- on a 16-camera 2 MP rig that is ~100 M
// pixels a second on a machine that already has no GPU to spare.
//
// Channel-order agnostic: RGBA and BGRA both carry alpha last, and the first
// three channels get the same curve either way.
inline void build_contrast_lut(uint8_t lut[256], float alpha, float beta,
                               bool pivot_midgray) {
    const float pivot = pivot_midgray ? 128.0f : 0.0f;
    for (int i = 0; i < 256; ++i) {
        float v = alpha * ((float)i - pivot) + pivot + beta;
        v = std::min(std::max(v, 0.0f), 255.0f);
        lut[i] = (uint8_t)(v + 0.5f);
    }
}

// Apply a table built by build_contrast_lut. dst and src may be the same
// buffer; the ring slots are never written in place, so callers pass a scratch
// buffer and leave the decoded frame intact for the next redraw.
inline void apply_contrast_lut(uint8_t *dst, const uint8_t *src, int width,
                               int height, const uint8_t lut[256]) {
    const size_t n = (size_t)width * (size_t)height;
    for (size_t p = 0; p < n; ++p) {
        const size_t i = p * 4;
        dst[i + 0] = lut[src[i + 0]];
        dst[i + 1] = lut[src[i + 1]];
        dst[i + 2] = lut[src[i + 2]];
        dst[i + 3] = src[i + 3];
    }
}
