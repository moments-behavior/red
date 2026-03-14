#pragma once
// aruco_detect.h — Self-contained ArUco and ChArUco detection using only
// the standard library and Eigen. Replaces OpenCV's ArUco module entirely.
//
// Supports 9 dictionaries: DICT_4X4_50/100/250, DICT_5X5_50/100/250,
// DICT_6X6_50/250, DICT_ARUCO_ORIGINAL.
// Implements: adaptive thresholding, contour finding, quad detection,
// perspective sampling, dictionary matching, ChArUco corner interpolation,
// and gradient-based subpixel corner refinement.
//
// All functions are inline (header-only). Namespace: aruco_detect.

#include "red_math.h"

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <future>
#include <map>
#include <numeric>
#include <utility>
#include <vector>

namespace aruco_detect {

// ─────────────────────────────────────────────────────────────────────────────
// ArUco dictionary bit patterns (auto-generated from OpenCV)
// ─────────────────────────────────────────────────────────────────────────────
// All patterns stored as uint64_t, row-major MSB-first.
// For NxN grid: bit (N*N-1) = top-left cell, bit 0 = bottom-right cell.
#include "aruco_dict_data.h"

// ─────────────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────────────

struct ArUcoDictionary {
    const uint64_t *patterns;  // bit patterns array (uint64_t holds 4x4..7x7)
    int num_markers;
    int marker_bits;           // side length of data grid (4, 5, or 6)
    int max_correction_bits;   // max Hamming distance for match

    bool valid() const { return patterns != nullptr && num_markers > 0; }
};

struct DetectedMarker {
    int id;
    std::array<Eigen::Vector2f, 4> corners; // TL, TR, BR, BL order
    int hamming_distance;                    // match quality (0 = perfect)
};

struct CharucoBoard {
    int squares_x, squares_y;  // e.g. 5, 5
    float square_length;       // in world units (mm)
    float marker_length;       // in world units (mm)
    int dictionary_id;         // 0 = DICT_4X4_50
};

struct CharucoResult {
    std::vector<Eigen::Vector2f> corners;
    std::vector<int> ids;  // corner IDs (0 to (squares_x-1)*(squares_y-1)-1)
};

// Optional GPU-accelerated batch threshold function pointer.
// When non-null, replaces CPU adaptive threshold in detectMarkers.
// The GPU computes box sums internally (separable filter) — no integral
// image needed, only the 7MB grayscale image is transferred.
// Parameters:
//   ctx:             opaque GPU context (e.g. ArucoMetalHandle cast to void*)
//   gray:            grayscale image (w*h bytes)
//   w, h:            image dimensions
//   window_sizes:    array of window sizes (num_passes elements)
//   C:               threshold constant
//   num_passes:      number of threshold passes
//   binary_outputs:  array of num_passes pointers, each pre-allocated to w*h bytes
using GpuThresholdFunc = void (*)(
    void *ctx,
    const uint8_t *gray, int w, int h,
    const int *window_sizes, int C, int num_passes,
    uint8_t **binary_outputs);

// ─────────────────────────────────────────────────────────────────────────────
// getDictionary
// ─────────────────────────────────────────────────────────────────────────────

inline ArUcoDictionary getDictionary(int id) {
    switch (id) {
    case  0: return {DICT_4X4_50,   50, 4, DICT_4X4_50_MAX_CORRECTION};
    case  1: return {DICT_4X4_100, 100, 4, DICT_4X4_100_MAX_CORRECTION};
    case  2: return {DICT_4X4_250, 250, 4, DICT_4X4_250_MAX_CORRECTION};
    case  4: return {DICT_5X5_50,   50, 5, DICT_5X5_50_MAX_CORRECTION};
    case  5: return {DICT_5X5_100, 100, 5, DICT_5X5_100_MAX_CORRECTION};
    case  6: return {DICT_5X5_250, 250, 5, DICT_5X5_250_MAX_CORRECTION};
    case  8: return {DICT_6X6_50,   50, 6, DICT_6X6_50_MAX_CORRECTION};
    case 10: return {DICT_6X6_250, 250, 6, DICT_6X6_250_MAX_CORRECTION};
    case 16: return {DICT_ARUCO_ORIGINAL, 1024, 5, DICT_ARUCO_ORIGINAL_MAX_CORRECTION};
    default: return {nullptr, 0, 0, 0};
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────
namespace detail {

// --- Integral image (for fast adaptive threshold) ---
// Uses 64-bit sums to avoid overflow on large images (3208*2200*255 < 2^32
// but row sums of the integral image exceed 2^32 easily).
struct IntegralImage {
    std::vector<int64_t> data;
    int w, h;

    void compute(const uint8_t *gray, int width, int height) {
        w = width;
        h = height;
        data.resize((size_t)(w + 1) * (h + 1), 0);

        // Row 0 and col 0 of the (w+1)*(h+1) table are zero.
        for (int y = 0; y < h; y++) {
            int64_t row_sum = 0;
            for (int x = 0; x < w; x++) {
                row_sum += gray[y * w + x];
                data[(y + 1) * (w + 1) + (x + 1)] =
                    row_sum + data[y * (w + 1) + (x + 1)];
            }
        }
    }

    // Sum of pixels in rectangle [x0, x1) x [y0, y1) (exclusive end).
    inline int64_t rectSum(int x0, int y0, int x1, int y1) const {
        // Clamp to valid range
        if (x0 < 0) x0 = 0;
        if (y0 < 0) y0 = 0;
        if (x1 > w) x1 = w;
        if (y1 > h) y1 = h;
        return data[y1 * (w + 1) + x1] - data[y0 * (w + 1) + x1] -
               data[y1 * (w + 1) + x0] + data[y0 * (w + 1) + x0];
    }
};

// --- Adaptive threshold using integral image ---
// Output: binary image where 255 = foreground (white), 0 = background.
// Uses block-mean: pixel is foreground if value > local_mean - C.
inline void adaptiveThreshold(const uint8_t *gray, int w, int h,
                              std::vector<uint8_t> &binary,
                              const IntegralImage &integral,
                              int window_size, int C) {
    binary.resize((size_t)w * h);
    int half = window_size / 2;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int x0 = x - half;
            int y0 = y - half;
            int x1 = x + half + 1;
            int y1 = y + half + 1;

            // Clamp window to image bounds for pixel count
            int cx0 = (x0 < 0) ? 0 : x0;
            int cy0 = (y0 < 0) ? 0 : y0;
            int cx1 = (x1 > w) ? w : x1;
            int cy1 = (y1 > h) ? h : y1;
            int count = (cx1 - cx0) * (cy1 - cy0);

            int64_t sum = integral.rectSum(x0, y0, x1, y1);
            // Mean threshold: pixel is white if val * count > sum - C * count
            // This avoids a division.
            int val = gray[y * w + x];
            binary[y * w + x] = (val * count > (int)(sum - (int64_t)C * count))
                                    ? 255
                                    : 0;
        }
    }
}

// --- Contour tracing (8-connectivity border following) ---
// Simple Moore boundary tracing. Returns outer contours of white regions.

// 8-connected neighbor offsets: dx, dy for directions 0..7.
// Direction 0 = right, 1 = down-right, 2 = down, etc. (clockwise).
static constexpr int dx8[8] = {1, 1, 0, -1, -1, -1, 0, 1};
static constexpr int dy8[8] = {0, 1, 1, 1, 0, -1, -1, -1};

inline std::vector<std::vector<Eigen::Vector2i>>
findContours(const std::vector<uint8_t> &binary, int w, int h,
             std::vector<uint8_t> *reuse_visited = nullptr,
             int max_contour_length = 0) {
    std::vector<std::vector<Eigen::Vector2i>> contours;

    // Visited mask to avoid re-tracing the same contour.
    // Reuse caller-provided buffer to avoid 7MB allocation per call.
    std::vector<uint8_t> local_visited;
    std::vector<uint8_t> &visited = reuse_visited ? *reuse_visited : local_visited;
    visited.assign(binary.size(), 0);

    // Default max contour length: no ArUco marker should have a perimeter
    // longer than ~4000 pixels on a 3208x2200 image. Skip huge foreground
    // regions (e.g. white paper) that waste tracing time.
    int max_steps = (max_contour_length > 0) ? max_contour_length : w * h;

    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int idx = y * w + x;
            if (binary[idx] == 0 || visited[idx])
                continue;

            // Check if this is a border pixel (has at least one background
            // neighbor in 4-connectivity, which means left pixel is background
            // for a left-to-right scan).
            if (binary[idx - 1] != 0)
                continue; // Not a left border pixel

            // Trace the contour using Moore boundary tracing.
            std::vector<Eigen::Vector2i> contour;
            int sx = x, sy = y;
            int cx = x, cy = y;
            int dir = 0; // Start looking right

            // The starting direction should be such that the background is to
            // the right of our forward direction. Since we entered from the
            // left (pixel to the left is background), start looking right
            // (direction 0), which means we came from direction 4 (left).
            dir = 7; // Start by checking up-right from the entry direction

            int steps = 0;
            bool exceeded_max = false;

            do {
                contour.push_back(Eigen::Vector2i(cx, cy));
                visited[cy * w + cx] = 1;

                // Look for next boundary pixel: rotate clockwise from
                // (dir + 5) % 8, which is the direction we came from + 1.
                int start_dir = (dir + 5) % 8; // backtrack direction + 1
                bool found = false;
                for (int i = 0; i < 8; i++) {
                    int d = (start_dir + i) % 8;
                    int nx = cx + dx8[d];
                    int ny = cy + dy8[d];
                    if (nx >= 0 && nx < w && ny >= 0 && ny < h &&
                        binary[ny * w + nx] != 0) {
                        cx = nx;
                        cy = ny;
                        dir = d;
                        found = true;
                        break;
                    }
                }
                if (!found)
                    break;

                steps++;
                if (steps > max_steps) {
                    exceeded_max = true;
                    // Continue tracing to mark all boundary pixels as visited
                    // (prevents re-entering the same contour from unvisited
                    // boundary pixels), but stop recording points.
                    do {
                        visited[cy * w + cx] = 1;
                        int sd = (dir + 5) % 8;
                        bool f2 = false;
                        for (int i = 0; i < 8; i++) {
                            int d = (sd + i) % 8;
                            int nx = cx + dx8[d];
                            int ny = cy + dy8[d];
                            if (nx >= 0 && nx < w && ny >= 0 && ny < h &&
                                binary[ny * w + nx] != 0) {
                                cx = nx;
                                cy = ny;
                                dir = d;
                                f2 = true;
                                break;
                            }
                        }
                        if (!f2) break;
                    } while (cx != sx || cy != sy);
                    visited[cy * w + cx] = 1; // mark start pixel too
                    break;
                }

                // Check termination: back at start with same direction
                if (cx == sx && cy == sy) {
                    // Standard termination: we've completed the loop
                    break;
                }
            } while (true);

            if (contour.size() >= 20 && !exceeded_max) {
                contours.push_back(std::move(contour));
            }
        }
    }
    return contours;
}

// --- Douglas-Peucker polygon approximation ---

inline float pointToSegmentDist(const Eigen::Vector2f &p,
                                const Eigen::Vector2f &a,
                                const Eigen::Vector2f &b) {
    Eigen::Vector2f ab = b - a;
    float len2 = ab.squaredNorm();
    if (len2 < 1e-12f) return (p - a).norm();
    float t = (p - a).dot(ab) / len2;
    t = std::max(0.0f, std::min(1.0f, t));
    Eigen::Vector2f proj = a + t * ab;
    return (p - proj).norm();
}

inline void douglasPuckerRecurse(const std::vector<Eigen::Vector2f> &pts,
                                  int start, int end, float epsilon,
                                  std::vector<bool> &keep) {
    float max_dist = 0.0f;
    int max_idx = start;
    const auto &a = pts[start];
    const auto &b = pts[end];

    for (int i = start + 1; i < end; i++) {
        float d = pointToSegmentDist(pts[i], a, b);
        if (d > max_dist) {
            max_dist = d;
            max_idx = i;
        }
    }

    if (max_dist > epsilon) {
        keep[max_idx] = true;
        if (max_idx - start > 1)
            douglasPuckerRecurse(pts, start, max_idx, epsilon, keep);
        if (end - max_idx > 1)
            douglasPuckerRecurse(pts, max_idx, end, epsilon, keep);
    }
}

inline std::vector<Eigen::Vector2f>
approxPolyDP(const std::vector<Eigen::Vector2f> &curve, float epsilon) {
    if (curve.size() < 3)
        return curve;

    int n = (int)curve.size();

    // For closed contours, find the two farthest points to use as split.
    int idx_a = 0, idx_b = 0;
    float max_dist = 0.0f;
    for (int i = 1; i < n; i++) {
        float d = (curve[i] - curve[0]).squaredNorm();
        if (d > max_dist) {
            max_dist = d;
            idx_b = i;
        }
    }

    std::vector<bool> keep(n, false);
    keep[idx_a] = true;
    keep[idx_b] = true;

    if (idx_b - idx_a > 1)
        douglasPuckerRecurse(curve, idx_a, idx_b, epsilon, keep);
    if (idx_a + n - idx_b > 1) {
        // Wrap-around segment: idx_b → end → start → idx_a
        // Linearize by appending curve[0..idx_a] after curve[idx_b..n-1]
        std::vector<Eigen::Vector2f> seg;
        for (int i = idx_b; i < n; i++) seg.push_back(curve[i]);
        for (int i = 0; i <= idx_a; i++) seg.push_back(curve[i]);

        std::vector<bool> seg_keep(seg.size(), false);
        seg_keep[0] = true;
        seg_keep[seg.size() - 1] = true;
        if ((int)seg.size() > 2)
            douglasPuckerRecurse(seg, 0, (int)seg.size() - 1, epsilon,
                                  seg_keep);

        // Map back to original indices
        for (int i = 0; i < (int)seg.size(); i++) {
            if (seg_keep[i]) {
                int orig = (idx_b + i) % n;
                keep[orig] = true;
            }
        }
    }

    std::vector<Eigen::Vector2f> result;
    for (int i = 0; i < n; i++) {
        if (keep[i])
            result.push_back(curve[i]);
    }
    return result;
}

// --- Contour perimeter ---
inline float contourPerimeter(const std::vector<Eigen::Vector2f> &pts) {
    float peri = 0.0f;
    int n = (int)pts.size();
    for (int i = 0; i < n; i++) {
        peri += (pts[(i + 1) % n] - pts[i]).norm();
    }
    return peri;
}

// --- Polygon area (shoelace formula, signed) ---
inline float polygonArea(const std::vector<Eigen::Vector2f> &pts) {
    float area = 0.0f;
    int n = (int)pts.size();
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        area += pts[i].x() * pts[j].y();
        area -= pts[j].x() * pts[i].y();
    }
    return area * 0.5f;
}

// --- Convexity check for a polygon ---
inline bool isConvex(const std::vector<Eigen::Vector2f> &pts) {
    int n = (int)pts.size();
    if (n < 3) return false;

    bool sign_set = false;
    bool positive = false;

    for (int i = 0; i < n; i++) {
        const auto &a = pts[i];
        const auto &b = pts[(i + 1) % n];
        const auto &c = pts[(i + 2) % n];
        float cross = (b.x() - a.x()) * (c.y() - b.y()) -
                      (b.y() - a.y()) * (c.x() - b.x());
        if (std::abs(cross) < 1e-6f) continue;
        if (!sign_set) {
            positive = (cross > 0);
            sign_set = true;
        } else if ((cross > 0) != positive) {
            return false;
        }
    }
    return sign_set;
}

// --- Order quad corners: ensure consistent winding ---
// Orders points in counter-clockwise order starting from top-left.
inline void orderQuadCorners(std::array<Eigen::Vector2f, 4> &corners) {
    // Sort by y first, then x to find TL
    std::sort(corners.begin(), corners.end(),
              [](const Eigen::Vector2f &a, const Eigen::Vector2f &b) {
                  return (a.y() < b.y()) ||
                         (a.y() == b.y() && a.x() < b.x());
              });

    // Top two points (smallest y)
    Eigen::Vector2f top1 = corners[0], top2 = corners[1];
    // Bottom two points (largest y)
    Eigen::Vector2f bot1 = corners[2], bot2 = corners[3];

    // TL has smaller x among top pair
    Eigen::Vector2f tl = (top1.x() <= top2.x()) ? top1 : top2;
    Eigen::Vector2f tr = (top1.x() <= top2.x()) ? top2 : top1;
    // BL has smaller x among bottom pair
    Eigen::Vector2f bl = (bot1.x() <= bot2.x()) ? bot1 : bot2;
    Eigen::Vector2f br = (bot1.x() <= bot2.x()) ? bot2 : bot1;

    corners = {tl, tr, br, bl};
}

// --- 4-point homography (DLT) ---
// Maps src[i] → dst[i] for exactly 4 point correspondences.
// Returns the 3x3 homography H such that dst ~ H * src (homogeneous).
inline Eigen::Matrix3d
computeHomography4pt(const std::array<Eigen::Vector2f, 4> &src,
                     const std::array<Eigen::Vector2f, 4> &dst) {
    // Build 8x9 DLT system.
    Eigen::Matrix<double, 8, 9> A;
    A.setZero();
    for (int i = 0; i < 4; i++) {
        double X = src[i].x(), Y = src[i].y();
        double u = dst[i].x(), v = dst[i].y();
        A(2 * i, 0) = X;
        A(2 * i, 1) = Y;
        A(2 * i, 2) = 1.0;
        A(2 * i, 6) = -u * X;
        A(2 * i, 7) = -u * Y;
        A(2 * i, 8) = -u;
        A(2 * i + 1, 3) = X;
        A(2 * i + 1, 4) = Y;
        A(2 * i + 1, 5) = 1.0;
        A(2 * i + 1, 6) = -v * X;
        A(2 * i + 1, 7) = -v * Y;
        A(2 * i + 1, 8) = -v;
    }

    Eigen::JacobiSVD<Eigen::Matrix<double, 8, 9>> svd(A, Eigen::ComputeFullV);
    Eigen::Matrix<double, 9, 1> h = svd.matrixV().col(8);
    Eigen::Matrix3d H;
    H << h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7), h(8);
    if (std::abs(H(2, 2)) > 1e-12)
        H /= H(2, 2);
    return H;
}

// Map a 2D point through a 3x3 homography (homogeneous division).
inline Eigen::Vector2f applyHomography(const Eigen::Matrix3d &H,
                                       const Eigen::Vector2f &pt) {
    Eigen::Vector3d p(pt.x(), pt.y(), 1.0);
    Eigen::Vector3d q = H * p;
    if (std::abs(q(2)) < 1e-12)
        return Eigen::Vector2f(0, 0);
    return Eigen::Vector2f((float)(q(0) / q(2)), (float)(q(1) / q(2)));
}

// --- Bilinear sampling of grayscale image ---
inline float sampleBilinear(const uint8_t *gray, int w, int h, float fx,
                            float fy) {
    if (fx < 0 || fy < 0 || fx >= w - 1 || fy >= h - 1)
        return 0.0f;

    int x0 = (int)fx;
    int y0 = (int)fy;
    float dx = fx - x0;
    float dy = fy - y0;

    float v00 = gray[y0 * w + x0];
    float v10 = gray[y0 * w + x0 + 1];
    float v01 = gray[(y0 + 1) * w + x0];
    float v11 = gray[(y0 + 1) * w + x0 + 1];

    return v00 * (1 - dx) * (1 - dy) + v10 * dx * (1 - dy) +
           v01 * (1 - dx) * dy + v11 * dx * dy;
}

// --- Rotate an NxN bit pattern 90 degrees clockwise ---
// Bit (N*N-1) = top-left cell, bit 0 = bottom-right cell.
inline uint64_t rotateBits90CW(uint64_t bits, int side) {
    uint64_t result = 0;
    for (int r = 0; r < side; r++) {
        for (int c = 0; c < side; c++) {
            int src_bit = (side * side - 1) - (r * side + c);
            int dst_r = c;
            int dst_c = side - 1 - r;
            int dst_bit = (side * side - 1) - (dst_r * side + dst_c);
            if (bits & (1ULL << src_bit))
                result |= (1ULL << dst_bit);
        }
    }
    return result;
}

// --- Hamming distance between two bit patterns ---
inline int hammingDistance(uint64_t a, uint64_t b) {
    return __builtin_popcountll(a ^ b);
}

// --- Warp quad candidate and read marker bits ---
// Returns true if the border check passes.
// bits_out: the inner marker_bits x marker_bits pattern as a uint64_t.
// The border is the outer ring of cells in the (marker_bits+2) grid.
inline bool readMarkerBits(const uint8_t *gray, int w, int h,
                           const std::array<Eigen::Vector2f, 4> &quad_corners,
                           int marker_bits, uint64_t &bits_out) {
    int grid_size = marker_bits + 2; // Include 1-cell border on each side

    // Map quad corners (image) to canonical grid corners.
    // Quad corners are in TL, TR, BR, BL order.
    // Grid coords: TL=(0,0), TR=(grid_size,0), BR=(grid_size,grid_size),
    //              BL=(0,grid_size).
    std::array<Eigen::Vector2f, 4> grid_corners = {
        Eigen::Vector2f(0.0f, 0.0f),
        Eigen::Vector2f((float)grid_size, 0.0f),
        Eigen::Vector2f((float)grid_size, (float)grid_size),
        Eigen::Vector2f(0.0f, (float)grid_size)};

    // Homography: grid → image (so we can sample image at grid cell centers)
    Eigen::Matrix3d H = computeHomography4pt(grid_corners, quad_corners);

    // Sample at cell centers: cell (r, c) center is at (c + 0.5, r + 0.5)
    std::vector<float> cell_values(grid_size * grid_size);
    for (int r = 0; r < grid_size; r++) {
        for (int c = 0; c < grid_size; c++) {
            Eigen::Vector2f grid_pt(c + 0.5f, r + 0.5f);
            Eigen::Vector2f img_pt = applyHomography(H, grid_pt);
            cell_values[r * grid_size + c] =
                sampleBilinear(gray, w, h, img_pt.x(), img_pt.y());
        }
    }

    // Compute threshold using Otsu's method on the sampled cells.
    // This gives much better black/white separation than a simple mean.
    float threshold;
    {
        // Build histogram (256 bins)
        int hist[256] = {};
        for (float v : cell_values)
            hist[std::max(0, std::min(255, (int)v))]++;

        int total_pixels = (int)cell_values.size();
        float sum = 0;
        for (int i = 0; i < 256; i++) sum += i * hist[i];

        float sumB = 0;
        int wB = 0;
        float max_var = 0;
        threshold = 128.0f;

        for (int t = 0; t < 256; t++) {
            wB += hist[t];
            if (wB == 0) continue;
            int wF = total_pixels - wB;
            if (wF == 0) break;

            sumB += t * hist[t];
            float mB = sumB / wB;
            float mF = (sum - sumB) / wF;
            float between = (float)wB * (float)wF * (mB - mF) * (mB - mF);
            if (between > max_var) {
                max_var = between;
                threshold = t;
            }
        }
    }

    // Check border: all border cells should be black (below threshold).
    int border_white = 0;
    int border_total = 0;
    for (int r = 0; r < grid_size; r++) {
        for (int c = 0; c < grid_size; c++) {
            if (r == 0 || r == grid_size - 1 || c == 0 || c == grid_size - 1) {
                border_total++;
                if (cell_values[r * grid_size + c] > threshold)
                    border_white++;
            }
        }
    }
    // Allow at most 25% of border cells to be wrong (noise tolerance)
    if (border_white > border_total / 4)
        return false;

    // Read inner bits
    bits_out = 0;
    for (int r = 0; r < marker_bits; r++) {
        for (int c = 0; c < marker_bits; c++) {
            float val = cell_values[(r + 1) * grid_size + (c + 1)];
            int bit_idx = (marker_bits * marker_bits - 1) - (r * marker_bits + c);
            if (val > threshold)
                bits_out |= (1ULL << bit_idx);
        }
    }

    return true;
}

// --- Downsample binary image 3x (OR: preserve all foreground edges) ---
// Specialized for N=3: unrolled 3×3 OR with no bounds checks for interior.
// 3208×2200 → 1069×733 (784KB) — fits in L2 cache.
inline void downsampleBinary3x(const uint8_t *src, int sw, int sh,
                               std::vector<uint8_t> &dst, int &dw, int &dh) {
    dw = sw / 3;
    dh = sh / 3;
    dst.resize((size_t)dw * dh);
    for (int y = 0; y < dh; y++) {
        int sy = y * 3;
        const uint8_t *r0 = src + sy * sw;
        const uint8_t *r1 = r0 + sw;
        const uint8_t *r2 = r1 + sw;
        for (int x = 0; x < dw; x++) {
            int sx = x * 3;
            dst[y * dw + x] = (r0[sx] | r0[sx+1] | r0[sx+2] |
                                r1[sx] | r1[sx+1] | r1[sx+2] |
                                r2[sx] | r2[sx+1] | r2[sx+2]) ? 255 : 0;
        }
    }
}

// Generic fallback for other downsample factors.
inline void downsampleBinaryNx(const uint8_t *src, int sw, int sh,
                               std::vector<uint8_t> &dst, int &dw, int &dh,
                               int N) {
    if (N == 3) { downsampleBinary3x(src, sw, sh, dst, dw, dh); return; }
    dw = sw / N;
    dh = sh / N;
    dst.resize((size_t)dw * dh);
    for (int y = 0; y < dh; y++) {
        int sy = y * N;
        for (int x = 0; x < dw; x++) {
            int sx = x * N;
            uint8_t val = 0;
            for (int dy = 0; dy < N && sy + dy < sh; dy++)
                for (int dx = 0; dx < N && sx + dx < sw; dx++)
                    val |= src[(sy + dy) * sw + (sx + dx)];
            dst[y * dw + x] = val ? 255 : 0;
        }
    }
}

} // namespace detail

// Forward declaration (defined after detectMarkers)
inline void cornerSubPix(const uint8_t *gray, int w, int h,
                         std::vector<Eigen::Vector2f> &corners,
                         int half_win, int max_iter,
                         float epsilon);

// ─────────────────────────────────────────────────────────────────────────────
// detectMarkers — detect ArUco markers in a grayscale image
// ─────────────────────────────────────────────────────────────────────────────

inline std::vector<DetectedMarker>
detectMarkers(const uint8_t *gray, int w, int h, const ArUcoDictionary &dict,
              GpuThresholdFunc gpu_thresh = nullptr, void *gpu_ctx = nullptr,
              const std::vector<std::vector<uint8_t>> *precomputed_ds = nullptr,
              int precomputed_num_passes = 0) {
    if (!dict.valid() || !gray || w < 10 || h < 10)
        return {};

    // ── Steps (a)+(b): Multi-scale adaptive threshold + contour finding ──

    // Two window sizes: small (marker-sized) and large (board-sized).
    // With 4x downsampled contour finding, the second pass adds only ~15ms.
    std::vector<int> window_sizes;
    {
        int small_win = std::max(3, std::min(w, h) / 40);
        if (small_win % 2 == 0) small_win++;
        int large_win = std::max(w, h) / 10;
        if (large_win % 2 == 0) large_win++;
        large_win = std::min(large_win, 255);
        window_sizes.push_back(small_win);
        if (large_win > small_win + 10)
            window_sizes.push_back(large_win);
    }
    int C = 7;
    int num_passes = (int)window_sizes.size();

    // Max contour length: ArUco markers have perimeter ~200-800px.
    // Abort tracing for noise contours (large foreground blobs) early.
    int max_contour_len = 800;

    std::vector<std::vector<Eigen::Vector2i>> all_contours;

    // Lambda: downsample binary 2x, find contours, scale back to full res.
    // The 2x downsample makes contour arrays fit in L2 cache (~1.76MB vs 7MB),
    // giving ~50x speedup over full-resolution contour tracing.
    // Downsample binary 3x, find contours on small image (784KB fits in L2),
    // then scale contour points back to full resolution.
    constexpr int ds_factor = 3;
    auto find_contours_downsampled = [&](const std::vector<uint8_t> &binary) {
        int dw = 0, dh = 0;
        std::vector<uint8_t> small;
        detail::downsampleBinaryNx(binary.data(), w, h, small, dw, dh, ds_factor);

        auto contours = detail::findContours(small, dw, dh,
                                             nullptr, max_contour_len / ds_factor);

        // Scale contour points back to full resolution
        for (auto &contour : contours) {
            for (auto &pt : contour) {
                pt.x() *= ds_factor;
                pt.y() *= ds_factor;
            }
        }
        return contours;
    };

    if (precomputed_ds && precomputed_num_passes > 0) {
        // Pre-computed binary path: binaries already produced by GPU video pipeline.
        // Skip both GPU threshold call and CPU threshold — just find contours.
        int dw = w / 3, dh = h / 3;
        auto find_ds_contours = [&](const std::vector<uint8_t> &ds) {
            auto contours = detail::findContours(ds, dw, dh,
                                                 nullptr, max_contour_len / ds_factor);
            for (auto &contour : contours)
                for (auto &pt : contour) { pt.x() *= ds_factor; pt.y() *= ds_factor; }
            return contours;
        };
        all_contours = find_ds_contours((*precomputed_ds)[0]);
        if (precomputed_num_passes > 1 && all_contours.size() >= 4) {
            auto pass2 = find_ds_contours((*precomputed_ds)[1]);
            all_contours.insert(all_contours.end(),
                               std::make_move_iterator(pass2.begin()),
                               std::make_move_iterator(pass2.end()));
        }
    } else if (gpu_thresh && gpu_ctx) {
        // GPU path: threshold + 3x downsample fused on GPU.
        // Only (w/3)*(h/3) bytes transferred back per pass (~784KB vs 7MB).
        int dw = w / 3, dh = h / 3;
        std::vector<std::vector<uint8_t>> ds_images(num_passes);
        std::vector<uint8_t *> ds_ptrs(num_passes);
        for (int p = 0; p < num_passes; p++) {
            ds_images[p].resize((size_t)dw * dh);
            ds_ptrs[p] = ds_images[p].data();
        }

        gpu_thresh(gpu_ctx, gray, w, h,
                   window_sizes.data(), C, num_passes,
                   ds_ptrs.data());

        // Contours directly on GPU-downsampled output (no CPU downsample needed)
        auto find_ds_contours = [&](const std::vector<uint8_t> &ds) {
            auto contours = detail::findContours(ds, dw, dh,
                                                 nullptr, max_contour_len / ds_factor);
            for (auto &contour : contours)
                for (auto &pt : contour) { pt.x() *= ds_factor; pt.y() *= ds_factor; }
            return contours;
        };

        // Run first pass contour finding; early-exit if no markers visible.
        all_contours = find_ds_contours(ds_images[0]);
        if (num_passes > 1 && all_contours.size() >= 4) {
            auto pass2 = find_ds_contours(ds_images[1]);
            all_contours.insert(all_contours.end(),
                               std::make_move_iterator(pass2.begin()),
                               std::make_move_iterator(pass2.end()));
        }
    } else {
        // CPU path: compute integral image, threshold + contour on downsampled
        detail::IntegralImage integral;
        integral.compute(gray, w, h);

        // First pass
        {
            std::vector<uint8_t> bin;
            detail::adaptiveThreshold(gray, w, h, bin, integral,
                                      window_sizes[0], C);
            all_contours = find_contours_downsampled(bin);
        }
        // Second pass only if first found enough candidates
        if (num_passes > 1 && all_contours.size() >= 4) {
            std::vector<uint8_t> bin;
            detail::adaptiveThreshold(gray, w, h, bin, integral,
                                      window_sizes[1], C);
            auto pass2 = find_contours_downsampled(bin);
            all_contours.insert(all_contours.end(),
                               std::make_move_iterator(pass2.begin()),
                               std::make_move_iterator(pass2.end()));
        }
    }
    auto &contours = all_contours;

    // ── Steps (c-h): For each contour, approximate polygon, filter quads,
    //    read bits, match dictionary, handle duplicates ──

    // Collect all valid marker detections.
    struct MarkerCandidate {
        int id;
        int rotation;         // 0, 1, 2, 3 (number of 90 CW rotations matched)
        int hamming;
        std::array<Eigen::Vector2f, 4> corners; // original quad corners, TL/TR/BR/BL
    };
    std::vector<MarkerCandidate> candidates;

    for (const auto &contour : contours) {
        // Convert contour to float
        std::vector<Eigen::Vector2f> fcontour(contour.size());
        for (size_t i = 0; i < contour.size(); i++)
            fcontour[i] = contour[i].cast<float>();

        // (c) Polygon approximation
        float peri = detail::contourPerimeter(fcontour);
        float epsilon = 0.05f * peri;
        auto poly = detail::approxPolyDP(fcontour, epsilon);

        // (d) Quad filtering
        if (poly.size() != 4)
            continue;

        // Convexity
        if (!detail::isConvex(poly))
            continue;

        // Area check (use absolute area since polygon may be CW or CCW)
        float area = std::abs(detail::polygonArea(poly));
        if (area < 100.0f)
            continue;

        // Side length checks
        float min_side = 1e9f, max_side = 0.0f;
        for (int i = 0; i < 4; i++) {
            float side = (poly[(i + 1) % 4] - poly[i]).norm();
            min_side = std::min(min_side, side);
            max_side = std::max(max_side, side);
        }
        if (min_side < 10.0f)
            continue;
        if (max_side > 4.0f * min_side)
            continue;

        // Order corners consistently: TL, TR, BR, BL
        std::array<Eigen::Vector2f, 4> quad = {poly[0], poly[1], poly[2],
                                                 poly[3]};
        detail::orderQuadCorners(quad);

        // Refine quad corners to subpixel accuracy on full-resolution
        // gray image. This recovers precision lost by the 3x downsampled
        // contour finding (~1.5px grid → subpixel accuracy).
        {
            std::vector<Eigen::Vector2f> qv(quad.begin(), quad.end());
            cornerSubPix(gray, w, h, qv, 5, 30, 0.01f);
            for (int i = 0; i < 4; i++) quad[i] = qv[i];
        }

        // Ensure clockwise winding (positive area in screen coords with y-down)
        float signed_area =
            (quad[1].x() - quad[0].x()) * (quad[2].y() - quad[0].y()) -
            (quad[2].x() - quad[0].x()) * (quad[1].y() - quad[0].y());
        // For TL, TR, BR, BL in y-down coords, the cross product
        // (TR-TL) x (BR-TL) should be positive for clockwise winding.
        // If negative, swap TR and BL to fix winding.
        if (signed_area < 0) {
            std::swap(quad[1], quad[3]); // swap TR and BL
        }

        // (e) Read marker bits
        uint64_t bits = 0;
        if (!detail::readMarkerBits(gray, w, h, quad, dict.marker_bits, bits))
            continue;

        // (f) Dictionary matching with all 4 rotations
        int best_id = -1;
        int best_rotation = 0;
        int best_hamming = dict.marker_bits * dict.marker_bits + 1;

        uint64_t rotated = bits;
        for (int rot = 0; rot < 4; rot++) {
            for (int mid = 0; mid < dict.num_markers; mid++) {
                int hd = detail::hammingDistance(rotated, dict.patterns[mid]);
                if (hd <= dict.max_correction_bits && hd < best_hamming) {
                    best_hamming = hd;
                    best_id = mid;
                    best_rotation = rot;
                }
            }
            if (rot < 3) {
                rotated =
                    detail::rotateBits90CW(rotated, dict.marker_bits);
            }
        }

        if (best_id < 0)
            continue;

        // (g) Rotate quad corners to match canonical orientation.
        // The bits were rotated 'best_rotation' times (90 CW each) to match
        // the dictionary. This means our reading frame (quad[0]=grid TL)
        // needs to be rotated to find the canonical TL. A single 90 CW
        // rotation of the reading frame maps: TL→TR, TR→BR, BR→BL, BL→TL.
        // So after R rotations, canonical[i] = quad[(i + 4 - R) % 4].
        std::array<Eigen::Vector2f, 4> ordered_corners;
        int shift = (4 - best_rotation) % 4;
        for (int i = 0; i < 4; i++) {
            ordered_corners[i] = quad[(i + shift) % 4];
        }

        candidates.push_back({best_id, best_rotation, best_hamming,
                               ordered_corners});
    }

    // (h) Duplicate removal: keep best hamming for each ID
    // Sort by ID then hamming distance
    std::sort(candidates.begin(), candidates.end(),
              [](const MarkerCandidate &a, const MarkerCandidate &b) {
                  if (a.id != b.id) return a.id < b.id;
                  return a.hamming < b.hamming;
              });

    std::vector<DetectedMarker> results;
    int prev_id = -1;
    for (const auto &c : candidates) {
        if (c.id == prev_id)
            continue; // Skip worse duplicate
        prev_id = c.id;
        results.push_back({c.id, c.corners, c.hamming});
    }

    // Sort by ID for consistent output
    std::sort(results.begin(), results.end(),
              [](const DetectedMarker &a, const DetectedMarker &b) {
                  return a.id < b.id;
              });

    return results;
}

// ─────────────────────────────────────────────────────────────────────────────
// cornerSubPix — gradient-based subpixel corner refinement
// ─────────────────────────────────────────────────────────────────────────────
// Replaces cv::cornerSubPix. Uses the gradient autocorrelation method:
// find the point p that minimizes sum_i [grad_i . (p - x_i)]^2 over a
// window, which gives the saddle-point / corner location.

inline void cornerSubPix(const uint8_t *gray, int w, int h,
                         std::vector<Eigen::Vector2f> &corners,
                         int half_win = 3, int max_iter = 30,
                         float epsilon = 0.01f) {
    for (auto &corner : corners) {
        float cx = corner.x();
        float cy = corner.y();

        for (int iter = 0; iter < max_iter; iter++) {
            int ix = (int)std::round(cx);
            int iy = (int)std::round(cy);

            if (ix - half_win - 1 < 0 || ix + half_win + 1 >= w ||
                iy - half_win - 1 < 0 || iy + half_win + 1 >= h)
                break;

            // Build the autocorrelation matrix M and vector b:
            // M = sum (g * g^T)
            // b = sum (g * g^T * [x, y]^T)
            // where g = [dI/dx, dI/dy] and (x, y) are pixel positions.
            double m00 = 0, m01 = 0, m11 = 0;
            double bx = 0, by = 0;

            for (int dy = -half_win; dy <= half_win; dy++) {
                for (int dx = -half_win; dx <= half_win; dx++) {
                    int px = ix + dx;
                    int py = iy + dy;

                    // Central differences for gradient
                    float gx = (float)gray[py * w + px + 1] -
                               (float)gray[py * w + px - 1];
                    float gy = (float)gray[(py + 1) * w + px] -
                               (float)gray[(py - 1) * w + px];

                    m00 += gx * gx;
                    m01 += gx * gy;
                    m11 += gy * gy;
                    bx += gx * gx * (float)px + gx * gy * (float)py;
                    by += gx * gy * (float)px + gy * gy * (float)py;
                }
            }

            // Solve M * p = b
            double det = m00 * m11 - m01 * m01;
            if (std::abs(det) < 1e-6)
                break;

            double inv_det = 1.0 / det;
            float new_x = (float)((m11 * bx - m01 * by) * inv_det);
            float new_y = (float)((-m01 * bx + m00 * by) * inv_det);

            float shift_x = new_x - cx;
            float shift_y = new_y - cy;

            cx = new_x;
            cy = new_y;

            if (shift_x * shift_x + shift_y * shift_y < epsilon * epsilon)
                break;
        }

        // Clamp to image bounds
        cx = std::max(0.0f, std::min((float)(w - 1), cx));
        cy = std::max(0.0f, std::min((float)(h - 1), cy));

        corner = Eigen::Vector2f(cx, cy);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Saddle-point subpixel corner refinement (ROCHADE-inspired)
// ─────────────────────────────────────────────────────────────────────────────
// Fits I(x,y) = a*x^2 + b*x*y + c*y^2 + d*x + e*y + f to a cone-weighted
// image patch around each corner, then computes the saddle point analytically.
// Non-iterative and affine-invariant. Better than cornerSubPix for checkerboard
// X-junctions. Based on Lucchese & Mitra (2002) and Placht et al. (ECCV 2014).

inline void saddlePointRefine(const uint8_t *gray, int w, int h,
                               std::vector<Eigen::Vector2f> &corners,
                               int half_win = 5) {
    const int win = 2 * half_win + 1;
    const float R = (float)half_win + 0.5f;

    // Build cone (Bartlett) filter kernel + design matrix entries
    struct PixelEntry { double row[6]; double weight; };
    std::vector<PixelEntry> entries(win * win);
    Eigen::Matrix<double, 6, 6> AtWA = Eigen::Matrix<double, 6, 6>::Zero();

    int idx = 0;
    for (int dy = -half_win; dy <= half_win; dy++) {
        for (int dx = -half_win; dx <= half_win; dx++) {
            double x = (double)dx, y = (double)dy;
            float r = std::sqrt((float)(dx * dx + dy * dy));
            double wt = std::max(0.0, 1.0 - r / R);
            entries[idx] = {{x*x, x*y, y*y, x, y, 1.0}, wt};
            for (int i = 0; i < 6; i++)
                for (int j = 0; j < 6; j++)
                    AtWA(i, j) += entries[idx].row[i] * wt * entries[idx].row[j];
            idx++;
        }
    }
    Eigen::Matrix<double, 6, 6> AtWA_inv = AtWA.inverse();

    for (auto &corner : corners) {
        int ix = (int)std::round(corner.x());
        int iy = (int)std::round(corner.y());
        if (ix - half_win < 0 || ix + half_win >= w ||
            iy - half_win < 0 || iy + half_win >= h)
            continue;

        Eigen::Matrix<double, 6, 1> AtWI = Eigen::Matrix<double, 6, 1>::Zero();
        idx = 0;
        for (int dy = -half_win; dy <= half_win; dy++) {
            for (int dx = -half_win; dx <= half_win; dx++) {
                double I_val = (double)gray[(iy + dy) * w + (ix + dx)];
                double wt = entries[idx].weight;
                for (int i = 0; i < 6; i++)
                    AtWI(i) += entries[idx].row[i] * wt * I_val;
                idx++;
            }
        }

        Eigen::Matrix<double, 6, 1> p = AtWA_inv * AtWI;
        double a = p(0), b = p(1), c = p(2), d = p(3), e = p(4);
        double det = 4.0 * a * c - b * b;
        if (det >= -1e-10) continue; // not a saddle point

        double x_s = (b * e - 2.0 * c * d) / det;
        double y_s = (b * d - 2.0 * a * e) / det;
        if (std::abs(x_s) > half_win || std::abs(y_s) > half_win) continue;

        float nx = std::max(0.0f, std::min((float)(w - 1), (float)ix + (float)x_s));
        float ny = std::max(0.0f, std::min((float)(h - 1), (float)iy + (float)y_s));
        corner = Eigen::Vector2f(nx, ny);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ChArUco board layout helpers
// ─────────────────────────────────────────────────────────────────────────────
namespace detail {

// Determine which markers are on the board and their grid positions.
// In a standard ChArUco board, markers fill the "white" squares in a
// checkerboard pattern. The board is squares_x columns by squares_y rows.
// Square (row, col) is white if (row + col) is even (following OpenCV's
// convention where (0,0) is a white/marker-containing square).
// Actually, OpenCV's convention: markers are placed in the *white* squares
// where the first square (0,0) is white. So (row+col) even = white = has marker.
// Marker IDs are assigned sequentially row by row among white squares.

// Returns: map from marker_id to (row, col) in the board grid.
// OpenCV convention: square (0,0) is black (no marker); markers go in
// "white" squares where (row + col) is odd.
inline std::map<int, std::pair<int, int>>
boardMarkerPositions(int squares_x, int squares_y) {
    std::map<int, std::pair<int, int>> positions;
    int id = 0;
    for (int r = 0; r < squares_y; r++) {
        for (int c = 0; c < squares_x; c++) {
            if ((r + c) % 2 == 1) {
                positions[id] = {r, c};
                id++;
            }
        }
    }
    return positions;
}

// Get the 4 corner positions of a marker in board coordinates (mm).
// The marker is centered in its square with the given marker_length.
// Square (row, col) occupies [col*sq_len, (col+1)*sq_len] x
//                             [row*sq_len, (row+1)*sq_len].
// The marker is centered within the square.
// Returns corners in TL, TR, BR, BL order (in board world coords).
inline std::array<Eigen::Vector2f, 4>
markerCornersInBoard(int grid_row, int grid_col, float sq_len,
                     float marker_len) {
    float cx = (grid_col + 0.5f) * sq_len;
    float cy = (grid_row + 0.5f) * sq_len;
    float half = marker_len * 0.5f;
    return {
        Eigen::Vector2f(cx - half, cy - half), // TL
        Eigen::Vector2f(cx + half, cy - half), // TR
        Eigen::Vector2f(cx + half, cy + half), // BR
        Eigen::Vector2f(cx - half, cy + half), // BL
    };
}

// Get the position of a ChArUco inner corner in board coordinates.
// Corner (cr, cc) is at position ((cc+1)*sq_len, (cr+1)*sq_len).
inline Eigen::Vector2f charucoCornerInBoard(int corner_row, int corner_col,
                                            float sq_len) {
    return Eigen::Vector2f((corner_col + 1) * sq_len,
                           (corner_row + 1) * sq_len);
}

// Get the 4 surrounding square positions (row, col) for an inner corner.
// Inner corner (cr, cc) is at the intersection of squares:
//   (cr, cc), (cr, cc+1), (cr+1, cc), (cr+1, cc+1)
// These are the 4 squares that share this corner vertex.
inline std::array<std::pair<int, int>, 4>
surroundingSquares(int corner_row, int corner_col) {
    return {std::pair<int, int>{corner_row, corner_col},
            std::pair<int, int>{corner_row, corner_col + 1},
            std::pair<int, int>{corner_row + 1, corner_col},
            std::pair<int, int>{corner_row + 1, corner_col + 1}};
}

} // namespace detail

// ─────────────────────────────────────────────────────────────────────────────
// detectCharucoBoard — detect ChArUco board corners
// ─────────────────────────────────────────────────────────────────────────────
// Detects ArUco markers on a ChArUco board, then interpolates the inner
// checkerboard corner positions using surrounding marker corners and known
// board geometry.

inline CharucoResult
detectCharucoBoard(const uint8_t *gray, int w, int h,
                   const CharucoBoard &board, const ArUcoDictionary &dict,
                   GpuThresholdFunc gpu_thresh = nullptr,
                   void *gpu_ctx = nullptr,
                   const std::vector<std::vector<uint8_t>> *precomputed_ds = nullptr,
                   int precomputed_num_passes = 0) {
    CharucoResult result;

    // Step 1: Detect ArUco markers
    auto markers = detectMarkers(gray, w, h, dict, gpu_thresh, gpu_ctx,
                                 precomputed_ds, precomputed_num_passes);
    if (markers.empty())
        return result;

    // Step 2: Build map from marker ID to grid position and detected corners
    auto marker_grid = detail::boardMarkerPositions(board.squares_x,
                                                     board.squares_y);

    // Filter: only keep markers whose IDs are valid for this board
    int max_markers_on_board = (int)marker_grid.size();
    {
        std::vector<DetectedMarker> valid;
        for (auto &m : markers) {
            if (marker_grid.count(m.id))
                valid.push_back(m);
        }
        markers = std::move(valid);
    }

    // Require at least 4 markers for a reliable detection
    if ((int)markers.size() < 4)
        return result;

    // Board-level spatial consistency check: verify marker centers form a
    // grid pattern consistent with known board geometry.
    // For each pair of detected markers, check that the ratio of image distance
    // to expected board distance is roughly consistent.
    {
        std::vector<double> scale_ratios;
        for (size_t i = 0; i < markers.size(); i++) {
            auto it_i = marker_grid.find(markers[i].id);
            if (it_i == marker_grid.end()) continue;
            Eigen::Vector2f ci = (markers[i].corners[0] + markers[i].corners[2]) * 0.5f;
            float bi_x = (it_i->second.second + 0.5f) * board.square_length;
            float bi_y = (it_i->second.first + 0.5f) * board.square_length;

            for (size_t j = i + 1; j < markers.size(); j++) {
                auto it_j = marker_grid.find(markers[j].id);
                if (it_j == marker_grid.end()) continue;
                Eigen::Vector2f cj = (markers[j].corners[0] + markers[j].corners[2]) * 0.5f;
                float bj_x = (it_j->second.second + 0.5f) * board.square_length;
                float bj_y = (it_j->second.first + 0.5f) * board.square_length;

                float img_dist = (ci - cj).norm();
                float board_dist = std::hypot(bi_x - bj_x, bi_y - bj_y);
                if (board_dist > 1e-6f && img_dist > 10.0f)
                    scale_ratios.push_back(img_dist / board_dist);
            }
        }

        if (scale_ratios.size() >= 3) {
            std::sort(scale_ratios.begin(), scale_ratios.end());
            double median = scale_ratios[scale_ratios.size() / 2];
            // Reject markers whose scale ratio deviates >50% from median
            std::vector<DetectedMarker> consistent;
            for (auto &m : markers) {
                auto it = marker_grid.find(m.id);
                if (it == marker_grid.end()) continue;
                Eigen::Vector2f cm = (m.corners[0] + m.corners[2]) * 0.5f;
                float bm_x = (it->second.second + 0.5f) * board.square_length;
                float bm_y = (it->second.first + 0.5f) * board.square_length;

                bool ok = true;
                for (auto &other : markers) {
                    if (other.id == m.id) continue;
                    auto it2 = marker_grid.find(other.id);
                    if (it2 == marker_grid.end()) continue;
                    Eigen::Vector2f co = (other.corners[0] + other.corners[2]) * 0.5f;
                    float bo_x = (it2->second.second + 0.5f) * board.square_length;
                    float bo_y = (it2->second.first + 0.5f) * board.square_length;

                    float img_d = (cm - co).norm();
                    float brd_d = std::hypot(bm_x - bo_x, bm_y - bo_y);
                    if (brd_d > 1e-6f && img_d > 10.0f) {
                        double ratio = img_d / brd_d;
                        if (ratio < median * 0.5 || ratio > median * 1.5) {
                            ok = false;
                            break;
                        }
                    }
                }
                if (ok) consistent.push_back(m);
            }
            if ((int)consistent.size() >= 3)
                markers = std::move(consistent);
        }
    }

    // Map from (grid_row, grid_col) of marker → detected corners in image
    std::map<std::pair<int, int>, const DetectedMarker *> detected_on_grid;
    for (const auto &m : markers) {
        auto it = marker_grid.find(m.id);
        if (it != marker_grid.end()) {
            detected_on_grid[it->second] = &m;
        }
    }

    // Step 3: Compute a global homography from ALL detected markers, then
    // use it to interpolate all inner corner positions. This is more robust
    // than per-corner local homographies.
    int num_inner_x = board.squares_x - 1;
    int num_inner_y = board.squares_y - 1;

    // Collect all board-coord and image-coord point pairs from all markers
    std::vector<Eigen::Vector2f> all_board_pts;
    std::vector<Eigen::Vector2f> all_image_pts;
    for (const auto &[grid_pos, marker_ptr] : detected_on_grid) {
        auto board_corners = detail::markerCornersInBoard(
            grid_pos.first, grid_pos.second,
            board.square_length, board.marker_length);
        for (int ci = 0; ci < 4; ci++) {
            all_board_pts.push_back(board_corners[ci]);
            all_image_pts.push_back(marker_ptr->corners[ci]);
        }
    }

    // Need at least 4 correspondences (1 marker) for homography
    if ((int)all_board_pts.size() < 4)
        return result;

    // Compute global homography: board → image (DLT with Hartley normalization)
    Eigen::Matrix3d H_global;
    {
        int n = (int)all_board_pts.size();
        std::vector<Eigen::Vector2d> src(n), dst(n);
        for (int i = 0; i < n; i++) {
            src[i] = all_board_pts[i].cast<double>();
            dst[i] = all_image_pts[i].cast<double>();
        }

        // Hartley normalize
        std::vector<Eigen::Vector2d> src_n, dst_n;
        Eigen::Matrix3d T_src = red_math::hartleyNormalize(src, src_n);
        Eigen::Matrix3d T_dst = red_math::hartleyNormalize(dst, dst_n);

        Eigen::MatrixXd A(2 * n, 9);
        for (int i = 0; i < n; i++) {
            double X = src_n[i].x(), Y = src_n[i].y();
            double u = dst_n[i].x(), v = dst_n[i].y();
            A(2*i,   0)=X; A(2*i,   1)=Y; A(2*i,   2)=1;
            A(2*i,   3)=0; A(2*i,   4)=0; A(2*i,   5)=0;
            A(2*i,   6)=-u*X; A(2*i,   7)=-u*Y; A(2*i,   8)=-u;
            A(2*i+1, 0)=0; A(2*i+1, 1)=0; A(2*i+1, 2)=0;
            A(2*i+1, 3)=X; A(2*i+1, 4)=Y; A(2*i+1, 5)=1;
            A(2*i+1, 6)=-v*X; A(2*i+1, 7)=-v*Y; A(2*i+1, 8)=-v;
        }
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
        Eigen::VectorXd hv = svd.matrixV().col(8);
        Eigen::Matrix3d H_n;
        H_n << hv(0), hv(1), hv(2), hv(3), hv(4), hv(5), hv(6), hv(7), hv(8);
        H_global = T_dst.inverse() * H_n * T_src;
        if (std::abs(H_global(2,2)) > 1e-12)
            H_global /= H_global(2,2);
    }

    for (int cr = 0; cr < num_inner_y; cr++) {
        for (int cc = 0; cc < num_inner_x; cc++) {
            int corner_id = cr * num_inner_x + cc;

            // Check that at least 1 surrounding marker was detected
            auto surr = detail::surroundingSquares(cr, cc);
            bool has_neighbor = false;
            for (const auto &sq : surr) {
                if ((sq.first + sq.second) % 2 == 1) {
                    if (detected_on_grid.count(sq)) {
                        has_neighbor = true;
                        break;
                    }
                }
            }
            if (!has_neighbor)
                continue;

            // Map corner position through the global homography
            Eigen::Vector2f corner_board = detail::charucoCornerInBoard(
                cr, cc, board.square_length);
            Eigen::Vector2f corner_img =
                detail::applyHomography(H_global, corner_board);

            // Sanity: corner should be within the image
            if (corner_img.x() >= 0 && corner_img.x() < w &&
                corner_img.y() >= 0 && corner_img.y() < h) {
                result.corners.push_back(corner_img);
                result.ids.push_back(corner_id);
            }
        }
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// matchImagePoints — map detected ChArUco corner IDs to 3D board points
// ─────────────────────────────────────────────────────────────────────────────
// Follows OpenCV convention: corner ID = row * (squares_x - 1) + col
// Object point: x = (col + 1) * square_length, y = (row + 1) * square_length,
// z = 0.

inline void matchImagePoints(const CharucoBoard &board,
                              const std::vector<Eigen::Vector2f> &corners,
                              const std::vector<int> &ids,
                              std::vector<Eigen::Vector3f> &obj_pts,
                              std::vector<Eigen::Vector2f> &img_pts) {
    obj_pts.clear();
    img_pts.clear();

    int num_inner_x = board.squares_x - 1;

    for (size_t i = 0; i < ids.size(); i++) {
        int cid = ids[i];
        int col = cid % num_inner_x;
        int row = cid / num_inner_x;

        float x = (col + 1) * board.square_length;
        float y = (row + 1) * board.square_length;

        obj_pts.push_back(Eigen::Vector3f(x, y, 0.0f));
        img_pts.push_back(corners[i]);
    }
}

} // namespace aruco_detect
