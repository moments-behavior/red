#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <tmmintrin.h>

typedef uint32_t u32;
typedef uint8_t u8;
typedef float f32;


inline void rgba_to_rgb_cpu(unsigned char *rgba_image, unsigned char *rgb_image, int length){
    // __m128i mask = _mm_set_epi8(-128, -128, -128, -128, 13, 14, 15, 9, 10, 11, 5, 6, 7, 1, 2, 3);
    __m128i mask = _mm_set_epi8(-128, -128, -128, -128, 15, 14, 13, 11, 10, 9, 7, 6, 5, 3, 2, 1);

    for (u32 p = 0; p < length / 4; p++) {
        u8 *dest = (u8 *)rgb_image + 12 * p;
        u8 *orig = (u8 *)rgba_image + 16 * p;
        _mm_storeu_si128((__m128i *)dest, _mm_shuffle_epi8(_mm_load_si128((__m128i *)orig), mask));
    }
}