#ifndef RED_SIMD
#define RED_SIMD

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <tmmintrin.h>

typedef uint32_t u32;
typedef uint8_t u8;
typedef float f32;


inline void rgba_to_rgb_cpu(unsigned char *rgba_image, unsigned char *rgb_image, int length){
    // __m128i mask = _mm_set_epi8(-128, -128, -128, -128, 13, 14, 15, 9, 10, 11, 5, 6, 7, 1, 2, 3);
    __m128i mask = _mm_set_epi8(-128, -128, -128, -128, 14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 0);

    for (u32 p = 0; p < length / 4; p++) {
        u8 *dest = (u8 *)rgb_image + 12 * p;
        u8 *orig = (u8 *)rgba_image + 16 * p;
        _mm_storeu_si128((__m128i *)dest, _mm_shuffle_epi8(_mm_load_si128((__m128i *)orig), mask));
    }

    // int last_pixel_rgba = (length-1) * 4;
    // int last_pixel_rgb = (length-1) * 3;

	// printf("rgba image first pixel: %d %d %d %d \n", rgba_image[0], rgba_image[1], rgba_image[2], rgba_image[3]);
	// printf("rgba image last pixel: %d %d %d %d \n", rgba_image[last_pixel_rgba], rgba_image[last_pixel_rgba+1], rgba_image[last_pixel_rgba+2], rgba_image[last_pixel_rgba+3]);

    // printf("rgb image first pixel: %d %d %d \n", rgb_image[0], rgb_image[1], rgb_image[2]);
	// printf("rgb image last pixel: %d %d %d \n", rgb_image[last_pixel_rgb], rgb_image[last_pixel_rgb+1], rgb_image[last_pixel_rgb+2]);


}


inline void rgba_to_bgr_cpu(unsigned char *rgba_image, unsigned char *rgb_image, int length){
    // __m128i mask = _mm_set_epi8(-128, -128, -128, -128, 13, 14, 15, 9, 10, 11, 5, 6, 7, 1, 2, 3);
    __m128i mask = _mm_set_epi8(-128, -128, -128, -128, 12, 13, 14, 8, 9, 10, 4, 5, 6, 0, 1, 2);

    for (u32 p = 0; p < length / 4; p++) {
        u8 *dest = (u8 *)rgb_image + 12 * p;
        u8 *orig = (u8 *)rgba_image + 16 * p;
        _mm_storeu_si128((__m128i *)dest, _mm_shuffle_epi8(_mm_load_si128((__m128i *)orig), mask));
    }

    // int last_pixel_rgba = (length-1) * 4;
    // int last_pixel_rgb = (length-1) * 3;

	// printf("rgba image first pixel: %d %d %d %d \n", rgba_image[0], rgba_image[1], rgba_image[2], rgba_image[3]);
	// printf("rgba image last pixel: %d %d %d %d \n", rgba_image[last_pixel_rgba], rgba_image[last_pixel_rgba+1], rgba_image[last_pixel_rgba+2], rgba_image[last_pixel_rgba+3]);

    // printf("rgb image first pixel: %d %d %d \n", rgb_image[0], rgb_image[1], rgb_image[2]);
	// printf("rgb image last pixel: %d %d %d \n", rgb_image[last_pixel_rgb], rgb_image[last_pixel_rgb+1], rgb_image[last_pixel_rgb+2]);


}

#endif