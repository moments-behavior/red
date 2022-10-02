#pragma once
#include <cuda.h>
#include "NvCodecUtils.h"

void GetImage(CUdeviceptr dpSrc, uint8_t* pDst, int nWidth, int nHeight);
void clear_buffer_with_constant_image(unsigned char* image_pt, int width, int height);
void print_one_display_buffer(unsigned char* image_pt, int width, int height, int channels);