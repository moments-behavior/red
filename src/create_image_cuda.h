#pragma once
#include "red_build_config.h"

#if defined(RED_HAVE_CUDA)
#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

void create_image_cuda(unsigned char *cuda_buffer);
#endif // RED_HAVE_CUDA
