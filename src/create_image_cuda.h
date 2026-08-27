#pragma once

#ifndef __APPLE__
#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

void create_image_cuda(unsigned char *cuda_buffer);
#endif // !__APPLE__
