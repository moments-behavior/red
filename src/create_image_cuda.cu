#include "create_image_cuda.h"

__global__ void create_image_cuda_kernel(unsigned char *cuda_buffer, double current_time) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	double multiplier = 0.5 * (sin(current_time * 0.000000001) + 1.0);
	if ((x < 854) && (y < 480)) {
		*(cuda_buffer + ((y * 854 * 4) + (x * 4))) = 200 + ((unsigned char) (((double) y / 854.0 * multiplier) * 55.0));
		*(cuda_buffer + ((y * 854 * 4) + (x * 4)) + 1) = ((unsigned char) (((double) x / 854.0 * multiplier) * 255.0));
		*(cuda_buffer + ((y * 854 * 4) + (x * 4)) + 2) = ((unsigned char) (((double) y / 480.0 * multiplier) * 255.0));
		*(cuda_buffer + ((y * 854 * 4) + (x * 4)) + 3) = 255;
	}
}

void create_image_cuda(unsigned char *cuda_buffer) {
    dim3 threads_per_block(16, 16);
    dim3 num_blocks((854 + 1) / threads_per_block.x, (480 + 1) / threads_per_block.y);
    double current_time = (double) (std::chrono::system_clock::now().time_since_epoch()).count();
    create_image_cuda_kernel<<<num_blocks, threads_per_block>>>(cuda_buffer, current_time);
}