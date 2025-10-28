#ifndef RED_RENDER
#define RED_RENDER
#include "decoder.h"
#include "gx_helper.h"

struct PBO_CUDA {
    GLuint pbo;
    unsigned char *cuda_buffer;
    cudaGraphicsResource_t cuda_resource;
    size_t cuda_pbo_storage_buffer_size;
};

struct RenderScene {
    u32 num_cams;
    u32 *image_width;
    u32 *image_height;
    u32 size_of_buffer;
    GLuint *image_texture;
    PBO_CUDA *pbo_cuda;
    PictureBuffer **display_buffer;
    SeekInfo *seek_context;
    bool use_cpu_buffer;
};

void render_initialize_target(gx_context *context);
void render_allocate_scene_memory(RenderScene *scene, u32 size_of_buffer);

#endif
