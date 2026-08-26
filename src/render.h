#ifndef RED_RENDER
#define RED_RENDER
#include "decoder.h"
#include "gx_helper.h"

struct PBO_CUDA {
#ifndef __APPLE__
    GLuint pbo;
#if defined(RED_HAVE_CUDA)
    unsigned char *cuda_buffer;
    cudaGraphicsResource_t cuda_resource;
    size_t cuda_pbo_storage_buffer_size;
#endif
#endif
};

struct RenderScene {
    u32 num_cams;
    u32 *image_width;
    u32 *image_height;
    u32 size_of_buffer;
#ifdef __APPLE__
    ImTextureID *image_descriptor;  // ImTextureID per camera (Metal MTLTexture)
#else
    GLuint *image_texture;
#endif
    PBO_CUDA *pbo_cuda;
    PictureBuffer **display_buffer;
    SeekInfo *seek_context;
    bool use_cpu_buffer;
    // False on the software backend: frames arrive in host memory, there is no
    // CUDA context to register a PBO with, and upload is a plain
    // glTexSubImage2D. Resolved once in render_allocate_scene_memory and read
    // by the render loop and the teardown path, which must agree with it --
    // freeing a calloc'd buffer with cudaFree is the failure mode here.
    bool gpu_upload;
};

void render_initialize_target(gx_context *context);
void render_allocate_scene_memory(RenderScene *scene, u32 size_of_buffer);

#endif
