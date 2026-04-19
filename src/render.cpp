#include "render.h"
#ifdef __APPLE__
#include "metal_context.h"
#else
#include <cuda_runtime.h>
#include <cstdio>
#endif

void render_initialize_target(gx_context *context) {
    GLFWwindow *render_target = gx_glfw_init_render_target(
        3, 3, context->width, context->height, "Red", context->glsl_version);
    gx_init(context, render_target);
    gx_imgui_init(context);
}

void render_allocate_scene_memory(RenderScene *scene, u32 size_of_buffer) {
    int num_cams = scene->num_cams;

#ifndef __APPLE__
    // VRAM budget check: if the requested display buffer would exceed ~70% of
    // free GPU memory, fall back to CPU-hosted buffers (or shrink the buffer).
    // Otherwise many cameras × high-res × size_of_buffer (e.g. 16 × 3216×2208
    // × 64 slots ≈ 29 GB) OOM the GPU and NvDecoder allocations fail later
    // with opaque "ILLEGAL_ADDRESS" errors.
    {
        size_t per_frame_bytes = 0;
        for (int j = 0; j < num_cams; j++) {
            per_frame_bytes += (size_t)scene->image_width[j] *
                               scene->image_height[j] * 4;
        }
        size_t needed = per_frame_bytes * size_of_buffer;

        size_t free_bytes = 0, total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        // Reserve budget for NvDecoder surfaces, PBOs, textures, JARVIS/ONNX,
        // MuJoCo, etc. Allow at most 70% of currently-free VRAM for the display
        // buffer allocations below.
        size_t budget = (size_t)(free_bytes * 0.70);

        if (!scene->use_cpu_buffer && needed > budget) {
            fprintf(stderr,
                    "[render] Display buffer would need %.2f GB of VRAM "
                    "(budget %.2f GB free of %.2f GB total); falling back "
                    "to CPU-hosted frames. Reduce default_buffer_size in "
                    "settings to keep frames on GPU.\n",
                    needed / 1e9, budget / 1e9, total_bytes / 1e9);
            scene->use_cpu_buffer = true;
        }
    }
#endif

    scene->size_of_buffer = size_of_buffer;

    scene->seek_context = (SeekInfo *)malloc(sizeof(SeekInfo) * num_cams);
    for (u32 j = 0; j < num_cams; j++) {
        scene->seek_context[j].use_seek = false;
        scene->seek_context[j].seek_frame = 0;
        scene->seek_context[j].seek_done = false;
    }

    scene->display_buffer =
        (PictureBuffer **)malloc(num_cams * sizeof(PictureBuffer *));

    for (u32 j = 0; j < num_cams; j++) {
        scene->display_buffer[j] = new PictureBuffer[size_of_buffer]();
    }

    scene->pbo_cuda = (PBO_CUDA *)malloc(sizeof(PBO_CUDA) * num_cams);
#ifndef __APPLE__
    for (u32 j = 0; j < num_cams; j++) {
        create_pbo(&scene->pbo_cuda[j].pbo, scene->image_width[j],
                   scene->image_height[j]);
        register_pbo_to_cuda(&scene->pbo_cuda[j].pbo,
                             &scene->pbo_cuda[j].cuda_resource);
        map_cuda_resource(&scene->pbo_cuda[j].cuda_resource);
        cuda_pointer_from_resource(
            &scene->pbo_cuda[j].cuda_buffer,
            &scene->pbo_cuda[j].cuda_pbo_storage_buffer_size,
            &scene->pbo_cuda[j].cuda_resource);
    }
#endif

    // allocate frame buffers
    for (u32 j = 0; j < num_cams; j++) {
        unsigned int size_pic = scene->image_width[j] * scene->image_height[j] *
                                4 * sizeof(unsigned char);
        for (u32 i = 0; i < size_of_buffer; i++) {
#ifdef __APPLE__
            // macOS: CPU frame buffer still needed for image_loader path.
            // Video decode path (Phase 2/3) uses pixel_buffer instead.
            scene->display_buffer[j][i].frame =
                (unsigned char *)malloc(size_pic);
            decoder_clear_buffer_with_constant_image(
                scene->display_buffer[j][i].frame, scene->image_width[j],
                scene->image_height[j]);
            scene->display_buffer[j][i].pixel_buffer = nullptr;
#else
            if (scene->use_cpu_buffer) {
                scene->display_buffer[j][i].frame =
                    (unsigned char *)malloc(size_pic);
                decoder_clear_buffer_with_constant_image(
                    scene->display_buffer[j][i].frame, scene->image_width[j],
                    scene->image_height[j]);
            } else {
                // gpu buffer
                cudaMalloc((void **)&scene->display_buffer[j][i].frame,
                           size_pic);
            }
#endif
            scene->display_buffer[j][i].frame_number = -1;
            scene->display_buffer[j][i].available_to_write = true;
        }
    }

#ifdef __APPLE__
    // Metal: create per-camera RGBA output textures used as ImTextureID
    metal_allocate_textures(num_cams, scene->image_width, scene->image_height);
    scene->image_descriptor =
        (ImTextureID *)malloc(sizeof(ImTextureID) * num_cams);
    for (int j = 0; j < num_cams; j++)
        scene->image_descriptor[j] = metal_get_texture_id(j);
#else
    scene->image_texture = (GLuint *)malloc(sizeof(GLuint) * num_cams);
    for (u32 j = 0; j < num_cams; j++) {
        glGenTextures(1, &scene->image_texture[j]);
        glBindTexture(GL_TEXTURE_2D, scene->image_texture[j]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, scene->image_width[j],
                     scene->image_height[j], 0, GL_RGBA, GL_UNSIGNED_BYTE,
                     NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
#endif
}
