#include "render.h"
#include <cstdio>
#include <cstdlib>
#ifdef __APPLE__
#include "metal_context.h"
#endif

void render_initialize_target(gx_context *context) {
    GLFWwindow *render_target = gx_glfw_init_render_target(
        3, 3, context->width, context->height, "Red", context->glsl_version);
    gx_init(context, render_target);
    gx_imgui_init(context);
}

void render_allocate_scene_memory(RenderScene *scene, u32 size_of_buffer) {
    int num_cams = scene->num_cams;
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
    //
    // The frame rings are the largest allocation the app makes
    // (width*height*4 bytes per slot per camera — ~220 MiB per slot for 8K
    // video), so allocation failure is a realistic outcome here, and a
    // silent failure surfaces much later as an inscrutable
    // cudaErrorIllegalAddress in an unrelated CUDA call. Check every
    // allocation; if the GPU ring cannot fit, fall back to CPU buffers.
    const double GiB = 1024.0 * 1024.0 * 1024.0;
    size_t ring_bytes = 0;
    for (u32 j = 0; j < num_cams; j++)
        ring_bytes += (size_t)scene->image_width[j] * scene->image_height[j] *
                      4 * size_of_buffer;

    auto alloc_cpu_slot = [&](u32 j, u32 i, size_t size_pic) {
        scene->display_buffer[j][i].frame = (unsigned char *)malloc(size_pic);
        if (!scene->display_buffer[j][i].frame) {
            fprintf(stderr,
                    "[render] FATAL: cannot allocate frame buffer (camera %u, "
                    "slot %u of %u, %zu bytes; %.2f GiB total across %d "
                    "cameras). Reduce default_buffer_size in user settings.\n",
                    j, i, size_of_buffer, size_pic, ring_bytes / GiB,
                    num_cams);
            exit(1);
        }
        decoder_clear_buffer_with_constant_image(
            scene->display_buffer[j][i].frame, scene->image_width[j],
            scene->image_height[j]);
    };

#ifndef __APPLE__
    if (!scene->use_cpu_buffer) {
        // Pre-flight: leave at least 25% of currently-free VRAM for decoder
        // surfaces, conversion buffers, PBOs and textures.
        size_t free_b = 0, total_b = 0;
        cudaError_t merr = cudaMemGetInfo(&free_b, &total_b);
        if (merr != cudaSuccess || ring_bytes > free_b - free_b / 4) {
            fprintf(stderr,
                    "[render] GPU frame buffers need %.2f GiB but only %.2f "
                    "GiB VRAM is free; falling back to CPU frame buffers\n",
                    ring_bytes / GiB,
                    merr == cudaSuccess ? free_b / GiB : 0.0);
            scene->use_cpu_buffer = true;
        }
    }
#endif

    bool restart = true;
    while (restart) {
        restart = false;
        for (u32 j = 0; j < num_cams && !restart; j++) {
            size_t size_pic = (size_t)scene->image_width[j] *
                              scene->image_height[j] * 4;
            for (u32 i = 0; i < size_of_buffer && !restart; i++) {
#ifdef __APPLE__
                // macOS: CPU frame buffer still needed for image_loader path.
                // Video decode path (Phase 2/3) uses pixel_buffer instead.
                alloc_cpu_slot(j, i, size_pic);
                scene->display_buffer[j][i].pixel_buffer = nullptr;
#else
                if (scene->use_cpu_buffer) {
                    alloc_cpu_slot(j, i, size_pic);
                } else {
                    // gpu buffer
                    cudaError_t err = cudaMalloc(
                        (void **)&scene->display_buffer[j][i].frame, size_pic);
                    if (err != cudaSuccess) {
                        // Free the GPU slots allocated so far and restart the
                        // whole ring on the CPU: unload_media frees by the
                        // scene-wide use_cpu_buffer flag, so modes can't mix.
                        fprintf(stderr,
                                "[render] cudaMalloc failed (%s) for camera "
                                "%u slot %u (%zu bytes); falling back to CPU "
                                "frame buffers\n",
                                cudaGetErrorString(err), j, i, size_pic);
                        for (u32 jj = 0; jj <= j; jj++) {
                            u32 lim = (jj == j) ? i : size_of_buffer;
                            for (u32 ii = 0; ii < lim; ii++) {
                                cudaFree(scene->display_buffer[jj][ii].frame);
                                scene->display_buffer[jj][ii].frame = nullptr;
                            }
                        }
                        scene->use_cpu_buffer = true;
                        restart = true;
                        break;
                    }
                }
#endif
                scene->display_buffer[j][i].frame_number = -1;
                scene->display_buffer[j][i].available_to_write = true;
                scene->display_buffer[j][i].dropped = false;
            }
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
