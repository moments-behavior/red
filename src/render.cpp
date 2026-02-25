#include "render.h"

void render_initialize_target(gx_context *context) {
    GLFWwindow *render_target = gx_glfw_init_render_target(
        3, 3, context->width, context->height, "Red", context->glsl_version);
    gx_init(context, render_target);
    gx_imgui_init(context);
}

void render_allocate_scene_memory(RenderScene *scene, u32 size_of_buffer) {
    int num_cams = scene->num_cams;
    scene->image_texture = (GLuint *)malloc(sizeof(GLuint) * num_cams);
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
        scene->display_buffer[j] =
            (PictureBuffer *)malloc(size_of_buffer * sizeof(PictureBuffer));
    }

    scene->pbo_cuda = (PBO_CUDA *)malloc(sizeof(PBO_CUDA) * num_cams);
    for (u32 j = 0; j < num_cams; j++) {
#ifndef __APPLE__
        create_pbo(&scene->pbo_cuda[j].pbo, scene->image_width[j],
                   scene->image_height[j]);
        register_pbo_to_cuda(&scene->pbo_cuda[j].pbo,
                             &scene->pbo_cuda[j].cuda_resource);
        map_cuda_resource(&scene->pbo_cuda[j].cuda_resource);
        cuda_pointer_from_resource(
            &scene->pbo_cuda[j].cuda_buffer,
            &scene->pbo_cuda[j].cuda_pbo_storage_buffer_size,
            &scene->pbo_cuda[j].cuda_resource);
#else
        scene->pbo_cuda[j].pbo = 0;
        scene->pbo_cuda[j].cuda_buffer = nullptr;
#endif
    }

    // allocate frame buffers
    for (u32 j = 0; j < num_cams; j++) {
        unsigned int size_pic = scene->image_width[j] * scene->image_height[j] *
                                4 * sizeof(unsigned char);
        for (u32 i = 0; i < size_of_buffer; i++) {
#ifdef __APPLE__
            // macOS always uses CPU buffers
            scene->display_buffer[j][i].frame =
                (unsigned char *)malloc(size_pic);
            decoder_clear_buffer_with_constant_image(
                scene->display_buffer[j][i].frame, scene->image_width[j],
                scene->image_height[j]);
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

    for (u32 j = 0; j < num_cams; j++) {
        glGenTextures(1, &scene->image_texture[j]);
        glBindTexture(GL_TEXTURE_2D, scene->image_texture[j]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, scene->image_width[j],
                     scene->image_height[j], 0, GL_RGBA, GL_UNSIGNED_BYTE,
                     NULL);
        // Setup filtering parameters for display
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                        GL_CLAMP_TO_EDGE); // This is required on WebGL for non
                                           // power-of-two textures
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,
                        GL_CLAMP_TO_EDGE); // Same
    }
}
