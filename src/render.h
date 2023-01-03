#ifndef RED_RENDER
#define RED_RENDER
#include "gx_helper.h"
#include "decoder.h"

struct scene {
    int num_cams;
    int size_pic;
    const int size_of_buffer;
    GLuint* image_texture;
    PictureBuffer** display_buffer;
    SeekInfo* seek_context;
};


void initialize_render_target(window_context *context) {
    GLFWwindow *render_target = glfw_init_render_target(3, 3, context->width, context->height, "RED Labeling Tool", context->glsl_version); 
    gx_init(context, render_target);
    gx_imgui_init(context);
}

#endif
