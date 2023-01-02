#ifndef RED_RENDER
#define RED_RENDER
#include "gx_helper.h"

void initialize_render_target(window_context *context) {
    GLFWwindow *render_target = glfw_init_render_target(3, 3, context->width, context->height, "RED Labeling Tool", context->glsl_version); 
    gx_init(context, render_target);
    gx_imgui_init(context);
}

#endif
