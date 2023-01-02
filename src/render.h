#ifndef RED_RENDER
#define RED_RENDER
#include "gl_helper.h"

void initialize_render_target(window_context *window_c) {
    GLFWwindow *render_target = glfw_init_render_target(3, 3, window_c->width, window_c->height, "RED Labeling Tool");
    gx_init(window_c, render_target);
}

#endif
