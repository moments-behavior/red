#ifndef GX_HELPER
#define GX_HELPER

#include "IconsForkAwesome.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "implot.h"
#include "types.h"
#include <GLFW/glfw3.h>
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include "vulkan_context.h"
#include "imgui_impl_vulkan.h"
#else
#include "imgui_impl_opengl3.h"
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#endif

typedef struct gx_context {
    u32 swap_interval;
    u32 width;
    u32 height;
    GLFWwindow *render_target;
    char *render_target_title;
    char *glsl_version;  // unused on macOS/Vulkan, kept for Linux compat
} gx_context;

static void gx_glfw_error_callback(int error, const char *description) {
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

#ifndef __APPLE__
static void gx_glew_error_callback(GLenum glew_error) {
    if (GLEW_OK != glew_error) {
        printf("GLEW error: %s\n", glewGetErrorString(glew_error));
    }
}
#endif

inline void gx_init(gx_context *context, GLFWwindow *render_target) {
    context->render_target = render_target;
#ifdef __APPLE__
    vk_init(render_target);
#else
    glfwMakeContextCurrent(render_target);
    gx_glew_error_callback(glewInit());
    glfwSwapInterval(context->swap_interval);
#endif
}

inline GLFWwindow *gx_glfw_init_render_target(u32 major_version,
                                              u32 minor_version, u32 width,
                                              u32 height, const char *title,
                                              char *glsl_version) {
    glfwSetErrorCallback(gx_glfw_error_callback);
    if (!glfwInit()) {
        printf("Could not initialize glfw!");
        exit(EXIT_FAILURE);
    }

#ifdef __APPLE__
    // Vulkan: no OpenGL context
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
#else
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    strcpy(glsl_version, "#version 130");
#endif

    GLFWwindow *window = glfwCreateWindow(1920, 1080, title, NULL, NULL);
    if (!window) {
        printf("Could not initialize window!");
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    return window;
}

inline void gx_imgui_init(gx_context *context) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlotContext *implotCtx = ImPlot::CreateContext();

    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    ImGui::StyleColorsClassic();

    ImGuiStyle &style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

#ifdef __APPLE__
    ImGui_ImplGlfw_InitForVulkan(context->render_target, true);
    vk_init_imgui();
#else
    ImGui_ImplGlfw_InitForOpenGL(context->render_target, true);
    ImGui_ImplOpenGL3_Init(context->glsl_version);
#endif

    io.Fonts->AddFontFromFileTTF("fonts/Roboto-Regular.ttf", 15.0f);
    static const ImWchar icons_ranges[] = {ICON_MIN_FK, ICON_MAX_16_FK, 0};
    ImFontConfig icons_config;
    icons_config.MergeMode = true;
    icons_config.PixelSnapH = true;
    io.Fonts->AddFontFromFileTTF("fonts/forkawesome-webfont.ttf", 15.0f,
                                 &icons_config, icons_ranges);
}

#ifndef __APPLE__
static void create_texture(GLuint *texture) {
    glGenTextures(1, texture);
    glBindTexture(GL_TEXTURE_2D, *texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

static void bind_texture(GLuint *texture) {
    glBindTexture(GL_TEXTURE_2D, *texture);
}

static void unbind_texture() { glBindTexture(GL_TEXTURE_2D, 0); }

static void create_pbo(GLuint *pbo, int image_width, int img_height) {
    glGenBuffers(1, pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER,
                 image_width * img_height * 4 * sizeof(unsigned char), 0,
                 GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

static void bind_pbo(GLuint *pbo) {
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
}

static void unbind_pbo() { glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); }

static void register_pbo_to_cuda(GLuint *pbo,
                                 cudaGraphicsResource_t *cuda_resource) {
    cudaGraphicsGLRegisterBuffer(cuda_resource, *pbo,
                                 cudaGraphicsRegisterFlagsNone);
}

static void map_cuda_resource(cudaGraphicsResource_t *cuda_resource) {
    cudaGraphicsMapResources(1, cuda_resource);
}

static void cuda_pointer_from_resource(unsigned char **cuda_buffer_p,
                                       size_t *size_p,
                                       cudaGraphicsResource_t *cuda_resource) {
    cudaGraphicsResourceGetMappedPointer((void **)cuda_buffer_p, size_p,
                                         *cuda_resource);
}

static void unmap_cuda_resource(cudaGraphicsResource_t *cuda_resource) {
    cudaGraphicsUnmapResources(1, cuda_resource);
}

static void upload_image_pbo_to_texture(int image_width, int img_height) {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, img_height, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, 0);
}

static void upload_texture(GLuint *image_texture, unsigned char *frame,
                           unsigned int width, unsigned int height) {
    bind_texture(image_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, frame);
    unbind_texture();
}
#endif // !__APPLE__

#endif // GX_HELPER
