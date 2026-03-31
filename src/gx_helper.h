#ifndef GX_HELPER
#define GX_HELPER

#include "IconsForkAwesome.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "implot.h"
#include "implot3d.h"
#include "types.h"
#include <cstdio>
#include <filesystem>
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <GLFW/glfw3.h>
#include "metal_context.h"
// imgui_impl_metal.h is ObjC-only; all Metal ImGui calls go through metal_context.mm
#else
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "imgui_impl_opengl3.h"
#include <cuda_gl_interop.h>
#endif

typedef struct gx_context {
    u32 swap_interval;
    u32 width;
    u32 height;
    GLFWwindow *render_target;
    char *render_target_title;
    char *glsl_version;  // unused on macOS/Metal, kept for Linux compat
    std::string exe_dir; // absolute path to directory containing the binary
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
    metal_init(render_target);
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
    // Metal: no OpenGL context needed; CAMetalLayer handles presentation
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
    ImPlot3D::CreateContext();

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
    ImGui_ImplGlfw_InitForOther(context->render_target, true);
    metal_init_imgui();
    // Note: macOS screen recording modifier key fix is in
    // imgui_impl_glfw_patched.cpp (CoreGraphics hardware key state query).
#else
    ImGui_ImplGlfw_InitForOpenGL(context->render_target, true);
    ImGui_ImplOpenGL3_Init(context->glsl_version);
#endif

    // Use absolute paths so fonts and ini work regardless of cwd
    static std::string ini_path;
    ini_path = context->exe_dir + "/imgui.ini";

    // Always reset to the shipped default layout on launch so that new projects
    // start with a clean arrangement rather than inheriting stale window positions.
    // Per-project layouts are handled separately by switch_ini_to_project().
    for (const auto &candidate : {
            context->exe_dir + "/../default_imgui_layout.ini",           // dev build
            context->exe_dir + "/../share/red/default_imgui_layout.ini", // Homebrew
        }) {
        if (std::filesystem::exists(candidate)) {
            std::filesystem::copy_file(candidate, ini_path,
                std::filesystem::copy_options::overwrite_existing);
            break;
        }
    }

    io.IniFilename = ini_path.c_str();

    // Search for the fonts directory in candidate locations so the binary
    // works both from a development build (./release/red → ./fonts) and
    // from a Homebrew install (/opt/homebrew/bin/red → /opt/homebrew/share/red/fonts).
    std::string font_dir;
    for (const auto &candidate : {
            context->exe_dir + "/../fonts",            // dev build
            context->exe_dir + "/../share/red/fonts",  // Homebrew install
        }) {
        if (std::filesystem::exists(candidate + "/Roboto-Regular.ttf")) {
            font_dir = candidate;
            break;
        }
    }
    if (font_dir.empty()) {
        fprintf(stderr, "[RED] Could not find fonts directory (searched relative to %s)\n",
                context->exe_dir.c_str());
        font_dir = context->exe_dir + "/../fonts"; // best-effort fallback
    }
    io.Fonts->AddFontFromFileTTF((font_dir + "/Roboto-Regular.ttf").c_str(), 15.0f);
    static const ImWchar icons_ranges[] = {ICON_MIN_FK, ICON_MAX_16_FK, 0};
    ImFontConfig icons_config;
    icons_config.MergeMode = true;
    icons_config.PixelSnapH = true;
    io.Fonts->AddFontFromFileTTF((font_dir + "/forkawesome-webfont.ttf").c_str(), 15.0f,
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
