#ifndef GL_HELPER
#define GL_HELPER
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <GL/glew.h>
#include <cuda_gl_interop.h>

typedef uint32_t u32;

typedef struct window_context
{
    u32 swap_interval;
    u32 width;
    u32 height;
    GLFWwindow *render_target;
} window_context;


static void glfw_error_callback(int error, const char *description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

static void glew_error_callback(GLenum glew_error)
{
    if (GLEW_OK != glew_error)
    {
        printf("GLEW error: %s\n", glewGetErrorString(glew_error));
    }
}

void gx_init(window_context *context, GLFWwindow *render_target){
    context->render_target = render_target;    
    glfwMakeContextCurrent(render_target);
    glew_error_callback(glewInit());
    glfwSwapInterval(1); // Enable vsync
}


GLFWwindow* glfw_init_render_target(u32 marjor_version, u32 minor_version, u32 width, u32 height, const char *title)
{
    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
    {
        printf("Could not initialize glfw!");
        exit(EXIT_FAILURE);
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

    // Create window with graphics context
    GLFWwindow *window = glfwCreateWindow(1920, 1080, title, NULL, NULL);
    if (!window)
    {
        printf("Could not initialize window!");
        glfwTerminate();
        exit(EXIT_FAILURE);
    };

    return window;
}

static void create_texture(GLuint *texture)
{
    // Create a OpenGL texture identifier
    glGenTextures(1, texture);
    glBindTexture(GL_TEXTURE_2D, *texture);

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

static void bind_texture(GLuint *texture)
{
    glBindTexture(GL_TEXTURE_2D, *texture);
}

static void unbind_texture()
{
    glBindTexture(GL_TEXTURE_2D, 0);
}

static void create_pbo(GLuint *pbo)
{
    glGenBuffers(1, pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 3208 * 2200 * 4 * sizeof(unsigned char), 0, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

static void bind_pbo(GLuint *pbo)
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
}

static void unbind_pbo()
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

static void register_pbo_to_cuda(GLuint *pbo, cudaGraphicsResource_t *cuda_resource)
{
    cudaGraphicsGLRegisterBuffer(cuda_resource, *pbo, cudaGraphicsRegisterFlagsNone);
}

static void map_cuda_resource(cudaGraphicsResource_t *cuda_resource)
{
    cudaGraphicsMapResources(1, cuda_resource);
}

static void cuda_pointer_from_resource(unsigned char **cuda_buffer_p, size_t *size_p, cudaGraphicsResource_t *cuda_resource)
{
    cudaGraphicsResourceGetMappedPointer((void **)cuda_buffer_p, size_p, *cuda_resource);
}

static void unmap_cuda_resource(cudaGraphicsResource_t *cuda_resource)
{
    cudaGraphicsUnmapResources(1, cuda_resource);
}

static void upload_image_pbo_to_texture()
{
    // Assume PBO is bound before this, therefore the last
    // argument is an offset into the PBO, not a pointer to a
    // buffer stored in CPU memory
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 3208, 2200, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
}

#endif
