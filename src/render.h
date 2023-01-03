#ifndef RED_RENDER
#define RED_RENDER
#include "gx_helper.h"
#include "decoder.h"

struct render_scene
{
    u32 num_cams;
    u32 image_width;
    u32 image_height;
    u32 size_of_buffer;
    GLuint *image_texture;
    PictureBuffer **display_buffer;
    SeekInfo *seek_context;
};

void render_initialize_target(gx_context *context)
{
    GLFWwindow *render_target = gx_glfw_init_render_target(3, 3, context->width, context->height, "RED Labeling Tool", context->glsl_version);
    gx_init(context, render_target);
    gx_imgui_init(context);
}

static void render_allocate_scene_memory(render_scene *scene, u32 image_width, u32 image_height, u32 num_cams, u32 size_of_buffer)
{

    scene->num_cams = num_cams;
    scene->image_width = image_width;
    scene->image_height = image_height;
    scene->image_texture = (GLuint *)malloc(sizeof(GLuint) * num_cams);
    scene->size_of_buffer = size_of_buffer;

    for (u32 j = 0; j < num_cams; j++)
    {
        glGenTextures(1, &scene->image_texture[j]);
        glBindTexture(GL_TEXTURE_2D, scene->image_texture[j]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, scene->image_width, scene->image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        // Setup filtering parameters for display
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same
    }

    scene->seek_context = (SeekInfo *)malloc(sizeof(SeekInfo) * num_cams);
    for (u32 j = 0; j < num_cams; j++)
    {
        scene->seek_context[j].use_seek = false;
        scene->seek_context[j].seek_frame = 0;
        scene->seek_context[j].seek_done = false;
    }

    scene->display_buffer = (PictureBuffer **)malloc(num_cams * sizeof(PictureBuffer *));
    for (u32 j = 0; j < num_cams; j++)
    {
        scene->display_buffer[j] = (PictureBuffer *)malloc(size_of_buffer * sizeof(PictureBuffer));
    }

    unsigned int size_pic = image_width * image_height * 4 * sizeof(unsigned char);
    for (u32 j = 0; j < num_cams; j++)
    {
        for (u32 i = 0; i < size_of_buffer; i++)
        {
            scene->display_buffer[j][i].frame = (unsigned char *)malloc(size_pic);
            decoder_clear_buffer_with_constant_image(scene->display_buffer[j][i].frame, 3208, 2200);
            scene->display_buffer[j][i].frame_number = 0;
            scene->display_buffer[j][i].available_to_write = true;
        }
    }
}

#endif
