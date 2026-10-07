#pragma once
// A picture loaded from disk and uploaded as a texture ImGui/ImPlot can draw,
// on whichever renderer this build uses: Metal on macOS, OpenGL elsewhere.
// Red's other textures belong to the camera views; this is for one-off images
// such as the Skeleton Creator's tracing background.

#include "imgui.h"
#include <string>

struct ImageTexture {
    ImTextureID id = (ImTextureID)0;
    int width = 0;
    int height = 0;
    bool valid() const { return id != (ImTextureID)0; }
};

// Load `path` (anything stb_image reads: JPG, PNG, BMP, ...) into *out,
// replacing and freeing whatever *out held. On failure *out is left empty and
// *err says why.
bool image_texture_load(const std::string &path, ImageTexture *out,
                        std::string *err);

// Upload w x h RGBA8 pixels (rows top to bottom) into *out, replacing and
// freeing whatever *out held.
bool image_texture_from_rgba(const unsigned char *rgba, int w, int h,
                             ImageTexture *out, std::string *err);

void image_texture_free(ImageTexture *tex);
