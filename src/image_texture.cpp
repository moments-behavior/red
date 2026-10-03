#include "image_texture.h"

#include "../lib/ImGuiFileDialog/stb/stb_image.h"  // implementation in red.cpp

#ifdef __APPLE__
#include "metal_context.h"
#else
#include <GL/glew.h>
#endif

#include <cstdint>

void image_texture_free(ImageTexture *tex) {
    if (!tex || !tex->valid()) return;
#ifdef __APPLE__
    metal_release_image_texture(tex->id);
#else
    GLuint name = (GLuint)(intptr_t)tex->id;
    glDeleteTextures(1, &name);
#endif
    *tex = ImageTexture{};
}

bool image_texture_load(const std::string &path, ImageTexture *out,
                        std::string *err) {
    image_texture_free(out);
    int w = 0, h = 0, channels = 0;
    unsigned char *rgba = stbi_load(path.c_str(), &w, &h, &channels, 4);
    if (!rgba) {
        if (err) *err = std::string("Could not read image: ") + stbi_failure_reason();
        return false;
    }
#ifdef __APPLE__
    ImTextureID id = metal_create_image_texture(rgba, (uint32_t)w, (uint32_t)h);
#else
    GLuint name = 0;
    glGenTextures(1, &name);
    glBindTexture(GL_TEXTURE_2D, name);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                 rgba);
    glBindTexture(GL_TEXTURE_2D, 0);
    ImTextureID id = (ImTextureID)(intptr_t)name;
#endif
    stbi_image_free(rgba);
    if (!id) {
        if (err) *err = "Could not create a texture for " + path;
        return false;
    }
    out->id = id;
    out->width = w;
    out->height = h;
    return true;
}
