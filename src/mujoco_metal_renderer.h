#pragma once
// mujoco_metal_renderer.h — Offscreen Metal rendering of MuJoCo body models
//
// Uses mjv_updateScene() to extract abstract geometry (no OpenGL needed),
// then renders capsules/spheres/boxes with Metal shaders into an offscreen
// texture that can be displayed via ImGui::Image().

#ifdef __APPLE__
#ifdef RED_HAS_MUJOCO

#include "imgui.h"
#include <mujoco.h>
#include <stdint.h>

struct MujocoContext;

// Opaque handle — implementation in .mm file
struct MujocoRenderer;

// Optional direct view/projection override (bypasses mjvCamera spherical coords).
// Used for calibration camera views where roll angle matters.
struct ViewOverride {
    bool active = false;
    float view[16];  // 4x4 column-major view matrix
    float proj[16];  // 4x4 column-major projection matrix
    float eye[3];    // eye position (for lighting)
};

// Create/destroy the renderer. Uses the system default MTLDevice.
MujocoRenderer *mujoco_renderer_create(uint32_t width, uint32_t height);
void            mujoco_renderer_destroy(MujocoRenderer *r);
void            mujoco_renderer_resize(MujocoRenderer *r, uint32_t width, uint32_t height);

// Render the current MuJoCo scene to the offscreen texture.
// If view_override is active, uses its view/proj matrices instead of deriving from cam.
// bg_texture: optional video frame to draw as background (pass 0 for none).
// Must be a Metal texture (id<MTLTexture> cast to void*).
void mujoco_renderer_render(MujocoRenderer *r, MujocoContext *mj,
                            mjvCamera *cam,
                            bool show_skin = true,
                            bool show_bodies = true,
                            bool show_sites = true,
                            bool show_arena = true,
                            const ViewOverride *view_override = nullptr,
                            bool show_arena_corners = false,
                            void *bg_texture = nullptr,
                            float scene_opacity = 1.0f,
                            float bg_zoom = 1.0f,
                            const float *bg_pan = nullptr,
                            float arena_width = 1.828f,
                            float arena_depth = 1.828f,
                            const float *arena_offset = nullptr,
                            float brightness = 0.0f,
                            float contrast = 1.0f);

// Get the rendered texture as an ImGui texture ID.
ImTextureID mujoco_renderer_get_texture(MujocoRenderer *r);

// Get current dimensions
void mujoco_renderer_get_size(MujocoRenderer *r, uint32_t *w, uint32_t *h);

#endif // RED_HAS_MUJOCO
#endif // __APPLE__
