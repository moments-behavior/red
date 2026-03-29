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

// Create/destroy the renderer. Uses the system default MTLDevice.
MujocoRenderer *mujoco_renderer_create(uint32_t width, uint32_t height);
void            mujoco_renderer_destroy(MujocoRenderer *r);
void            mujoco_renderer_resize(MujocoRenderer *r, uint32_t width, uint32_t height);

// Render the current MuJoCo scene to the offscreen texture.
// Call after mj_forward() has been called (site positions are up to date).
// camera_lookat/distance/azimuth/elevation control the viewpoint.
// Render using a MuJoCo camera (pass the mjvCamera directly).
void mujoco_renderer_render(MujocoRenderer *r, MujocoContext *mj,
                            mjvCamera *cam,
                            bool show_skin = true,
                            bool show_bodies = true,
                            bool show_sites = true,
                            bool show_arena = true);

// Get the rendered texture as an ImGui texture ID.
ImTextureID mujoco_renderer_get_texture(MujocoRenderer *r);

// Get current dimensions
void mujoco_renderer_get_size(MujocoRenderer *r, uint32_t *w, uint32_t *h);

#endif // RED_HAS_MUJOCO
#endif // __APPLE__
