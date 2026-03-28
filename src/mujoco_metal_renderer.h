#pragma once
// mujoco_metal_renderer.h — Offscreen Metal rendering of MuJoCo body models
//
// Uses mjv_updateScene() to extract abstract geometry (no OpenGL needed),
// then renders capsules/spheres/boxes with Metal shaders into an offscreen
// texture that can be displayed via ImGui::Image().

#ifdef __APPLE__
#ifdef RED_HAS_MUJOCO

#include "imgui.h"
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
// show_skin: render the smooth skin mesh (if model has one).
// show_sites: render keypoint site markers.
// show_arena: render a 1828mm arena floor plane with ramp.
void mujoco_renderer_render(MujocoRenderer *r, MujocoContext *mj,
                            const float lookat[3], float distance,
                            float azimuth, float elevation,
                            bool show_skin = true,
                            bool show_sites = true,
                            bool show_arena = true);

// Get the rendered texture as an ImGui texture ID.
ImTextureID mujoco_renderer_get_texture(MujocoRenderer *r);

// Get current dimensions
void mujoco_renderer_get_size(MujocoRenderer *r, uint32_t *w, uint32_t *h);

#endif // RED_HAS_MUJOCO
#endif // __APPLE__
