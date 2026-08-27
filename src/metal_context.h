#pragma once
#ifdef __APPLE__

#include <stdint.h>
#include "imgui.h"

#ifdef __cplusplus
extern "C" {
#endif

struct GLFWwindow;

// Phase 1: Metal init + RGBA CPU upload + frame loop
void metal_init(GLFWwindow *window);
void metal_init_imgui();
void metal_allocate_textures(int num_cams, uint32_t *widths, uint32_t *heights);
bool metal_begin_frame();   // acquires drawable, calls ImGui_ImplMetal_NewFrame
void metal_upload_texture(int cam_idx, const uint8_t *rgba, uint32_t w, uint32_t h);
ImTextureID metal_get_texture_id(int cam_idx);
void metal_end_frame();     // ImGui render + present + commit
void metal_cleanup();

// Phase 2: GPU NV12→RGBA via Metal compute (CVPixelBuffer path)
#include <CoreVideo/CoreVideo.h>
void metal_upload_pixelbuf(int cam_idx, CVPixelBufferRef pb, uint32_t w, uint32_t h);

// Phase 3: contrast/brightness compute shader (in-place on display texture)
void metal_apply_contrast_brightness(int cam_idx, float contrast, float brightness, bool pivot_midgray);

#ifdef __cplusplus
}
#endif

#endif // __APPLE__
