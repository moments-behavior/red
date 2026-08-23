#pragma once
// label_palette.h — role-named colors for label-state display, shared by the
// Labeling Tool (grid squares + timeline ticks) and the Frame Buffer window.
//
// One mutable process-wide instance: display code looks colors up by ROLE
// every frame, so the Settings panel can recolor states without the render
// sites knowing. Defaults match the historical hardcoded values (the old
// green/yellow/purple/teal/orange/lilac literals).
#include <imgui.h>
#include <map>
#include <string>
#include <vector>

struct LabelPalette {
    // Keypoint-label frame states (see KpState in labeling_tool_window.h)
    ImVec4 kp_complete{0.2f, 0.8f, 0.3f, 1.0f};    // all placed & triangulated
    ImVec4 kp_all_triangulated{0.63f, 0.35f, 0.86f, 1.0f}; // every placed keypoint triangulated, but not complete
    ImVec4 kp_needs_triangulation{0.95f, 0.85f, 0.15f, 1.0f}; // some placed keypoint untriangulated
    ImVec4 kp_partial{0.2f, 0.7f, 0.7f, 1.0f};     // Frame Buffer: labeled but incomplete
    ImVec4 needs_improvement{0.90f, 0.28f, 0.28f, 1.0f}; // promoted prediction awaiting a manual fix
    // Other annotation types
    ImVec4 mask{0.9f, 0.55f, 0.12f, 1.0f};         // SAM masks
    ImVec4 bbox{0.63f, 0.35f, 0.86f, 1.0f};        // axis-aligned boxes
    ImVec4 obb{0.78f, 0.59f, 1.0f, 1.0f};          // oriented boxes
};

inline LabelPalette &label_palette() {
    static LabelPalette p;
    return p;
}

// The editable roles, for the Settings panel and user-settings persistence.
// `key` is the stable identifier used in settings JSON — never rename one.
struct LabelPaletteRole {
    const char *key;   // stable settings key
    const char *label; // Settings-panel label
    ImVec4 LabelPalette::*color;
};

inline const std::vector<LabelPaletteRole> &label_palette_roles() {
    static const std::vector<LabelPaletteRole> roles = {
        {"kp_complete", "Keypoints: complete", &LabelPalette::kp_complete},
        {"kp_all_triangulated", "Keypoints: all placed triangulated",
         &LabelPalette::kp_all_triangulated},
        {"kp_needs_triangulation", "Keypoints: needs triangulation",
         &LabelPalette::kp_needs_triangulation},
        {"kp_partial", "Keypoints: partial (Frame Buffer)",
         &LabelPalette::kp_partial},
        {"needs_improvement", "Needs improvement", &LabelPalette::needs_improvement},
        {"mask", "SAM masks", &LabelPalette::mask},
        {"bbox", "Bounding boxes", &LabelPalette::bbox},
        {"obb", "Oriented boxes", &LabelPalette::obb},
    };
    return roles;
}

// Apply user-settings overrides (role key -> RGB triplet) onto the palette.
// Roles absent from the map keep their defaults; alpha is always 1.
inline void apply_label_color_overrides(
    const std::map<std::string, std::vector<float>> &overrides) {
    LabelPalette &pal = label_palette();
    for (const auto &role : label_palette_roles()) {
        auto it = overrides.find(role.key);
        if (it != overrides.end() && it->second.size() >= 3)
            pal.*role.color =
                ImVec4(it->second[0], it->second[1], it->second[2], 1.0f);
    }
}
