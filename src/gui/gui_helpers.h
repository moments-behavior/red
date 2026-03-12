#pragma once
#include "imgui.h"

// Optional: tiny helper for inline help tooltips
inline void HelpMarker(const char *desc) {
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", desc);
}
