#pragma once
#include "imgui.h"
#include <functional>

// Wraps ImGui Begin/End boilerplate for standard panels.
// content_fn is called only when the window is visible.
// always_fn (optional) is called every frame regardless (e.g. file dialog handlers).
inline void DrawPanel(const char *name, bool &open,
                      std::function<void()> content_fn,
                      std::function<void()> always_fn = nullptr,
                      ImVec2 default_size = ImVec2(500, 400),
                      ImGuiWindowFlags flags = 0) {
    // Always-run logic (file dialogs, etc.) executes regardless of visibility
    if (always_fn)
        always_fn();

    if (!open)
        return;

    ImGui::SetNextWindowSize(default_size, ImGuiCond_FirstUseEver);
    if (ImGui::Begin(name, &open, flags)) {
        content_fn();
    }
    ImGui::End();
}
