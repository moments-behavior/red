#pragma once
#include "imgui.h"
#include <chrono>
#include <string>
#include <vector>

struct Toast {
    enum Level { Info, Success, Warning, Error };
    std::string message;
    Level level = Info;
    std::chrono::steady_clock::time_point created;
    float duration_sec = 4.0f;
};

struct ToastQueue {
    std::vector<Toast> toasts;

    void push(const std::string &msg, Toast::Level level = Toast::Info,
              float duration = 4.0f) {
        toasts.push_back({msg, level,
                          std::chrono::steady_clock::now(), duration});
    }

    void pushSuccess(const std::string &msg) {
        push(msg, Toast::Success, 5.0f);
    }

    void pushError(const std::string &msg) {
        push(msg, Toast::Error, 8.0f);
    }

    size_t size() const { return toasts.size(); }
};

// Call once per frame. Renders bottom-right, stacked upward, with fade-out.
inline void drawToasts(ToastQueue &queue) {
    if (queue.toasts.empty())
        return;

    auto now = std::chrono::steady_clock::now();
    const float fade_duration = 0.5f;

    // Get main viewport for positioning
    const ImGuiViewport *vp = ImGui::GetMainViewport();
    float padding = 16.0f;
    float y_offset = padding;

    // Iterate in reverse so newest are at the bottom
    for (int i = (int)queue.toasts.size() - 1; i >= 0; i--) {
        auto &t = queue.toasts[i];
        float elapsed = std::chrono::duration<float>(now - t.created).count();
        if (elapsed > t.duration_sec) {
            queue.toasts.erase(queue.toasts.begin() + i);
            continue;
        }

        // Alpha: 1.0 during display, fade out in last fade_duration seconds
        float alpha = 1.0f;
        float remaining = t.duration_sec - elapsed;
        if (remaining < fade_duration)
            alpha = remaining / fade_duration;

        // Color based on level
        ImVec4 col;
        switch (t.level) {
        case Toast::Success: col = ImVec4(0.1f, 0.7f, 0.1f, alpha); break;
        case Toast::Warning: col = ImVec4(0.9f, 0.7f, 0.0f, alpha); break;
        case Toast::Error:   col = ImVec4(0.9f, 0.2f, 0.2f, alpha); break;
        default:             col = ImVec4(0.2f, 0.2f, 0.2f, alpha); break;
        }

        // Calculate text size to position window
        ImVec2 text_size = ImGui::CalcTextSize(t.message.c_str());
        float win_w = text_size.x + 24.0f;
        float win_h = text_size.y + 16.0f;

        ImVec2 pos(vp->WorkPos.x + vp->WorkSize.x - win_w - padding,
                   vp->WorkPos.y + vp->WorkSize.y - win_h - y_offset);

        char id[32];
        snprintf(id, sizeof(id), "##toast_%d", i);

        ImGui::SetNextWindowPos(pos);
        ImGui::SetNextWindowSize(ImVec2(win_w, win_h));
        ImGui::SetNextWindowBgAlpha(alpha * 0.85f);

        ImGuiWindowFlags flags =
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
            ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoFocusOnAppearing |
            ImGuiWindowFlags_NoNav | ImGuiWindowFlags_AlwaysAutoResize |
            ImGuiWindowFlags_NoSavedSettings;

        ImGui::PushStyleColor(ImGuiCol_WindowBg, col);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 6.0f);
        if (ImGui::Begin(id, nullptr, flags)) {
            ImGui::PushStyleColor(ImGuiCol_Text,
                                  ImVec4(1.0f, 1.0f, 1.0f, alpha));
            ImGui::TextUnformatted(t.message.c_str());
            ImGui::PopStyleColor();
        }
        ImGui::End();
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();

        y_offset += win_h + 4.0f;
    }
}
