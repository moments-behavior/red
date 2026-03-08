#pragma once
#include "imgui.h"
#include <functional>
#include <string>
#include <vector>

struct PopupEntry {
    std::string title;
    std::string message;
    enum Type { Error, Info, Confirm } type = Error;
    std::function<void()> on_confirm; // for Confirm type
};

struct PopupStack {
    std::vector<PopupEntry> pending;
    PopupEntry active;
    bool has_active = false;

    void pushError(const std::string &msg) {
        pending.push_back({"Error", msg, PopupEntry::Error, nullptr});
    }

    void pushInfo(const std::string &title, const std::string &msg) {
        pending.push_back({title, msg, PopupEntry::Info, nullptr});
    }

    void pushConfirm(const std::string &title, const std::string &msg,
                     std::function<void()> on_confirm) {
        pending.push_back({title, msg, PopupEntry::Confirm, std::move(on_confirm)});
    }
};

// Call once per frame, after all window drawing, before ImGui::Render().
inline void drawPopups(PopupStack &stack) {
    // Activate the next pending popup if none is active
    if (!stack.has_active && !stack.pending.empty()) {
        stack.active = std::move(stack.pending.front());
        stack.pending.erase(stack.pending.begin());
        stack.has_active = true;
        ImGui::OpenPopup(stack.active.title.c_str());
    }

    if (!stack.has_active)
        return;

    if (ImGui::BeginPopupModal(stack.active.title.c_str(), NULL,
                               ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("%s", stack.active.message.c_str());
        ImGui::Separator();

        if (stack.active.type == PopupEntry::Confirm) {
            if (ImGui::Button("OK")) {
                if (stack.active.on_confirm)
                    stack.active.on_confirm();
                ImGui::CloseCurrentPopup();
                stack.has_active = false;
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel")) {
                ImGui::CloseCurrentPopup();
                stack.has_active = false;
            }
        } else {
            if (ImGui::Button("OK")) {
                ImGui::CloseCurrentPopup();
                stack.has_active = false;
            }
        }

        ImGui::EndPopup();
    }
}
