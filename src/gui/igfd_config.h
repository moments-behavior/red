#pragma once
// red's ImGuiFileDialog settings (CUSTOM_IMGUIFILEDIALOG_CONFIG, set in
// CMakeLists.txt): the library's defaults, plus hooks one dialog can set while
// it draws (Save Project): the name field's label, and a line under the name
// that can grey out OK. Whoever sets them resets them after Display().
#include "ImGuiFileDialogConfig.h"
#include <cstring>
#include <functional>
// First to include imgui.h here, so turn on what the library needs of it.
#ifndef IMGUI_DEFINE_MATH_OPERATORS
#define IMGUI_DEFINE_MATH_OPERATORS
#endif
#include <imgui.h>

struct IgfdHooks {
    const char *name_label = "File Name:";
    // Draws the line under the name field; false greys out OK.
    std::function<bool()> under_name;
    bool ok_blocked = false;   // under_name's last answer, read by the OK button
};
inline IgfdHooks &igfd_hooks() {
    static IgfdHooks h;
    return h;
}

// Used only as ImGui::Text(fileNameString), so it carries its own "%s".
#define fileNameString "%s", igfd_hooks().name_label

// The library reads this alignment just before drawing OK/Cancel, below the
// name field -- where the line goes. Text moves the buttons down a row.
inline float igfd_under_name() {
    IgfdHooks &h = igfd_hooks();
    h.ok_blocked = h.under_name && !h.under_name();
    return 0.0f;   // the library's default: buttons on the left
}
#define okCancelButtonAlignement igfd_under_name()

// Every button the library draws; OK is drawn greyed out when blocked.
inline bool igfd_button(const char *label, const ImVec2 &size = ImVec2(0, 0)) {
    if (igfd_hooks().ok_blocked && std::strstr(label, "##validationdialog") &&
        std::strncmp(label, "OK", 2) == 0) {
        ImGui::BeginDisabled();
        ImGui::Button(label, size);
        ImGui::EndDisabled();
        return false;
    }
    return ImGui::Button(label, size);
}
#define IMGUI_BUTTON igfd_button
