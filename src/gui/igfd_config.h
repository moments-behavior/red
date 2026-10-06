#pragma once
// red's ImGuiFileDialog settings (CUSTOM_IMGUIFILEDIALOG_CONFIG, set in
// CMakeLists.txt): the library's defaults, plus a name-field label a dialog
// can change -- Save Project asks for a folder name, not a file name.
#include "ImGuiFileDialogConfig.h"

// The label beside the name field; reset to "File Name:" by whoever changed it.
inline const char *&igfd_file_name_label() {
    static const char *label = "File Name:";
    return label;
}
// Used only as ImGui::Text(fileNameString), so it carries its own "%s".
#define fileNameString "%s", igfd_file_name_label()
