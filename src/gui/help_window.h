#pragma once
#include "gui/panel.h"

inline void DrawHelpWindow(bool &show) {
    drawPanel("Help Menu", show, []() {
        ImGui::SeparatorText("General");
        ImGui::Text("<h>: toggle this help menu");
        ImGui::Text("<Space>: toggle play and pause");
        ImGui::Text("<Left Arrow>    : Seek backward");
        ImGui::Text("<Shift+Left>    : Seek backward (x10)");
        ImGui::Text("<Right Arrow>   : Seek forward");
        ImGui::Text("<Shift+Right>   : Seek forward (x10)");
        ImGui::Text("<t>: triangulate");
        ImGui::Text("<Ctrl+s>: Save labels");

        ImGui::SeparatorText("When paused");
        ImGui::Text("<,>: previous image in buffer");
        ImGui::Text("<.>: next image in buffer");

        ImGui::SeparatorText("While hovering image");
        ImGui::Text("<b>: create keypoints on frame");
        ImGui::Text("<w>: drop active keypoint");
        ImGui::Text("<a>: active keypoint-- ");
        ImGui::Text("<d>: active keypoint++");
        ImGui::Text("<q>: active keypoint set to first node");
        ImGui::Text("<e>: active keypoint set to last node");
        ImGui::Text("<Backspace>: delete all keypoints");

        ImGui::SeparatorText("While hovering keypoints");
        ImGui::Text("<r>: delete active keypoint");
        ImGui::Text("<f>: delete active keypoint on all cameras");
        ImGui::Text("Click keypoint to activate it");
    });
}
