#pragma once
#include "imgui.h"
#include "implot.h"
#include <algorithm>

// ImPlot v1.0 obsoleted SetNextLineStyle(); item styling is now passed per-call
// via ImPlotSpec. This builds the common line-colour/weight case in a
// C++17-friendly way (designated initialisers would need C++20).
inline ImPlotSpec red_line_spec(const ImVec4 &col, float weight) {
    ImPlotSpec s;
    s.LineColor = col;
    s.LineWeight = weight;
    return s;
}

// Optional: tiny helper for inline help tooltips
inline void HelpMarker(const char *desc) {
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", desc);
}

// Overlay drawn INSIDE an ImPlot plot over a camera image whose current ring
// slot is a duplicate standing in for a frame the camera dropped (desync-fix
// mode): red border + badge, so held frames are never mistaken for real ones.
inline void DrawDroppedFrameBadge(float image_width, float image_height) {
    ImDrawList *dl = ImPlot::GetPlotDrawList();
    ImVec2 a = ImPlot::PlotToPixels(ImPlotPoint(0, 0));
    ImVec2 b = ImPlot::PlotToPixels(ImPlotPoint(image_width, image_height));
    ImVec2 mn(std::min(a.x, b.x), std::min(a.y, b.y));
    ImVec2 mx(std::max(a.x, b.x), std::max(a.y, b.y));
    const ImU32 red = IM_COL32(230, 60, 60, 255);
    dl->AddRect(mn, mx, red, 0.0f, 0, 3.0f);
    const char *msg = "DROPPED (frame held)";
    ImVec2 ts = ImGui::CalcTextSize(msg);
    ImVec2 tp(mn.x + 8.0f, mn.y + 8.0f);
    dl->AddRectFilled(ImVec2(tp.x - 4.0f, tp.y - 2.0f),
                      ImVec2(tp.x + ts.x + 4.0f, tp.y + ts.y + 2.0f),
                      IM_COL32(0, 0, 0, 160));
    dl->AddText(tp, red, msg);
}
