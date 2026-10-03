#pragma once
// Skeleton Creator — draw a skeleton by placing nodes and joining them, and
// write it out as the .json red loads elsewhere.
//
// Ported from the fetch_paper branch, where it lived as ~300 lines inline in
// red.cpp's render loop. The logic is unchanged; it is a window of its own
// here, with its state in a struct rather than in main().
//
// The background-image tracing from that version is NOT carried over: it
// uploaded through a raw GLuint, and this tree renders through Metal on macOS
// as well, so it needs a backend-neutral texture path first.

#include "app_context.h"
#include "gui/panel.h"
#include "gui/gui_helpers.h"
#include "gui/shortcuts.h"
#include "image_texture.h"
#include "video_files.h"
#include "imgui.h"
#include "implot.h"
#include "json.hpp"
#include "misc/cpp/imgui_stdlib.h"
#include <ImGuiFileDialog.h>

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

struct SkeletonCreatorNode {
    ImPlotPoint position{0.5, 0.5};
    std::string name;
    ImVec4 color{1, 1, 1, 1};
    int id = -1;

    SkeletonCreatorNode() = default;
    SkeletonCreatorNode(double x, double y, int node_id)
        : position(x, y), id(node_id) {
        name = "Node" + std::to_string(node_id);
        color = (ImVec4)ImColor::HSV(node_id / 10.0f, 1.0f, 1.0f);
    }
};

struct SkeletonCreatorEdge {
    int node1_id = -1;
    int node2_id = -1;
    SkeletonCreatorEdge() = default;
    SkeletonCreatorEdge(int a, int b) : node1_id(a), node2_id(b) {}
};

struct SkeletonCreatorState {
    bool show = false;
    std::vector<SkeletonCreatorNode> nodes;
    std::vector<SkeletonCreatorEdge> edges;
    int next_node_id = 0;
    int selected_for_edge = -1;
    std::string name = "CustomSkeleton";
    std::string status;
    // Right-click menu: the node it is for, and a request to open it raised
    // inside the plot and acted on after EndPlot, where the popup lives.
    int menu_node = -1;
    bool open_menu = false;
    // Height of the drawing pad (and the nodes table beside it), in pixels.
    // 0 = fit the window: take whatever the help text below does not need,
    // so none of it is pushed out of view. Dragging the splitter under them
    // sets a height; double-click goes back to 0.
    float editor_height = 0.0f;
    // A picture to trace the skeleton over, drawn behind the nodes and
    // scaled to fit the pad with its proportions kept.
    ImageTexture background;
    float background_opacity = 0.5f;
    // Show the whole pad (0..1) next frame -- on first open and on Reset View.
    bool reset_view = true;
};

// The .json red loads: names and edges by index, plus positions so this window
// can reopen its own output with the layout intact.
inline nlohmann::json skeleton_creator_to_json(const SkeletonCreatorState &st) {
    nlohmann::json j;
    j["name"] = st.name;
    j["has_skeleton"] = true;
    j["num_nodes"] = (int)st.nodes.size();

    std::vector<std::string> names;
    std::vector<std::vector<double>> positions;
    for (const auto &n : st.nodes) {
        names.push_back(n.name);
        positions.push_back({n.position.x, n.position.y});
    }
    j["node_names"] = names;
    // Not read back by load_skeleton_json, which only wants names and edges --
    // kept so this window can reopen its own output with the layout intact
    // rather than restacking it in a line.
    j["node_positions"] = positions;

    std::vector<std::vector<int>> edges_out;
    for (const auto &e : st.edges) {
        int a = -1, b = -1;
        for (size_t i = 0; i < st.nodes.size(); i++) {
            if (st.nodes[i].id == e.node1_id) a = (int)i;
            if (st.nodes[i].id == e.node2_id) b = (int)i;
        }
        if (a >= 0 && b >= 0) edges_out.push_back({a, b});
    }
    j["edges"] = edges_out;
    j["num_edges"] = (int)edges_out.size();
    return j;
}

inline void skeleton_creator_delete_node(SkeletonCreatorState &st, int id) {
    st.nodes.erase(std::remove_if(st.nodes.begin(), st.nodes.end(),
                                  [id](const SkeletonCreatorNode &n) {
                                      return n.id == id;
                                  }),
                   st.nodes.end());
    st.edges.erase(std::remove_if(st.edges.begin(), st.edges.end(),
                                  [id](const SkeletonCreatorEdge &e) {
                                      return e.node1_id == id || e.node2_id == id;
                                  }),
                   st.edges.end());
    if (st.selected_for_edge == id) st.selected_for_edge = -1;
}

// Join a and b, or unjoin them if they already are.
inline void skeleton_creator_toggle_edge(SkeletonCreatorState &st, int a, int b) {
    auto joins = [a, b](const SkeletonCreatorEdge &e) {
        return (e.node1_id == a && e.node2_id == b) ||
               (e.node1_id == b && e.node2_id == a);
    };
    auto it = std::find_if(st.edges.begin(), st.edges.end(), joins);
    if (it == st.edges.end())
        st.edges.emplace_back(a, b);
    else
        st.edges.erase(std::remove_if(st.edges.begin(), st.edges.end(), joins),
                       st.edges.end());
}

inline void DrawSkeletonCreatorWindow(SkeletonCreatorState &st, AppContext &ctx) {
    const auto &skeleton_dir = ctx.skeleton_dir;
    ImGuiIO &io = ImGui::GetIO();

    DrawPanel("Skeleton Creator", st.show, [&]() {
        ImGui::SetNextItemWidth(240.0f);
        ImGui::InputText("Name", &st.name);

        if (ImGui::Button("Clear All")) {
            st.nodes.clear();
            st.edges.clear();
            st.next_node_id = 0;
            st.selected_for_edge = -1;
            st.menu_node = -1;
            st.status.clear();
        }
        ImGui::SameLine();
        if (ImGui::Button("Load from JSON")) {
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 1;
            cfg.path = skeleton_dir;
            cfg.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog("LoadSkeletonForEdit",
                                                    "Load Skeleton", ".json", cfg);
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(st.nodes.empty() || st.name.empty());
        if (ImGui::Button("Save to JSON")) {
            // Ask where, starting in the skeleton folder with <name>.json.
            IGFD::FileDialogConfig cfg;
            cfg.path = skeleton_dir;
            cfg.fileName = st.name + ".json";
            cfg.flags = ImGuiFileDialogFlags_Modal |
                        ImGuiFileDialogFlags_ConfirmOverwrite;
            ImGuiFileDialog::Instance()->OpenDialog("SaveSkeletonFromEdit",
                                                    "Save Skeleton", ".json", cfg);
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("Reset View")) st.reset_view = true;
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
            ImGui::SetTooltip("Show the whole pad again after zooming or panning");
        ImGui::SameLine();
        if (ImGui::Button("Background Image...")) {
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 1;
            cfg.path = ctx.pm.media_folder.empty() ? skeleton_dir
                                                   : ctx.pm.media_folder;
            cfg.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog("LoadSkeletonBackground",
                                                    "Background Image",
                                                    image_ext_filter(), cfg);
        }
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
            ImGui::SetTooltip("Show a picture behind the nodes to trace over. "
                              "It is not saved with the skeleton.");
        if (st.background.valid()) {
            ImGui::SameLine();
            if (ImGui::Button("Clear Image")) image_texture_free(&st.background);
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            ImGui::SliderFloat("Opacity", &st.background_opacity, 0.1f, 1.0f,
                               "%.1f");
        }

        // Editor on the left, its nodes on the right; drag the divider
        // between them to share the width.
        // What sits below the pad: the splitter, a pending-join hint, the Help
        // heading and its seven bullets, and a status line. Reserved whether or
        // not the hint and status are showing, so the pad does not jump.
        const float line_h = ImGui::GetTextLineHeightWithSpacing();
        const ImGuiStyle &style = ImGui::GetStyle();
        const float below = 6.0f + 10.0f * line_h + 2.0f * style.ItemSpacing.y +
                            style.SeparatorTextPadding.y * 2.0f;
        const float fit_h = ImGui::GetContentRegionAvail().y - below;
        const float editor_h =
            ImMax(st.editor_height > 0.0f ? st.editor_height : fit_h, 200.0f);
        const bool layout = ImGui::BeginTable(
            "##skel_layout", 2,
            ImGuiTableFlags_Resizable | ImGuiTableFlags_BordersInnerV);
        if (layout) {
            ImGui::TableSetupColumn("editor", ImGuiTableColumnFlags_WidthStretch, 0.6f);
            ImGui::TableSetupColumn("nodes", ImGuiTableColumnFlags_WidthStretch, 0.4f);
            ImGui::TableNextColumn();
        }

        // NoMenus/NoBoxSelect: right-click belongs to the nodes here, not to
        // ImPlot's own context menu and box zoom.
        //
        // Zoom with the scroll wheel; pan by RIGHT-dragging, since the left
        // button already adds and drags nodes. Double-click-to-fit moves to
        // the middle button so a double-click on the pad does not refit the
        // view under the node it just added. The input map is global to
        // ImPlot, so it is swapped in for this plot only and restored after.
        ImPlotInputMap &input = ImPlot::GetInputMap();
        const ImPlotInputMap saved_input = input;
        input.Pan = ImGuiMouseButton_Right;
        input.PanMod = ImGuiMod_None;
        input.Fit = ImGuiMouseButton_Middle;
        if (ImPlot::BeginPlot("##skelcreator", ImVec2(-1, editor_h),
                              ImPlotFlags_Equal | ImPlotFlags_NoMenus |
                                  ImPlotFlags_NoBoxSelect)) {
            ImPlot::SetupAxes("", "");
            ImPlot::SetupAxesLimits(0.0, 1.0, 0.0, 1.0,
                                    st.reset_view ? ImPlotCond_Always
                                                  : ImPlotCond_Once);
            st.reset_view = false;
            ImPlot::SetupAxisTicks(ImAxis_X1, nullptr, 0);
            ImPlot::SetupAxisTicks(ImAxis_Y1, nullptr, 0);

            // Click on empty space adds a node -- but not while a node is
            // waiting to be joined, or every missed Ctrl+Click would litter
            // the canvas.
            if (ImPlot::IsPlotHovered() &&
                ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !io.KeyCtrl &&
                st.selected_for_edge < 0) {
                ImPlotPoint m = ImPlot::GetPlotMousePos();
                st.nodes.emplace_back(m.x, m.y, st.next_node_id++);
            }

            if (ImGui::IsKeyPressed(ImGuiKey_Escape) && !io.WantTextInput)
                st.selected_for_edge = -1;

            // Background picture: fit inside the unit square, centred, with
            // its aspect ratio kept (the plot is ImPlotFlags_Equal).
            if (st.background.valid() && st.background.height > 0) {
                const double aspect = (double)st.background.width /
                                      (double)st.background.height;
                const double bw = aspect >= 1.0 ? 1.0 : aspect;
                const double bh = aspect >= 1.0 ? 1.0 / aspect : 1.0;
                ImPlot::PlotImage("##skel_background", st.background.id,
                                  ImPlotPoint(0.5 - bw / 2, 0.5 - bh / 2),
                                  ImPlotPoint(0.5 + bw / 2, 0.5 + bh / 2),
                                  ImVec2(0, 0), ImVec2(1, 1),
                                  ImVec4(1, 1, 1, st.background_opacity));
            }

            if (st.nodes.empty()) {
                ImPlot::PushStyleColor(ImPlotCol_InlayText,
                                       ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
                ImPlot::PlotText("Click here to add a node", 0.5, 0.5);
                ImPlot::PopStyleColor();
            }

            for (const auto &e : st.edges) {
                const SkeletonCreatorNode *a = nullptr, *b = nullptr;
                for (const auto &n : st.nodes) {
                    if (n.id == e.node1_id) a = &n;
                    if (n.id == e.node2_id) b = &n;
                }
                if (a && b) {
                    double xs[2]{a->position.x, b->position.x};
                    double ys[2]{a->position.y, b->position.y};
                    ImPlot::PlotLine("##edge", xs, ys, 2,
                                     red_line_spec(ImVec4(0.8f, 0.8f, 0.8f, 1.0f),
                                                   2.0f));
                }
            }

            for (size_t i = 0; i < st.nodes.size(); i++) {
                auto &node = st.nodes[i];
                bool clicked = false, hovered = false;
                ImVec4 col = st.selected_for_edge == node.id
                                 ? ImVec4(1.0f, 1.0f, 0.0f, 1.0f)
                                 : node.color;

                ImPlot::DragPoint(node.id, &node.position.x, &node.position.y,
                                  col, 8.0f, ImPlotDragToolFlags_None, &clicked,
                                  &hovered);

                // Name to the right of the node, always. PlotText centres on
                // the point, so shift it by half its width plus a gap.
                const float half_w = ImGui::CalcTextSize(node.name.c_str()).x * 0.5f;
                ImPlot::PlotText(node.name.c_str(), node.position.x,
                                 node.position.y, ImVec2(half_w + 10.0f, 0.0f));

                if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
                    st.menu_node = node.id;
                    st.open_menu = true;
                }
                if (hovered && ImGui::IsKeyPressed(ImGuiKey_R, false) &&
                    !io.WantTextInput) {
                    skeleton_creator_delete_node(st, node.id);
                    break;
                }

                // Picking the second node of a join: a plain click once one
                // is pending (Join... in the menu), or Ctrl+Click (Cmd on macOS) as before.
                // Clicking the pending node itself cancels.
                if (clicked && (io.KeyCtrl || st.selected_for_edge >= 0)) {
                    if (st.selected_for_edge < 0) {
                        st.selected_for_edge = node.id;
                    } else if (st.selected_for_edge != node.id) {
                        skeleton_creator_toggle_edge(st, st.selected_for_edge,
                                                     node.id);
                        st.selected_for_edge = -1;
                    } else {
                        st.selected_for_edge = -1;
                    }
                }
            }
            ImPlot::EndPlot();
        }
        input = saved_input;

        if (layout) {
            ImGui::TableNextColumn();
            // Same height as the editor, scrolling past it.
            if (ImGui::BeginTable("##skelnodes", 3,
                                  ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                      ImGuiTableFlags_ScrollY,
                                  ImVec2(0, editor_h))) {
                ImGui::TableSetupScrollFreeze(0, 1);
                ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, 30.0f);
                ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("Position", ImGuiTableColumnFlags_WidthFixed,
                                        100.0f);
                ImGui::TableHeadersRow();
                for (size_t i = 0; i < st.nodes.size(); i++) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::Text("%d", st.nodes[i].id);
                    ImGui::TableSetColumnIndex(1);
                    ImGui::PushID((int)i);
                    ImGui::SetNextItemWidth(-FLT_MIN);
                    ImGui::InputText("##name", &st.nodes[i].name);
                    ImGui::PopID();
                    ImGui::TableSetColumnIndex(2);
                    ImGui::Text("%.3f, %.3f", st.nodes[i].position.x,
                                st.nodes[i].position.y);
                }
                // Position: the header or any cell of the column.
                if (ImGui::TableGetHoveredColumn() == 2 &&
                    ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows))
                    ImGui::SetTooltip(
                        "Where the node sits on this pad (0-1). Saved only so "
                        "the\neditor can reopen the layout; red does not use it "
                        "for\nlabelling, export or anything else.");
                if (st.nodes.empty()) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(1);
                    ImGui::TextDisabled("No nodes yet");
                }
                ImGui::EndTable();
            }
            ImGui::EndTable();
        }

        // Splitter under the pad and the table, as under the Labeling Tool's
        // keypoints table.
        {
            const float splitter_h = 6.0f;
            ImGui::InvisibleButton("##skel_editor_splitter",
                                   ImVec2(-1.0f, splitter_h));
            const bool active = ImGui::IsItemActive();
            const bool hover = ImGui::IsItemHovered();
            if (active || hover)
                ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
            if (active)
                st.editor_height = editor_h + ImGui::GetIO().MouseDelta.y;
            if (hover && ImGui::IsMouseDoubleClicked(0))
                st.editor_height = 0.0f;
            if (hover && !active)
                ImGui::SetTooltip("Drag to resize the drawing pad, double-click "
                                  "to fit the window");
            const ImVec2 mn = ImGui::GetItemRectMin();
            const ImVec2 mx = ImGui::GetItemRectMax();
            const float y = (mn.y + mx.y) * 0.5f;
            const ImU32 col = ImGui::GetColorU32(
                active  ? ImGuiCol_SeparatorActive
                : hover ? ImGuiCol_SeparatorHovered
                        : ImGuiCol_Separator);
            ImGui::GetWindowDrawList()->AddLine(ImVec2(mn.x, y), ImVec2(mx.x, y),
                                                col, active ? 3.0f : 2.0f);
        }

        // Right-click menu for one node.
        if (st.open_menu) {
            ImGui::OpenPopup("##skel_node_menu");
            st.open_menu = false;
        }
        if (ImGui::BeginPopup("##skel_node_menu")) {
            auto it = std::find_if(st.nodes.begin(), st.nodes.end(),
                                   [&](const SkeletonCreatorNode &n) {
                                       return n.id == st.menu_node;
                                   });
            if (it == st.nodes.end()) {
                ImGui::CloseCurrentPopup();
            } else {
                SkeletonCreatorNode &node = *it;
                ImGui::TextDisabled("Node %d", node.id);
                ImGui::SetNextItemWidth(200.0f);
                if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
                if (ImGui::InputText("Name", &node.name,
                                     ImGuiInputTextFlags_EnterReturnsTrue))
                    ImGui::CloseCurrentPopup();
                ImGui::Separator();
                if (ImGui::MenuItem("Join to another node...")) {
                    st.selected_for_edge = node.id;
                }
                // Unjoin, one item per neighbour.
                bool any_edge = false;
                for (const auto &e : st.edges) {
                    int other = e.node1_id == node.id   ? e.node2_id
                                : e.node2_id == node.id ? e.node1_id
                                                        : -1;
                    if (other < 0) continue;
                    auto o = std::find_if(st.nodes.begin(), st.nodes.end(),
                                          [other](const SkeletonCreatorNode &n) {
                                              return n.id == other;
                                          });
                    if (o == st.nodes.end()) continue;
                    if (!any_edge) ImGui::Separator();
                    any_edge = true;
                    if (ImGui::MenuItem(("Unjoin from " + o->name).c_str())) {
                        skeleton_creator_toggle_edge(st, node.id, other);
                        break;
                    }
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Delete node", "R")) {
                    skeleton_creator_delete_node(st, node.id);
                    st.menu_node = -1;
                }
            }
            ImGui::EndPopup();
        }

        if (st.selected_for_edge >= 0)
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                               "Node selected. Click another to join them "
                               "(or to unjoin), Esc to cancel.");

        ImGui::SeparatorText("Help");
        ImGui::BulletText("Click empty space to add a node");
        ImGui::BulletText("Drag a node to move it");
        ImGui::BulletText("Scroll to zoom, right-drag to pan, Reset View to see it all");
        ImGui::BulletText("Right-click a node to rename, join, unjoin or delete it");
        ImGui::BulletText(RED_MOD_KEY "+Click two nodes to join or unjoin them");
        ImGui::BulletText("Esc cancels a pending join");
        ImGui::BulletText("R while hovering a node deletes it and its edges");

        if (!st.status.empty())
            ImGui::TextDisabled("%s", st.status.c_str());
        },
        [&]() {
        // Always: the dialogs have to be pumped whether the window is up.
        if (ImGuiFileDialog::Instance()->Display("LoadSkeletonBackground",
                                                 ImGuiWindowFlags_NoCollapse,
                                                 ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                const std::string path =
                    ImGuiFileDialog::Instance()->GetFilePathName();
                std::string err;
                if (image_texture_load(path, &st.background, &err))
                    st.status = "Background: " + path;
                else {
                    st.status = err;
                    ctx.popups.pushError(err);
                }
            }
            ImGuiFileDialog::Instance()->Close();
        }
        if (ImGuiFileDialog::Instance()->Display("SaveSkeletonFromEdit",
                                                 ImGuiWindowFlags_NoCollapse,
                                                 ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                const std::string path =
                    ImGuiFileDialog::Instance()->GetFilePathName();
                std::ofstream f(path);
                if (f) {
                    f << skeleton_creator_to_json(st).dump(4);
                    st.status = "Saved " + path;
                    ctx.toasts.pushSuccess("Saved skeleton " + st.name);
                } else {
                    st.status = "Could not write " + path;
                    ctx.popups.pushError(st.status);
                }
            }
            ImGuiFileDialog::Instance()->Close();
        }
        if (ImGuiFileDialog::Instance()->Display("LoadSkeletonForEdit",
                                                 ImGuiWindowFlags_NoCollapse,
                                                 ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                const std::string path =
                    ImGuiFileDialog::Instance()->GetFilePathName();
                std::ifstream f(path);
                nlohmann::json j;
                bool ok = false;
                if (f) { try { f >> j; ok = true; } catch (...) { ok = false; } }
                if (!ok) {
                    st.status = "Could not read " + path;
                    ctx.popups.pushError(st.status);
                } else {
                    st.nodes.clear();
                    st.edges.clear();
                    st.selected_for_edge = -1;
                    st.next_node_id = 0;
                    if (j.contains("name")) st.name = j["name"].get<std::string>();

                    std::vector<std::string> names =
                        j.value("node_names", std::vector<std::string>{});
                    std::vector<std::vector<double>> pos =
                        j.value("node_positions", std::vector<std::vector<double>>{});
                    // A skeleton red shipped has names and edges but no
                    // positions; lay those out in a row so they can be dragged
                    // into shape rather than refusing to open them.
                    const double spacing = names.empty() ? 0.0
                                                         : 0.8 / (double)(names.size() + 1);
                    for (size_t i = 0; i < names.size(); i++) {
                        SkeletonCreatorNode n;
                        n.id = st.next_node_id++;
                        n.name = names[i];
                        n.position = (i < pos.size() && pos[i].size() >= 2)
                                         ? ImPlotPoint(pos[i][0], pos[i][1])
                                         : ImPlotPoint(0.1 + spacing * (double)(i + 1), 0.5);
                        n.color = (ImVec4)ImColor::HSV(n.id / 10.0f, 1.0f, 1.0f);
                        st.nodes.push_back(n);
                    }
                    for (const auto &e : j.value("edges", std::vector<std::vector<int>>{}))
                        if (e.size() >= 2 && e[0] >= 0 && e[1] >= 0 &&
                            e[0] < (int)st.nodes.size() && e[1] < (int)st.nodes.size())
                            st.edges.emplace_back(st.nodes[(size_t)e[0]].id,
                                                  st.nodes[(size_t)e[1]].id);
                    st.status = "Loaded " + path;
                }
            }
            ImGuiFileDialog::Instance()->Close();
        }
        },
        ImVec2(900, 740));
}
