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
    bool has_bbox = false;
    std::string status;
};

inline void DrawSkeletonCreatorWindow(SkeletonCreatorState &st, AppContext &ctx) {
    const auto &skeleton_dir = ctx.skeleton_dir;
    ImGuiIO &io = ImGui::GetIO();

    DrawPanel("Skeleton Creator", st.show, [&]() {
        ImGui::SeparatorText("Skeleton");
        ImGui::SetNextItemWidth(240.0f);
        ImGui::InputText("Name", &st.name);
        ImGui::SameLine();
        ImGui::Checkbox("Has bounding box", &st.has_bbox);

        ImGui::SeparatorText("Editor");
        if (ImPlot::BeginPlot("##skelcreator", ImVec2(-1, 400), ImPlotFlags_Equal)) {
            ImPlot::SetupAxes("", "");
            ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, 1.0, ImGuiCond_Always);
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.0, ImGuiCond_Always);
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

                if (hovered) {
                    ImPlot::PlotText(node.name.c_str(), node.position.x,
                                     node.position.y + 0.03);
                    if (ImGui::IsKeyPressed(ImGuiKey_R, false) && !io.WantTextInput) {
                        const int dead = node.id;
                        st.nodes.erase(st.nodes.begin() + (long)i);
                        st.edges.erase(
                            std::remove_if(st.edges.begin(), st.edges.end(),
                                           [dead](const SkeletonCreatorEdge &e) {
                                               return e.node1_id == dead ||
                                                      e.node2_id == dead;
                                           }),
                            st.edges.end());
                        if (st.selected_for_edge == dead) st.selected_for_edge = -1;
                        break;
                    }
                }

                // Ctrl+Click picks a node, then joins to the second -- or
                // removes the edge if those two are already joined.
                if (clicked && io.KeyCtrl) {
                    if (st.selected_for_edge < 0) {
                        st.selected_for_edge = node.id;
                    } else if (st.selected_for_edge != node.id) {
                        const int a = st.selected_for_edge, b = node.id;
                        auto joins = [a, b](const SkeletonCreatorEdge &e) {
                            return (e.node1_id == a && e.node2_id == b) ||
                                   (e.node1_id == b && e.node2_id == a);
                        };
                        auto it = std::find_if(st.edges.begin(), st.edges.end(), joins);
                        if (it == st.edges.end())
                            st.edges.emplace_back(a, b);
                        else
                            st.edges.erase(std::remove_if(st.edges.begin(),
                                                          st.edges.end(), joins),
                                           st.edges.end());
                        st.selected_for_edge = -1;
                    } else {
                        st.selected_for_edge = -1;
                    }
                }
            }
            ImPlot::EndPlot();
        }

        if (st.selected_for_edge >= 0)
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                               "Node selected. Ctrl+Click another to join them "
                               "(or to unjoin), Esc to cancel.");

        ImGui::SeparatorText("Help");
        ImGui::BulletText("Click empty space to add a node");
        ImGui::BulletText("Drag a node to move it");
        ImGui::BulletText("Ctrl+Click two nodes to join or unjoin them");
        ImGui::BulletText("Esc cancels a pending join");
        ImGui::BulletText("R while hovering a node deletes it and its edges");

        ImGui::SeparatorText("Actions");
        if (ImGui::Button("Clear All")) {
            st.nodes.clear();
            st.edges.clear();
            st.next_node_id = 0;
            st.selected_for_edge = -1;
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
            nlohmann::json j;
            j["name"] = st.name;
            j["has_skeleton"] = true;
            j["has_bbox"] = st.has_bbox;
            j["num_nodes"] = (int)st.nodes.size();

            std::vector<std::string> names;
            std::vector<std::vector<double>> positions;
            for (const auto &n : st.nodes) {
                names.push_back(n.name);
                positions.push_back({n.position.x, n.position.y});
            }
            j["node_names"] = names;
            // Not read back by load_skeleton_json, which only wants names and
            // edges -- kept so this window can reopen its own output with the
            // layout intact rather than restacking it in a line.
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

            const std::string path = skeleton_dir + "/" + st.name + ".json";
            std::ofstream f(path);
            if (f) {
                f << j.dump(4);
                st.status = "Saved " + path;
                ctx.toasts.pushSuccess("Saved skeleton " + st.name);
            } else {
                st.status = "Could not write " + path;
                ctx.popups.pushError(st.status);
            }
        }
        ImGui::EndDisabled();

        if (!st.nodes.empty()) {
            ImGui::SeparatorText("Nodes");
            if (ImGui::BeginTable("##skelnodes", 3,
                                  ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, 40.0f);
                ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("Position", ImGuiTableColumnFlags_WidthFixed,
                                        120.0f);
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
                ImGui::EndTable();
            }
        }

        if (!st.status.empty())
            ImGui::TextDisabled("%s", st.status.c_str());
        },
        [&]() {
        // Always: the load dialog has to be pumped whether the window is up.
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
                    if (j.contains("has_bbox")) st.has_bbox = j["has_bbox"].get<bool>();

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
        ImVec2(820, 640));
}
