#pragma once
#include "imgui.h"
#include "implot.h"
#include "skeleton.h"
#include "json.hpp"
#include <ImGuiFileDialog.h>
#include <misc/cpp/imgui_stdlib.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifndef __APPLE__
#include <GL/gl.h>
#endif

struct SkeletonCreatorState {
    bool show = false;
    std::vector<SkeletonCreatorNode> nodes;
    std::vector<SkeletonCreatorEdge> edges;
    int next_node_id = 0;
    int selected_node_for_edge = -1;
    std::string name = "CustomSkeleton";
    bool has_bbox = false;

    // Background image
    std::string background_image_path;
    bool background_image_selected = false;
#ifndef __APPLE__
    GLuint background_texture = 0;
#else
    uintptr_t background_texture = 0;
#endif
    int background_width = 0;
    int background_height = 0;
};

inline void DrawSkeletonCreatorWindow(SkeletonCreatorState &state,
                                      const std::string &skeleton_dir) {
    // Always process file dialogs (even when window is hidden)

    // Handle background image selection
    if (ImGuiFileDialog::Instance()->Display("ChooseBackgroundImage", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            auto file_selection =
                ImGuiFileDialog::Instance()->GetSelection();
            if (!file_selection.empty()) {
                state.background_image_path = file_selection.begin()->second;
                state.background_image_selected = true;

                // Load the background image texture
#ifndef __APPLE__
                if (state.background_texture != 0) {
                    glDeleteTextures(1, &state.background_texture);
                    state.background_texture = 0;
                }

                // Load image using stb_image
                int channels;
                unsigned char *image_data = stbi_load(
                    state.background_image_path.c_str(), &state.background_width,
                    &state.background_height, &channels, 0);
                if (image_data) {
                    glGenTextures(1, &state.background_texture);
                    glBindTexture(GL_TEXTURE_2D, state.background_texture);

                    GLenum format = GL_RGB;
                    if (channels == 4)
                        format = GL_RGBA;
                    else if (channels == 1)
                        format = GL_RED;

                    glTexImage2D(GL_TEXTURE_2D, 0, format, state.background_width,
                                 state.background_height, 0, format,
                                 GL_UNSIGNED_BYTE, image_data);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                                    GL_LINEAR);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                                    GL_LINEAR);

                    stbi_image_free(image_data);
                }
#else
                // Background texture not supported on macOS/Metal yet
                (void)state.background_width; (void)state.background_height;
#endif
            }
        }
        ImGuiFileDialog::Instance()->Close();
    }

    // Handle skeleton load dialog for editing
    if (ImGuiFileDialog::Instance()->Display("LoadSkeletonForEdit", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::string file_path =
                ImGuiFileDialog::Instance()->GetFilePathName();

            std::ifstream file(file_path);
            if (file.is_open()) {
                nlohmann::json skeleton_json;
                file >> skeleton_json;
                file.close();

                // Clear existing data
                state.nodes.clear();
                state.edges.clear();
                state.selected_node_for_edge = -1;
                state.next_node_id = 0;

                // Load skeleton data
                if (skeleton_json.contains("name")) {
                    state.name = skeleton_json["name"];
                }

                if (skeleton_json.contains("has_bbox")) {
                    state.has_bbox = skeleton_json["has_bbox"];
                }

                if (skeleton_json.contains("node_names") &&
                    skeleton_json.contains("node_positions")) {
                    std::vector<std::string> node_names =
                        skeleton_json["node_names"];
                    std::vector<std::vector<double>> node_positions =
                        skeleton_json["node_positions"];

                    for (size_t i = 0;
                         i < node_names.size() && i < node_positions.size();
                         i++) {
                        if (node_positions[i].size() >= 2) {
                            SkeletonCreatorNode node;
                            node.id = state.next_node_id++;
                            node.name = node_names[i];
                            node.position = ImPlotPoint(
                                node_positions[i][0], node_positions[i][1]);
                            node.color = (ImVec4)ImColor::HSV(
                                node.id / 10.0f, 1.0f, 1.0f);
                            state.nodes.push_back(node);
                        }
                    }
                } else if (skeleton_json.contains("node_names")) {
                    // Fallback for skeletons without saved positions
                    std::vector<std::string> node_names =
                        skeleton_json["node_names"];
                    double spacing = 0.8 / (node_names.size() + 1);

                    for (size_t i = 0; i < node_names.size(); i++) {
                        SkeletonCreatorNode node;
                        node.id = state.next_node_id++;
                        node.name = node_names[i];
                        node.position =
                            ImPlotPoint(0.1 + spacing * (i + 1), 0.5);
                        node.color = (ImVec4)ImColor::HSV(node.id / 10.0f,
                                                          1.0f, 1.0f);
                        state.nodes.push_back(node);
                    }
                }

                if (skeleton_json.contains("edges")) {
                    std::vector<std::vector<int>> edges_array =
                        skeleton_json["edges"];
                    for (const auto &edge : edges_array) {
                        if (edge.size() >= 2 &&
                            edge[0] < (int)state.nodes.size() &&
                            edge[1] < (int)state.nodes.size()) {
                            SkeletonCreatorEdge creator_edge;
                            creator_edge.node1_id =
                                state.nodes[edge[0]].id;
                            creator_edge.node2_id =
                                state.nodes[edge[1]].id;
                            state.edges.push_back(creator_edge);
                        }
                    }
                }

                std::cout << "Skeleton loaded from: " << file_path
                          << " (with " << state.nodes.size() << " nodes)"
                          << std::endl;
            }
        }
        ImGuiFileDialog::Instance()->Close();
    }

    if (!state.show)
        return;

    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Skeleton Creator", &state.show)) {
        ImGui::SeparatorText("Skeleton Configuration");
        ImGui::InputText("Skeleton Name", &state.name);
        ImGui::Checkbox("Has Bounding Box", &state.has_bbox);

        if (ImGui::Button("Select Background Image")) {
            IGFD::FileDialogConfig config;
            config.countSelectionMax = 1;
            config.path = std::filesystem::current_path().string();
            config.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseBackgroundImage", "Choose Background Image",
                ".png,.jpg,.jpeg,.tiff,.bmp,.tga", config);
        }
        ImGui::SameLine();
        if (state.background_image_selected && state.background_texture != 0) {
            std::filesystem::path path(state.background_image_path);
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f),
                               "Image: %s (%dx%d)",
                               path.filename().string().c_str(),
                               state.background_width, state.background_height);
            ImGui::SameLine();
            if (ImGui::Button("Clear Background")) {
#ifndef __APPLE__
                if (state.background_texture != 0) {
                    glDeleteTextures(1, &state.background_texture);
                    state.background_texture = 0;
                }
#endif
                state.background_image_path = "";
                state.background_image_selected = false;
                state.background_width = 0;
                state.background_height = 0;
                std::cout << "Background image cleared" << std::endl;
            }
        } else {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                               "No background image");
        }

        ImGui::SeparatorText("Interactive Editor");

        ImGuiIO &io = ImGui::GetIO();

        if (ImPlot::BeginPlot("Skeleton Creator", ImVec2(-1, 400),
                              ImPlotFlags_Equal)) {
            ImPlot::SetupAxes("", "");
            ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, 1.0,
                                    ImGuiCond_Always);
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.0,
                                    ImGuiCond_Always);

            ImPlot::SetupAxisTicks(ImAxis_X1, nullptr, 0);
            ImPlot::SetupAxisTicks(ImAxis_Y1, nullptr, 0);

            if (state.background_image_selected && state.background_texture != 0) {
                ImPlot::PlotImage(
                    "##background",
                    (ImTextureID)(intptr_t)state.background_texture,
                    ImPlotPoint(0, 0), ImPlotPoint(1, 1));
            }

            if (ImPlot::IsPlotHovered() &&
                ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
                !ImGui::GetIO().KeyCtrl) {
                if (state.selected_node_for_edge < 0) {
                    ImPlotPoint mouse_pos = ImPlot::GetPlotMousePos();
                    state.nodes.emplace_back(mouse_pos.x, mouse_pos.y,
                                             state.next_node_id++);
                }
            }

            if (ImGui::IsKeyPressed(ImGuiKey_Escape) &&
                !io.WantTextInput) {
                state.selected_node_for_edge = -1;
            }

            for (const auto &edge : state.edges) {
                const SkeletonCreatorNode *node1 = nullptr;
                const SkeletonCreatorNode *node2 = nullptr;

                for (const auto &node : state.nodes) {
                    if (node.id == edge.node1_id)
                        node1 = &node;
                    if (node.id == edge.node2_id)
                        node2 = &node;
                }

                if (node1 && node2) {
                    double xs[2] = {node1->position.x,
                                    node2->position.x};
                    double ys[2] = {node1->position.y,
                                    node2->position.y};
                    ImPlot::SetNextLineStyle(
                        ImVec4(0.8f, 0.8f, 0.8f, 1.0f), 2.0f);
                    ImPlot::PlotLine("##edge", xs, ys, 2);
                }
            }

            for (size_t i = 0; i < state.nodes.size(); i++) {
                auto &node = state.nodes[i];

                bool clicked = false, hovered = false, held = false;
                ImVec4 node_color = node.color;

                if (state.selected_node_for_edge == node.id) {
                    node_color = ImVec4(1.0f, 1.0f, 0.0f, 1.0f);
                }

                bool modified = ImPlot::DragPoint(
                    node.id, &node.position.x, &node.position.y,
                    node_color, 8.0f, ImPlotDragToolFlags_None,
                    &clicked, &hovered, &held);

                if (hovered) {
                    ImPlot::PlotText(node.name.c_str(), node.position.x,
                                     node.position.y + 0.03);

                    if (ImGui::IsKeyPressed(ImGuiKey_R, false) &&
                        !io.WantTextInput) {
                        int node_id_to_delete = node.id;

                        state.nodes.erase(state.nodes.begin() + i);

                        state.edges.erase(
                            std::remove_if(
                                state.edges.begin(),
                                state.edges.end(),
                                [node_id_to_delete](
                                    const SkeletonCreatorEdge &edge) {
                                    return edge.node1_id ==
                                               node_id_to_delete ||
                                           edge.node2_id ==
                                               node_id_to_delete;
                                }),
                            state.edges.end());

                        if (state.selected_node_for_edge ==
                            node_id_to_delete) {
                            state.selected_node_for_edge = -1;
                        }

                        break;
                    }
                }

                if (clicked && ImGui::GetIO().KeyCtrl) {
                    if (state.selected_node_for_edge < 0) {
                        state.selected_node_for_edge = node.id;
                    } else if (state.selected_node_for_edge != node.id) {
                        bool edge_exists = false;
                        for (const auto &existing_edge :
                             state.edges) {
                            if ((existing_edge.node1_id ==
                                     state.selected_node_for_edge &&
                                 existing_edge.node2_id == node.id) ||
                                (existing_edge.node1_id == node.id &&
                                 existing_edge.node2_id ==
                                     state.selected_node_for_edge)) {
                                edge_exists = true;
                                break;
                            }
                        }

                        if (!edge_exists) {
                            state.edges.emplace_back(
                                state.selected_node_for_edge, node.id);
                        } else {
                            // Delete the existing edge
                            state.edges.erase(
                                std::remove_if(
                                    state.edges.begin(),
                                    state.edges.end(),
                                    [&state,
                                     node_id = node.id](
                                        const SkeletonCreatorEdge
                                            &edge) {
                                        return (edge.node1_id ==
                                                    state.selected_node_for_edge &&
                                                edge.node2_id ==
                                                    node_id) ||
                                               (edge.node1_id ==
                                                    node_id &&
                                                edge.node2_id ==
                                                    state.selected_node_for_edge);
                                    }),
                                state.edges.end());
                        }

                        state.selected_node_for_edge = -1;
                    } else {
                        state.selected_node_for_edge = -1;
                    }
                }
            }

            ImPlot::EndPlot();
        }

        if (state.selected_node_for_edge >= 0) {
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                               "Selected node for edge creation. "
                               "Ctrl+Click another node to "
                               "create edge or remove existing "
                               "edge, or press ESC to cancel.");
        }

        ImGui::SeparatorText("Help");
        ImGui::BulletText(
            "Left-click in the plot area to add a new node");
        ImGui::BulletText("Drag nodes to reposition them");
        ImGui::BulletText(
            "Ctrl+Click a node to select it for edge creation");
        ImGui::BulletText(
            "Ctrl+Click another node to create an edge or remove "
            "an existing edge between them");
        ImGui::BulletText("Press ESC to cancel edge creation");
        ImGui::BulletText("Press R while hovering a node to delete "
                          "it and its edges");

        ImGui::SeparatorText("Actions");

        if (ImGui::Button("Clear All")) {
            state.nodes.clear();
            state.edges.clear();
            state.next_node_id = 0;
            state.selected_node_for_edge = -1;
        }

        ImGui::SameLine();
        if (ImGui::Button("Load from JSON")) {
            IGFD::FileDialogConfig config;
            config.countSelectionMax = 1;
            config.path = skeleton_dir;
            config.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "LoadSkeletonForEdit", "Load Skeleton", ".json",
                config);
        }

        ImGui::SameLine();
        if (ImGui::Button("Save to JSON")) {
            if (!state.nodes.empty()) {
                nlohmann::json skeleton_json;
                skeleton_json["name"] = state.name;
                skeleton_json["has_skeleton"] = true;
                skeleton_json["has_bbox"] = state.has_bbox;
                skeleton_json["num_nodes"] = (int)state.nodes.size();
                skeleton_json["num_edges"] = (int)state.edges.size();

                std::vector<std::string> node_names;
                std::vector<std::vector<double>> node_positions;
                for (const auto &node : state.nodes) {
                    node_names.push_back(node.name);
                    node_positions.push_back(
                        {node.position.x, node.position.y});
                }
                skeleton_json["node_names"] = node_names;
                skeleton_json["node_positions"] = node_positions;

                std::vector<std::vector<int>> edges_array;
                for (const auto &edge : state.edges) {
                    int idx1 = -1, idx2 = -1;
                    for (size_t i = 0; i < state.nodes.size(); i++) {
                        if (state.nodes[i].id == edge.node1_id)
                            idx1 = (int)i;
                        if (state.nodes[i].id == edge.node2_id)
                            idx2 = (int)i;
                    }
                    if (idx1 >= 0 && idx2 >= 0) {
                        edges_array.push_back({idx1, idx2});
                    }
                }
                skeleton_json["edges"] = edges_array;

                std::string filename;
                filename = skeleton_dir + "/" + state.name +
                           ".json";

                std::ofstream file(filename);
                file << skeleton_json.dump(4);
                file.close();
                std::cout << "Skeleton saved to: " << filename
                          << " (with node positions)" << std::endl;
            }
        }

        if (!state.nodes.empty()) {
            ImGui::SeparatorText("Nodes");
            if (ImGui::BeginTable("NodeTable", 3,
                                  ImGuiTableFlags_Borders |
                                      ImGuiTableFlags_RowBg)) {
                ImGui::TableSetupColumn(
                    "ID", ImGuiTableColumnFlags_WidthFixed, 40.0f);
                ImGui::TableSetupColumn(
                    "Name", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn(
                    "Position", ImGuiTableColumnFlags_WidthFixed,
                    120.0f);
                ImGui::TableHeadersRow();

                for (size_t i = 0; i < state.nodes.size(); i++) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::Text("%d", state.nodes[i].id);

                    ImGui::TableSetColumnIndex(1);
                    ImGui::PushID(i);
                    ImGui::InputText("##name", &state.nodes[i].name);
                    ImGui::PopID();

                    ImGui::TableSetColumnIndex(2);
                    ImGui::Text("(%.2f, %.2f)",
                                state.nodes[i].position.x,
                                state.nodes[i].position.y);
                }
                ImGui::EndTable();
            }
        }
    }
    ImGui::End();
}
