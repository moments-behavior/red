#pragma once
#include "gui/panel.h"
#include "gui/help_content.h"
#include <cctype>
#include <cstring>
#include <string>

// ---------------------------------------------------------------------------
// Small rendering helpers
// ---------------------------------------------------------------------------
namespace help_ui {

inline std::string lower(const char *s) {
    std::string r(s ? s : "");
    for (char &c : r) c = (char)std::tolower((unsigned char)c);
    return r;
}
inline bool match(const std::string &q, const char *hay) {
    if (q.empty()) return true;
    return lower(hay).find(q) != std::string::npos;
}
// match against several fields at once
inline bool match_any(const std::string &q, std::initializer_list<const char *> fields) {
    if (q.empty()) return true;
    for (const char *f : fields)
        if (lower(f).find(q) != std::string::npos) return true;
    return false;
}

// Is an entry visible given the current project/build context? A Need3D entry
// is hidden only when a project IS open and it is 2D (with no project open we
// show everything as a general reference).
inline bool gate_ok(help::Gate g, const help::Context &c) {
    switch (g) {
        case help::Gate::Need3D:      return !c.project_open || c.is_3d;
        default:                      return true;
    }
}
inline bool tool_active(help::Gate g, const help::Context &c) {
    switch (g) {
        case help::Gate::ToolBbox:    return c.bbox_on;
        case help::Gate::ToolObb:     return c.obb_on;
        case help::Gate::ToolMidline: return c.midline_on;
        default:                      return false;
    }
}

// A boxed, monospace-ish "key chip".
inline void key_chip(const char *txt) {
    ImDrawList *dl = ImGui::GetWindowDrawList();
    const ImVec2 pad(7.0f, 3.0f);
    ImVec2 ts = ImGui::CalcTextSize(txt);
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImVec2 rmin = p;
    ImVec2 rmax = ImVec2(p.x + ts.x + pad.x * 2, p.y + ts.y + pad.y * 2);
    dl->AddRectFilled(rmin, rmax, ImGui::GetColorU32(ImGuiCol_FrameBg), 4.0f);
    dl->AddRect(rmin, rmax, ImGui::GetColorU32(ImGuiCol_Border), 4.0f);
    dl->AddText(ImVec2(p.x + pad.x, p.y + pad.y),
                ImGui::GetColorU32(ImGuiCol_Text), txt);
    ImGui::Dummy(ImVec2(ts.x + pad.x * 2, ts.y + pad.y * 2));
}

// Key label for a shortcut row: derived from the binding table (single source
// of truth) for bound keys, or the literal string for mouse/multi-step rows.
inline std::string sc_label(const help::Shortcut &s) {
    return (s.sc != keys::Sc::COUNT) ? keys::display(s.sc)
                                     : std::string(s.lit ? s.lit : "");
}

// Render one shortcut group as a table. Returns whether it drew anything.
inline bool draw_group(const help::Group &g, const help::Context &c,
                       const std::string &q, bool active_badge) {
    // Collect visible + matching rows first so we can skip empty groups.
    std::vector<const help::Shortcut *> rows;
    for (const auto &s : g.items)
        if (gate_ok(s.gate, c) &&
            (match_any(q, {g.title, sc_label(s).c_str(), s.action, s.note})))
            rows.push_back(&s);
    if (rows.empty()) return false;

    ImGui::SeparatorText(g.title);
    if (active_badge) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.35f, 0.80f, 0.45f, 1.0f), "[active]");
    }
    if (g.subtitle && g.subtitle[0]) ImGui::TextDisabled("%s", g.subtitle);

    if (ImGui::BeginTable(g.title, 2,
            ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_PadOuterX)) {
        ImGui::TableSetupColumn("key", ImGuiTableColumnFlags_WidthFixed, 150.0f);
        ImGui::TableSetupColumn("action", ImGuiTableColumnFlags_WidthStretch);
        for (const auto *s : rows) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            key_chip(sc_label(*s).c_str());
            ImGui::TableNextColumn();
            ImGui::TextWrapped("%s", s->action);
            if (s->note && s->note[0]) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
                ImGui::TextWrapped("%s", s->note);
                ImGui::PopStyleColor();
            }
        }
        ImGui::EndTable();
    }
    ImGui::Spacing();
    return true;
}

inline void tab_shortcuts(const help::Context &c, const std::string &q) {
    bool any = false;
    // Context-aware: float the enabled tool's group(s) to the top with a badge.
    for (const auto &g : help::shortcut_groups())
        if (tool_active(g.group_gate, c)) any |= draw_group(g, c, q, true);
    // Then everything else in declared order (skip tool groups already shown).
    for (const auto &g : help::shortcut_groups()) {
        if (tool_active(g.group_gate, c)) continue;   // already drawn on top
        any |= draw_group(g, c, q, false);
    }
    if (!any) ImGui::TextDisabled("No shortcuts match \"%s\".", q.c_str());
}

inline void tab_mouse(const std::string &q) {
    bool any = false;
    for (const auto &mg : help::mouse_groups()) {
        std::vector<const help::MouseAction *> rows;
        for (const auto &m : mg.items)
            if (match_any(q, {mg.title, m.input, m.effect, m.note}))
                rows.push_back(&m);
        if (rows.empty()) continue;
        any = true;
        ImGui::SeparatorText(mg.title);
        if (mg.subtitle && mg.subtitle[0]) ImGui::TextDisabled("%s", mg.subtitle);
        if (ImGui::BeginTable(mg.title, 2,
                ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_PadOuterX)) {
            ImGui::TableSetupColumn("in", ImGuiTableColumnFlags_WidthFixed, 220.0f);
            ImGui::TableSetupColumn("eff", ImGuiTableColumnFlags_WidthStretch);
            for (const auto *m : rows) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(m->input);
                ImGui::TableNextColumn();
                ImGui::TextWrapped("%s", m->effect);
                if (m->note && m->note[0]) {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
                    ImGui::TextWrapped("%s", m->note);
                    ImGui::PopStyleColor();
                }
            }
            ImGui::EndTable();
        }
        ImGui::Spacing();
    }
    if (!any) ImGui::TextDisabled("No mouse controls match \"%s\".", q.c_str());
}

inline void tab_tools(const help::Context &c, const std::string &q) {
    if (ImGui::BeginTable("tools", 3,
            ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH |
            ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Tool", ImGuiTableColumnFlags_WidthFixed, 190.0f);
        ImGui::TableSetupColumn("What it does");
        ImGui::TableSetupColumn("Needs", ImGuiTableColumnFlags_WidthFixed, 190.0f);
        ImGui::TableHeadersRow();
        int shown = 0;
        for (const auto &t : help::tools()) {
            if (!gate_ok(t.gate, c)) continue;
            if (!match_any(q, {t.name, t.open, t.desc, t.needs})) continue;
            ++shown;
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(t.name);
            ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
            ImGui::TextWrapped("%s", t.open);
            ImGui::PopStyleColor();
            ImGui::TableNextColumn();
            ImGui::TextWrapped("%s", t.desc);
            ImGui::TableNextColumn();
            ImGui::TextWrapped("%s", t.needs);
        }
        ImGui::EndTable();
        if (!shown) ImGui::TextDisabled("No tools match \"%s\".", q.c_str());
    }
}

inline void tab_workflows(const std::string &q) {
    bool any = false;
    for (const auto &w : help::workflows()) {
        bool title_hit = match(q, w.title);
        bool step_hit = false;
        for (const char *s : w.steps) step_hit |= match(q, s);
        if (!title_hit && !step_hit) continue;
        any = true;
        ImGui::SeparatorText(w.title);
        int i = 1;
        for (const char *s : w.steps)
            ImGui::TextWrapped("%d.  %s", i++, s);
        ImGui::Spacing();
    }
    if (!any) ImGui::TextDisabled("No workflows match \"%s\".", q.c_str());
}

inline void tab_concepts(const std::string &q) {
    bool any = false;
    for (const auto &cc : help::concepts()) {
        if (!match_any(q, {cc.term, cc.def})) continue;
        any = true;
        ImGui::SeparatorText(cc.term);
        ImGui::TextWrapped("%s", cc.def);
        ImGui::Spacing();
    }
    if (!any) ImGui::TextDisabled("No concepts match \"%s\".", q.c_str());
}

inline void tab_about() {
    ImGui::TextUnformatted("RED \xE2\x80\x94 Multi-Camera Keypoint Labeling Tool");
    ImGui::TextDisabled("Press  H  any time to toggle this window.");
    ImGui::Spacing();
    ImGui::TextWrapped("Help adapts to your project: 3D-only entries are hidden "
                       "for 2D projects, and the shortcuts for an enabled tool "
                       "are floated to the top of the Shortcuts tab.");
}

} // namespace help_ui

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
inline void DrawHelpWindow(bool &show, const help::Context &ctx = {}) {
    DrawPanel("Help Menu", show, [&]() {
        static char filter[64] = "";
        ImGui::SetNextItemWidth(-1.0f);
        ImGui::InputTextWithHint("##help_search",
            "Search shortcuts, mouse, tools, workflows, concepts...",
            filter, IM_ARRAYSIZE(filter));
        std::string q = help_ui::lower(filter);
        ImGui::Spacing();

        if (ImGui::BeginTabBar("##help_tabs", ImGuiTabBarFlags_None)) {
            if (ImGui::BeginTabItem("Shortcuts")) {
                if (ImGui::BeginChild("##sc")) help_ui::tab_shortcuts(ctx, q);
                ImGui::EndChild();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Mouse")) {
                if (ImGui::BeginChild("##ms")) help_ui::tab_mouse(q);
                ImGui::EndChild();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Tools")) {
                if (ImGui::BeginChild("##tl")) help_ui::tab_tools(ctx, q);
                ImGui::EndChild();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Workflows")) {
                if (ImGui::BeginChild("##wf")) help_ui::tab_workflows(q);
                ImGui::EndChild();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Concepts")) {
                if (ImGui::BeginChild("##cn")) help_ui::tab_concepts(q);
                ImGui::EndChild();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("About")) {
                if (ImGui::BeginChild("##ab")) help_ui::tab_about();
                ImGui::EndChild();
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
    }, nullptr, ImVec2(760, 580));
}
