#pragma once
// Single source of truth for keyboard shortcuts (Phase 4). Each action's
// trigger key lives here exactly once. Both sides read this table:
//   - input handling calls keys::pressed()/held() instead of hardcoding
//     ImGui::IsKeyPressed(ImGuiKey_X, ...),
//   - the Help window derives its displayed key label via keys::display().
// So the key that fires an action and the key shown in Help are the same
// datum and cannot drift apart.
//
// Scope: the shortcuts handled in red.cpp + Ctrl+S + the peek key. The
// tool-mode keys (bbox/OBB/SAM) are migrated in a later pass; until then they
// keep literal labels in help_content.h.
#include <imgui.h>
#include <string>

namespace keys {

enum class Sc {
    ToggleHelp,
    PlayPause,
    SeekBack,
    SeekFwd,
    SaveLabels,
    PredictCurrent,
    BufferPrev,
    BufferNext,
    CreateFrame,
    PlaceKeypoint,
    ActivePrev,
    ActiveNext,
    ActiveFirst,
    ActiveLast,
    DeleteAllKp,
    Triangulate,
    PlotMenu,
    PeekRaw,
    SelectAllKeypoints, // Keypoints window: select every keypoint column (toggle)
    CopyKeypoints,    // Keypoints window: copy the selected node set
    PasteKeypoints,   // Keypoints window: paste the copied node set onto this frame
    DeleteKeypoint,   // Keypoints window: delete (hovered cell / hovered column / selection)
    COUNT  // sentinel: "no single bound key" (help rows that use a literal label)
};

struct Binding {
    ImGuiKey key;
    bool ctrl;    // required modifier
    bool shift;   // required modifier
    bool repeat;  // IsKeyPressed repeat flag (preserves each site's original)
    bool hold;    // trigger is IsKeyDown (momentary hold) rather than IsKeyPressed
};

// The canonical table. Order MUST match enum Sc. Repeat flags mirror the
// original call sites exactly.
inline const Binding &binding(Sc s) {
    static const Binding table[] = {
        /* ToggleHelp     */ {ImGuiKey_H, false, false, false, false},
        /* PlayPause      */ {ImGuiKey_Space, false, false, false, false},
        /* SeekBack       */ {ImGuiKey_LeftArrow, false, false, false, false},
        /* SeekFwd        */ {ImGuiKey_RightArrow, false, false, false, false},
        /* SaveLabels     */ {ImGuiKey_S, true, false, false, false},
        /* PredictCurrent */ {ImGuiKey_6, false, false, false, false},
        /* BufferPrev     */ {ImGuiKey_Comma, false, false, true, false},
        /* BufferNext     */ {ImGuiKey_Period, false, false, true, false},
        /* CreateFrame    */ {ImGuiKey_B, false, false, false, false},
        /* PlaceKeypoint  */ {ImGuiKey_W, false, false, false, false},
        /* ActivePrev     */ {ImGuiKey_A, false, false, true, false},
        /* ActiveNext     */ {ImGuiKey_D, false, false, true, false},
        /* ActiveFirst    */ {ImGuiKey_Q, false, false, false, false},
        /* ActiveLast     */ {ImGuiKey_E, false, false, false, false},
        /* DeleteAllKp    */ {ImGuiKey_Backspace, false, false, false, false},
        /* Triangulate    */ {ImGuiKey_T, false, false, false, false},
        /* PlotMenu       */ {ImGuiKey_2, false, false, false, false},
        /* PeekRaw        */ {ImGuiKey_P, false, false, false, true},
        /* SelectAllKeypoints */ {ImGuiKey_A, true, false, false, false},
        /* CopyKeypoints  */ {ImGuiKey_C, true, false, false, false},
        /* PasteKeypoints */ {ImGuiKey_V, true, false, false, false},
        /* DeleteKeypoint */ {ImGuiKey_Delete, false, false, false, false},
    };
    static_assert(sizeof(table) / sizeof(table[0]) == (size_t)Sc::COUNT,
                  "keys::binding table is out of sync with enum Sc");
    return table[(int)s];
}

// Modifier match: a required modifier (ctrl/shift) must be held. We only
// ENFORCE modifiers the binding requires; we don't forbid extra ones, which
// preserves the original behavior (e.g. plain 'W' fired regardless of Shift,
// and Left/Right read Shift at the site to pick x1 vs x10).
inline bool mods_ok(const Binding &b) {
    const ImGuiIO &io = ImGui::GetIO();
    if (b.ctrl && !io.KeyCtrl) return false;
    if (b.shift && !io.KeyShift) return false;
    return true;
}

// True on the frame the shortcut is pressed. Mirrors the original
// `IsKeyPressed(key, repeat) && !io.WantTextInput` (+ required modifiers).
inline bool pressed(Sc s, bool allow_text_input = false) {
    const Binding &b = binding(s);
    if (!allow_text_input && ImGui::GetIO().WantTextInput) return false;
    if (!mods_ok(b)) return false;
    return ImGui::IsKeyPressed(b.key, b.repeat);
}

// True while the shortcut key is held (for momentary "hold" bindings, e.g. peek).
inline bool held(Sc s, bool allow_text_input = false) {
    const Binding &b = binding(s);
    if (!allow_text_input && ImGui::GetIO().WantTextInput) return false;
    if (!mods_ok(b)) return false;
    return ImGui::IsKeyDown(b.key);
}

// Pretty name for a key. Self-contained (no ImGui::GetKeyName dependency);
// ImGuiKey_A..Z and _0..9 are contiguous in ImGui's enum.
inline std::string key_name(ImGuiKey k) {
    if (k >= ImGuiKey_A && k <= ImGuiKey_Z)
        return std::string(1, (char)('A' + (k - ImGuiKey_A)));
    if (k >= ImGuiKey_0 && k <= ImGuiKey_9)
        return std::string(1, (char)('0' + (k - ImGuiKey_0)));
    switch (k) {
        case ImGuiKey_LeftArrow:  return "\xE2\x86\x90"; // <-
        case ImGuiKey_RightArrow: return "\xE2\x86\x92"; // ->
        case ImGuiKey_UpArrow:    return "\xE2\x86\x91";
        case ImGuiKey_DownArrow:  return "\xE2\x86\x93";
        case ImGuiKey_Comma:      return ",";
        case ImGuiKey_Period:     return ".";
        case ImGuiKey_Space:      return "Space";
        case ImGuiKey_Backspace:  return "Backspace";
        case ImGuiKey_Enter:      return "Enter";
        case ImGuiKey_Escape:     return "Esc";
        case ImGuiKey_Delete:     return "Delete";
        case ImGuiKey_Tab:        return "Tab";
        default:                  return "?";
    }
}

// Human-readable label shown in Help, derived from the binding.
inline std::string display(Sc s) {
    const Binding &b = binding(s);
    std::string out;
    if (b.ctrl)  out += "Ctrl + ";
    if (b.shift) out += "Shift + ";
    out += key_name(b.key);
    if (b.hold)  out += "  (hold)";
    return out;
}

} // namespace keys
