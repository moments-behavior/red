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
#include "IconsForkAwesome.h"
#include <imgui.h>
#include <imgui_internal.h>  // InputEventsQueue (drain_queued_arrows)
#include <string>

// What to call the modifier ImGui reports as Ctrl. On macOS ImGui swaps Cmd
// and Ctrl (io.ConfigMacOSXBehaviors, on by default under __APPLE__), so
// io.KeyCtrl and every "Ctrl" binding fire on Command there. A macro so it
// can join string literals in the help tables.
#if defined(__APPLE__)
#define RED_MOD_KEY "Cmd"
#else
#define RED_MOD_KEY "Ctrl"
#endif

namespace keys {

inline const char *mod_name() { return RED_MOD_KEY; }

enum class Sc {
    ToggleHelp,
    PlayPause,
    SeekBack,
    SeekFwd,
    JumpBack,       // a keyframe interval back (video), ten frames (images)
    JumpFwd,
    SaveLabels,
    CreateFrame,
    PlaceKeypoint,
    MarkOccluded,
    NextInstance,
    PrevInstance,
    NextView,
    ActivePrev,
    ActiveNext,
    ActiveFirst,
    ActiveLast,
    DeleteAllKp,
    Triangulate,
    PeekRaw,
    SelectAllKeypoints, // keypoints table: select every keypoint column (toggle)
    CopyKeypoints,    // keypoints table: copy the selected node set
    PasteKeypoints,   // keypoints table: paste the copied node set onto this frame
    DeleteKeypoint,   // keypoints table: delete (hovered cell / hovered column / selection)
    TextLarger,       // UI text size (View > Text Size)
    TextSmaller,
    TextReset,
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
        /* SeekBack       */ {ImGuiKey_LeftArrow, false, false, true, false},
        /* SeekFwd        */ {ImGuiKey_RightArrow, false, false, true, false},
        /* JumpBack       */ {ImGuiKey_UpArrow, false, false, true, false},
        /* JumpFwd        */ {ImGuiKey_DownArrow, false, false, true, false},
        /* SaveLabels     */ {ImGuiKey_S, true, false, false, false},
        /* CreateFrame    */ {ImGuiKey_B, false, false, false, false},
        /* PlaceKeypoint  */ {ImGuiKey_W, false, false, false, false},
        /* MarkOccluded   */ {ImGuiKey_M, false, false, false, false},
        /* NextInstance   */ {ImGuiKey_X, false, false, false, false},
        /* PrevInstance   */ {ImGuiKey_Z, false, false, false, false},
        /* NextView       */ {ImGuiKey_Tab, false, false, true, false},
        /* ActivePrev     */ {ImGuiKey_A, false, false, true, false},
        /* ActiveNext     */ {ImGuiKey_D, false, false, true, false},
        /* ActiveFirst    */ {ImGuiKey_Q, false, false, false, false},
        /* ActiveLast     */ {ImGuiKey_E, false, false, false, false},
        /* DeleteAllKp    */ {ImGuiKey_Backspace, false, false, false, false},
        /* Triangulate    */ {ImGuiKey_T, false, false, false, false},
        /* PeekRaw        */ {ImGuiKey_P, false, false, false, true},
        /* SelectAllKeypoints */ {ImGuiKey_A, true, false, false, false},
        /* CopyKeypoints  */ {ImGuiKey_C, true, false, false, false},
        /* PasteKeypoints */ {ImGuiKey_V, true, false, false, false},
        /* DeleteKeypoint */ {ImGuiKey_Delete, false, false, false, false},
        /* TextLarger     */ {ImGuiKey_Equal, true, false, true, false},
        /* TextSmaller    */ {ImGuiKey_Minus, true, false, true, false},
        /* TextReset      */ {ImGuiKey_0, true, false, false, false},
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
        // Arrow keys use the ForkAwesome glyphs: Roboto has no U+2190-2193
        // block, but the icon font is merged into the same atlas (gx_helper.h),
        // so these render as real arrows rather than the fallback '?'.
        case ImGuiKey_LeftArrow:  return ICON_FK_ARROW_LEFT;
        case ImGuiKey_RightArrow: return ICON_FK_ARROW_RIGHT;
        case ImGuiKey_UpArrow:    return ICON_FK_ARROW_UP;
        case ImGuiKey_DownArrow:  return ICON_FK_ARROW_DOWN;
        case ImGuiKey_Comma:      return ",";
        case ImGuiKey_Equal:      return "=";
        case ImGuiKey_Minus:      return "-";
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
    if (b.ctrl)  out += RED_MOD_KEY " + ";
    if (b.shift) out += "Shift + ";
    out += key_name(b.key);
    if (b.hold)  out += "  (hold)";
    return out;
}

// The frame move one arrow press makes: Left/Right one frame, Up/Down
// `jump_frames`. 0 for any other key.
inline int arrow_delta(ImGuiKey k, int jump_frames) {
    switch (k) {
    case ImGuiKey_LeftArrow:  return -1;
    case ImGuiKey_RightArrow: return 1;
    case ImGuiKey_UpArrow:    return -jump_frames;
    case ImGuiKey_DownArrow:  return jump_frames;
    default:                  return 0;
    }
}

// Arrow presses still waiting in ImGui's input queue -- made while a blocking
// seek held the main thread, and otherwise handed out one per frame from here
// on -- taken out, so they do not each run a seek. Returns the move they add
// up to. Only the key-downs are removed; their key-ups stay and change
// nothing. *only_jumps (optional) is cleared if any was Left/Right. Uses
// ImGui internals (the queue is not public API).
inline int drain_queued_arrows(int jump_frames, bool *only_jumps) {
    ImVector<ImGuiInputEvent> &q = ImGui::GetCurrentContext()->InputEventsQueue;
    int total = 0;
    for (int n = 0; n < q.Size;) {
        const ImGuiInputEvent &e = q[n];
        const int d = e.Type == ImGuiInputEventType_Key && e.Key.Down
                          ? arrow_delta(e.Key.Key, jump_frames)
                          : 0;
        if (d == 0) { ++n; continue; }
        total += d;
        if (only_jumps && e.Key.Key != ImGuiKey_UpArrow &&
            e.Key.Key != ImGuiKey_DownArrow)
            *only_jumps = false;
        q.erase(q.Data + n);
    }
    return total;
}

} // namespace keys
