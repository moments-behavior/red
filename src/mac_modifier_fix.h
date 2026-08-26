#pragma once
// mac_modifier_fix.h — corrects stuck modifier keys on macOS.
//
// The problem: imgui_impl_glfw's ImGui_ImplGlfw_UpdateKeyModifiers() reads
// modifier state with glfwGetKey(), which returns state GLFW *cached* from
// key events. macOS screen recording (Cmd+Shift+5) can swallow the key-UP
// event, so GLFW keeps believing a modifier is held — Ctrl/Shift/Cmd stay
// stuck until the user presses and releases them again. Every consumer of
// glfwGetKey() inherits this; it is not really an ImGui bug, and upstream has
// not addressed it (still glfwGetKey() as of 1.93.0 WIP).
//
// The fix: query the real hardware state via CoreGraphics and re-send the
// modifier events. This deliberately does NOT patch the backend. The backend
// only touches modifiers from its mouse/key callbacks, which run during
// glfwPollEvents(); ImGui applies its input-event queue in order at
// ImGui::NewFrame(), so events appended afterwards win. Calling this between
// ImGui_ImplGlfw_NewFrame() and ImGui::NewFrame() therefore overrides the
// cached values without forking imgui_impl_glfw.cpp — which previously meant
// re-deriving a 1400-line copy on every ImGui bump.
//
// No-op off macOS.
#include "imgui.h"

#ifdef __APPLE__
#include <CoreGraphics/CoreGraphics.h>
#endif

inline void red_sync_mac_modifiers() {
#ifdef __APPLE__
    ImGuiIO &io = ImGui::GetIO();
    // Physical-key -> ImGui modifier, matching the backend's own mapping
    // (ImGui swaps Ctrl/Super internally on macOS, so Command -> Super).
    CGEventFlags flags =
        CGEventSourceFlagsState(kCGEventSourceStateCombinedSessionState);
    io.AddKeyEvent(ImGuiMod_Ctrl,  (flags & kCGEventFlagMaskControl)   != 0);
    io.AddKeyEvent(ImGuiMod_Shift, (flags & kCGEventFlagMaskShift)     != 0);
    io.AddKeyEvent(ImGuiMod_Alt,   (flags & kCGEventFlagMaskAlternate) != 0);
    io.AddKeyEvent(ImGuiMod_Super, (flags & kCGEventFlagMaskCommand)   != 0);
#endif
}
