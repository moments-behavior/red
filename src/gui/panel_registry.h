#pragma once
#include <functional>
#include <string>
#include <vector>

struct PanelEntry {
    std::string name;
    std::function<void()> draw;
    std::function<bool()> visible;  // nullptr = always draw
};

struct PanelRegistry {
    std::vector<PanelEntry> entries;

    void add(PanelEntry e) { entries.push_back(std::move(e)); }

    void drawAll() {
        for (auto &e : entries)
            if (!e.visible || e.visible()) e.draw();
    }
};
