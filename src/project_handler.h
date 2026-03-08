#pragma once
#include "json.hpp"
#include <functional>
#include <string>
#include <vector>

// Subsystems register save/load callbacks for named JSON sections
// within the .redproj file. Enables the unified project format where
// calibration, annotation, and future subsystems each own their data.
struct ProjectSection {
    std::string key;                                  // JSON key
    std::function<nlohmann::json()> save;             // produce JSON
    std::function<void(const nlohmann::json &)> load; // consume JSON
};

struct ProjectHandlerRegistry {
    std::vector<ProjectSection> sections;

    void add(ProjectSection s) { sections.push_back(std::move(s)); }
    size_t size() const { return sections.size(); }
};

// Merge registered sections into an existing JSON object.
inline void project_handlers_save(const ProjectHandlerRegistry &reg,
                                  nlohmann::json &j) {
    for (const auto &s : reg.sections) {
        if (s.save)
            j[s.key] = s.save();
    }
}

// Extract registered sections from a JSON object.
// Missing sections are silently skipped (backward compatible).
inline void project_handlers_load(const ProjectHandlerRegistry &reg,
                                  const nlohmann::json &j) {
    for (const auto &s : reg.sections) {
        if (s.load && j.contains(s.key))
            s.load(j.at(s.key));
    }
}
