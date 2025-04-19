module;

export module Cinder.Render.Vulkan:Queue;

import std;

import Cinder.Globals;

export namespace zrx {
struct QueueFamilyIndices {
    std::optional<uint32_t> graphics_compute_family;
    std::optional<uint32_t> present_family;

    [[nodiscard]] bool isComplete() const {
        return graphics_compute_family.has_value() && present_family.has_value();
    }
};
} // zrx

