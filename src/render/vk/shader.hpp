#pragma once

#include <filesystem>
#include <utility>

#include "src/render/libs.hpp"
#include "src/render/globals.hpp"

namespace zrx {

struct RendererContext;

class SpirvShaderPair {
    std::filesystem::path vert_path, frag_path;
    vector<vk::raii::DescriptorSetLayout> cached_descriptor_set_layouts;

public:
    explicit SpirvShaderPair(std::filesystem::path vert_path, std::filesystem::path frag_path)
        : vert_path(std::move(vert_path)), frag_path(std::move(frag_path)) {}

    [[nodiscard]]
    const vector<vk::raii::DescriptorSetLayout>& get_descriptor_set_layouts(const RendererContext& ctx);
};

} // zrx
