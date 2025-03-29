#pragma once

#include <filesystem>

#include "src/render/globals.hpp"

struct SpvReflectShaderModule;
struct SpvReflectDescriptorSet;
struct SpvReflectDescriptorBinding;

namespace zrx {
class SpirvReflectModuleWrapper {
    unique_ptr<SpvReflectShaderModule> module = nullptr;

public:
    explicit SpirvReflectModuleWrapper(const std::filesystem::path& path);

    ~SpirvReflectModuleWrapper();

    [[nodiscard]] vector<SpvReflectDescriptorSet*> descriptor_sets() const;

    [[nodiscard]] vector<SpvReflectDescriptorBinding*> descriptor_bindings() const;
};
} // zrx
