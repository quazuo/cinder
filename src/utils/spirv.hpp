#pragma once

#include <SPIRV-Reflect/spirv_reflect.h>
#include <filesystem>

#include "src/render/globals.hpp"

class SpirvReflectModuleWrapper {
    SpvReflectShaderModule module{};

public:
    explicit SpirvReflectModuleWrapper(const std::filesystem::path& path);

    ~SpirvReflectModuleWrapper();

    [[nodiscard]] vector<SpvReflectDescriptorSet*> descriptor_sets() const;

    [[nodiscard]] vector<SpvReflectDescriptorBinding*> descriptor_bindings() const;
};
