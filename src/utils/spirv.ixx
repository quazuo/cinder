module;

export module Cinder.Utils:Spirv;

import spirv_reflect;
import std;

import Cinder.Globals;

export namespace zrx {
class SpirvReflectModuleWrapper {
    unique_ptr<SpvReflectShaderModule> module = nullptr;

public:
    explicit SpirvReflectModuleWrapper(const std::filesystem::path& path);

    ~SpirvReflectModuleWrapper();

    [[nodiscard]] vector<SpvReflectDescriptorSet*> descriptor_sets() const;

    [[nodiscard]] vector<SpvReflectDescriptorBinding*> descriptor_bindings() const;

    [[nodiscard]] vector<SpvReflectBlockVariable*> push_constant_blocks() const;
};
} // zrx
