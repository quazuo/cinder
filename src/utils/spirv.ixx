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

    auto descriptor_sets() const -> vector<SpvReflectDescriptorSet*>;

    auto descriptor_bindings() const -> vector<SpvReflectDescriptorBinding*>;

    auto push_constant_blocks() const -> vector<SpvReflectBlockVariable*>;
};
} // zrx
