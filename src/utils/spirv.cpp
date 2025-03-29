#include "spirv.hpp"

#include <SPIRV-Reflect/spirv_reflect.h>

#include <fstream>
#include <functional>

#include "src/utils/logger.hpp"

namespace zrx {
static vector<char> read_file(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        Logger::error("failed to open file!");
    }

    const size_t file_size = file.tellg();
    vector<char> buffer(file_size);
    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(file_size));

    return buffer;
}

static void check_spv_result(const SpvReflectResult result) {
    if (result != SPV_REFLECT_RESULT_SUCCESS) {
        Logger::error("call to SPV library function failed with code: " + std::to_string(result));
    }
}

template<typename T>
using SpvReflectFn = std::function<SpvReflectResult(const SpvReflectShaderModule *, uint32_t *, T **)>;

template<typename T>
static vector<T *> enumerate_spv_objects(const SpvReflectShaderModule *module, SpvReflectFn<T> spv_fn) {
    uint32_t obj_count = 0;
    check_spv_result(spv_fn(module, &obj_count, nullptr));
    vector<T *> objects(obj_count);
    check_spv_result(spv_fn(module, &obj_count, objects.data()));
    return objects;
}

SpirvReflectModuleWrapper::SpirvReflectModuleWrapper(const std::filesystem::path &path) {
    module = make_unique<SpvReflectShaderModule>();
    const auto file_buffer = read_file(path);
    check_spv_result(spvReflectCreateShaderModule(file_buffer.size(), file_buffer.data(), &*module));
}

SpirvReflectModuleWrapper::~SpirvReflectModuleWrapper() {
    spvReflectDestroyShaderModule(&*module);
}

vector<SpvReflectDescriptorSet*> SpirvReflectModuleWrapper::descriptor_sets() const {
    return enumerate_spv_objects<SpvReflectDescriptorSet>(&*module, spvReflectEnumerateDescriptorSets);
}

vector<SpvReflectDescriptorBinding*> SpirvReflectModuleWrapper::descriptor_bindings() const {
    return enumerate_spv_objects<SpvReflectDescriptorBinding>(&*module, spvReflectEnumerateDescriptorBindings);
}
} // zrx
