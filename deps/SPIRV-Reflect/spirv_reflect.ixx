module;

#include "spirv_reflect.h"

export module SpirvReflect;

export {
    using ::SpvReflectShaderModule;
    using ::SpvReflectResult;
    using ::SpvReflectDescriptorSet;
    using ::SpvReflectDescriptorBinding;
    using ::SpvReflectBlockVariable;

    using ::SPV_REFLECT_RESULT_SUCCESS;

    using ::spvReflectCreateShaderModule;
    using ::spvReflectDestroyShaderModule;
    using ::spvReflectEnumerateDescriptorSets;
    using ::spvReflectEnumerateDescriptorBindings;
    using ::spvReflectEnumeratePushConstantBlocks;
}
