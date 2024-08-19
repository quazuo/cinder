#include "descriptor.hpp"

#include "buffer.hpp"
#include "image.hpp"

namespace zrx {

DescriptorLayoutBuilder &
DescriptorLayoutBuilder::addBinding(const vk::DescriptorType type, const vk::ShaderStageFlags stages,
                                    const uint32_t descriptorCount) {
    bindings.emplace_back(vk::DescriptorSetLayoutBinding{
        .binding = static_cast<uint32_t>(bindings.size()),
        .descriptorType = type,
        .descriptorCount = descriptorCount,
        .stageFlags = stages,
    });

    return *this;
}

DescriptorLayoutBuilder &
DescriptorLayoutBuilder::addRepeatedBindings(const size_t count, const vk::DescriptorType type,
                                                  const vk::ShaderStageFlags stages, const uint32_t descriptorCount) {
    for (size_t i = 0; i < count; i++) {
        addBinding(type, stages, descriptorCount);
    }

    return *this;
}

vk::raii::DescriptorSetLayout DescriptorLayoutBuilder::create(const RendererContext &ctx) {
    const vk::DescriptorSetLayoutCreateInfo setLayoutInfo{
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };

    return {*ctx.device, setLayoutInfo};
}

} // zrx
