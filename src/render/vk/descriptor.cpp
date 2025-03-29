#include "descriptor.hpp"

#include "src/render/renderer.hpp"
#include "buffer.hpp"
#include "image.hpp"

namespace zrx {
DescriptorLayoutBuilder &
DescriptorLayoutBuilder::add_binding(const vk::DescriptorType type, const vk::ShaderStageFlags stages,
                                     const uint32_t descriptor_count) {
    bindings.emplace_back(vk::DescriptorSetLayoutBinding{
        .binding = static_cast<uint32_t>(bindings.size()),
        .descriptorType = type,
        .descriptorCount = descriptor_count,
        .stageFlags = stages,
    });

    return *this;
}

DescriptorLayoutBuilder &
DescriptorLayoutBuilder::add_repeated_bindings(const size_t count, const vk::DescriptorType type,
                                               const vk::ShaderStageFlags stages, const uint32_t descriptor_count) {
    for (size_t i = 0; i < count; i++) {
        add_binding(type, stages, descriptor_count);
    }

    return *this;
}

vk::raii::DescriptorSetLayout DescriptorLayoutBuilder::create(const RendererContext &ctx) {
    const vk::DescriptorSetLayoutCreateInfo set_layout_info{
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };

    return {*ctx.device, set_layout_info};
}

DescriptorSet &DescriptorSet::queue_update(const uint32_t binding, const Buffer &buffer, const vk::DescriptorType type,
                                           const vk::DeviceSize size, const vk::DeviceSize offset,
                                           const uint32_t array_element) {
    const vk::DescriptorBufferInfo buffer_info{
        .buffer = *buffer,
        .offset = offset,
        .range = size,
    };

    queued_updates.emplace_back(DescriptorUpdate{
        .binding = binding,
        .array_element = array_element,
        .type = type,
        .info = buffer_info,
    });

    return *this;
}

DescriptorSet &DescriptorSet::queue_update(const RendererContext &ctx, const uint32_t binding, const Texture &texture,
                                           const vk::DescriptorType type, const uint32_t array_element) {
    const vk::DescriptorImageInfo image_info{
        .sampler = *texture.get_sampler(),
        .imageView = **texture.get_image().get_view(ctx),
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    queued_updates.emplace_back(DescriptorUpdate{
        .binding = binding,
        .array_element = array_element,
        .type = type,
        .info = image_info,
    });

    return *this;
}

DescriptorSet &DescriptorSet::queue_update(const uint32_t binding, const vk::raii::ImageView &view,
                                           const uint32_t array_element) {
    const vk::DescriptorImageInfo image_info{
        .imageView = *view,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    queued_updates.emplace_back(DescriptorUpdate{
        .binding = binding,
        .array_element = array_element,
        .type = vk::DescriptorType::eStorageImage,
        .info = image_info,
    });

    return *this;
}

DescriptorSet &DescriptorSet::queue_update(const uint32_t binding, const AccelerationStructure &accel,
                                           const uint32_t array_element) {
    const vk::WriteDescriptorSetAccelerationStructureKHR accel_info{
        .accelerationStructureCount = 1u,
        .pAccelerationStructures = &**accel, // todo - dangling pointer?
    };

    queued_updates.emplace_back(DescriptorUpdate{
        .binding = binding,
        .array_element = array_element,
        .type = vk::DescriptorType::eAccelerationStructureKHR,
        .info = accel_info,
    });

    return *this;
}

void DescriptorSet::commit_updates(const RendererContext &ctx) {
    vector<vk::WriteDescriptorSet> descriptor_writes;

    for (const auto &update: queued_updates) {
        vk::WriteDescriptorSet write{
            .dstSet = **set,
            .dstBinding = update.binding,
            .dstArrayElement = update.array_element,
            .descriptorCount = 1,
            .descriptorType = update.type,
        };

        if (std::holds_alternative<vk::DescriptorBufferInfo>(update.info)) {
            write.pBufferInfo = &std::get<vk::DescriptorBufferInfo>(update.info);
        } else if (std::holds_alternative<vk::DescriptorImageInfo>(update.info)) {
            write.pImageInfo = &std::get<vk::DescriptorImageInfo>(update.info);
        } else if (std::holds_alternative<vk::WriteDescriptorSetAccelerationStructureKHR>(update.info)) {
            write.pNext = &std::get<vk::WriteDescriptorSetAccelerationStructureKHR>(update.info);
        } else {
            Logger::error("unexpected variant in DescriptorSet::commitUpdates");
        }

        descriptor_writes.emplace_back(write);
    }

    ctx.device->updateDescriptorSets(descriptor_writes, nullptr);

    queued_updates.clear();
}

void DescriptorSet::update_binding(const RendererContext &ctx, const uint32_t binding, const Buffer &buffer,
                                   const vk::DescriptorType type, const vk::DeviceSize size,
                                   const vk::DeviceSize offset, const uint32_t array_element) const {
    const vk::DescriptorBufferInfo buffer_info{
        .buffer = *buffer,
        .offset = offset,
        .range = size,
    };

    const vk::WriteDescriptorSet write{
        .dstSet = **set,
        .dstBinding = binding,
        .dstArrayElement = array_element,
        .descriptorCount = 1,
        .descriptorType = type,
        .pBufferInfo = &buffer_info,
    };

    ctx.device->updateDescriptorSets(write, nullptr);
}

void DescriptorSet::update_binding(const RendererContext &ctx, const uint32_t binding, const Texture &texture,
                                   const vk::DescriptorType type, const uint32_t array_element) const {
    const vk::DescriptorImageInfo image_info{
        .sampler = *texture.get_sampler(),
        .imageView = **texture.get_image().get_view(ctx),
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    const vk::WriteDescriptorSet write{
        .dstSet = **set,
        .dstBinding = binding,
        .dstArrayElement = array_element,
        .descriptorCount = 1,
        .descriptorType = type,
        .pImageInfo = &image_info,
    };

    ctx.device->updateDescriptorSets(write, nullptr);
}

void DescriptorSet::update_binding(const RendererContext &ctx, const uint32_t binding, const vk::raii::ImageView &view,
                                   const uint32_t array_element) const {
    const vk::DescriptorImageInfo image_info{
        .imageView = *view,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    const vk::WriteDescriptorSet write{
        .dstSet = **set,
        .dstBinding = binding,
        .dstArrayElement = array_element,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eStorageImage,
        .pImageInfo = &image_info,
    };

    ctx.device->updateDescriptorSets(write, nullptr);
}

void DescriptorSet::update_binding(const RendererContext &ctx, const uint32_t binding,
                                   const AccelerationStructure &accel, const uint32_t array_element) const {
    const vk::WriteDescriptorSetAccelerationStructureKHR accel_info{
        .accelerationStructureCount = 1,
        .pAccelerationStructures = &**accel,
    };

    const vk::WriteDescriptorSet write{
        .pNext = &accel_info,
        .dstSet = **set,
        .dstBinding = binding,
        .dstArrayElement = array_element,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eAccelerationStructureKHR,
    };

    ctx.device->updateDescriptorSets(write, nullptr);
}

vector<DescriptorSet>
utils::desc::create_descriptor_sets(const RendererContext &ctx, const vk::raii::DescriptorPool &pool,
                                    const shared_ptr<vk::raii::DescriptorSetLayout> &layout, const uint32_t count) {
    const vector set_layouts(count, **layout);

    const vk::DescriptorSetAllocateInfo alloc_info{
        .descriptorPool = *pool,
        .descriptorSetCount = count,
        .pSetLayouts = set_layouts.data(),
    };

    vector<vk::raii::DescriptorSet> descriptor_sets = ctx.device->allocateDescriptorSets(alloc_info);

    vector<DescriptorSet> final_sets;

    for (size_t i = 0; i < count; i++) {
        final_sets.emplace_back(layout, std::move(descriptor_sets[i]));
    }

    return final_sets;
}

DescriptorSet utils::desc::create_descriptor_set(const RendererContext &ctx, const vk::raii::DescriptorPool &pool,
                                                 const shared_ptr<vk::raii::DescriptorSetLayout> &layout) {
    auto sets = create_descriptor_sets(ctx, pool, layout, 1);
    auto set = std::move(sets[0]);
    return set;
}
} // zrx
