module;

#include "src/utils/macros.hpp"

module Cinder.Render.Graph;

import glm;
import std;

import Cinder.Render.Vulkan;
import Cinder.Globals;
import Cinder.Utils;

namespace zrx {
BindlessHandle::IDType BindlessHandle::next_free_handle_id = 0;
BindlessHandle::IDType BindlessHandle::next_free_special_handle_id = -1;

ResourceManager::ResourceManager(const RendererContext& ctx, const uint32_t frames_in_flight)
    : renderer_ctx(ctx), frames_in_flight(frames_in_flight), queued_for_removal_resources(frames_in_flight)
{
    for (uint32_t i = 0; i < BINDLESS_ARRAY_SIZE; i++) {
        free_bindless_handles.push(BindlessHandle::get_new());
    }

    for (uint32_t i = 0; i < MAX_RESOURCE_COUNT; i++) {
        free_resource_handles.push(ResourceHandle::get_new());
    }

    const vector<vk::DescriptorPoolSize> pool_sizes = {
        {
            .type = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = BINDLESS_ARRAY_SIZE,
        },
        {
            .type = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 2 * BINDLESS_ARRAY_SIZE,
        },
        {
            .type = vk::DescriptorType::eStorageImage,
            .descriptorCount = BINDLESS_ARRAY_SIZE,
        },
        {
            .type = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = BINDLESS_ARRAY_SIZE,
        },
        {
            .type = vk::DescriptorType::eAccelerationStructureKHR,
            .descriptorCount = BINDLESS_ARRAY_SIZE,
        },
    };

    const vk::DescriptorPoolCreateInfo pool_info{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet
                 | vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind,
        .maxSets = static_cast<uint32_t>(frames_in_flight) * 6 + 5,
        .poolSizeCount = static_cast<uint32_t>(pool_sizes.size()),
        .pPoolSizes = pool_sizes.data(),
    };

    descriptor_pool = make_unique<vk::raii::DescriptorPool>(*ctx.device, pool_info);

    constexpr vk::DescriptorBindingFlags binding_flags = vk::DescriptorBindingFlagBits::ePartiallyBound
                                                         | vk::DescriptorBindingFlagBits::eUpdateAfterBind;

    bindless_descriptor_set = make_unique<BindlessDescriptorSet>(
        ctx,
        *descriptor_pool,
        ResourcePack<Image> {
            BINDLESS_ARRAY_SIZE,
            vk::ShaderStageFlagBits::eAllGraphics,
            vk::DescriptorType::eCombinedImageSampler,
            binding_flags
        },
        ResourcePack<Image> {
            BINDLESS_ARRAY_SIZE,
            vk::ShaderStageFlagBits::eCompute,
            vk::DescriptorType::eStorageImage,
            binding_flags
        },
        ResourcePack<Buffer> {
            BINDLESS_ARRAY_SIZE,
            vk::ShaderStageFlagBits::eAll,
            vk::DescriptorType::eUniformBuffer,
            binding_flags
        }
    );
}

void ResourceManager::clear_removal_queue() {
    // todo - not sure if that's correct...
    queued_for_removal_resources[renderer_ctx.get().current_frame_idx].clear();
}

void ResourceManager::recreate(const ResourceHandle handle) {
    if (!handle_to_kind_mapping.contains(handle)) {
        LOG_ERROR_WITH_FUNC("tried to recreate a non-existent resource");
    }

    const auto error_missing_builder = [&] {
        LOG_ERROR_WITH_FUNC("tried to recreate a resource without a registered builder for that resource");
    };

    switch (handle_to_kind_mapping.at(handle)) {
        case ResourceKind::IMAGE:
            if (!image_builders.contains(handle)) error_missing_builder();
            attach_raw(handle, image_builders.at(handle).create(renderer_ctx.get()));
            break;
        case ResourceKind::GRAPHICS_PIPELINE:
            if (!graphics_pipeline_builders.contains(handle)) error_missing_builder();
            attach_raw(handle, graphics_pipeline_builders.at(handle).create(renderer_ctx.get()));
            break;
        case ResourceKind::COMPUTE_PIPELINE:
            if (!compute_pipeline_builders.contains(handle)) error_missing_builder();
            attach_raw(handle, compute_pipeline_builders.at(handle).create(renderer_ctx.get()));
            break;
        case ResourceKind::BUFFER:
            if (!buffer_builders.contains(handle)) error_missing_builder();
            attach_raw(handle, buffer_builders.at(handle).create(renderer_ctx.get()));
            break;
        default:
            LOG_ERROR_WITH_FUNC("unexpected resource kind");
    }
}

void ResourceManager::reload_all_pipelines() {
    for (const auto& [handle, builder] : graphics_pipeline_builders) {
        recreate(handle);
    }

    for (const auto& [handle, builder] : compute_pipeline_builders) {
        recreate(handle);
    }
}

auto ResourceManager::get_name(const ResourceHandle handle) const -> const std::string& {
    if (handle == FINAL_IMAGE_HANDLE) return FINAL_IMAGE_NAME;
    return resource_names.at(handle);
}
} // zrx
