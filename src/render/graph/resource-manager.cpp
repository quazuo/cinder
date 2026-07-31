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

ResourceManager::ResourceManager(
    const RendererContext& ctx, const uint32_t max_bindless_handles, const uint32_t frames_in_flight
) : renderer_ctx(ctx), queued_for_removal_resources(frames_in_flight) {
    for (uint32_t i = 0; i < max_bindless_handles; i++) {
        free_bindless_handles.push(BindlessHandle::get_new());
    }

    for (uint32_t i = 0; i < MAX_RESOURCE_COUNT; i++) {
        free_resource_handles.push(ResourceHandle::get_new());
    }
}

void ResourceManager::clear_removal_queue() {
    // todo - not sure if that's correct...
    queued_for_removal_resources[renderer_ctx.get().current_frame_idx].clear();
}

void ResourceManager::recreate(const ResourceHandle handle) {
    auto& removal_queue = queued_for_removal_resources[renderer_ctx.get().current_frame_idx];

    if (!handle_to_kind_mapping.contains(handle)) {
        LOG_ERROR_WITH_FUNC("tried to recreate a non-existent resource");
    }

    const auto error_missing_builder = [&] {
        LOG_ERROR_WITH_FUNC("tried to recreate a resource without a registered builder for that resource");
    };

    switch (handle_to_kind_mapping.at(handle)) {
        case ResourceKind::IMAGE:
            removal_queue.emplace_back(std::move(images.extract(handle).mapped()));
            if (!image_builders.contains(handle)) error_missing_builder();
            images.emplace(handle, image_builders.at(handle).create(renderer_ctx.get()));
            break;
        case ResourceKind::GRAPHICS_PIPELINE:
            removal_queue.emplace_back(std::move(graphics_pipelines.extract(handle).mapped()));
            if (!graphics_pipeline_builders.contains(handle)) error_missing_builder();
            graphics_pipelines.emplace(handle, graphics_pipeline_builders.at(handle).create(renderer_ctx.get()));
            break;
        case ResourceKind::COMPUTE_PIPELINE:
            removal_queue.emplace_back(std::move(compute_pipelines.extract(handle).mapped()));
            if (!compute_pipeline_builders.contains(handle)) error_missing_builder();
            compute_pipelines.emplace(handle, compute_pipeline_builders.at(handle).create(renderer_ctx.get()));
            break;
        case ResourceKind::BUFFER:
            LOG_ERROR_WITH_FUNC("tried to recreate a resource without an associated builder type");
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
