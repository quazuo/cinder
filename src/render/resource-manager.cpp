#include "resource-manager.hpp"

#include "vk/buffer.hpp"
#include "mesh/model.hpp"
#include "vk/image.hpp"

namespace zrx {
ResourceManager::ResourceManager(const uint32_t max_bindless_handles) {
    for (uint32_t i = 0; i < max_bindless_handles; i++) {
        free_texture_bindless_handles.push(i);
        free_ubo_bindless_handles.push(i);
    }
}

void ResourceManager::add(const ResourceHandle handle, unique_ptr<Buffer>&& buffer) {
    buffers.emplace(handle, std::move(buffer));

    const auto bindless_handle = free_ubo_bindless_handles.top();
    bindless_handle_mapping.emplace(handle, bindless_handle);
    free_ubo_bindless_handles.pop();
}

void ResourceManager::add(const ResourceHandle handle, unique_ptr<Texture>&& texture) {
    textures.emplace(handle, std::move(texture));

    const auto bindless_handle = free_texture_bindless_handles.top();
    bindless_handle_mapping.emplace(handle, bindless_handle);
    free_texture_bindless_handles.pop();
}

void ResourceManager::add(const ResourceHandle handle, unique_ptr<Model>&& model) {
    models.emplace(handle, std::move(model));
}
}
