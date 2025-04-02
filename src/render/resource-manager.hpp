#pragma once

#include <map>
#include <queue>

#include "globals.hpp"

namespace zrx {
class Buffer;
class Texture;
class Model;

class ResourceManager {
    std::map<ResourceHandle, unique_ptr<Buffer> > buffers;
    std::map<ResourceHandle, unique_ptr<Texture> > textures;
    std::map<ResourceHandle, unique_ptr<Model> > models;

    using HandlePrioQueue = std::priority_queue<BindlessHandle, std::vector<BindlessHandle>, std::greater<>>;

    std::map<ResourceHandle, BindlessHandle> bindless_handle_mapping;
    HandlePrioQueue free_texture_bindless_handles;
    HandlePrioQueue free_ubo_bindless_handles;

public:
    explicit ResourceManager(uint32_t max_bindless_handles);

    void add(ResourceHandle handle, unique_ptr<Buffer>&& buffer);
    void add(ResourceHandle handle, unique_ptr<Texture>&& texture);
    void add(ResourceHandle handle, unique_ptr<Model>&& model);

    [[nodiscard]] BindlessHandle get_bindless_handle(const ResourceHandle handle) const { return bindless_handle_mapping.at(handle); }

    [[nodiscard]] const Buffer& get_buffer(const ResourceHandle handle) const { return *buffers.at(handle); }
    [[nodiscard]] const Texture& get_texture(const ResourceHandle handle) const { return *textures.at(handle); }
    [[nodiscard]] const Model& get_model(const ResourceHandle handle) const { return *models.at(handle); }

    [[nodiscard]] Buffer& get_buffer(const ResourceHandle handle) { return *buffers.at(handle); }
    [[nodiscard]] Texture& get_texture(const ResourceHandle handle) { return *textures.at(handle); }
    [[nodiscard]] Model& get_model(const ResourceHandle handle) { return *models.at(handle); }

    [[nodiscard]] bool contains_buffer(const ResourceHandle handle) const { return buffers.contains(handle); }
    [[nodiscard]] bool contains_texture(const ResourceHandle handle) const { return textures.contains(handle); }
    [[nodiscard]] bool contains_model(const ResourceHandle handle) const { return models.contains(handle); }
};
} // zrx