#pragma once

#include <map>

#include "globals.hpp"

namespace zrx {
class Buffer;
class Texture;
class Model;

class ResourceManager {
    std::map<ResourceHandle, unique_ptr<Buffer> > buffers;
    std::map<ResourceHandle, unique_ptr<Texture> > textures;
    std::map<ResourceHandle, unique_ptr<Model> > models;

public:
    void add(const ResourceHandle handle, unique_ptr<Buffer>&& buffer) { buffers.emplace(handle, std::move(buffer)); }
    void add(const ResourceHandle handle, unique_ptr<Texture>&& texture) { textures.emplace(handle, std::move(texture)); }
    void add(const ResourceHandle handle, unique_ptr<Model>&& model) { models.emplace(handle, std::move(model)); }

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