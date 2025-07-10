module;

export module Cinder.Render.Graph:ResourceManager;

import std;

import Cinder.Render.Vulkan;
import Cinder.Render.Mesh;
import Cinder.Globals;

export namespace zrx {
class ResourceManager {
    std::map<ResourceHandle, unique_ptr<Buffer> > buffers;
    std::map<ResourceHandle, unique_ptr<Texture> > textures;
    std::map<ResourceHandle, unique_ptr<Model> > models;

    using HandlePrioQueue = std::priority_queue<BindlessHandle, std::vector<BindlessHandle>, std::greater<>>;

    std::map<ResourceHandle, BindlessHandle> bindless_handle_mapping;
    HandlePrioQueue free_bindless_handles;

    std::map<ResourceHandle, std::string> resource_names;

    template<typename T>
    struct is_valid_resource_type : std::disjunction<
        std::is_same<T, Texture>,
        std::is_same<T, Buffer>,
        std::is_same<T, Model>> {
    };

public:
    explicit ResourceManager(uint32_t max_bindless_handles);

    template <typename T>
    void add(ResourceHandle handle, unique_ptr<T>&& resource, const std::string& name = "NO_NAME") {
        get_resource_map<T>().emplace(handle, std::move(resource));

        const auto bindless_handle = free_bindless_handles.top();
        bindless_handle_mapping.emplace(handle, bindless_handle);
        free_bindless_handles.pop();

        resource_names.emplace(handle, name);
    }

    [[nodiscard]] const std::string& get_name(const ResourceHandle handle) const { return resource_names.at(handle); }

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

private:
    template <typename T>
        requires is_valid_resource_type<T>::value
    std::map<ResourceHandle, unique_ptr<T> >& get_resource_map() {
        if constexpr (std::is_same_v<T, Buffer>) {
            return buffers;
        } else if constexpr (std::is_same_v<T, Texture>) {
            return textures;
        } else if constexpr (std::is_same_v<T, Model>) {
            return models;
        } else {
            static_assert(false, "invalid type in ResourceManager::get_resource_map");
            return {};
        }
    }
};
} // zrx