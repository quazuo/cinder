module;

export module Cinder.Render.Graph:ResourceManager;

import std;

import :Resource;

import Cinder.Render.Vulkan;
import Cinder.Render.Mesh;
import Cinder.Globals;

export namespace zrx {
class ResourceManager {
    // todo - replace by vectors
    map<ResourceHandle, unique_ptr<Buffer> > buffers;
    map<ResourceHandle, unique_ptr<Texture> > textures;
    map<ResourceHandle, unique_ptr<Model> > models;

    using HandlePrioQueue = std::priority_queue<BindlessHandle, std::vector<BindlessHandle>, std::greater<>>;

    map<ResourceHandle, BindlessHandle> bindless_handle_mapping;
    HandlePrioQueue free_bindless_handles;

    map<ResourceHandle, std::string> resource_names;

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

    auto get_name(const ResourceHandle handle) const -> const std::string& {
        if (handle == FINAL_IMAGE_HANDLE) return FINAL_IMAGE_NAME;
        return resource_names.at(handle);
    }

    auto get_bindless_handle(const ResourceHandle handle) const -> BindlessHandle { return bindless_handle_mapping.at(handle); }

    auto get_buffer(const ResourceHandle handle) const  -> const Buffer&  { return *buffers.at(handle); }
    auto get_texture(const ResourceHandle handle) const -> const Texture& { return *textures.at(handle); }
    auto get_model(const ResourceHandle handle) const   -> const Model&   { return *models.at(handle); }

    auto get_buffer(const ResourceHandle handle)  -> Buffer&  { return *buffers.at(handle); }
    auto get_texture(const ResourceHandle handle) -> Texture& { return *textures.at(handle); }
    auto get_model(const ResourceHandle handle)   -> Model&   { return *models.at(handle); }

    auto contains_buffer(const ResourceHandle handle) const  -> bool { return buffers.contains(handle); }
    auto contains_texture(const ResourceHandle handle) const -> bool { return textures.contains(handle); }
    auto contains_model(const ResourceHandle handle) const   -> bool { return models.contains(handle); }

private:
    template <typename T>
        requires is_valid_resource_type<T>::value
    auto get_resource_map() -> map<ResourceHandle, unique_ptr<T> >& {
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