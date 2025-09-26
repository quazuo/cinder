module;

export module Cinder.Render.Graph:ResourceManager;

import std;

import :Resource;

import Cinder.Render.Vulkan;
import Cinder.Render.Mesh;
import Cinder.Globals;

namespace zrx {
template<typename T>
concept is_valid_resource_type =
    std::same_as<T, Texture>
    || std::same_as<T, Buffer>
    || std::same_as<T, Model>;

template <typename Builder, typename Resource>
concept is_valid_builder_type = requires (Builder builder) {
    { builder.create(RendererContext()) } -> std::convertible_to<unique_ptr<Resource>>;
} && is_valid_resource_type<Resource>;
}

export namespace zrx {
constexpr BindlessHandle EMPTY_TEXTURE_BINDLESS_HANDLE = 0xffffffff;

class ResourceManager {
public:
    struct MaterialTextureHandles {
        BindlessHandle base_color;
        BindlessHandle normal;
        BindlessHandle orm;
    };

private:
    reference_wrapper<const RendererContext> renderer_ctx;

    // todo - replace by vectors
    map<ResourceHandle, unique_ptr<Buffer> > buffers;
    map<ResourceHandle, unique_ptr<Texture> > textures;
    map<ResourceHandle, unique_ptr<Model> > models;

    using ModelMaterialHandle = uint32_t;
    map<ResourceHandle, vector<ModelMaterialHandle>> models_to_materials;
    map<ModelMaterialHandle, MaterialTextureHandles> materials;

    map<ResourceHandle, TextureBuilder> texture_builders;

    template <typename HandleType>
    using HandlePrioQueue = std::priority_queue<HandleType, std::vector<HandleType>, std::greater<>>;

    map<ResourceHandle, BindlessHandle> bindless_handle_mapping;
    HandlePrioQueue<BindlessHandle> free_bindless_handles;

    HandlePrioQueue<ModelMaterialHandle> free_model_mat_handles;

    map<ResourceHandle, std::string> resource_names;

public:
    explicit ResourceManager(const RendererContext& ctx, uint32_t max_bindless_handles);

    template <typename T>
        requires is_valid_resource_type<T>
    void add(const ResourceHandle handle, unique_ptr<T>&& resource, const std::string& name = "NO_NAME") {
        if constexpr (std::is_same_v<T, Model>) {
            add_model_materials(handle, *resource);
        }

        get_resource_map<T>().emplace(handle, std::move(resource));

        const auto bindless_handle = get_new_handle(free_bindless_handles);
        bindless_handle_mapping.emplace(handle, bindless_handle);

        resource_names.emplace(handle, name);
    }

    template <typename T, typename U>
        requires is_valid_builder_type<T, U>
    void add_from_builder(const ResourceHandle handle, T&& builder, const std::string& name = "NO_NAME") {
        texture_builders.emplace(handle, std::move(builder));
        add(handle, builder.create(renderer_ctx.get()), name);
    }

    void add_model_materials(const ResourceHandle handle, const Model& model) {
        vector<ModelMaterialHandle> material_handles;

        for (size_t i = 0; i < model.get_materials().size(); i++) {
            const Material& material = model.get_materials()[i];
            const ModelMaterialHandle new_mat_handle = get_new_handle(free_model_mat_handles);

            material_handles.emplace_back(new_mat_handle);

            const auto base_color_handle = material.base_color ? get_new_handle(free_bindless_handles) : EMPTY_TEXTURE_BINDLESS_HANDLE;
            const auto normal_handle     = material.normal     ? get_new_handle(free_bindless_handles) : EMPTY_TEXTURE_BINDLESS_HANDLE;
            const auto orm_handle        = material.orm        ? get_new_handle(free_bindless_handles) : EMPTY_TEXTURE_BINDLESS_HANDLE;

            materials.emplace(new_mat_handle, MaterialTextureHandles {
                .base_color = base_color_handle,
                .normal = normal_handle,
                .orm = orm_handle,
            });
        }

        models_to_materials.emplace(handle, material_handles);
    }

    void recreate(const ResourceHandle handle) {
        if (texture_builders.contains(handle)) {
            textures[handle] = texture_builders[handle].create(renderer_ctx.get());

        } else {
            throw std::runtime_error("unsupported resource type in ResourceManager::recreate");
        }
    }

    auto get_name(const ResourceHandle handle) const -> const std::string& {
        if (handle == FINAL_IMAGE_HANDLE) return FINAL_IMAGE_NAME;
        return resource_names.at(handle);
    }

    auto get_bindless_handle(const ResourceHandle handle) const -> BindlessHandle { return bindless_handle_mapping.at(handle); }

    auto get_buffer(const ResourceHandle handle) const      -> const Buffer&  { return *buffers.at(handle); }
    auto get_texture(const ResourceHandle handle) const     -> const Texture& { return *textures.at(handle); }
    auto get_model(const ResourceHandle handle) const       -> const Model&   { return *models.at(handle); }
    auto get_tex_builder(const ResourceHandle handle) const -> const TextureBuilder& { return texture_builders.at(handle); }

    auto get_buffer(const ResourceHandle handle)        -> Buffer&  { return *buffers.at(handle); }
    auto get_texture(const ResourceHandle handle)       -> Texture& { return *textures.at(handle); }
    auto get_model(const ResourceHandle handle)         -> Model&   { return *models.at(handle); }
    auto get_tex_builder(const ResourceHandle handle)   -> TextureBuilder& { return texture_builders.at(handle); }

    auto get_model_material_handles(const ResourceHandle handle) const { return models_to_materials.at(handle); }
    auto get_material_tex_handles(const ModelMaterialHandle handle) const { return materials.at(handle); }

    auto get_model_mat_tex_handles(const ResourceHandle handle) const {
        vector<MaterialTextureHandles> result;

        for (auto& material_handle : models_to_materials.at(handle)) {
            result.emplace_back(materials.at(material_handle));
        }

        return result;
    }

    auto contains_buffer(const ResourceHandle handle) const         -> bool { return buffers.contains(handle); }
    auto contains_texture(const ResourceHandle handle) const        -> bool { return textures.contains(handle); }
    auto contains_model(const ResourceHandle handle) const          -> bool { return models.contains(handle); }
    auto contains_tex_builder(const ResourceHandle handle) const    -> bool { return texture_builders.contains(handle); }

private:
    template <typename T>
        requires is_valid_resource_type<T>
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

    template <typename HandleType>
    auto get_new_handle(HandlePrioQueue<HandleType>& handle_prio_queue) {
        const auto handle = handle_prio_queue.top();
        handle_prio_queue.pop();
        return handle;
    }
};
} // zrx