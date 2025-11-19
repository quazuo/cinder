module;

export module Cinder.Render.Graph:ResourceManager;

import std;

import :Resource;

import Cinder.Render.Vulkan;
import Cinder.Render.Mesh;
import Cinder.Globals;
import Cinder.Utils;

namespace zrx {
#define NON_PIPELINE_RESOURCE_TYPES Texture, Buffer, Model
#define PIPELINE_RESOURCE_TYPES GraphicsPipeline, ComputePipeline
#define RESOURCE_TYPES NON_PIPELINE_RESOURCE_TYPES, PIPELINE_RESOURCE_TYPES

using ResourceVariant = std::variant<RESOURCE_TYPES>;

template<typename T>
concept is_resource_type = is_one_of<T, RESOURCE_TYPES>;

template <typename T>
concept is_builder_type = requires (T builder) {
    { builder.create(RendererContext()) } -> is_resource_type;
};

template <typename T>
concept is_pipeline_type = is_one_of<T, PIPELINE_RESOURCE_TYPES>;

enum class ResourceKind {
    TEXTURE,
    BUFFER,
    MODEL,
    GRAPHICS_PIPELINE,
    COMPUTE_PIPELINE,
};

template <typename T>
struct ResourceTypeToKind;

template <>
struct ResourceTypeToKind<Texture> {
    static constexpr auto value = ResourceKind::TEXTURE;
};

template <>
struct ResourceTypeToKind<Buffer> {
    static constexpr auto value = ResourceKind::BUFFER;
};

template <>
struct ResourceTypeToKind<Model> {
    static constexpr auto value = ResourceKind::MODEL;
};

template <>
struct ResourceTypeToKind<GraphicsPipeline> {
    static constexpr auto value = ResourceKind::GRAPHICS_PIPELINE;
};

template <>
struct ResourceTypeToKind<ComputePipeline> {
    static constexpr auto value = ResourceKind::COMPUTE_PIPELINE;
};

template <typename T>
inline constexpr ResourceKind resource_type_to_kind_v = ResourceTypeToKind<T>::value;
} // zrx

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

    map<ResourceHandle, ResourceKind> handle_to_kind_mapping;

    // todo - replace by vectors
    map<ResourceHandle, Buffer> buffers;
    map<ResourceHandle, Texture> textures;
    map<ResourceHandle, Model> models;
    map<ResourceHandle, GraphicsPipeline> graphics_pipelines;
    map<ResourceHandle, ComputePipeline> compute_pipelines;

    using ModelMaterialHandle = uint32_t;
    map<ResourceHandle, vector<ModelMaterialHandle>> models_to_materials;
    map<ModelMaterialHandle, MaterialTextureHandles> materials;

    map<ResourceHandle, TextureBuilder> texture_builders;
    map<ResourceHandle, GraphicsPipelineBuilder> graphics_pipeline_builders;
    map<ResourceHandle, ComputePipelineBuilder> compute_pipeline_builders;

    template <typename HandleType>
    using HandlePrioQueue = std::priority_queue<HandleType, std::vector<HandleType>, std::greater<>>;

    map<ResourceHandle, BindlessHandle> bindless_handle_mapping;
    HandlePrioQueue<BindlessHandle> free_bindless_handles;

    HandlePrioQueue<ModelMaterialHandle> free_model_mat_handles;

    map<ResourceHandle, std::string> resource_names;

    vector<vector<ResourceVariant>> queued_for_removal_resources; // one queue for each frame in flight

public:
    explicit ResourceManager(const RendererContext& ctx, uint32_t max_bindless_handles, uint32_t frames_in_flight);

    template <typename T>
        requires is_resource_type<T>
    void add(const ResourceHandle handle, T&& resource, const std::string& name = "NO_NAME") {
        handle_to_kind_mapping.emplace(handle, resource_type_to_kind_v<T>);

        if constexpr (std::is_same_v<T, Model>) {
            add_model_materials(handle, resource);
        }

        auto& resource_map = get_resource_map<T>();

        if (resource_map.contains(handle)) {
            queued_for_removal_resources[renderer_ctx.get().current_frame_idx]
                .emplace_back(std::move(resource_map.extract(handle).mapped()));
        }

        resource_map.emplace(handle, std::move(resource));

        const auto bindless_handle = get_new_handle(free_bindless_handles);
        bindless_handle_mapping.emplace(handle, bindless_handle);

        resource_names.emplace(handle, name);
    }

    template <typename T>
        requires is_builder_type<T>
    void add_from_builder(const ResourceHandle handle, T&& builder, const std::string& name = "NO_NAME") {
        using BuiltResourceType = std::invoke_result_t<decltype(&T::create), T, const RendererContext&>;
        handle_to_kind_mapping.emplace(handle, resource_type_to_kind_v<BuiltResourceType>);

        add(handle, builder.create(renderer_ctx.get()), name);
        get_builder_map<T>().emplace(handle, std::move(builder));
    }

    void add_model_materials(ResourceHandle handle, const Model& model);

    void recreate(ResourceHandle handle);

    void reload_all_pipelines();

    void clear_removal_queue();

    auto get_name(ResourceHandle handle) const -> const std::string&;

    auto get_bindless_handle(const ResourceHandle handle) const -> BindlessHandle { return bindless_handle_mapping.at(handle); }

    template <typename T>
        requires is_resource_type<T>
    auto get(const ResourceHandle handle) const -> const T& { return get_resource_map<T>().at(handle); }

    template <typename T>
        requires is_resource_type<T>
    auto get(const ResourceHandle handle) -> T& { return get_resource_map<T>().at(handle); }

    template <typename T>
        requires is_builder_type<T>
    auto get(const ResourceHandle handle) const -> const T& { return get_builder_map<T>().at(handle); }

    template <typename T>
        requires is_builder_type<T>
    auto get(const ResourceHandle handle) -> T& { return get_builder_map<T>().at(handle); }

    auto get_model_material_handles(const ResourceHandle handle) const { return models_to_materials.at(handle); }
    auto get_material_tex_handles(const ModelMaterialHandle handle) const { return materials.at(handle); }

    auto get_model_mat_tex_handles(ResourceHandle handle) const -> vector<MaterialTextureHandles>;

    template <typename T>
        requires is_resource_type<T>
    auto contains(const ResourceHandle handle) const -> bool { return get_resource_map<T>().contains(handle); }

    template <typename T>
        requires is_builder_type<T>
    auto contains(const ResourceHandle handle) const -> bool { return get_builder_map<T>().contains(handle); }

private:
    template <typename T>
        requires is_resource_type<T>
    auto get_resource_map() const -> const map<ResourceHandle, T>& {
        if constexpr (std::is_same_v<T, Buffer>) {
            return buffers;
        } else if constexpr (std::is_same_v<T, Texture>) {
            return textures;
        } else if constexpr (std::is_same_v<T, Model>) {
            return models;
        } else if constexpr (std::is_same_v<T, GraphicsPipeline>) {
            return graphics_pipelines;
        } else if constexpr (std::is_same_v<T, ComputePipeline>) {
            return compute_pipelines;
        } else {
            static_assert(false, "invalid type in ResourceManager::get_resource_map");
            return {};
        }
    }

    // non-const version of the above fn
    template <typename T>
        requires is_resource_type<T>
    auto get_resource_map() -> map<ResourceHandle, T>& {
        return const_cast<map<ResourceHandle, T> &>(
            static_cast<const ResourceManager *>(this)->get_resource_map<T>()
        );
    }

    template <typename T>
        requires is_builder_type<T>
    auto get_builder_map() const -> const map<ResourceHandle, T>& {
        if constexpr (std::is_same_v<T, TextureBuilder>) {
            return texture_builders;
        } else if constexpr (std::is_same_v<T, GraphicsPipelineBuilder>) {
            return graphics_pipeline_builders;
        } else if constexpr (std::is_same_v<T, ComputePipelineBuilder>) {
            return compute_pipeline_builders;
        } else {
            static_assert(false, "invalid type in ResourceManager::get_builder_map");
            return {};
        }
    }

    // non-const version of the above fn
    template <typename T>
        requires is_builder_type<T>
    auto get_builder_map() -> map<ResourceHandle, T>& {
        return const_cast<map<ResourceHandle, T> &>(
            static_cast<const ResourceManager *>(this)->get_builder_map<T>()
        );
    }

    template <typename HandleType>
    auto get_new_handle(HandlePrioQueue<HandleType>& handle_prio_queue) {
        const auto handle = handle_prio_queue.top();
        handle_prio_queue.pop();
        return handle;
    }
};
} // zrx