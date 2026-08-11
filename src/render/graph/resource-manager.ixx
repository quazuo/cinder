module;

export module Cinder.Render.Graph:ResourceManager;

import std;

import :Resource;

import Cinder.Render.Vulkan;
import Cinder.Globals;
import Cinder.Utils;

namespace zrx {
#define NON_PIPELINE_RESOURCE_TYPES Image, Buffer
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
    IMAGE,
    BUFFER,
    GRAPHICS_PIPELINE,
    COMPUTE_PIPELINE,
};

template <typename T>
struct ResourceTypeToKind;

template <>
struct ResourceTypeToKind<Image> {
    static constexpr auto value = ResourceKind::IMAGE;
};

template <>
struct ResourceTypeToKind<Buffer> {
    static constexpr auto value = ResourceKind::BUFFER;
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
struct BindlessHandleTag {};
using BindlessHandle = UniqueHandle<BindlessHandleTag>;

const BindlessHandle PLACEHOLDER_BINDLESS_HANDLE = BindlessHandle::get_new_special();

class ResourceManager {
    reference_wrapper<const RendererContext> renderer_ctx;

    // todo - put everything into one map, big struct in value type

    map<ResourceHandle, ResourceKind> handle_to_kind_mapping;

    map<ResourceHandle, ResourceDescVariant> descriptions;

    // todo - replace by vectors??
    map<ResourceHandle, Buffer> buffers;
    map<ResourceHandle, Image> images;
    map<ResourceHandle, GraphicsPipeline> graphics_pipelines;
    map<ResourceHandle, ComputePipeline> compute_pipelines;

    map<ResourceHandle, ImageBuilder> image_builders;
    map<ResourceHandle, GraphicsPipelineBuilder> graphics_pipeline_builders;
    map<ResourceHandle, ComputePipelineBuilder> compute_pipeline_builders;

    template <typename HandleType>
    using HandlePrioQueue = std::priority_queue<HandleType, std::vector<HandleType>, std::greater<>>;

    map<ResourceHandle, BindlessHandle> bindless_handle_mapping;
    HandlePrioQueue<BindlessHandle> free_bindless_handles;

    HandlePrioQueue<ResourceHandle> free_resource_handles;

    map<ResourceHandle, std::string> resource_names;

    vector<vector<ResourceVariant>> queued_for_removal_resources; // one queue for each frame in flight

public:
    explicit ResourceManager(const RendererContext& ctx, uint32_t max_bindless_handles, uint32_t frames_in_flight);

    template <typename T>
        requires is_resource_type<T>
    auto attach_raw(const ResourceHandle& handle, T&& resource) {
        if (handle_to_kind_mapping.contains(handle)) {
            if (handle_to_kind_mapping.at(handle) != resource_type_to_kind_v<T>) {
                Logger::error("Invalid resource type in ResourceManager::attach_raw : "
                              "resource type doesn't match the type of previously attached resource");
            }
        } else {
            handle_to_kind_mapping.emplace(handle, resource_type_to_kind_v<T>);
        }

        auto& resource_map = get_resource_map<T>();

        if (resource_map.contains(handle)) {
            auto& removal_queue = queued_for_removal_resources[renderer_ctx.get().current_frame_idx];
            removal_queue.emplace_back(std::move(resource_map.extract(handle).mapped()));
        }

        resource_map.emplace(handle, std::move(resource));

        if constexpr (std::is_same_v<T, Image> || std::is_same_v<T, Buffer>) {
            if (!bindless_handle_mapping.contains(handle)) {
                const auto bindless_handle = get_new_handle(free_bindless_handles);
                bindless_handle_mapping.emplace(handle, bindless_handle);
            }
        }
    }

    template <typename T>
        requires is_builder_type<T>
    auto attach_builder(const RendererContext& ctx, const ResourceHandle& handle, T&& builder) {
        attach_raw(handle, builder.create(ctx));

        auto& builder_map = get_builder_map<T>();
        
        if (builder_map.contains(handle)) {
            builder_map.erase(handle);
        }

        builder_map.emplace(handle, std::move(builder));
    }

    template <typename T>
        requires is_resource_desc_type<T>
    auto add_from_desc(T&& desc) -> ResourceHandle {
        const ResourceHandle handle = get_new_handle(free_resource_handles);

        handle_to_kind_mapping.emplace(handle, resource_type_to_kind_v<resource_desc_to_type_t<T>>);
        descriptions.emplace(handle, desc);

        if constexpr (!is_pipeline_resource_desc_type<T>) {
            resource_names.emplace(handle, desc.name);
        }

        if constexpr (is_image_resource_desc_type<T> || is_buffer_resource_desc_type<T>) {
            const auto bindless_handle = get_new_handle(free_bindless_handles);
            bindless_handle_mapping.emplace(handle, bindless_handle);
        }

        return handle;
    }

    void recreate(ResourceHandle handle);

    void reload_all_pipelines();

    void clear_removal_queue();

    auto get_name(ResourceHandle handle) const -> const std::string&;
    
    auto get_bindless_handle(const ResourceHandle handle) const -> BindlessHandle { return bindless_handle_mapping.at(handle); }

    auto get_desc_variant(const ResourceHandle handle) const -> const ResourceDescVariant& { return descriptions.at(handle); }

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

    auto get_all_resource_handles_range() const { return handle_to_kind_mapping | views::keys; }

    template <typename T>
        requires is_resource_type<T>
    auto contains(const ResourceHandle handle) const -> bool { return get_resource_map<T>().contains(handle); }

    template <typename T>
        requires is_builder_type<T>
    auto contains(const ResourceHandle handle) const -> bool { return get_builder_map<T>().contains(handle); }

    auto has_attached_resource(const ResourceHandle handle) const -> bool {
        switch (handle_to_kind_mapping.at(handle)) {
            case ResourceKind::BUFFER:
                return buffers.contains(handle);
            case ResourceKind::IMAGE:
                return images.contains(handle);
            case ResourceKind::GRAPHICS_PIPELINE:
                return graphics_pipelines.contains(handle);
            case ResourceKind::COMPUTE_PIPELINE:
                return compute_pipelines.contains(handle);
            default:
                Logger::error("unimplemented switch case in ResourceManager::has_attached_resource");
                return true; // silence warning
        }
    }

private:
    template <typename T>
        requires is_resource_type<T>
    auto get_resource_map() const -> const map<ResourceHandle, T>& {
        if constexpr (std::is_same_v<T, Buffer>) {
            return buffers;
        } else if constexpr (std::is_same_v<T, Image>) {
            return images;
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
        if constexpr (std::is_same_v<T, ImageBuilder>) {
            return image_builders;
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