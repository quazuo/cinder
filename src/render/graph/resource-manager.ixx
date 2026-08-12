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
    uint32_t frames_in_flight;

    // todo - put everything into one map, big struct in value type

    map<ResourceHandle, ResourceKind> handle_to_kind_mapping;

    map<ResourceHandle, ResourceDescVariant> descriptions;

    // todo - replace by vectors??
    map<ResourceHandle, Buffer> buffers;
    map<ResourceHandle, Image> images;
    map<ResourceHandle, GraphicsPipeline> graphics_pipelines;
    map<ResourceHandle, ComputePipeline> compute_pipelines;

    map<ResourceHandle, BufferBuilder> buffer_builders;
    map<ResourceHandle, ImageBuilder> image_builders;
    map<ResourceHandle, GraphicsPipelineBuilder> graphics_pipeline_builders;
    map<ResourceHandle, ComputePipelineBuilder> compute_pipeline_builders;

    template <typename HandleType>
    using HandlePrioQueue = std::priority_queue<HandleType, std::vector<HandleType>, std::greater<>>;

    HandlePrioQueue<ResourceHandle> free_resource_handles;

    map<ResourceHandle, std::string> resource_names;

    map<ResourceHandle, std::vector<ResourceHandle>> duplicated_resource_proxy_map;
    map<ResourceHandle, ResourceHandle> parent_proxy_handles; // todo: this REALLY sucks..

    vector<vector<ResourceVariant>> queued_for_removal_resources; // one queue for each frame in flight

    // bindless resources

    using BindlessDescriptorSet = FixedDescriptorSet<Image, Image, Buffer>;
    unique_ptr<vk::raii::DescriptorPool> descriptor_pool;
    unique_ptr<BindlessDescriptorSet> bindless_descriptor_set;

    map<ResourceHandle, BindlessHandle> bindless_handle_mapping;
    HandlePrioQueue<BindlessHandle> free_bindless_handles;

    static constexpr uint32_t BINDLESS_ARRAY_SIZE = 256;

    static constexpr uint32_t BINDLESS_SAMPLER_BINDING         = 0;
    static constexpr uint32_t BINDLESS_STORAGE_TEXTURE_BINDING = 1;
    static constexpr uint32_t BINDLESS_UBO_BINDING             = 2;

public:
    explicit ResourceManager(const RendererContext& ctx, uint32_t frames_in_flight);

    template <typename T>
        requires is_resource_type<T>
    auto attach_raw(const ResourceHandle& handle, T&& resource) {
        if (handle_to_kind_mapping.contains(handle)) {
            if (handle_to_kind_mapping.at(handle) != resource_type_to_kind_v<T>) {
                Logger::error("Invalid resource type in ResourceManager::attach_raw: "
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
        const T& created_resource = resource_map.at(handle);

        if constexpr (std::is_same_v<T, Image> || std::is_same_v<T, Buffer>) {
            if (!bindless_handle_mapping.contains(handle)) {
                const auto bindless_handle = get_new_handle(free_bindless_handles);
                bindless_handle_mapping.emplace(handle, bindless_handle);
            }
        }

        if constexpr (std::is_same_v<T, Image>) {
            const auto bindless_handle = bindless_handle_mapping.at(handle);
            bindless_descriptor_set->update_binding<BINDLESS_SAMPLER_BINDING>(created_resource, static_cast<uint32_t>(bindless_handle));

            if (created_resource.is_storage()) {
                bindless_descriptor_set->update_binding<BINDLESS_STORAGE_TEXTURE_BINDING>(created_resource, static_cast<uint32_t>(bindless_handle));
            }
        }

        if constexpr (std::is_same_v<T, Buffer>) {
            // todo: YIKES this sucks i need to rewrite this some time...
            const bool is_ubo = std::holds_alternative<UniformBufferResourceDesc>(descriptions.at(
                parent_proxy_handles.contains(handle) ? parent_proxy_handles.at(handle) : handle
            ));

            if (is_ubo) {
                const auto bindless_handle = bindless_handle_mapping.at(handle);
                bindless_descriptor_set->update_binding<BINDLESS_UBO_BINDING>(created_resource, static_cast<uint32_t>(bindless_handle));
            }
        }
    }

    template <typename T>
        requires is_builder_type<T>
    auto attach_builder(const RendererContext& ctx, const ResourceHandle& handle, T&& builder) {
        if constexpr (std::is_same_v<T, GraphicsPipelineBuilder> || std::is_same_v<T, ComputePipelineBuilder>) {
            vector<vk::DescriptorSetLayout> descriptor_set_layouts;
            descriptor_set_layouts.push_back(*bindless_descriptor_set->get_layout());
            builder.with_descriptor_layouts(descriptor_set_layouts);
        }

        if (duplicated_resource_proxy_map.contains(handle)) {
            for (const auto& sub_handle : duplicated_resource_proxy_map.at(handle)) {
                attach_raw(sub_handle, builder.create(ctx));
            }
        } else {
            attach_raw(handle, builder.create(ctx));
        }

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

        const bool duplicated = needs_duplication(desc);

        if (duplicated) {
            duplicated_resource_proxy_map[handle] = {};

            for (uint32_t i = 0; i < frames_in_flight; i++) {
                const ResourceHandle sub_handle = get_new_handle(free_resource_handles);
                duplicated_resource_proxy_map.at(handle).push_back(sub_handle);
                parent_proxy_handles.emplace(sub_handle, handle);
            }
        }

        if constexpr (is_image_resource_desc_type<T> || is_buffer_resource_desc_type<T>) {
            if (duplicated) {
                for (uint32_t i = 0; i < frames_in_flight; i++) {
                    const auto bindless_handle = get_new_handle(free_bindless_handles);
                    bindless_handle_mapping.emplace(duplicated_resource_proxy_map.at(handle)[i], bindless_handle);
                }
            } else {
                const auto bindless_handle = get_new_handle(free_bindless_handles);
                bindless_handle_mapping.emplace(handle, bindless_handle);
            }
        }

        return handle;
    }

    void commit_bindless_descriptor_updates() const { bindless_descriptor_set->commit_updates(); }

    void recreate(ResourceHandle handle);

    void reload_all_pipelines();

    void clear_removal_queue();

    auto get_name(ResourceHandle handle) const -> const std::string&;

    auto get_bindless_descriptor_set() const -> const vk::raii::DescriptorSet& { return **bindless_descriptor_set; }
    
    auto get_bindless_handle(const ResourceHandle handle) const -> BindlessHandle {
        if (duplicated_resource_proxy_map.contains(handle)) {
            return bindless_handle_mapping.at(get_duplicated_resource_handle(handle));
        }
        return bindless_handle_mapping.at(handle);
    }

    auto get_desc_variant(const ResourceHandle handle) const -> const ResourceDescVariant& { return descriptions.at(handle); }

    template <typename T, typename Self>
        requires is_resource_type<T>
    auto get(this Self&& self, const ResourceHandle handle) -> decltype(auto) {
        decltype(auto) resource_map = std::forward<Self>(self).template get_resource_map<T>();

        if (self.duplicated_resource_proxy_map.contains(handle)) {
            return resource_map.at(self.get_duplicated_resource_handle(handle));
        }

        return resource_map.at(handle);
    }

    template <typename T, typename Self>
        requires is_builder_type<T>
    auto get(this Self&& self, const ResourceHandle handle) -> decltype(auto) {
        return std::forward<Self>(self).template get_builder_map<T>().at(handle);
    }

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

    template <typename T>
        requires is_resource_desc_type<T>
    static auto needs_duplication(const T& desc) -> bool {
        (void) desc;
        return false;
    }

    static auto needs_duplication(const UniformBufferResourceDesc& desc) -> bool {
        return !(desc.flags & UniformBufferFlags::IS_NEVER_UPDATED);
    }

private:
    auto get_duplicated_resource_handle(const ResourceHandle handle) const -> ResourceHandle {
        return duplicated_resource_proxy_map.at(handle)[renderer_ctx.get().current_frame_idx];
    }

    template <typename T, typename Self>
        requires is_resource_type<T>
    auto get_resource_map(this Self&& self) -> decltype(auto) {
        if constexpr (std::is_same_v<T, Buffer>) {
            return (std::forward<Self>(self).buffers);
        } else if constexpr (std::is_same_v<T, Image>) {
            return (std::forward<Self>(self).images);
        } else if constexpr (std::is_same_v<T, GraphicsPipeline>) {
            return (std::forward<Self>(self).graphics_pipelines);
        } else if constexpr (std::is_same_v<T, ComputePipeline>) {
            return (std::forward<Self>(self).compute_pipelines);
        } else {
            static_assert(false, "invalid type in ResourceManager::get_resource_map");
        }
    }

    template <typename T, typename Self>
        requires is_builder_type<T>
    auto get_builder_map(this Self&& self) -> decltype(auto) {
        if constexpr (std::is_same_v<T, BufferBuilder>) {
            return (std::forward<Self>(self).buffer_builders);
        } else if constexpr (std::is_same_v<T, ImageBuilder>) {
            return (std::forward<Self>(self).image_builders);
        } else if constexpr (std::is_same_v<T, GraphicsPipelineBuilder>) {
            return (std::forward<Self>(self).graphics_pipeline_builders);
        } else if constexpr (std::is_same_v<T, ComputePipelineBuilder>) {
            return (std::forward<Self>(self).compute_pipeline_builders);
        } else {
            static_assert(false, "invalid type in ResourceManager::get_builder_map");
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