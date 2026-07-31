module;

export module Cinder.Render.Graph:Resource;

import std;
import vulkan;

import Cinder.Render.Vulkan;
import Cinder.Globals;
import Cinder.Utils;

export namespace zrx {
struct ResourceHandleTag {};
using ResourceHandle = UniqueHandle<ResourceHandleTag>;

const ResourceHandle FINAL_IMAGE_HANDLE  = ResourceHandle::get_new_special();

constexpr uint32_t MAX_RESOURCE_COUNT = 1 << 16;

const string FINAL_IMAGE_NAME = "final-image";
constexpr std::monostate EMPTY_DESCRIPTOR_SET_BINDING = {};

struct VertexBufferResourceDesc {
    string name{};
    vk::DeviceSize size = 0;
    const void *data;
};

struct IndexBufferResourceDesc {
    string name{};
    vk::DeviceSize size = 0;
    const void *data;
};

struct UniformBufferResourceDesc {
    string name{};
    vk::DeviceSize size = 0;
    optional<const void *> data;
};

struct ExternalTextureResourceDesc {
    string name{};
    vector<std::filesystem::path> paths{};
    vk::Format format{};
    ImageOverrides overrides{};
    ImageFlags flags{};
    optional<SwizzleDesc> swizzle{};
};

struct TargetTextureResourceDesc {
    string name{};
    vk::Format format{};
    vk::Extent2D extent = {0, 0}; // {0, 0} means we're using the swapchain image's extent
    ImageOverrides overrides{};
    ImageFlags flags{};
};

struct PersistentTextureResourceDesc {
    string name{};
    vk::Format format{};
    vk::Extent2D extent = {0, 0}; // {0, 0} means we're using the swapchain image's extent
    ImageOverrides overrides{};
    ImageFlags flags{};
};

struct TransientTextureResourceDesc {
    string name{};
    vk::Format format{};
    vk::Extent2D extent = {0, 0}; // {0, 0} means we're using the swapchain image's extent
    ImageOverrides overrides{};
    ImageFlags flags{};
};

// basically same purpose as std::monostate but with a specific name
struct FinalImageFormatPlaceholder {};
constexpr auto FINAL_FORMAT = FinalImageFormatPlaceholder();

struct GraphicsPipelineResourceDesc {
    using AttachmentFormat = std::variant<vk::Format, FinalImageFormatPlaceholder>;

    std::filesystem::path vertex_path;
    std::filesystem::path fragment_path;
    vector<vk::VertexInputBindingDescription> vertex_bindings;
    vector<vk::VertexInputAttributeDescription> vertex_attributes;
    vector<AttachmentFormat> color_formats;
    optional<AttachmentFormat> depth_format;

    struct CustomProperties {
        bool use_msaa                  = false;
        bool disable_depth_test        = false;
        bool disable_depth_write       = false;
        vk::CompareOp depth_compare_op = vk::CompareOp::eLess;
        vk::CullModeFlagBits cull_mode = vk::CullModeFlagBits::eBack;
        uint32_t multiview_count       = 1;
    } custom_properties;
};

struct ComputePipelineResourceDesc {
    std::filesystem::path path;

    struct CustomProperties {
    } custom_properties;
};

#define BUFFER_RESOURCE_DESC_TYPES      \
    VertexBufferResourceDesc,           \
    IndexBufferResourceDesc,            \
    UniformBufferResourceDesc

#define IMAGE_RESOURCE_DESC_TYPES       \
    ExternalTextureResourceDesc,        \
    TargetTextureResourceDesc,          \
    PersistentTextureResourceDesc,      \
    TransientTextureResourceDesc

#define PIPELINE_RESOURCE_DESC_TYPES    \
    GraphicsPipelineResourceDesc,       \
    ComputePipelineResourceDesc

#define ALL_RESOURCE_DESC_TYPES         \
    BUFFER_RESOURCE_DESC_TYPES,         \
    IMAGE_RESOURCE_DESC_TYPES,          \
    PIPELINE_RESOURCE_DESC_TYPES

using BufferResourceDescVariant   = variant<BUFFER_RESOURCE_DESC_TYPES>;
using TextureResourceDescVariant  = variant<IMAGE_RESOURCE_DESC_TYPES>;
using PipelineResourceDescVariant = variant<PIPELINE_RESOURCE_DESC_TYPES>;
using ResourceDescVariant         = variant<ALL_RESOURCE_DESC_TYPES>;

template <typename T>
concept is_buffer_resource_desc_type = is_one_of<T, BUFFER_RESOURCE_DESC_TYPES>;

template <typename T>
concept is_image_resource_desc_type = is_one_of<T, IMAGE_RESOURCE_DESC_TYPES>;

template <typename T>
concept is_pipeline_resource_desc_type = is_one_of<T, PIPELINE_RESOURCE_DESC_TYPES>;

template <typename T>
concept is_resource_desc_type =
    is_buffer_resource_desc_type<T>
    || is_image_resource_desc_type<T>
    || is_pipeline_resource_desc_type<T>;

template <typename T>
struct ResourceDescToType;

template <typename T>
    requires is_buffer_resource_desc_type<T>
struct ResourceDescToType<T> {
    using type = Buffer;
};

template <typename T>
    requires is_image_resource_desc_type<T>
struct ResourceDescToType<T> {
    using type = Image;
};

template <>
struct ResourceDescToType<GraphicsPipelineResourceDesc> {
    using type = GraphicsPipeline;
};

template <>
struct ResourceDescToType<ComputePipelineResourceDesc> {
    using type = ComputePipeline;
};

template <typename T>
using resource_desc_to_type_t = ResourceDescToType<T>::type;

enum class ShaderBindingType {
    Empty,
    SampledTexture,
    StorageTexture,
    UniformBuffer,
    StorageBuffer,
};
} // zrx
