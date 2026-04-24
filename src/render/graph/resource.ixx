module;

export module Cinder.Render.Graph:Resource;

import std;
import vulkan_hpp;

import Cinder.Render.Vulkan;
import Cinder.Render.Mesh;
import Cinder.Globals;
import Cinder.Utils;

export namespace zrx {
struct ResourceHandleTag {};
using ResourceHandle = UniqueHandle<ResourceHandleTag>;

const ResourceHandle FINAL_IMAGE_HANDLE      = ResourceHandle::get_new_special();
const ResourceHandle CURRENT_MATERIAL_HANDLE = ResourceHandle::get_new_special();

constexpr uint32_t MAX_RESOURCE_COUNT = 1 << 16;

const string FINAL_IMAGE_NAME = "final-image";
constexpr std::monostate EMPTY_DESCRIPTOR_SET_BINDING = {};

struct VertexBufferResourceDesc {
    string name{};
    vk::DeviceSize size = 0;
    const void *data;
};

struct UniformBufferResourceDesc {
    string name{};
    vk::DeviceSize size = 0;
};

struct ExternalTextureResourceDesc {
    string name{};
    vector<std::filesystem::path> paths{};
    vk::Format format{};
    TextureOverrides overrides{};
    TextureFlags flags{};
    optional<SwizzleDesc> swizzle{};
};

struct TargetTextureResourceDesc {
    string name{};
    vk::Format format{};
    vk::Extent2D extent = {0, 0}; // {0, 0} means we're using the swapchain image's extent
    TextureOverrides overrides{};
    TextureFlags flags{};
};

struct PersistentTextureResourceDesc {
    string name{};
    vk::Format format{};
    vk::Extent2D extent = {0, 0}; // {0, 0} means we're using the swapchain image's extent
    TextureOverrides overrides{};
    TextureFlags flags{};
};

struct TransientTextureResourceDesc {
    string name{};
    vk::Format format{};
    vk::Extent2D extent = {0, 0}; // {0, 0} means we're using the swapchain image's extent
    TextureOverrides overrides{};
    TextureFlags flags{};
};

struct ModelResourceDesc {
    string name{};
    std::filesystem::path path{};
    bool has_materials = false;
};

concept is_buffer_resource_desc_type = is_one_of<T,
    VertexBufferResourceDesc,
    UniformBufferResourceDesc>;

concept is_texture_resource_desc_type = is_one_of<T,
    ExternalTextureResourceDesc,
    TargetTextureResourceDesc,
    PersistentTextureResourceDesc,
    TransientTextureResourceDesc>;

concept is_model_resource_desc_type = is_one_of<T,
    ModelResourceDesc>;

concept is_resource_desc_type = is_one_of<T,
    is_buffer_resource_desc_type,
    is_texture_resource_desc_type,
    is_model_resource_desc_type>;

// basically same purpose as std::monostate but with a specific name
struct FinalImageFormatPlaceholder {};

constexpr auto FINAL_FORMAT = FinalImageFormatPlaceholder();

enum class ShaderBindingType {
    Empty,
    SampledTexture,
    StorageTexture,
    UniformBuffer,
    StorageBuffer,
};

struct GraphicsPipelineDesc {
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

struct ComputePipelineDesc {
    std::filesystem::path path;

    struct CustomProperties {
    } custom_properties;
};
} // zrx
