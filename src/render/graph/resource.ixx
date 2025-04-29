module;

export module Cinder.Render.Graph:Resource;

import std;
import vulkan_hpp;

import Cinder.Render.Vulkan;
import Cinder.Render.Mesh;
import Cinder.Globals;

export namespace zrx {
constexpr ResourceHandle FINAL_IMAGE_RESOURCE_HANDLE  = -1;
constexpr std::monostate EMPTY_DESCRIPTOR_SET_BINDING = {};

template<typename T>
concept ResourceLike = requires(T t) {
    { t.name } -> std::same_as<string&>;
};

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
    vk::TextureFlagsZRX flags = vk::TextureFlagBitsZRX::MIPMAPS;
    std::optional<SwizzleDesc> swizzle{};
};

struct TargetTextureResourceDesc {
    string name{};
    vk::Format format{};
    vk::Extent2D extent = {0, 0}; // {0, 0} means we're using the swapchain image's extent
    vk::TextureFlagsZRX flags = vk::TextureFlagBitsZRX::MIPMAPS;
};

struct TransientTextureResourceDesc {
    string name{};
    vk::Format format{};
    vk::Extent2D extent = {0, 0}; // {0, 0} means we're using the swapchain image's extent
    vk::TextureFlagsZRX flags{};
};

struct ModelResourceDesc {
    string name{};
    std::filesystem::path path{};
};

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
    vector<vk::VertexInputBindingDescription> binding_descriptions;
    vector<vk::VertexInputAttributeDescription> attr_descriptions;
    vector<AttachmentFormat> color_formats;
    std::optional<AttachmentFormat> depth_format;

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
    vector<ResourceHandle> used_resources;

    struct CustomProperties {
    } custom_properties;
};
} // zrx
