#pragma once

#include <filesystem>
#include <string>
#include <variant>
#include <set>

#include "src/render/mesh/model.hpp"
#include "src/render/vk/image.hpp"
#include "src/render/vk/buffer.hpp"

namespace zrx {
static constexpr ResourceHandle FINAL_IMAGE_RESOURCE_HANDLE  = -1;
static constexpr std::monostate EMPTY_DESCRIPTOR_SET_BINDING = {};

template<typename T>
concept ResourceLike = requires(T t) {
    { t.name } -> std::same_as<string&>;
};

struct VertexBufferResourceDesc {
    string name;
    vk::DeviceSize size;
    const void *data;
};

struct UniformBufferResourceDesc {
    string name;
    vk::DeviceSize size;
};

struct ExternalTextureResourceDesc {
    string name;
    vector<std::filesystem::path> paths;
    vk::Format format;
    vk::TextureFlagsZRX tex_flags = vk::TextureFlagBitsZRX::MIPMAPS;
    std::optional<SwizzleDesc> swizzle{};
};

struct TargetTextureResourceDesc {
    string name;
    vk::Format format;
    vk::Extent2D extent = {0, 0}; // {0, 0} means we're using the swapchain image's extent
    vk::TextureFlagsZRX flags = vk::TextureFlagBitsZRX::MIPMAPS;
};

struct TransientTextureResourceDesc {
    string name;
    vk::Format format;
    vk::Extent2D extent = {0, 0}; // {0, 0} means we're using the swapchain image's extent
    vk::TextureFlagsZRX flags{};
};

struct ModelResourceDesc {
    string name;
    std::filesystem::path path;
};

// basically same purpose as std::monostate but with a specific name
struct FinalImageFormatPlaceholder {
};

struct GraphicsPipelineDesc {
    using AttachmentFormat = std::variant<vk::Format, FinalImageFormatPlaceholder>;

    std::filesystem::path vertex_path;
    std::filesystem::path fragment_path;
    vector<ResourceHandle> used_resources;
    vector<vk::VertexInputBindingDescription> binding_descriptions;
    vector<vk::VertexInputAttributeDescription> attribute_descriptions;
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

    template<typename VertexType>
        requires VertexLike<VertexType>
    GraphicsPipelineDesc(
        std::filesystem::path &&vertex_path_,
        std::filesystem::path &&fragment_path_,
        vector<ResourceHandle> &&used_resources_,
        // it's not possible to explicitly specialize the ctor :( todo: change this
        [[maybe_unused]] VertexType &&vertex_example,
        vector<AttachmentFormat> colors,
        const std::optional<AttachmentFormat> depth_format = {},
        CustomProperties &&custom_properties               = {}
    )
        : vertex_path(vertex_path_), fragment_path(fragment_path_),
          used_resources(used_resources_),
          binding_descriptions(VertexType::get_binding_descriptions()),
          attribute_descriptions(VertexType::get_attribute_descriptions()),
          color_formats(std::move(colors)), depth_format(depth_format),
          custom_properties(custom_properties) {
    }

    [[nodiscard]] std::set<ResourceHandle> get_bound_resources_set() const;
};

struct ComputePipelineDesc {
    std::filesystem::path path;
    vector<ResourceHandle> used_resources;

    struct CustomProperties {
    } custom_properties;
};
} // zrx
