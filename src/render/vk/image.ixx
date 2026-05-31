module;

#define VULKAN_HPP_ENABLE_STD_MODULE
#define VULKAN_HPP_STD_MODULE
#include <vulkan/vulkan_hpp_macros.hpp>

export module Cinder.Render.Vulkan:Image;

import vk_mem_alloc;
import vulkan;
import std;

import :Buffer;
import :Context;

import Cinder.Globals;
import Cinder.Utils;

export namespace zrx {
/**
 * Parameters defining which mip levels and layers of a given image are available for a given view.
 * This struct is used mainly for caching views to eliminate creating multiple identical views.
 */
struct ViewParams {
    uint32_t base_mip_level;
    uint32_t mip_levels;
    uint32_t base_layer;
    uint32_t layer_count;

    // `unordered_map` requirement
    bool operator==(const ViewParams &other) const {
        return base_mip_level == other.base_mip_level
               && mip_levels == other.mip_levels
               && base_layer == other.base_layer
               && layer_count == other.layer_count;
    }
};
} // zrx

// `unordered_map` requirement
template<>
struct std::hash<zrx::ViewParams> {
    size_t operator()(zrx::ViewParams const &params) const noexcept {
        return (hash<uint32_t>()(params.mip_levels) >> 1) ^
               (hash<uint32_t>()(params.base_mip_level) << 1) ^
               (hash<uint32_t>()(params.base_layer) << 1) ^
               (hash<uint32_t>()(params.layer_count) << 1);
    }
};

export namespace zrx {
class ImageBuilder;

/**
 * Abstraction over a Vulkan image, making it easier to manage by hiding all the Vulkan API calls.
 * These images are allocated using VMA and as such are not suited for swap chain images.
 */
class Image {
    vk::raii::Image image;
    shared_ptr<vma::raii::Allocation> allocation;

    vk::Extent3D extent;
    vk::Format format;
    uint32_t mip_level_count;
    uint32_t layer_count;
    vk::ImageAspectFlags aspect_mask;
    bool is_cubemap;

    mutable std::unordered_map<ViewParams, shared_ptr<vk::raii::ImageView> > cached_views;

    optional<vk::raii::Sampler> sampler;

public:
    explicit Image(const RendererContext &ctx, const vk::ImageCreateInfo &image_info,
                   vk::ImageAspectFlags aspect, shared_ptr<vma::raii::Allocation>&& allocation);

    explicit Image(const RendererContext &ctx, const vk::ImageCreateInfo &image_info,
                   vk::ImageAspectFlags aspect, vk::MemoryPropertyFlags properties);

    /**
     * Returns a raw handle to the actual Vulkan image.
     * @return Handle to the image.
     */
    auto operator*() const -> const vk::raii::Image & { return image; }

    /**
     * Returns an image view containing all mip levels and all layers of this image.
     */
    auto get_full_view(const RendererContext &ctx) const -> shared_ptr<vk::raii::ImageView>;

    /**
     * Returns an image view containing a single mip level and all layers of this image.
     */
    auto get_mip_view(const RendererContext &ctx, uint32_t mip_level) const -> shared_ptr<vk::raii::ImageView>;

    /**
     * Returns an image view containing all mip levels and a single specified layer of this image.
     */
    auto get_layer_view(const RendererContext &ctx, uint32_t layer) const -> shared_ptr<vk::raii::ImageView>;

    /**
     * Returns an image view containing a single mip level and a single specified layer of this image.
     */
    auto get_layer_mip_view(const RendererContext &ctx, uint32_t layer, uint32_t mip_level) const -> shared_ptr<vk::raii::ImageView>;

    auto get_extent() const -> vk::Extent3D { return extent; }

    auto get_extent_2d() const -> vk::Extent2D { return {extent.width, extent.height}; }

    auto get_format() const -> vk::Format { return format; }

    auto get_mip_levels() const -> uint32_t { return mip_level_count; }

    auto get_layer_count() const -> uint32_t { return layer_count; }

    auto has_sampler() const -> bool { return sampler.has_value(); }

    auto get_sampler() const -> const vk::raii::Sampler & { return *sampler; }

    void attach_sampler(vk::raii::Sampler&& s) { sampler = std::move(s); }

    /**
     * Records commands that copy the contents of a given buffer to this image.
     */
    void copy_from_buffer(const Buffer& buffer, const vk::raii::CommandBuffer &command_buffer);

    /**
     * Records commands that transition this image's layout.
     * A valid old layout must be provided, as the image's current layout is not being tracked.
     */
    void transition_layout(vk::ImageLayout old_layout, vk::ImageLayout new_layout,
                           const vk::raii::CommandBuffer &command_buffer) const;

    /**
     * Records commands that transition this image's layout, also specifying a specific subresource range
     * on which the transition should occur.
     * A valid old layout must be provided, as the image's current layout is not being tracked.
     */
    void transition_layout(vk::ImageLayout old_layout, vk::ImageLayout new_layout,
                           vk::ImageSubresourceRange range, const vk::raii::CommandBuffer &command_buffer) const;

    void generate_mipmaps(const RendererContext &ctx, vk::ImageLayout final_layout, const vk::raii::CommandBuffer& command_buffer) const;

private:
    /**
     * Checks if a given view is cached already and if so, returns it without creating a new one.
     * Otherwise, creates the view and caches it for later.
     */
    auto get_cached_view(const RendererContext &ctx, ViewParams params) const -> shared_ptr<vk::raii::ImageView>;
};

enum class SwizzleComponent {
    R,
    G,
    B,
    A,
    ZERO,
    ONE,
    MAX,
    HALF_MAX
};

using SwizzleDesc = array<SwizzleComponent, 4>;

constexpr SwizzleDesc default_swizzle = {
    SwizzleComponent::R,
    SwizzleComponent::G,
    SwizzleComponent::B,
    SwizzleComponent::A
};

struct ImageOverrides {
    optional<vk::Filter>            mag_filter;
    optional<vk::Filter>            min_filter;
    optional<vk::SamplerMipmapMode> mipmap_mode;
    optional<float>                 mip_lod_bias;
};

enum class ImageFlags : uint32_t {
    CUBEMAP    = 1 << 0,
    HDR        = 1 << 1,
    NO_MIPMAPS = 1 << 2,
};

template <>
struct enable_bitmask_operators<ImageFlags> : std::true_type {};

/**
 * Builder used to streamline texture creation due to a huge amount of different parameters.
 * Currently only some specific scenarios are supported and some parameter combinations
 * might not be implemented, due to them not being needed at the moment.
 */
class ImageBuilder {
    ImageOverrides default_config = {
        .mag_filter   = vk::Filter::eLinear,
        .min_filter   = vk::Filter::eLinear,
        .mipmap_mode  = vk::SamplerMipmapMode::eLinear,
        .mip_lod_bias = 0.0f,
    };

    ImageOverrides config;

    optional<vk::Format> format{};
    vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eTransferSrc
                                | vk::ImageUsageFlagBits::eTransferDst
                                | vk::ImageUsageFlagBits::eSampled;

    ImageFlags tex_flags{};
    bool is_separate_channels = false;
    bool is_uninitialized = false;
    bool is_window_sized = false;

    optional<SwizzleDesc> swizzle;

    vk::SamplerAddressMode address_mode = vk::SamplerAddressMode::eRepeat;

    optional<vk::Extent3D> desired_extent;

    vector<std::filesystem::path> paths;
    void *memory_source = nullptr;
    bool is_from_swizzle_fill = false;

    const char *name = nullptr;

    optional<shared_ptr<vma::raii::Allocation>> allocation;

    struct LoadedImageData {
        vector<void *> sources;
        vk::Extent3D extent;
        uint32_t layer_count;
    };

    optional<LoadedImageData> loaded_texture_data;

    bool is_locked = false;

public:
    auto with_format(vk::Format f)                              -> ImageBuilder&;
    auto with_layout(vk::ImageLayout l)                         -> ImageBuilder&;
    auto with_usage(vk::ImageUsageFlags u)                      -> ImageBuilder&;
    auto with_config(const ImageOverrides &c)                   -> ImageBuilder&;
    auto with_mag_filter(vk::Filter f)                          -> ImageBuilder&;
    auto with_min_filter(vk::Filter f)                          -> ImageBuilder&;
    auto with_mipmap_mode(vk::SamplerMipmapMode m)              -> ImageBuilder&;
    auto with_mip_lod_bias(float lod_bias)                      -> ImageBuilder&;
    auto with_flags(ImageFlags flags)                           -> ImageBuilder&;
    auto as_separate_channels()                                 -> ImageBuilder&;
    auto with_sampler_address_mode(vk::SamplerAddressMode mode) -> ImageBuilder&;
    auto as_uninitialized()                                     -> ImageBuilder&;
    auto with_extent(vk::Extent3D extent)                       -> ImageBuilder&;
    auto with_window_size()                                     -> ImageBuilder&;
    auto with_swizzle(const SwizzleDesc &sw)                    -> ImageBuilder&;
    auto with_name(const char *n)                               -> ImageBuilder&;
    auto with_allocation(shared_ptr<vma::raii::Allocation> a)   -> ImageBuilder&;

    /**
     * Designates the texture's contents to be initialized with data stored in a given file.
     * This requires 6 different paths for cubemap textures.
     */
    auto from_paths(const vector<std::filesystem::path> &sources) -> ImageBuilder&;

    /**
     * Designates the texture's contents to be initialized with data stored in memory.
     */
    auto from_memory(void *ptr, vk::Extent3D extent) -> ImageBuilder&;

    /**
     * Designates the texture's contents to be initialized with static data defined using `with_swizzle`.
     */
    auto from_swizzle_fill(vk::Extent3D extent) -> ImageBuilder&;

    auto get_image_create_info(const RendererContext& ctx) -> vk::ImageCreateInfo;

    auto create(const RendererContext &ctx) -> Image;

private:
    void check_params() const;

    void check_if_locked() const;

    auto get_layer_count() const -> uint32_t;

    void load_image_data(const RendererContext& ctx);

    auto load_from_paths() const -> LoadedImageData;

    auto load_from_memory() const -> LoadedImageData;

    auto load_from_swizzle_fill(vk::Extent3D extent) const -> LoadedImageData;

    auto make_staging_buffer(const RendererContext &ctx, const LoadedImageData &data) const -> Buffer;

    static auto merge_channels(const vector<void *> &channels_data, size_t texture_size, size_t component_count) -> void*;

    void perform_swizzle(uint8_t *data, size_t size) const;

    auto create_sampler(const RendererContext &ctx) const -> vk::raii::Sampler;
};

/**
 * Convenience wrapper around image views which are used as render targets.
 * This is primarily an abstraction to unify textures and swapchain images, so that they're used
 * in an uniform way.
 */
class RenderTarget {
    shared_ptr<vk::raii::ImageView> view;
    shared_ptr<vk::raii::ImageView> resolve_view;
    vk::Format format{};

    vk::AttachmentLoadOp load_op = vk::AttachmentLoadOp::eClear;
    vk::AttachmentStoreOp store_op = vk::AttachmentStoreOp::eStore;

public:
    RenderTarget(shared_ptr<vk::raii::ImageView> view, vk::Format format);

    RenderTarget(shared_ptr<vk::raii::ImageView> view, shared_ptr<vk::raii::ImageView> resolve_view, vk::Format format);

    RenderTarget(const RendererContext &ctx, Image &image);

    auto operator*() const -> const vk::raii::ImageView& { return *view; }

    auto get_format() const -> vk::Format { return format; }

    auto get_attachment_info() const -> vk::RenderingAttachmentInfo;

    void override_attachment_config(vk::AttachmentLoadOp load_op,
                                    vk::AttachmentStoreOp store_op = vk::AttachmentStoreOp::eStore);
};

namespace utils::img {
    auto get_format_attachment_type(vk::Format format) -> vk::ImageUsageFlagBits;
}
} // zrx
