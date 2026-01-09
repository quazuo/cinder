module;

#define VULKAN_HPP_ENABLE_STD_MODULE
#define VULKAN_HPP_STD_MODULE
#include <vulkan/vulkan_hpp_macros.hpp>

export module Cinder.Render.Vulkan:Image;

import vma;
import vulkan_hpp;
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
class TextureBuilder;

/**
 * Abstraction over a Vulkan image, making it easier to manage by hiding all the Vulkan API calls.
 * These images are allocated using VMA and as such are not suited for swap chain images.
 */
class Image {
protected:
    VmaAllocator allocator;
    VmaAllocation allocation;
    vk::Image image;

    vk::Extent3D extent;
    vk::Format format;
    uint32_t mip_levels;
    vk::ImageAspectFlags aspect_mask;

    std::unordered_map<ViewParams, shared_ptr<vk::raii::ImageView> > cached_views;

public:
    explicit Image(const RendererContext &ctx, const vk::ImageCreateInfo &image_info,
                   vk::MemoryPropertyFlags properties, vk::ImageAspectFlags aspect);

    virtual ~Image();

    Image(const Image &other) = delete;

    Image(Image &&other) = delete;

    Image &operator=(const Image &other) = delete;

    Image &operator=(Image &&other) = delete;

    /**
     * Returns a raw handle to the actual Vulkan image.
     * @return Handle to the image.
     */
    const vk::Image &operator*() const { return image; }

    /**
     * Returns an image view containing all mip levels and all layers of this image.
     */
    virtual shared_ptr<vk::raii::ImageView>
    get_view(const RendererContext &ctx);

    /**
     * Returns an image view containing a single mip level and all layers of this image.
     */
    virtual shared_ptr<vk::raii::ImageView>
    get_mip_view(const RendererContext &ctx, uint32_t mip_level);

    /**
     * Returns an image view containing all mip levels and a single specified layer of this image.
     */
    shared_ptr<vk::raii::ImageView>
    get_layer_view(const RendererContext &ctx, uint32_t layer);

    /**
     * Returns an image view containing a single mip level and a single specified layer of this image.
     */
    shared_ptr<vk::raii::ImageView>
    get_layer_mip_view(const RendererContext &ctx, uint32_t layer, uint32_t mip_level);

    vk::Extent3D get_extent() const { return extent; }

    vk::Extent2D get_extent_2d() const { return {extent.width, extent.height}; }

    vk::Format get_format() const { return format; }

    uint32_t get_mip_levels() const { return mip_levels; }

    /**
     * Records commands that copy the contents of a given buffer to this image.
     */
    virtual void copy_from_buffer(vk::Buffer buffer, const vk::raii::CommandBuffer &command_buffer);

    /**
     * Records commands that transition this image's layout.
     * A valid old layout must be provided, as the image's current layout is not being tracked.
     */
    virtual void transition_layout(vk::ImageLayout old_layout, vk::ImageLayout new_layout,
                                   const vk::raii::CommandBuffer &command_buffer) const;

    /**
     * Records commands that transition this image's layout, also specifying a specific subresource range
     * on which the transition should occur.
     * A valid old layout must be provided, as the image's current layout is not being tracked.
     */
    void transition_layout(vk::ImageLayout old_layout, vk::ImageLayout new_layout,
                           vk::ImageSubresourceRange range, const vk::raii::CommandBuffer &command_buffer) const;

    /**
     * Writes the contents of this image to a file on a given path.
     *
     * Disclaimer: this might not work very well as it wasn't tested very well
     * (nor do I care about it working perfectly) and was created purely to debug a single thing in the past.
     * However, I'm not removing this as I might use it (and make it work better) again in the future.
     */
    void save_to_file(const RendererContext &ctx, const std::filesystem::path &path) const;

protected:
    /**
     * Checks if a given view is cached already and if so, returns it without creating a new one.
     * Otherwise, creates the view and caches it for later.
     */
    shared_ptr<vk::raii::ImageView> get_cached_view(const RendererContext &ctx, ViewParams params);
};

class CubeImage final : public Image {
public:
    explicit CubeImage(const RendererContext &ctx, const vk::ImageCreateInfo &image_info,
                       vk::MemoryPropertyFlags properties);

    shared_ptr<vk::raii::ImageView>
    get_view(const RendererContext &ctx) override;

    shared_ptr<vk::raii::ImageView>
    get_mip_view(const RendererContext &ctx, uint32_t mip_level) override;

    void copy_from_buffer(vk::Buffer buffer, const vk::raii::CommandBuffer &command_buffer) override;

    void transition_layout(vk::ImageLayout old_layout, vk::ImageLayout new_layout,
                           const vk::raii::CommandBuffer &command_buffer) const override;
};

class Texture {
    unique_ptr<Image> image;
    unique_ptr<vk::raii::Sampler> sampler;

    friend TextureBuilder;

    Texture() = default;

public:
    Image &get_image() const { return *image; }

    const vk::raii::Sampler &get_sampler() const { return *sampler; }

    uint32_t get_mip_levels() const { return image->get_mip_levels(); }

    vk::Format get_format() const { return image->get_format(); }

    void generate_mipmaps(const RendererContext &ctx, vk::ImageLayout final_layout, const vk::raii::CommandBuffer& command_buffer) const;
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

struct TextureOverrides {
    optional<vk::Filter>            mag_filter;
    optional<vk::Filter>            min_filter;
    optional<vk::SamplerMipmapMode> mipmap_mode;
    optional<float>                 mip_lod_bias;
};

enum class TextureFlags : uint32_t {
    CUBEMAP    = 1 << 0,
    HDR        = 1 << 1,
    NO_MIPMAPS = 1 << 2,
};

template <>
struct enable_bitmask_operators<TextureFlags> : std::true_type {};

/**
 * Builder used to streamline texture creation due to a huge amount of different parameters.
 * Currently only some specific scenarios are supported and some parameter combinations
 * might not be implemented, due to them not being needed at the moment.
 */
class TextureBuilder {
    TextureOverrides default_config = {
        .mag_filter   = vk::Filter::eLinear,
        .min_filter   = vk::Filter::eLinear,
        .mipmap_mode  = vk::SamplerMipmapMode::eLinear,
        .mip_lod_bias = 0.0f,
    };

    TextureOverrides config;

    optional<vk::Format> format{};
    vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eTransferSrc
                                | vk::ImageUsageFlagBits::eTransferDst
                                | vk::ImageUsageFlagBits::eSampled;

    TextureFlags tex_flags{};
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

    struct LoadedTextureData {
        vector<void *> sources;
        vk::Extent3D extent;
        uint32_t layer_count;
    };

public:
    auto with_config(const TextureOverrides &config)            -> TextureBuilder&;
    auto with_format(vk::Format f)                              -> TextureBuilder&;
    auto with_layout(vk::ImageLayout l)                         -> TextureBuilder&;
    auto with_usage(vk::ImageUsageFlags u)                      -> TextureBuilder&;
    auto with_mag_filter(vk::Filter f)                          -> TextureBuilder&;
    auto with_min_filter(vk::Filter f)                          -> TextureBuilder&;
    auto with_mipmap_mode(vk::SamplerMipmapMode m)              -> TextureBuilder&;
    auto with_mip_lod_bias(float lod_bias)                      -> TextureBuilder&;
    auto with_flags(TextureFlags flags)                         -> TextureBuilder&;
    auto as_separate_channels()                                 -> TextureBuilder&;
    auto with_sampler_address_mode(vk::SamplerAddressMode mode) -> TextureBuilder&;
    auto as_uninitialized()                                     -> TextureBuilder&;
    auto with_extent(vk::Extent3D extent)                       -> TextureBuilder&;
    auto with_window_size()                                     -> TextureBuilder&;
    auto with_swizzle(const SwizzleDesc &sw)                    -> TextureBuilder&;
    auto with_name(const char *n)                               -> TextureBuilder&;

    /**
     * Designates the texture's contents to be initialized with data stored in a given file.
     * This requires 6 different paths for cubemap textures.
     */
    auto from_paths(const vector<std::filesystem::path> &sources) -> TextureBuilder&;

    /**
     * Designates the texture's contents to be initialized with data stored in memory.
     */
    auto from_memory(void *ptr, vk::Extent3D extent) -> TextureBuilder&;

    /**
     * Designates the texture's contents to be initialized with static data defined using `with_swizzle`.
     */
    auto from_swizzle_fill(vk::Extent3D extent) -> TextureBuilder&;

    auto create(const RendererContext &ctx) const -> Texture;

private:
    void check_params() const;

    auto get_layer_count() const -> uint32_t;

    auto load_from_paths() const -> LoadedTextureData;

    auto load_from_memory() const -> LoadedTextureData;

    auto load_from_swizzle_fill(vk::Extent3D extent) const -> LoadedTextureData;

    auto make_staging_buffer(const RendererContext &ctx, const LoadedTextureData &data) const -> unique_ptr<Buffer>;

    static auto merge_channels(const vector<void *> &channels_data, size_t texture_size, size_t component_count) -> void*;

    void perform_swizzle(uint8_t *data, size_t size) const;

    void create_sampler(const RendererContext &ctx, Texture& texture) const;
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

    RenderTarget(const RendererContext &ctx, const Texture &texture);

    auto operator*() const -> const vk::raii::ImageView& { return *view; }

    auto get_format() const -> vk::Format { return format; }

    auto get_attachment_info() const -> vk::RenderingAttachmentInfo;

    void override_attachment_config(vk::AttachmentLoadOp load_op,
                                    vk::AttachmentStoreOp store_op = vk::AttachmentStoreOp::eStore);
};

namespace utils::img {
    auto create_image_view(const RendererContext &ctx, vk::Image image,
                           vk::Format format, vk::ImageAspectFlags aspect_flags,
                           uint32_t base_mip_level = 0, uint32_t mip_levels = 1,
                           uint32_t layer = 0) -> vk::raii::ImageView;

    auto create_cube_image_view(const RendererContext &ctx, vk::Image image,
                                vk::Format format, vk::ImageAspectFlags aspect_flags,
                                uint32_t base_mip_level = 0, uint32_t mip_levels = 1) -> vk::raii::ImageView;

    auto get_format_attachment_type(vk::Format format) -> vk::ImageUsageFlagBits;
}
} // zrx
