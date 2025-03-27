#pragma once

#include <filesystem>
#include <map>

#include <vma/vk_mem_alloc.h>
#include "src/render/libs.hpp"
#include "src/render/globals.hpp"


// for these bits, we're leveraging the already available flag system from vulkan-hpp.
// for this reason, the following code needs to be in the vulkan-hpp namespace.
namespace
VULKAN_HPP_NAMESPACE {
enum class TextureFlagBitsZRX : uint32_t {
    CUBEMAP = 1 << 0,
    HDR     = 1 << 1,
    MIPMAPS = 1 << 2,
};

using TextureFlagsZRX = Flags<TextureFlagBitsZRX>;

template<>
struct FlagTraits<TextureFlagBitsZRX> {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR TextureFlagsZRX allFlags =
            TextureFlagBitsZRX::CUBEMAP | TextureFlagBitsZRX::HDR | TextureFlagBitsZRX::MIPMAPS;
};
}

namespace zrx {
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

namespace zrx {
class Buffer;
struct RendererContext;

/**
 * Abstraction over a Vulkan image, making it easier to manage by hiding all the Vulkan API calls.
 * These images are allocated using VMA and as such are not suited for swap chain images.
 */
class Image {
protected:
    VmaAllocator allocator{};
    unique_ptr<VmaAllocation> allocation{};
    unique_ptr<vk::raii::Image> image;
    vk::Extent3D extent;
    vk::Format format{};
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
    [[nodiscard]] const vk::raii::Image &operator*() const { return *image; }

    /**
     * Returns an image view containing all mip levels and all layers of this image.
     */
    [[nodiscard]] virtual shared_ptr<vk::raii::ImageView>
    get_view(const RendererContext &ctx);

    /**
     * Returns an image view containing a single mip level and all layers of this image.
     */
    [[nodiscard]] virtual shared_ptr<vk::raii::ImageView>
    get_mip_view(const RendererContext &ctx, uint32_t mip_level);

    /**
     * Returns an image view containing all mip levels and a single specified layer of this image.
     */
    [[nodiscard]] shared_ptr<vk::raii::ImageView>
    get_layer_view(const RendererContext &ctx, uint32_t layer);

    /**
     * Returns an image view containing a single mip level and a single specified layer of this image.
     */
    [[nodiscard]] shared_ptr<vk::raii::ImageView>
    get_layer_mip_view(const RendererContext &ctx, uint32_t layer, uint32_t mip_level);

    [[nodiscard]] vk::Extent3D get_extent() const { return extent; }

    [[nodiscard]] vk::Extent2D get_extent_2d() const { return {extent.width, extent.height}; }

    [[nodiscard]] vk::Format get_format() const { return format; }

    [[nodiscard]] uint32_t get_mip_levels() const { return mip_levels; }

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
    [[nodiscard]] shared_ptr<vk::raii::ImageView> get_cached_view(const RendererContext &ctx, ViewParams params);
};

class CubeImage final : public Image {
public:
    explicit CubeImage(const RendererContext &ctx, const vk::ImageCreateInfo &image_info,
                       vk::MemoryPropertyFlags properties);

    [[nodiscard]] shared_ptr<vk::raii::ImageView>
    get_view(const RendererContext &ctx) override;

    [[nodiscard]] shared_ptr<vk::raii::ImageView>
    get_mip_view(const RendererContext &ctx, uint32_t mip_level) override;

    void copy_from_buffer(vk::Buffer buffer, const vk::raii::CommandBuffer &command_buffer) override;

    void transition_layout(vk::ImageLayout old_layout, vk::ImageLayout new_layout,
                           const vk::raii::CommandBuffer &command_buffer) const override;
};

class Texture {
    unique_ptr<Image> image;
    unique_ptr<vk::raii::Sampler> sampler;

    friend class TextureBuilder;

    Texture() = default;

public:
    [[nodiscard]] Image &get_image() const { return *image; }

    [[nodiscard]] const vk::raii::Sampler &get_sampler() const { return *sampler; }

    [[nodiscard]] uint32_t get_mip_levels() const { return image->get_mip_levels(); }

    [[nodiscard]] vk::Format get_format() const { return image->get_format(); }

    void generate_mipmaps(const RendererContext &ctx, vk::ImageLayout final_layout) const;

private:
    void create_sampler(const RendererContext &ctx, vk::SamplerAddressMode address_mode);
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

using SwizzleDesc = std::array<SwizzleComponent, 4>;

static constexpr SwizzleDesc default_swizzle = {
    SwizzleComponent::R,
    SwizzleComponent::G,
    SwizzleComponent::B,
    SwizzleComponent::A
};

/**
 * Builder used to streamline texture creation due to a huge amount of different parameters.
 * Currently only some specific scenarios are supported and some parameter combinations
 * might not be implemented, due to them not being needed at the moment.
 */
class TextureBuilder {
    vk::Format format = vk::Format::eR8G8B8A8Srgb;
    vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eTransferSrc
                                | vk::ImageUsageFlagBits::eTransferDst
                                | vk::ImageUsageFlagBits::eSampled;
    vk::TextureFlagsZRX tex_flags{};
    bool is_separate_channels = false;
    bool is_uninitialized = false;

    std::optional<SwizzleDesc> swizzle;

    vk::SamplerAddressMode address_mode = vk::SamplerAddressMode::eRepeat;

    std::optional<vk::Extent3D> desired_extent;

    vector<std::filesystem::path> paths;
    void *memory_source = nullptr;
    bool is_from_swizzle_fill = false;

    struct LoadedTextureData {
        vector<void *> sources;
        vk::Extent3D extent;
        uint32_t layer_count;
    };

public:
    TextureBuilder &use_format(vk::Format f);

    TextureBuilder &use_layout(vk::ImageLayout l);

    TextureBuilder &use_usage(vk::ImageUsageFlags u);

    TextureBuilder &with_flags(vk::TextureFlagsZRX flags);

    TextureBuilder &as_separate_channels();

    TextureBuilder &with_sampler_address_mode(vk::SamplerAddressMode mode);

    TextureBuilder &as_uninitialized(vk::Extent3D extent);

    TextureBuilder &with_swizzle(const SwizzleDesc &sw);

    /**
     * Designates the texture's contents to be initialized with data stored in a given file.
     * This requires 6 different paths for cubemap textures.
     */
    TextureBuilder &from_paths(const vector<std::filesystem::path> &sources);

    /**
     * Designates the texture's contents to be initialized with data stored in memory.
     */
    TextureBuilder &from_memory(void *ptr, vk::Extent3D extent);

    /**
     * Designates the texture's contents to be initialized with static data defined using `withSwizzle`.
     */
    TextureBuilder &from_swizzle_fill(vk::Extent3D extent);

    [[nodiscard]] unique_ptr<Texture>
    create(const RendererContext &ctx) const;

private:
    void check_params() const;

    [[nodiscard]] uint32_t get_layer_count() const;

    [[nodiscard]] LoadedTextureData load_from_paths() const;

    [[nodiscard]] LoadedTextureData load_from_memory() const;

    [[nodiscard]] LoadedTextureData load_from_swizzle_fill() const;

    [[nodiscard]] unique_ptr<Buffer> make_staging_buffer(const RendererContext &ctx,
                                                         const LoadedTextureData &data) const;

    static void *merge_channels(const vector<void *> &channels_data, size_t texture_size, size_t component_count);

    void perform_swizzle(uint8_t *data, size_t size) const;
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

    [[nodiscard]] const vk::raii::ImageView &operator*() const { return *view; }

    [[nodiscard]] vk::Format get_format() const { return format; }

    [[nodiscard]] vk::RenderingAttachmentInfo get_attachment_info() const;

    void override_attachment_config(vk::AttachmentLoadOp load_op,
                                    vk::AttachmentStoreOp store_op = vk::AttachmentStoreOp::eStore);
};

namespace utils::img {
    [[nodiscard]] vk::raii::ImageView
    create_image_view(const RendererContext &ctx, vk::Image image, vk::Format format, vk::ImageAspectFlags aspect_flags,
                      uint32_t basemip_level = 0, uint32_t mip_levels = 1, uint32_t layer = 0);

    [[nodiscard]] vk::raii::ImageView
    create_cube_image_view(const RendererContext &ctx, vk::Image image, vk::Format format,
                           vk::ImageAspectFlags aspect_flags, uint32_t base_mip_level = 0, uint32_t mip_levels = 1);

    [[nodiscard]] bool is_depth_format(vk::Format format);

    [[nodiscard]] size_t get_format_size_in_bytes(vk::Format format);

    [[nodiscard]] vk::ImageUsageFlagBits get_format_attachment_type(vk::Format format);
}
} // zrx
