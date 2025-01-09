#include "image.hpp"

#include <filesystem>
#include <map>

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

#include "buffer.hpp"
#include "cmd.hpp"
#include "ctx.hpp"

struct ImageBarrierInfo {
    vk::AccessFlagBits src_access_mask;
    vk::AccessFlagBits dst_access_mask;
    vk::PipelineStageFlagBits src_stage;
    vk::PipelineStageFlagBits dst_stage;
};

/**
 * List of stages and access masks for image layout transitions.
 * Currently there's no need for more fine-grained customization of these parameters during transitions,
 * so they're defined statically and used depeneding on the transition's start and end layouts.
 */
static const std::map<std::pair<vk::ImageLayout, vk::ImageLayout>, ImageBarrierInfo> transition_barrier_schemes{
    {
        {vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferSrcOptimal},
        {
            .src_access_mask = {},
            .dst_access_mask = vk::AccessFlagBits::eTransferRead,
            .src_stage = vk::PipelineStageFlagBits::eTopOfPipe,
            .dst_stage = vk::PipelineStageFlagBits::eTransfer,
        }
    },
    {
        {vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal},
        {
            .src_access_mask = {},
            .dst_access_mask = vk::AccessFlagBits::eTransferWrite,
            .src_stage = vk::PipelineStageFlagBits::eTopOfPipe,
            .dst_stage = vk::PipelineStageFlagBits::eTransfer,
        }
    },
    {
        {vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral},
        {
            .src_access_mask = {},
            .dst_access_mask = {},
            .src_stage = vk::PipelineStageFlagBits::eTopOfPipe,
            .dst_stage = vk::PipelineStageFlagBits::eBottomOfPipe,
        }
    },
    {
        {vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
        {
            .src_access_mask = {},
            .dst_access_mask = vk::AccessFlagBits::eTransferRead,
            .src_stage = vk::PipelineStageFlagBits::eTopOfPipe,
            .dst_stage = vk::PipelineStageFlagBits::eTransfer,
        }
    },
    {
        {vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal},
        {
            .src_access_mask = vk::AccessFlagBits::eTransferRead,
            .dst_access_mask = vk::AccessFlagBits::eShaderRead,
            .src_stage = vk::PipelineStageFlagBits::eTransfer,
            .dst_stage = vk::PipelineStageFlagBits::eFragmentShader,
        }
    },
    {
        {vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal},
        {
            .src_access_mask = vk::AccessFlagBits::eTransferWrite,
            .dst_access_mask = vk::AccessFlagBits::eShaderRead,
            .src_stage = vk::PipelineStageFlagBits::eTransfer,
            .dst_stage = vk::PipelineStageFlagBits::eFragmentShader,
        }
    },
    {
        {vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferSrcOptimal},
        {
            .src_access_mask = vk::AccessFlagBits::eShaderRead,
            .dst_access_mask = vk::AccessFlagBits::eTransferRead,
            .src_stage = vk::PipelineStageFlagBits::eFragmentShader,
            .dst_stage = vk::PipelineStageFlagBits::eTransfer,
        }
    },
    {
        {vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferDstOptimal},
        {
            .src_access_mask = vk::AccessFlagBits::eShaderRead,
            .dst_access_mask = vk::AccessFlagBits::eTransferWrite,
            .src_stage = vk::PipelineStageFlagBits::eFragmentShader,
            .dst_stage = vk::PipelineStageFlagBits::eTransfer,
        }
    },
    {
        {vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eGeneral},
        {
            .src_access_mask = vk::AccessFlagBits::eTransferWrite,
            .dst_access_mask = vk::AccessFlagBits::eMemoryRead,
            .src_stage = vk::PipelineStageFlagBits::eTransfer,
            .dst_stage = vk::PipelineStageFlagBits::eBottomOfPipe,
        }
    },
};

namespace zrx {
Image::Image(const RendererContext &ctx, const vk::ImageCreateInfo &image_info,
             const vk::MemoryPropertyFlags properties, const vk::ImageAspectFlags aspect)
    : allocator(**ctx.allocator),
      extent(image_info.extent),
      format(image_info.format),
      mip_levels(image_info.mipLevels),
      aspect_mask(aspect) {
    VmaAllocationCreateFlags flags;
    if (properties & vk::MemoryPropertyFlagBits::eDeviceLocal) {
        flags = 0;
    } else {
        flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    }

    const VmaAllocationCreateInfo alloc_info{
        .flags = flags,
        .usage = VMA_MEMORY_USAGE_AUTO,
        .requiredFlags = static_cast<VkMemoryPropertyFlags>(properties)
    };

    VkImage new_image;
    VmaAllocation new_allocation;

    const auto result = vmaCreateImage(
        allocator,
        reinterpret_cast<const VkImageCreateInfo *>(&image_info),
        &alloc_info,
        &new_image,
        &new_allocation,
        nullptr
    );

    if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer!");
    }

    image      = make_unique<vk::raii::Image>(*ctx.device, new_image);
    allocation = make_unique<VmaAllocation>(new_allocation);
}

Image::~Image() {
    vmaFreeMemory(allocator, *allocation);
}

shared_ptr<vk::raii::ImageView> Image::get_view(const RendererContext &ctx) {
    return get_cached_view(ctx, {0, mip_levels, 0, 1});
}

shared_ptr<vk::raii::ImageView> Image::get_mip_view(const RendererContext &ctx, const uint32_t mip_level) {
    return get_cached_view(ctx, {mip_level, 1, 0, 1});
}

shared_ptr<vk::raii::ImageView> Image::get_layer_view(const RendererContext &ctx, const uint32_t layer) {
    return get_cached_view(ctx, {0, mip_levels, layer, 1});
}

shared_ptr<vk::raii::ImageView> Image::get_layer_mip_view(const RendererContext &ctx, const uint32_t layer,
                                                          const uint32_t mip_level) {
    return get_cached_view(ctx, {mip_level, 1, layer, 1});
}

shared_ptr<vk::raii::ImageView> Image::get_cached_view(const RendererContext &ctx, ViewParams params) {
    if (cached_views.contains(params)) {
        return cached_views.at(params);
    }

    const auto &[base_mip, mip_count, base_layer, layer_count] = params;

    auto view = layer_count == 1
                    ? utils::img::create_image_view(ctx, **image, format, aspect_mask, base_mip, mip_count, base_layer)
                    : utils::img::create_cube_image_view(ctx, **image, format, aspect_mask, base_mip, mip_count);
    auto view_ptr = make_shared<vk::raii::ImageView>(std::move(view));
    cached_views.emplace(params, view_ptr);
    return view_ptr;
}

void Image::copy_from_buffer(const vk::Buffer buffer, const vk::raii::CommandBuffer &command_buffer) {
    const vk::BufferImageCopy region{
        .bufferOffset = 0U,
        .bufferRowLength = 0U,
        .bufferImageHeight = 0U,
        .imageSubresource = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .imageOffset = {0, 0, 0},
        .imageExtent = extent,
    };

    command_buffer.copyBufferToImage(
        buffer,
        **image,
        vk::ImageLayout::eTransferDstOptimal,
        region
    );
}

void Image::transition_layout(const vk::ImageLayout old_layout, const vk::ImageLayout new_layout,
                              const vk::raii::CommandBuffer &command_buffer) const {
    const vk::ImageSubresourceRange range{
        .aspectMask = aspect_mask,
        .baseMipLevel = 0,
        .levelCount = mip_levels,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };

    transition_layout(old_layout, new_layout, range, command_buffer);
}

void Image::transition_layout(vk::ImageLayout old_layout, vk::ImageLayout new_layout,
                              vk::ImageSubresourceRange range, const vk::raii::CommandBuffer &command_buffer) const {
    if (!transition_barrier_schemes.contains({old_layout, new_layout})) {
        throw std::invalid_argument("unsupported layout transition!");
    }

    const auto &[src_access_mask, dst_access_mask, src_stage, dst_stage] =
            transition_barrier_schemes.at({old_layout, new_layout});

    range.aspectMask = aspect_mask;

    const vk::ImageMemoryBarrier barrier{
        .srcAccessMask = src_access_mask,
        .dstAccessMask = dst_access_mask,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = **image,
        .subresourceRange = range,
    };

    command_buffer.pipelineBarrier(
        src_stage,
        dst_stage,
        {},
        nullptr,
        nullptr,
        barrier
    );
}

void Image::save_to_file(const RendererContext &ctx, const std::filesystem::path &path) const {
    const vk::ImageCreateInfo temp_image_info{
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR8G8B8A8Unorm,
        .extent = extent,
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eLinear,
        .usage = vk::ImageUsageFlagBits::eTransferDst,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    const Image temp_image{
        ctx,
        temp_image_info,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        vk::ImageAspectFlagBits::eColor
    };

    utils::cmd::do_single_time_commands(ctx, [&](const auto &cmd_buffer) {
        transition_layout(
            vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::ImageLayout::eTransferSrcOptimal,
            cmd_buffer
        );

        transition_layout(
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eTransferDstOptimal,
            cmd_buffer
        );
    });

    const vk::ImageCopy image_copy_region{
        .srcSubresource = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .layerCount = 1
        },
        .dstSubresource = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .layerCount = 1
        },
        .extent = extent
    };

    const vk::Offset3D blit_offset{
        .x = static_cast<int32_t>(extent.width),
        .y = static_cast<int32_t>(extent.height),
        .z = static_cast<int32_t>(extent.depth)
    };

    const vk::ImageMemoryBarrier2 image_memory_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
        .oldLayout = vk::ImageLayout::eTransferDstOptimal,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = **temp_image,
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .levelCount = 1,
            .layerCount = 1,
        }
    };

    const vk::DependencyInfo dependency_info{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &image_memory_barrier
    };

    const vk::ImageBlit blit_info{
        .srcSubresource = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .layerCount = 1
        },
        .srcOffsets = {{vk::Offset3D(), blit_offset}},
        .dstSubresource = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .layerCount = 1
        },
        .dstOffsets = {{vk::Offset3D(), blit_offset}}
    };

    bool supports_blit = true;

    // check if the device supports blitting from this image's format
    const vk::FormatProperties src_format_properties = ctx.physical_device->getFormatProperties(format);
    if (!(src_format_properties.linearTilingFeatures & vk::FormatFeatureFlagBits::eBlitSrc)) {
        supports_blit = false;
    }

    // check if the device supports blitting to linear images
    const vk::FormatProperties dst_format_properties = ctx.physical_device->getFormatProperties(temp_image.format);
    if (!(dst_format_properties.linearTilingFeatures & vk::FormatFeatureFlagBits::eBlitDst)) {
        supports_blit = false;
    }

    utils::cmd::do_single_time_commands(ctx, [&](const auto &commandBuffer) {
        if (supports_blit) {
            commandBuffer.blitImage(
                **image,
                vk::ImageLayout::eTransferSrcOptimal,
                **temp_image,
                vk::ImageLayout::eTransferDstOptimal,
                blit_info,
                vk::Filter::eLinear
            );
        } else {
            commandBuffer.copyImage(
                **image,
                vk::ImageLayout::eTransferSrcOptimal,
                **temp_image,
                vk::ImageLayout::eTransferDstOptimal,
                image_copy_region
            );
        }

        commandBuffer.pipelineBarrier2(dependency_info);
    });

    void *data;
    vmaMapMemory(temp_image.allocator, *temp_image.allocation, &data);

    stbi_write_png(
        path.string().c_str(),
        static_cast<int>(temp_image.extent.width),
        static_cast<int>(temp_image.extent.height),
        STBI_rgb_alpha,
        data,
        utils::img::get_format_size_in_bytes(temp_image.format) * temp_image.extent.width
    );

    vmaUnmapMemory(temp_image.allocator, *temp_image.allocation);

    utils::cmd::do_single_time_commands(ctx, [&](const auto &cmd_buffer) {
        transition_layout(
            vk::ImageLayout::eTransferSrcOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal,
            cmd_buffer
        );
    });
}

// ==================== CubeImage ====================

CubeImage::CubeImage(const RendererContext &ctx, const vk::ImageCreateInfo &image_info,
                     const vk::MemoryPropertyFlags properties)
    : Image(ctx, image_info, properties, vk::ImageAspectFlagBits::eColor) {
}

shared_ptr<vk::raii::ImageView> CubeImage::get_view(const RendererContext &ctx) {
    return get_cached_view(ctx, {0, mip_levels, 0, 6});
}

shared_ptr<vk::raii::ImageView> CubeImage::get_mip_view(const RendererContext &ctx, const uint32_t mip_level) {
    return get_cached_view(ctx, {mip_level, 1, 0, 6});
}

void CubeImage::copy_from_buffer(const vk::Buffer buffer, const vk::raii::CommandBuffer &command_buffer) {
    const vk::BufferImageCopy region{
        .bufferOffset = 0U,
        .bufferRowLength = 0U,
        .bufferImageHeight = 0U,
        .imageSubresource = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 6,
        },
        .imageOffset = {0, 0, 0},
        .imageExtent = extent,
    };

    command_buffer.copyBufferToImage(
        buffer,
        **image,
        vk::ImageLayout::eTransferDstOptimal,
        region
    );
}

void CubeImage::transition_layout(const vk::ImageLayout old_layout, const vk::ImageLayout new_layout,
                                  const vk::raii::CommandBuffer &command_buffer) const {
    const vk::ImageSubresourceRange range{
        .aspectMask = aspect_mask,
        .baseMipLevel = 0,
        .levelCount = mip_levels,
        .baseArrayLayer = 0,
        .layerCount = 6,
    };

    Image::transition_layout(old_layout, new_layout, range, command_buffer);
}

// ==================== Texture ====================

void Texture::generate_mipmaps(const RendererContext &ctx, const vk::ImageLayout final_layout) const {
    const vk::FormatProperties format_properties = ctx.physical_device->getFormatProperties(get_format());

    if (!(format_properties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
        throw std::runtime_error("texture image format does not support linear blitting!");
    }

    const vk::raii::CommandBuffer command_buffer = utils::cmd::begin_single_time_commands(ctx);

    const bool is_cube_map     = dynamic_cast<CubeImage *>(&*image) != nullptr;
    const uint32_t layer_count = is_cube_map ? 6 : 1;

    const vk::ImageMemoryBarrier barrier{
        .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
        .dstAccessMask = vk::AccessFlagBits::eTransferRead,
        .oldLayout = vk::ImageLayout::eTransferDstOptimal,
        .newLayout = vk::ImageLayout::eTransferSrcOptimal,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = ***image,
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = layer_count,
        }
    };

    int32_t mip_width  = image->get_extent().width;
    int32_t mip_height = image->get_extent().height;

    for (uint32_t i = 1; i < image->get_mip_levels(); i++) {
        vk::ImageMemoryBarrier curr_barrier        = barrier;
        curr_barrier.subresourceRange.baseMipLevel = i - 1;

        command_buffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
            {},
            nullptr,
            nullptr,
            curr_barrier
        );

        const std::array<vk::Offset3D, 2> src_offsets = {
            {
                {0, 0, 0},
                {mip_width, mip_height, 1},
            }
        };

        const std::array<vk::Offset3D, 2> dst_offsets = {
            {
                {0, 0, 0},
                {mip_width > 1 ? mip_width / 2 : 1, mip_height > 1 ? mip_height / 2 : 1, 1},
            }
        };

        const vk::ImageBlit blit{
            .srcSubresource = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = i - 1,
                .baseArrayLayer = 0,
                .layerCount = layer_count,
            },
            .srcOffsets = src_offsets,
            .dstSubresource = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = i,
                .baseArrayLayer = 0,
                .layerCount = layer_count,
            },
            .dstOffsets = dst_offsets
        };

        command_buffer.blitImage(
            ***image, vk::ImageLayout::eTransferSrcOptimal,
            ***image, vk::ImageLayout::eTransferDstOptimal,
            blit,
            vk::Filter::eLinear
        );

        vk::ImageMemoryBarrier trans_barrier = curr_barrier;
        trans_barrier.oldLayout              = vk::ImageLayout::eTransferSrcOptimal;
        trans_barrier.newLayout              = final_layout;
        trans_barrier.srcAccessMask          = vk::AccessFlagBits::eTransferRead;
        trans_barrier.dstAccessMask          = vk::AccessFlagBits::eShaderRead;

        command_buffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader,
            {},
            nullptr,
            nullptr,
            trans_barrier
        );

        if (mip_width > 1) mip_width /= 2;
        if (mip_height > 1) mip_height /= 2;
    }

    vk::ImageMemoryBarrier trans_barrier        = barrier;
    trans_barrier.subresourceRange.baseMipLevel = image->get_mip_levels() - 1;
    trans_barrier.oldLayout                     = vk::ImageLayout::eTransferDstOptimal;
    trans_barrier.newLayout                     = final_layout;
    trans_barrier.srcAccessMask                 = vk::AccessFlagBits::eTransferWrite;
    trans_barrier.dstAccessMask                 = vk::AccessFlagBits::eShaderRead;

    command_buffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader,
        {},
        nullptr,
        nullptr,
        trans_barrier
    );

    utils::cmd::end_single_time_commands(command_buffer, *ctx.graphics_queue);
}

void Texture::create_sampler(const RendererContext &ctx, const vk::SamplerAddressMode address_mode) {
    const vk::PhysicalDeviceProperties properties = ctx.physical_device->getProperties();

    const vk::SamplerCreateInfo sampler_info{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eLinear,
        .addressModeU = address_mode,
        .addressModeV = address_mode,
        .addressModeW = address_mode,
        .mipLodBias = 0.0f,
        .anisotropyEnable = vk::True,
        .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
        .compareEnable = vk::False,
        .compareOp = vk::CompareOp::eAlways,
        .minLod = 0.0f,
        .maxLod = static_cast<float>(image->get_mip_levels()),
        .borderColor = vk::BorderColor::eIntOpaqueBlack,
        .unnormalizedCoordinates = vk::False,
    };

    sampler = make_unique<vk::raii::Sampler>(*ctx.device, sampler_info);
}

// ==================== TextureBuilder ====================

TextureBuilder &TextureBuilder::use_format(const vk::Format f) {
    format = f;
    return *this;
}

TextureBuilder &TextureBuilder::use_layout(const vk::ImageLayout l) {
    layout = l;
    return *this;
}

TextureBuilder &TextureBuilder::use_usage(const vk::ImageUsageFlags u) {
    usage = u;
    return *this;
}

TextureBuilder &TextureBuilder::with_flags(const vk::TextureFlagsZRX flags) {
    tex_flags = flags;
    return *this;
}

TextureBuilder &TextureBuilder::as_separate_channels() {
    is_separate_channels = true;
    return *this;
}

TextureBuilder &TextureBuilder::with_sampler_address_mode(const vk::SamplerAddressMode mode) {
    address_mode = mode;
    return *this;
}

TextureBuilder &TextureBuilder::as_uninitialized(const vk::Extent3D extent) {
    is_uninitialized = true;
    desired_extent   = extent;
    return *this;
}

TextureBuilder &TextureBuilder::with_swizzle(const SwizzleDesc &sw) {
    swizzle = sw;
    return *this;
}

TextureBuilder &TextureBuilder::from_paths(const std::vector<std::filesystem::path> &sources) {
    paths = sources;
    return *this;
}

TextureBuilder &TextureBuilder::from_memory(void *ptr, const vk::Extent3D extent) {
    if (!ptr) {
        throw std::invalid_argument("cannot specify null memory source!");
    }

    memory_source  = ptr;
    desired_extent = extent;
    return *this;
}

TextureBuilder &TextureBuilder::from_swizzle_fill(vk::Extent3D extent) {
    is_from_swizzle_fill = true;
    desired_extent       = extent;
    return *this;
}

unique_ptr<Texture> TextureBuilder::create(const RendererContext &ctx) const {
    check_params();

    // stupid workaround because std::unique_ptr doesn't have access to the Texture ctor
    unique_ptr<Texture> texture; {
        Texture t;
        texture = make_unique<Texture>(std::move(t));
    }

    LoadedTextureData loaded_tex_data;

    if (is_uninitialized) loaded_tex_data = {{}, *desired_extent, get_layer_count()};
    else if (!paths.empty()) loaded_tex_data = load_from_paths();
    else if (memory_source) loaded_tex_data = load_from_memory();
    else if (is_from_swizzle_fill) loaded_tex_data = load_from_swizzle_fill();

    const auto extent         = loaded_tex_data.extent;
    const auto staging_buffer = is_uninitialized ? nullptr : make_staging_buffer(ctx, loaded_tex_data);

    uint32_t mip_levels = 1;
    if (tex_flags & vk::TextureFlagBitsZRX::MIPMAPS) {
        mip_levels = static_cast<uint32_t>(std::floor(std::log2(std::max(extent.width, extent.height)))) + 1;
    }

    const vk::ImageCreateInfo image_info{
        .flags = tex_flags & vk::TextureFlagBitsZRX::CUBEMAP
                     ? vk::ImageCreateFlagBits::eCubeCompatible
                     : static_cast<vk::ImageCreateFlags>(0),
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = extent,
        .mipLevels = mip_levels,
        .arrayLayers = loaded_tex_data.layer_count,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    const bool is_depth     = !!(usage & vk::ImageUsageFlagBits::eDepthStencilAttachment);
    const auto aspect_flags = is_depth ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor;

    if (tex_flags & vk::TextureFlagBitsZRX::CUBEMAP) {
        texture->image = make_unique<CubeImage>(
            ctx,
            image_info,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );
    } else {
        texture->image = make_unique<Image>(
            ctx,
            image_info,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            aspect_flags
        );
    }

    texture->create_sampler(ctx, address_mode);

    if (is_uninitialized && !(tex_flags & vk::TextureFlagBitsZRX::MIPMAPS)) {
        utils::cmd::do_single_time_commands(ctx, [&](const auto &cmd_buffer) {
            texture->image->transition_layout(
                vk::ImageLayout::eUndefined,
                layout,
                cmd_buffer
            );
        });
    } else {
        utils::cmd::do_single_time_commands(ctx, [&](const auto &cmd_buffer) {
            texture->image->transition_layout(
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eTransferDstOptimal,
                cmd_buffer
            );

            if (!is_uninitialized) {
                texture->image->copy_from_buffer(**staging_buffer, cmd_buffer);
            }

            if (!(tex_flags & vk::TextureFlagBitsZRX::MIPMAPS)) {
                texture->image->transition_layout(
                    vk::ImageLayout::eTransferDstOptimal,
                    layout,
                    cmd_buffer
                );
            }
        });

        if (tex_flags & vk::TextureFlagBitsZRX::MIPMAPS) {
            texture->generate_mipmaps(ctx, layout);
        }
    }

    return texture;
}

void TextureBuilder::check_params() const {
    if (paths.empty() && !memory_source && !is_from_swizzle_fill && !is_uninitialized) {
        throw std::invalid_argument("no specified data source for texture!");
    }

    size_t sources_count = 0;
    if (!paths.empty()) sources_count++;
    if (memory_source) sources_count++;
    if (is_from_swizzle_fill) sources_count++;

    if (sources_count > 1) {
        throw std::invalid_argument("cannot specify more than one texture source!");
    }

    if (sources_count != 0 && is_uninitialized) {
        throw std::invalid_argument("cannot simultaneously set texture as uninitialized and specify sources!");
    }

    if (tex_flags & vk::TextureFlagBitsZRX::CUBEMAP) {
        if (memory_source) {
            throw std::invalid_argument("cubemaps from a memory source are currently not supported!");
        }

        if (is_separate_channels) {
            throw std::invalid_argument("cubemaps from separated channels are currently not supported!");
        }

        if (is_from_swizzle_fill) {
            throw std::invalid_argument("cubemaps from swizzle fill are currently not supported!");
        }

        if (usage & vk::ImageUsageFlagBits::eDepthStencilAttachment) {
            throw std::invalid_argument("cubemaps cannot be depth/stencil attachments!");
        }

        if (paths.size() != 6 && !is_uninitialized) {
            throw std::invalid_argument("invalid layer count for cubemap texture!");
        }
    } else {
        // non-cubemap
        if (is_separate_channels) {
            if (paths.size() != 3) {
                throw std::invalid_argument("unsupported channel count for separate-channelled non-cubemap texture!");
            }
        } else if (!memory_source && !is_from_swizzle_fill && !is_uninitialized && paths.size() != 1) {
            throw std::invalid_argument("invalid layer count for non-cubemap texture!");
        }
    }

    if (is_separate_channels) {
        if (paths.empty()) {
            throw std::invalid_argument("separate-channeled textures must provide path sources!");
        }

        if (utils::img::get_format_size_in_bytes(format) != 4) {
            throw std::invalid_argument(
                "currently only 4-byte formats are supported when using separate channel mode!");
        }

        if (utils::img::get_format_size_in_bytes(format) % 4 != 0) {
            throw std::invalid_argument(
                "currently only 4-component formats are supported when using separate channel mode!"
            );
        }

        if (swizzle) {
            for (size_t comp = 0; comp < 3; comp++) {
                if (paths[comp].empty()
                    && (*swizzle)[comp] != SwizzleComponent::ZERO
                    && (*swizzle)[comp] != SwizzleComponent::ONE
                    && (*swizzle)[comp] != SwizzleComponent::MAX
                    && (*swizzle)[comp] != SwizzleComponent::HALF_MAX) {
                    throw std::invalid_argument("invalid swizzle component for channel provided by an empty path!");
                }
            }
        }
    }

    if (is_from_swizzle_fill) {
        if (!swizzle) {
            throw std::invalid_argument("textures filled from swizzle must provide a swizzle!");
        }

        for (size_t comp = 0; comp < 3; comp++) {
            if ((*swizzle)[comp] != SwizzleComponent::ZERO
                && (*swizzle)[comp] != SwizzleComponent::ONE
                && (*swizzle)[comp] != SwizzleComponent::MAX
                && (*swizzle)[comp] != SwizzleComponent::HALF_MAX) {
                throw std::invalid_argument("invalid swizzle component for swizzle-filled texture!");
            }
        }
    }
}

uint32_t TextureBuilder::get_layer_count() const {
    if (memory_source || is_from_swizzle_fill) return 1;

    const uint32_t sources_count = is_uninitialized
                                       ? (tex_flags & vk::TextureFlagBitsZRX::CUBEMAP ? 6 : 1)
                                       : paths.size();
    return is_separate_channels ? sources_count / 3 : sources_count;
}

TextureBuilder::LoadedTextureData TextureBuilder::load_from_paths() const {
    std::vector<void *> data_sources;
    int tex_width           = 0, tex_height = 0, tex_channels;
    bool is_first_non_empty = true;

    for (const auto &path: paths) {
        if (path.empty()) {
            data_sources.push_back(nullptr);
            continue;
        }

        stbi_set_flip_vertically_on_load(tex_flags & vk::TextureFlagBitsZRX::HDR ? 1 : 0);
        const int desired_channels = is_separate_channels ? STBI_grey : STBI_rgb_alpha;
        void *src;

        int curr_tex_width, curr_tex_height;

        if (tex_flags & vk::TextureFlagBitsZRX::HDR) {
            src = stbi_loadf(path.string().c_str(), &curr_tex_width, &curr_tex_height, &tex_channels, desired_channels);
        } else {
            src = stbi_load(path.string().c_str(), &curr_tex_width, &curr_tex_height, &tex_channels, desired_channels);
        }

        if (!src) {
            throw std::runtime_error("failed to load texture image at path: " + path.string());
        }

        if (is_first_non_empty && !desired_extent) {
            tex_width          = curr_tex_width;
            tex_height         = curr_tex_height;
            is_first_non_empty = false;
        } else if (tex_width != curr_tex_width || tex_height != curr_tex_height) {
            throw std::runtime_error("size mismatch while loading a texture from paths!");
        }

        data_sources.push_back(src);
    }

    const uint32_t layer_count        = get_layer_count();
    const vk::DeviceSize format_size  = utils::img::get_format_size_in_bytes(format);
    const vk::DeviceSize layer_size   = tex_width * tex_height * format_size;
    const vk::DeviceSize texture_size = layer_size * layer_count;

    constexpr uint32_t component_count = 4;
    if (format_size % component_count != 0) {
        throw std::runtime_error("texture formats with component count other than 4 are currently unsupported!");
    }

    if (is_separate_channels) {
        data_sources = {merge_channels(data_sources, texture_size, component_count)};
    }

    if (swizzle) {
        for (const auto &source: data_sources) {
            perform_swizzle(static_cast<uint8_t *>(source), layer_size);
        }
    }

    return {
        .sources = data_sources,
        .extent = {
            .width = static_cast<uint32_t>(tex_width),
            .height = static_cast<uint32_t>(tex_height),
            .depth = 1u
        },
        .layer_count = layer_count
    };
}

TextureBuilder::LoadedTextureData TextureBuilder::load_from_memory() const {
    const std::vector<void *> data_sources = {memory_source};

    const uint32_t tex_width  = desired_extent->width;
    const uint32_t tex_height = desired_extent->height;

    const uint32_t layer_count       = get_layer_count();
    const vk::DeviceSize format_size = utils::img::get_format_size_in_bytes(format);
    const vk::DeviceSize layer_size  = tex_width * tex_height * format_size;

    constexpr uint32_t component_count = 4;
    if (format_size % component_count != 0) {
        throw std::runtime_error("texture formats with component count other than 4 are currently unsupported!");
    }

    if (swizzle) {
        for (const auto &source: data_sources) {
            perform_swizzle(static_cast<uint8_t *>(source), layer_size);
        }
    }

    return {
        .sources = data_sources,
        .extent = {
            .width = static_cast<uint32_t>(tex_width),
            .height = static_cast<uint32_t>(tex_height),
            .depth = 1u
        },
        .layer_count = layer_count
    };
}

TextureBuilder::LoadedTextureData TextureBuilder::load_from_swizzle_fill() const {
    const uint32_t tex_width          = desired_extent->width;
    const uint32_t tex_height         = desired_extent->height;
    const uint32_t layer_count        = get_layer_count();
    const vk::DeviceSize format_size  = utils::img::get_format_size_in_bytes(format);
    const vk::DeviceSize layer_size   = tex_width * tex_height * format_size;
    const vk::DeviceSize texture_size = layer_size * layer_count;

    constexpr uint32_t component_count = 4;
    if (format_size % component_count != 0) {
        throw std::runtime_error("texture formats with component count other than 4 are currently unsupported!");
    }

    const std::vector<void *> data_sources = {malloc(texture_size)};

    for (const auto &source: data_sources) {
        perform_swizzle(static_cast<uint8_t *>(source), layer_size);
    }

    return {
        .sources = data_sources,
        .extent = {
            .width = static_cast<uint32_t>(tex_width),
            .height = static_cast<uint32_t>(tex_height),
            .depth = 1u
        },
        .layer_count = layer_count
    };
}

unique_ptr<Buffer>
TextureBuilder::make_staging_buffer(const RendererContext &ctx, const LoadedTextureData &data) const {
    const uint32_t layer_count        = get_layer_count();
    const vk::DeviceSize format_size  = utils::img::get_format_size_in_bytes(format);
    const vk::DeviceSize layer_size   = data.extent.width * data.extent.height * format_size;
    const vk::DeviceSize texture_size = layer_size * layer_count;

    auto staging_buffer = make_unique<Buffer>(
        **ctx.allocator,
        texture_size,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );

    void *mapped = staging_buffer->map();

    for (size_t i = 0; i < get_layer_count(); i++) {
        const size_t offset = layer_size * i;
        memcpy(static_cast<char *>(mapped) + offset, data.sources[i], static_cast<size_t>(layer_size));

        if (is_separate_channels || is_from_swizzle_fill) {
            free(data.sources[i]);
        } else if (!memory_source) {
            stbi_image_free(data.sources[i]);
        }
    }

    staging_buffer->unmap();

    return staging_buffer;
}

void *TextureBuilder::merge_channels(const std::vector<void *> &channels_data, const size_t texture_size,
                                     const size_t component_count) {
    auto *merged = static_cast<uint8_t *>(malloc(texture_size));
    if (!merged) {
        throw std::runtime_error("malloc failed");
    }

    for (size_t i = 0; i < texture_size; i++) {
        if (i % component_count == component_count - 1 || !channels_data[i % component_count]) {
            merged[i] = 0; // todo - utilize alpha
        } else {
            merged[i] = static_cast<uint8_t *>(channels_data[i % component_count])[i / component_count];
        }
    }

    return merged;
}

void TextureBuilder::perform_swizzle(uint8_t *data, const size_t size) const {
    if (!swizzle) {
        throw std::runtime_error("unexpected empty swizzle optional in TextureBuilder::performSwizzle");
    }

    constexpr size_t COMPONENT_COUNT = 4;

    for (size_t i = 0; i < size / COMPONENT_COUNT; i++) {
        const uint8_t r = data[COMPONENT_COUNT * i];
        const uint8_t g = data[COMPONENT_COUNT * i + 1];
        const uint8_t b = data[COMPONENT_COUNT * i + 2];
        const uint8_t a = data[COMPONENT_COUNT * i + 3];

        for (size_t comp = 0; comp < COMPONENT_COUNT; comp++) {
            switch ((*swizzle)[comp]) {
                case SwizzleComponent::R:
                    data[COMPONENT_COUNT * i + comp] = r;
                    break;
                case SwizzleComponent::G:
                    data[COMPONENT_COUNT * i + comp] = g;
                    break;
                case SwizzleComponent::B:
                    data[COMPONENT_COUNT * i + comp] = b;
                    break;
                case SwizzleComponent::A:
                    data[COMPONENT_COUNT * i + comp] = a;
                    break;
                case SwizzleComponent::ZERO:
                    data[COMPONENT_COUNT * i + comp] = 0;
                    break;
                case SwizzleComponent::ONE:
                    data[COMPONENT_COUNT * i + comp] = 1;
                    break;
                case SwizzleComponent::MAX:
                    data[COMPONENT_COUNT * i + comp] = std::numeric_limits<uint8_t>::max();
                    break;
                case SwizzleComponent::HALF_MAX:
                    data[COMPONENT_COUNT * i + comp] = std::numeric_limits<uint8_t>::max() / 2;
                    break;
            }
        }
    }
}

// ==================== RenderTarget ====================

RenderTarget::RenderTarget(shared_ptr<vk::raii::ImageView> view, const vk::Format format)
    : view(std::move(view)), format(format) {
}

RenderTarget::RenderTarget(shared_ptr<vk::raii::ImageView> view, shared_ptr<vk::raii::ImageView> resolve_view,
                           const vk::Format format)
    : view(std::move(view)), resolve_view(std::move(resolve_view)), format(format) {
}

RenderTarget::RenderTarget(const RendererContext &ctx, const Texture &texture)
    : view(texture.get_image().get_view(ctx)), format(texture.get_format()) {
}

vk::RenderingAttachmentInfo RenderTarget::get_attachment_info() const {
    const auto layout = utils::img::is_depth_format(format)
                            ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                            : vk::ImageLayout::eColorAttachmentOptimal;

    vk::ClearValue clear_value = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 1.0f};
    if (utils::img::is_depth_format(format)) {
        clear_value = vk::ClearDepthStencilValue{
            .depth = 1.0f,
            .stencil = 0,
        };
    }

    vk::RenderingAttachmentInfo info{
        .imageView = **view,
        .imageLayout = layout,
        .loadOp = load_op,
        .storeOp = store_op,
        .clearValue = clear_value,
    };

    if (resolve_view) {
        info.resolveMode        = vk::ResolveModeFlagBits::eAverage;
        info.resolveImageView   = **resolve_view;
        info.resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal;
    }

    return info;
}

void RenderTarget::override_attachment_config(const vk::AttachmentLoadOp load_op,
                                              const vk::AttachmentStoreOp store_op) {
    this->load_op  = load_op;
    this->store_op = store_op;
}

// ==================== utils ====================

namespace utils::img {
    vk::raii::ImageView
    utils::img::create_image_view(const RendererContext &ctx, const vk::Image image, const vk::Format format,
                                  const vk::ImageAspectFlags aspect_flags, const uint32_t base_mip_level,
                                  const uint32_t mip_levels, const uint32_t layer) {
        const vk::ImageViewCreateInfo create_info{
            .image = image,
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            .subresourceRange = {
                .aspectMask = aspect_flags,
                .baseMipLevel = base_mip_level,
                .levelCount = mip_levels,
                .baseArrayLayer = layer,
                .layerCount = 1,
            },
        };

        return {*ctx.device, create_info};
    }

    vk::raii::ImageView create_cube_image_view(const RendererContext &ctx, const vk::Image image,
                                               const vk::Format format, const vk::ImageAspectFlags aspect_flags,
                                               const uint32_t base_mip_level, const uint32_t mip_levels) {
        const vk::ImageViewCreateInfo create_info{
            .image = image,
            .viewType = vk::ImageViewType::eCube,
            .format = format,
            .subresourceRange = {
                .aspectMask = aspect_flags,
                .baseMipLevel = base_mip_level,
                .levelCount = mip_levels,
                .baseArrayLayer = 0,
                .layerCount = 6,
            }
        };

        return {*ctx.device, create_info};
    }

    bool is_depth_format(const vk::Format format) {
        switch (format) {
            case vk::Format::eD16Unorm:
            case vk::Format::eD32Sfloat:
            case vk::Format::eD16UnormS8Uint:
            case vk::Format::eD24UnormS8Uint:
            case vk::Format::eD32SfloatS8Uint:
                return true;
            default:
                return false;
        }
    }

    size_t get_format_size_in_bytes(const vk::Format format) {
        switch (format) {
            case vk::Format::eB8G8R8A8Srgb:
            case vk::Format::eR8G8B8A8Srgb:
            case vk::Format::eR8G8B8A8Unorm:
                return 4;
            case vk::Format::eR16G16B16Sfloat:
                return 6;
            case vk::Format::eR16G16B16A16Sfloat:
                return 8;
            case vk::Format::eR32G32B32Sfloat:
                return 12;
            case vk::Format::eR32G32B32A32Sfloat:
                return 16;
            default:
                throw std::runtime_error("unexpected_format_in_utils::img::get_format_size_in_bytes");
        }
    }

    vk::ImageUsageFlagBits get_format_attachment_type(const vk::Format format) {
        return utils::img::is_depth_format(format)
               ? vk::ImageUsageFlagBits::eDepthStencilAttachment
               : vk::ImageUsageFlagBits::eColorAttachment;
    }
} // utils::img
} // zrx
