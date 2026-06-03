module;

module Cinder.Render.Vulkan;

import std;
import vulkan;
import stb_image;

import Cinder.Globals;

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
 *
 * todo: remove this altogether and do it all properly.
 */
static const map<pair<vk::ImageLayout, vk::ImageLayout>, ImageBarrierInfo> transition_barrier_schemes{
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
        {vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal},
        {
            .src_access_mask = {},
            .dst_access_mask = vk::AccessFlagBits::eColorAttachmentWrite,
            .src_stage = vk::PipelineStageFlagBits::eTopOfPipe,
            .dst_stage = vk::PipelineStageFlagBits::eAllGraphics,
        }
    },
    {
        {vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal},
        {
            .src_access_mask = {},
            .dst_access_mask = vk::AccessFlagBits::eDepthStencilAttachmentWrite,
            .src_stage = vk::PipelineStageFlagBits::eTopOfPipe,
            .dst_stage = vk::PipelineStageFlagBits::eAllGraphics,
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
        {vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferDstOptimal},
        {
            .src_access_mask = vk::AccessFlagBits::eShaderWrite,
            .dst_access_mask = vk::AccessFlagBits::eTransferWrite,
            .src_stage = vk::PipelineStageFlagBits::eFragmentShader,
            .dst_stage = vk::PipelineStageFlagBits::eTransfer,
        }
    },
    {
        {vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal},
        {
            .src_access_mask = vk::AccessFlagBits::eShaderWrite,
            .dst_access_mask = vk::AccessFlagBits::eShaderRead,
            .src_stage = vk::PipelineStageFlagBits::eFragmentShader,
            .dst_stage = vk::PipelineStageFlagBits::eFragmentShader,
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
             const vk::ImageAspectFlags aspect, shared_ptr<vma::raii::Allocation>&& allocation)
    : image(*ctx.device, image_info),
      allocation(allocation),
      extent(image_info.extent),
      format(image_info.format),
      mip_level_count(image_info.mipLevels),
      layer_count(image_info.arrayLayers),
      is_cubemap(image_info.flags & vk::ImageCreateFlagBits::eCubeCompatible),
      aspect_mask(aspect)
{
    allocation->bindImage(*image);
}

Image::Image(const RendererContext &ctx, const vk::ImageCreateInfo &image_info,
             vk::ImageAspectFlags aspect, vk::MemoryPropertyFlags properties)
    : image(*ctx.device, image_info),
      extent(image_info.extent),
      format(image_info.format),
      mip_level_count(image_info.mipLevels),
      layer_count(image_info.arrayLayers),
      is_cubemap(image_info.flags & vk::ImageCreateFlagBits::eCubeCompatible),
      aspect_mask(aspect)
{
    vma::AllocationCreateFlags flags {};
    if (!(properties & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
        flags = vma::AllocationCreateFlagBits::eHostAccessRandom;
    }

    const vma::AllocationCreateInfo alloc_info{
        .flags = flags,
        .usage = vma::MemoryUsage::eAuto,
        .requiredFlags = properties
    };

    auto [allocation, image] = ctx.allocator->createImage(image_info, alloc_info).split();
    this->allocation = make_shared<vma::raii::Allocation>(std::move(allocation));
    this->image = std::move(image);
}

auto Image::get_full_view(const RendererContext &ctx) const -> shared_ptr<vk::raii::ImageView> {
    return get_cached_view(ctx, {0, mip_level_count, 0, layer_count});
}

auto Image::get_mip_view(const RendererContext &ctx, const uint32_t mip_level) const -> shared_ptr<vk::raii::ImageView> {
    return get_cached_view(ctx, {mip_level, 1, 0, layer_count});
}

auto Image::get_layer_view(const RendererContext &ctx, const uint32_t layer) const -> shared_ptr<vk::raii::ImageView> {
    return get_cached_view(ctx, {0, mip_level_count, layer, 1});
}

auto Image::get_layer_mip_view(const RendererContext &ctx, const uint32_t layer,
                                                          const uint32_t mip_level) const -> shared_ptr<vk::raii::ImageView> {
    return get_cached_view(ctx, {mip_level, 1, layer, 1});
}

auto Image::get_cached_view(const RendererContext &ctx, ViewParams params) const -> shared_ptr<vk::raii::ImageView> {
    if (cached_views.contains(params)) {
        return cached_views.at(params);
    }

    const auto &[base_mip, mip_count, base_layer, layer_count] = params;

    const vk::ImageViewCreateInfo create_info{
        .image = image,
        .viewType = is_cubemap && layer_count == 6 ? vk::ImageViewType::eCube : vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = {
            .aspectMask = aspect_mask,
            .baseMipLevel = base_mip,
            .levelCount = mip_count,
            .baseArrayLayer = base_layer,
            .layerCount = layer_count,
        },
    };

    auto view_ptr = make_shared<vk::raii::ImageView>(*ctx.device, create_info);
    cached_views.emplace(params, view_ptr);
    return view_ptr;
}

void Image::copy_from_buffer(const Buffer& buffer, const vk::raii::CommandBuffer &command_buffer) {
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
        *buffer,
        *image,
        vk::ImageLayout::eTransferDstOptimal,
        region
    );
}

void Image::transition_layout(const vk::ImageLayout old_layout, const vk::ImageLayout new_layout,
                              const vk::raii::CommandBuffer &command_buffer) const {
    const vk::ImageSubresourceRange range{
        .aspectMask = aspect_mask,
        .baseMipLevel = 0,
        .levelCount = mip_level_count,
        .baseArrayLayer = 0,
        .layerCount = layer_count,
    };

    transition_layout(old_layout, new_layout, range, command_buffer);
}

void Image::transition_layout(vk::ImageLayout old_layout, vk::ImageLayout new_layout,
                              vk::ImageSubresourceRange range, const vk::raii::CommandBuffer &command_buffer) const {
    if (!transition_barrier_schemes.contains({old_layout, new_layout})) {
        Logger::error("unsupported layout transition!");
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
        .image = *image,
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

void Image::generate_mipmaps(const RendererContext &ctx, const vk::ImageLayout final_layout,
                             const vk::raii::CommandBuffer& command_buffer) const {
    const vk::FormatProperties format_properties = ctx.physical_device->getFormatProperties(get_format());

    if (!(format_properties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
        Logger::error("texture image format does not support linear blitting!");
    }

    const vk::ImageMemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .oldLayout = vk::ImageLayout::eTransferDstOptimal,
        .newLayout = vk::ImageLayout::eTransferSrcOptimal,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = *image,
        .subresourceRange = vk::ImageSubresourceRange {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = layer_count,
        }
    };

    int32_t mip_width  = extent.width;
    int32_t mip_height = extent.height;

    for (uint32_t i = 1; i < mip_level_count; i++) {
        vk::ImageMemoryBarrier2 curr_barrier = barrier;
        curr_barrier.subresourceRange.baseMipLevel = i - 1;

        command_buffer.pipelineBarrier2(vk::DependencyInfo {
            .imageMemoryBarrierCount = 1u,
            .pImageMemoryBarriers = &curr_barrier,
        });

        const array<vk::Offset3D, 2> src_offsets = {
            {
                {0, 0, 0},
                {mip_width, mip_height, 1},
            }
        };

        const array<vk::Offset3D, 2> dst_offsets = {
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
            *image, vk::ImageLayout::eTransferSrcOptimal,
            *image, vk::ImageLayout::eTransferDstOptimal,
            blit,
            vk::Filter::eLinear
        );

        vk::ImageMemoryBarrier2 trans_barrier = curr_barrier;
        trans_barrier.oldLayout     = vk::ImageLayout::eTransferSrcOptimal;
        trans_barrier.newLayout     = final_layout;
        trans_barrier.srcAccessMask = vk::AccessFlagBits2::eTransferRead;
        trans_barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
        trans_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eFragmentShader;

        command_buffer.pipelineBarrier2(vk::DependencyInfo {
            .imageMemoryBarrierCount = 1u,
            .pImageMemoryBarriers = &trans_barrier,
        });

        if (mip_width > 1) mip_width /= 2;
        if (mip_height > 1) mip_height /= 2;
    }

    vk::ImageMemoryBarrier2 trans_barrier = barrier;
    trans_barrier.subresourceRange.baseMipLevel = mip_level_count - 1;
    trans_barrier.oldLayout                     = vk::ImageLayout::eTransferDstOptimal;
    trans_barrier.newLayout                     = final_layout;
    trans_barrier.srcAccessMask                 = vk::AccessFlagBits2::eTransferWrite;
    trans_barrier.dstAccessMask                 = vk::AccessFlagBits2::eShaderRead;
    trans_barrier.dstStageMask                  = vk::PipelineStageFlagBits2::eFragmentShader;

    command_buffer.pipelineBarrier2(vk::DependencyInfo {
        .imageMemoryBarrierCount = 1u,
        .pImageMemoryBarriers = &trans_barrier,
    });
}

// ==================== TextureBuilder ====================

auto ImageBuilder::with_format(const vk::Format f) -> ImageBuilder& {
    check_if_locked();
    format = f;
    return *this;
}

auto ImageBuilder::with_layout(const vk::ImageLayout l) -> ImageBuilder& {
    check_if_locked();
    layout = l;
    return *this;
}

auto ImageBuilder::with_usage(const vk::ImageUsageFlags u) -> ImageBuilder& {
    check_if_locked();
    usage = u;
    return *this;
}

auto ImageBuilder::with_config(const ImageOverrides &c) -> ImageBuilder& {
    check_if_locked();
    config = c;
    return *this;
}

auto ImageBuilder::with_mag_filter(const vk::Filter f) -> ImageBuilder& {
    check_if_locked();
    config.mag_filter = f;
    return *this;
}

auto ImageBuilder::with_min_filter(const vk::Filter f) -> ImageBuilder& {
    check_if_locked();
    config.min_filter = f;
    return *this;
}

auto ImageBuilder::with_mipmap_mode(const vk::SamplerMipmapMode m) -> ImageBuilder& {
    check_if_locked();
    config.mipmap_mode = m;
    return *this;
}

auto ImageBuilder::with_mip_lod_bias(const float lod_bias) -> ImageBuilder& {
    check_if_locked();
    config.mip_lod_bias = lod_bias;
    return *this;
}

auto ImageBuilder::with_flags(const ImageFlags flags) -> ImageBuilder& {
    check_if_locked();
    tex_flags = flags;
    return *this;
}

auto ImageBuilder::as_separate_channels() -> ImageBuilder& {
    check_if_locked();
    is_separate_channels = true;
    return *this;
}

auto ImageBuilder::with_sampler_address_mode(const vk::SamplerAddressMode mode) -> ImageBuilder& {
    check_if_locked();
    address_mode = mode;
    return *this;
}

auto ImageBuilder::as_uninitialized() -> ImageBuilder & {
    check_if_locked();
    is_uninitialized = true;
    return *this;
}

auto ImageBuilder::with_extent(vk::Extent3D extent) -> ImageBuilder & {
    check_if_locked();
    desired_extent = extent;
    is_window_sized = false;
    return *this;
}

auto ImageBuilder::with_window_size() -> ImageBuilder& {
    check_if_locked();
    desired_extent = {};
    is_window_sized = true;
    return *this;
}

auto ImageBuilder::with_swizzle(const SwizzleDesc &sw) -> ImageBuilder& {
    check_if_locked();
    swizzle = sw;
    return *this;
}

auto ImageBuilder::with_name(const char *n) -> ImageBuilder& {
    check_if_locked();
    name = n;
    return *this;
}

auto ImageBuilder::with_allocation(shared_ptr<vma::raii::Allocation> a) -> ImageBuilder& {
    check_if_locked();
    allocation = a;
    return *this;
}

auto ImageBuilder::from_paths(const vector<std::filesystem::path> &sources) -> ImageBuilder& {
    check_if_locked();
    paths = sources;
    return *this;
}

auto ImageBuilder::from_memory(void *ptr, const vk::Extent3D extent) -> ImageBuilder& {
    check_if_locked();

    if (!ptr) {
        Logger::error("cannot specify null memory source!");
    }

    memory_source  = ptr;
    desired_extent = extent;
    return *this;
}

auto ImageBuilder::from_swizzle_fill(vk::Extent3D extent) -> ImageBuilder& {
    check_if_locked();
    is_from_swizzle_fill = true;
    desired_extent       = extent;
    return *this;
}

auto ImageBuilder::get_image_create_info(const RendererContext& ctx) -> vk::ImageCreateInfo {
    load_image_data(ctx);

    const uint32_t mip_levels = !!(tex_flags & ImageFlags::NO_MIPMAPS)
        ? 1u
        : 1u + static_cast<uint32_t>(std::floor(std::log2(std::max(loaded_texture_data->extent.width, loaded_texture_data->extent.height))));

    return vk::ImageCreateInfo {
        .flags = !!(tex_flags & ImageFlags::CUBEMAP)
                     ? vk::ImageCreateFlagBits::eCubeCompatible
                     : static_cast<vk::ImageCreateFlags>(0),
        .imageType = vk::ImageType::e2D,
        .format = *format,
        .extent = loaded_texture_data->extent,
        .mipLevels = mip_levels,
        .arrayLayers = loaded_texture_data->layer_count,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    };
}

auto ImageBuilder::create(const RendererContext &ctx) -> Image {
    check_params();
    load_image_data(ctx);

    const vk::ImageCreateInfo image_create_info = get_image_create_info(ctx);

    const auto aspect_flags = vk::hasDepthComponent(*format) ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor;

    shared_ptr<vma::raii::Allocation> actual_allocation;

    if (allocation) {
        actual_allocation = *allocation;
    } else {
        const vk::MemoryRequirements2 mem_reqs = ctx.device->getImageMemoryRequirements(vk::DeviceImageMemoryRequirements {
            .pCreateInfo = &image_create_info,
            .planeAspect = aspect_flags,
        });

        constexpr vma::AllocationCreateInfo alloc_info{
            .flags = vma::AllocationCreateFlagBits::eHostAccessRandom,
            .requiredFlags = vk::MemoryPropertyFlagBits::eDeviceLocal
        };

        actual_allocation = make_shared<vma::raii::Allocation>(*ctx.allocator, mem_reqs.memoryRequirements, alloc_info);
    }

    Image image { ctx, image_create_info, aspect_flags, std::move(actual_allocation) };
    image.attach_sampler(create_sampler(ctx));

    optional<Buffer> staging_buffer;
    utils::cmd::do_single_time_commands(ctx, [&](const auto &cmd_buffer) {
        if (is_uninitialized && !!(tex_flags & ImageFlags::NO_MIPMAPS)) {
            image.transition_layout(
                vk::ImageLayout::eUndefined,
                layout,
                cmd_buffer
            );
        } else {
            image.transition_layout(
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eTransferDstOptimal,
                cmd_buffer
            );

            if (!is_uninitialized) {
                staging_buffer = make_staging_buffer(ctx, *loaded_texture_data);
                image.copy_from_buffer(*staging_buffer, cmd_buffer);
            }

            if (!!(tex_flags & ImageFlags::NO_MIPMAPS)) {
                image.transition_layout(
                    vk::ImageLayout::eTransferDstOptimal,
                    layout,
                    cmd_buffer
                );
            } else {
                image.generate_mipmaps(ctx, layout, cmd_buffer);
            }
        }
    });

    if (name) {
        ctx.device->setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT {
            .objectType = vk::ObjectType::eImage,
            .objectHandle = reinterpret_cast<uint64_t>(static_cast<VkImage>(**image)),
            .pObjectName = name,
        });
    }

    return image;
}

void ImageBuilder::check_params() const {
    auto params_error = [&](const char* msg) {
        Logger::error("error when creating texture [{}]: {}", name, msg);
    };

    if (!format) {
        params_error("missing format");
    }

    if (paths.empty() && !memory_source && !is_from_swizzle_fill && !is_uninitialized) {
        params_error("unspecified data source");
    }

    size_t sources_count = 0;
    if (!paths.empty()) sources_count++;
    if (memory_source) sources_count++;
    if (is_from_swizzle_fill) sources_count++;

    if (sources_count > 1) {
        params_error("cannot specify more than one texture source");
    }

    if (is_uninitialized) {
        if (sources_count != 0) {
            params_error("cannot simultaneously set texture as uninitialized and specify sources");
        }

        if (!is_window_sized && !desired_extent) {
            params_error("uninitialized textures must specify an extent or be window-sized");
        }
    }

    if (!!(tex_flags & ImageFlags::CUBEMAP)) {
        if (memory_source) {
            params_error("cubemaps from a memory source are currently not supported");
        }

        if (is_separate_channels) {
            params_error("cubemaps from separated channels are currently not supported");
        }

        if (is_from_swizzle_fill) {
            params_error("cubemaps from swizzle fill are currently not supported");
        }

        if (usage & vk::ImageUsageFlagBits::eDepthStencilAttachment) {
            params_error("cubemaps cannot be depth/stencil attachments");
        }

        if (paths.size() != 6 && !is_uninitialized) {
            params_error("invalid layer count for cubemap texture");
        }
    } else {
        // non-cubemap
        if (is_separate_channels) {
            if (paths.size() != 3) {
                params_error("unsupported channel count for separate-channelled non-cubemap texture");
            }
        } else if (!memory_source && !is_from_swizzle_fill && !is_uninitialized && paths.size() != 1) {
            params_error("invalid layer count for non-cubemap texture");
        }
    }

    if (is_separate_channels) {
        if (paths.empty()) {
            params_error("separate-channeled textures must provide path sources");
        }

        if (vk::blockSize(*format) != 4) {
            params_error("currently only 4-byte formats are supported when using separate channel mode");
        }

        if (vk::blockSize(*format) % 4 != 0) {
            params_error("currently only 4-component formats are supported when using separate channel mode");
        }

        if (swizzle) {
            for (size_t comp = 0; comp < 3; comp++) {
                if (paths[comp].empty()
                    && (*swizzle)[comp] != SwizzleComponent::ZERO
                    && (*swizzle)[comp] != SwizzleComponent::ONE
                    && (*swizzle)[comp] != SwizzleComponent::MAX
                    && (*swizzle)[comp] != SwizzleComponent::HALF_MAX) {
                    params_error("invalid swizzle component for channel provided by an empty path");
                }
            }
        }
    }

    if (is_from_swizzle_fill) {
        if (!swizzle) {
            params_error("textures filled from swizzle must provide a swizzle");
        }

        for (size_t comp = 0; comp < 3; comp++) {
            if ((*swizzle)[comp] != SwizzleComponent::ZERO
                && (*swizzle)[comp] != SwizzleComponent::ONE
                && (*swizzle)[comp] != SwizzleComponent::MAX
                && (*swizzle)[comp] != SwizzleComponent::HALF_MAX) {
                params_error("invalid swizzle component for swizzle-filled texture");
            }
        }

        if (!is_window_sized && !desired_extent) {
            params_error("textures filled from swizzle must specify an extent or be window-sized");
        }
    }
}

void ImageBuilder::check_if_locked() const {
    if (is_locked) {
        Logger::error("locked texture builder may not be modified");
    }
}

uint32_t ImageBuilder::get_layer_count() const {
    if (memory_source || is_from_swizzle_fill) return 1;

    const uint32_t sources_count = is_uninitialized
                                       ? (!!(tex_flags & ImageFlags::CUBEMAP) ? 6 : 1)
                                       : paths.size();
    return is_separate_channels ? sources_count / 3 : sources_count;
}

void ImageBuilder::load_image_data(const RendererContext& ctx) {
    if (loaded_texture_data) return;

    vk::Extent3D extent;

    if (is_uninitialized || is_from_swizzle_fill) {
        if (is_window_sized) {
            int width, height;
            glfwGetWindowSize(ctx.window, &width, &height);
            extent.width = width;
            extent.height = height;
            extent.depth = 1;
        } else {
            extent = *desired_extent;
        }
    }

    if (is_uninitialized)          loaded_texture_data = {{}, extent, get_layer_count()};
    else if (!paths.empty())       loaded_texture_data = load_from_paths();
    else if (memory_source)        loaded_texture_data = load_from_memory();
    else if (is_from_swizzle_fill) loaded_texture_data = load_from_swizzle_fill(extent);

    is_locked = true;
}

auto ImageBuilder::load_from_paths() const -> LoadedImageData {
    vector<void *> data_sources;
    int tex_width = 0, tex_height = 0, tex_channels;
    bool is_first_non_empty = true;

    for (const auto &path: paths) {
        if (path.empty()) {
            data_sources.push_back(nullptr);
            continue;
        }

        stbi_set_flip_vertically_on_load(!!(tex_flags & ImageFlags::HDR) ? 1 : 0);
        const int desired_channels = is_separate_channels ? STBI_grey : STBI_rgb_alpha;
        void *src;

        int curr_tex_width, curr_tex_height;

        if (!!(tex_flags & ImageFlags::HDR)) {
            src = stbi_loadf(path.string().c_str(), &curr_tex_width, &curr_tex_height, &tex_channels, desired_channels);
        } else {
            src = stbi_load(path.string().c_str(), &curr_tex_width, &curr_tex_height, &tex_channels, desired_channels);
        }

        if (!src) {
            Logger::error("failed to load texture image at path: {}", path.string());
        }

        if (is_first_non_empty && !desired_extent) {
            tex_width          = curr_tex_width;
            tex_height         = curr_tex_height;
            is_first_non_empty = false;
        } else if (tex_width != curr_tex_width || tex_height != curr_tex_height) {
            Logger::error("size mismatch while loading a texture from paths");
        }

        data_sources.push_back(src);
    }

    const uint32_t layer_count        = get_layer_count();
    const vk::DeviceSize format_size  = vk::blockSize(*format);
    const vk::DeviceSize layer_size   = tex_width * tex_height * format_size;
    const vk::DeviceSize texture_size = layer_size * layer_count;

    constexpr uint32_t component_count = 4;
    if (format_size % component_count != 0) {
        Logger::error("texture formats with component count other than 4 are currently unsupported");
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

auto ImageBuilder::load_from_memory() const -> LoadedImageData {
    const vector<void *> data_sources = {memory_source};

    const uint32_t tex_width  = desired_extent->width;
    const uint32_t tex_height = desired_extent->height;

    const uint32_t layer_count       = get_layer_count();
    const vk::DeviceSize format_size = vk::blockSize(*format);
    const vk::DeviceSize layer_size  = tex_width * tex_height * format_size;

    constexpr uint32_t component_count = 4;
    if (format_size % component_count != 0) {
        Logger::error("texture formats with component count other than 4 are currently unsupported");
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

auto ImageBuilder::load_from_swizzle_fill(const vk::Extent3D extent) const -> LoadedImageData {
    const uint32_t tex_width          = extent.width;
    const uint32_t tex_height         = extent.height;
    const uint32_t layer_count        = get_layer_count();
    const vk::DeviceSize format_size  = vk::blockSize(*format);
    const vk::DeviceSize layer_size   = tex_width * tex_height * format_size;
    const vk::DeviceSize texture_size = layer_size * layer_count;

    constexpr uint32_t component_count = 4;
    if (format_size % component_count != 0) {
        Logger::error("texture formats with component count other than 4 are currently unsupported");
    }

    const vector<void *> data_sources = {std::malloc(texture_size)};
    if (!data_sources[0]) {
        Logger::error("malloc failed");
    }

    for (const auto &source: data_sources) {
        perform_swizzle(static_cast<uint8_t *>(source), layer_size);
    }

    return {
        .sources = data_sources,
        .extent = extent,
        .layer_count = layer_count
    };
}

auto ImageBuilder::make_staging_buffer(const RendererContext &ctx, const LoadedImageData &data) const -> Buffer {
    const uint32_t layer_count        = get_layer_count();
    const vk::DeviceSize format_size  = vk::blockSize(*format);
    const vk::DeviceSize layer_size   = data.extent.width * data.extent.height * format_size;
    const vk::DeviceSize texture_size = layer_size * layer_count;

    Buffer staging_buffer {
        ctx,
        texture_size,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    };

    for (size_t i = 0; i < get_layer_count(); i++) {
        staging_buffer.copy_from_ptr(data.sources[i], layer_size, layer_size * i);

        if (is_separate_channels || is_from_swizzle_fill) {
            std::free(data.sources[i]);
        } else if (!memory_source) {
            stbi_image_free(data.sources[i]);
        }
    }

    return staging_buffer;
}

auto ImageBuilder::merge_channels(const vector<void *> &channels_data,
                                    const size_t texture_size,
                                    const size_t component_count) -> void * {
    auto *merged = static_cast<uint8_t *>(std::malloc(texture_size));
    if (!merged) {
        Logger::error("malloc failed");
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

void ImageBuilder::perform_swizzle(uint8_t *data, const size_t size) const {
    if (!swizzle) {
        Logger::error("unexpected empty swizzle optional in TextureBuilder::performSwizzle");
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
                    data[COMPONENT_COUNT * i + comp] = numeric_limits<uint8_t>::max();
                    break;
                case SwizzleComponent::HALF_MAX:
                    data[COMPONENT_COUNT * i + comp] = numeric_limits<uint8_t>::max() / 2;
                    break;
            }
        }
    }
}

auto ImageBuilder::create_sampler(const RendererContext &ctx) const -> vk::raii::Sampler {
    const vk::PhysicalDeviceProperties properties = ctx.physical_device->getProperties();

    const uint32_t mip_levels = !!(tex_flags & ImageFlags::NO_MIPMAPS)
        ? 1u
        : 1u + static_cast<uint32_t>(std::floor(std::log2(std::max(loaded_texture_data->extent.width, loaded_texture_data->extent.height))));

    const vk::SamplerCreateInfo sampler_info{
        .magFilter = config.mag_filter ? *config.mag_filter : *default_config.mag_filter,
        .minFilter = config.min_filter ? *config.min_filter : *default_config.min_filter,
        .mipmapMode = config.mipmap_mode ? *config.mipmap_mode : *default_config.mipmap_mode,
        .addressModeU = address_mode,
        .addressModeV = address_mode,
        .addressModeW = address_mode,
        .mipLodBias = config.mip_lod_bias ? *config.mip_lod_bias : *default_config.mip_lod_bias,
        .anisotropyEnable = vk::True,
        .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
        .compareEnable = vk::False,
        .compareOp = vk::CompareOp::eAlways,
        .minLod = 0.0f,
        .maxLod = static_cast<float>(mip_levels),
        .borderColor = vk::BorderColor::eIntOpaqueBlack,
        .unnormalizedCoordinates = vk::False,
    };

    return {*ctx.device, sampler_info};
}

// ==================== RenderTarget ====================

RenderTarget::RenderTarget(shared_ptr<vk::raii::ImageView> view, const vk::Format format)
    : view(std::move(view)), format(format) {
}

RenderTarget::RenderTarget(shared_ptr<vk::raii::ImageView> view, shared_ptr<vk::raii::ImageView> resolve_view,
                           const vk::Format format)
    : view(std::move(view)), resolve_view(std::move(resolve_view)), format(format) {
}

RenderTarget::RenderTarget(const RendererContext &ctx, Image &image)
    : view(image.get_full_view(ctx)), format(image.get_format()) {
}

vk::RenderingAttachmentInfo RenderTarget::get_attachment_info() const {
    const auto layout = vk::hasDepthComponent(format)
                        ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                        : vk::ImageLayout::eColorAttachmentOptimal;

    vk::ClearValue clear_value = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
    if (vk::hasDepthComponent(format)) {
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
    auto get_format_attachment_type(const vk::Format format) -> vk::ImageUsageFlagBits {
        return vk::hasDepthComponent(format)
               ? vk::ImageUsageFlagBits::eDepthStencilAttachment
               : vk::ImageUsageFlagBits::eColorAttachment;
    }
} // utils::img
} // zrx
