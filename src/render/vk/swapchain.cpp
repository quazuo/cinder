#include "swapchain.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "src/render/renderer.hpp"
#include "ctx.hpp"

namespace zrx {
SwapChainSupportDetails::SwapChainSupportDetails(const vk::raii::PhysicalDevice &physical_device,
                                                 const vk::raii::SurfaceKHR &surface)
    : capabilities(physical_device.getSurfaceCapabilitiesKHR(*surface)),
      formats(physical_device.getSurfaceFormatsKHR(*surface)),
      present_modes(physical_device.getSurfacePresentModesKHR(*surface)) {
}

SwapChain::SwapChain(const RendererContext &ctx, const vk::raii::SurfaceKHR &surface,
                     const QueueFamilyIndices &queue_families, GLFWwindow *window,
                     vk::SampleCountFlagBits sample_count) : msaa_sample_count(sample_count) {
    const auto [capabilities, formats, present_modes] = SwapChainSupportDetails{*ctx.physical_device, surface};

    extent = choose_extent(capabilities, window);

    const vk::SurfaceFormatKHR surface_format = choose_surface_format(formats);
    image_format = surface_format.format;

    const vk::PresentModeKHR present_mode = choose_present_mode(present_modes);

    const auto &[graphics_compute_family, present_family] = queue_families;
    const uint32_t queue_family_indices[] = {graphics_compute_family.value(), present_family.value()};
    const bool is_uniform_family = graphics_compute_family == present_family;

    const vk::SwapchainCreateInfoKHR create_info{
        .surface = *surface,
        .minImageCount = get_image_count(ctx, surface),
        .imageFormat = surface_format.format,
        .imageColorSpace = surface_format.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
        .imageSharingMode = is_uniform_family ? vk::SharingMode::eExclusive : vk::SharingMode::eConcurrent,
        .queueFamilyIndexCount = is_uniform_family ? 0u : 2u,
        .pQueueFamilyIndices = is_uniform_family ? nullptr : queue_family_indices,
        .preTransform = capabilities.currentTransform,
        .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
        .presentMode = present_mode,
        .clipped = vk::True,
    };

    swap_chain = make_unique<vk::raii::SwapchainKHR>(ctx.device->createSwapchainKHR(create_info));
    images = swap_chain->getImages();

    create_color_resources(ctx);

    depth_format = find_depth_format(ctx);
    create_depth_resources(ctx);
}

uint32_t SwapChain::get_image_count(const RendererContext &ctx, const vk::raii::SurfaceKHR &surface) {
    const auto [capabilities, formats, present_modes] = SwapChainSupportDetails{*ctx.physical_device, surface};

    uint32_t image_count = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && image_count > capabilities.maxImageCount) {
        image_count = capabilities.maxImageCount;
    }

    return image_count;
}

void SwapChain::transition_to_attachment_layout(const vk::raii::CommandBuffer &command_buffer) const {
    const vk::ImageMemoryBarrier barrier{
        .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = images[current_image_index],
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        }
    };

    command_buffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        {},
        nullptr,
        nullptr,
        barrier
    );
}

void SwapChain::transition_to_present_layout(const vk::raii::CommandBuffer &command_buffer) const {
    const vk::ImageMemoryBarrier barrier{
        .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
        .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .newLayout = vk::ImageLayout::ePresentSrcKHR,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = images[current_image_index],
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        }
    };

    command_buffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::PipelineStageFlagBits::eBottomOfPipe,
        {},
        nullptr,
        nullptr,
        barrier
    );
}

std::pair<vk::Result, uint32_t> SwapChain::acquire_next_image(const vk::raii::Semaphore &semaphore) {
    try {
        const auto &[result, image_index] = swap_chain->acquireNextImage(UINT64_MAX, *semaphore);
        current_image_index = image_index;
        return {result, image_index};
    } catch (...) {
        return {vk::Result::eErrorOutOfDateKHR, 0};
    }
}

vk::Extent2D SwapChain::choose_extent(const vk::SurfaceCapabilitiesKHR &capabilities, GLFWwindow *window) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    }

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    const uint32_t actual_extent_width = std::clamp(
        static_cast<uint32_t>(width),
        capabilities.minImageExtent.width,
        capabilities.maxImageExtent.width
    );
    const uint32_t actual_extent_height = std::clamp(
        static_cast<uint32_t>(height),
        capabilities.minImageExtent.height,
        capabilities.maxImageExtent.height
    );

    return {
        actual_extent_width,
        actual_extent_height
    };
}

vk::SurfaceFormatKHR SwapChain::choose_surface_format(const vector<vk::SurfaceFormatKHR> &available_formats) {
    if (available_formats.empty()) {
        Logger::error("unexpected empty list of available formats");
    }

    for (const auto &available_format: available_formats) {
        if (
            available_format.format == vk::Format::eB8G8R8A8Unorm
            && available_format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear
        ) {
            return available_format;
        }
    }

    return available_formats[0];
}

vk::PresentModeKHR SwapChain::choose_present_mode(const vector<vk::PresentModeKHR> &available_present_modes) {
    for (const auto &available_present_mode: available_present_modes) {
        if (available_present_mode == vk::PresentModeKHR::eMailbox) {
            return available_present_mode;
        }
    }

    return vk::PresentModeKHR::eFifo;
}

void SwapChain::create_color_resources(const RendererContext &ctx) {
    const vk::Format color_format = image_format;

    const vk::ImageCreateInfo image_info{
        .imageType = vk::ImageType::e2D,
        .format = color_format,
        .extent = {
            .width = extent.width,
            .height = extent.height,
            .depth = 1,
        },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = msaa_sample_count,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    color_image = make_unique<Image>(
        ctx,
        image_info,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        vk::ImageAspectFlagBits::eColor
    );
}

void SwapChain::create_depth_resources(const RendererContext &ctx) {
    const vk::ImageCreateInfo image_info{
        .imageType = vk::ImageType::e2D,
        .format = depth_format,
        .extent = {
            .width = extent.width,
            .height = extent.height,
            .depth = 1,
        },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = msaa_sample_count,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    depth_image = make_unique<Image>(
        ctx,
        image_info,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        vk::ImageAspectFlagBits::eDepth
    );
}

vector<SwapChainRenderTargets> SwapChain::get_render_targets(const RendererContext &ctx) {
    vector<SwapChainRenderTargets> targets;

    if (cached_views.empty()) {
        for (const auto &image: images) {
            auto view = make_shared<vk::raii::ImageView>(utils::img::create_image_view(
                ctx,
                image,
                image_format,
                vk::ImageAspectFlagBits::eColor
            ));

            cached_views.emplace_back(view);
        }
    }

    for (const auto &view: cached_views) {
        const bool is_msaa = msaa_sample_count != vk::SampleCountFlagBits::e1;

        auto color_target = is_msaa
                               ? RenderTarget{
                                   color_image->get_view(ctx),
                                   view,
                                   image_format
                               }
                               : RenderTarget{
                                   view,
                                   image_format
                               };

        RenderTarget depth_target{
            depth_image->get_view(ctx),
            depth_format
        };

        targets.emplace_back(std::move(color_target), std::move(depth_target));
    }

    return targets;
}

vk::Format SwapChain::find_depth_format(const RendererContext &ctx) {
    return find_supported_format(
        ctx,
        {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eDepthStencilAttachment
    );
}

vk::Format SwapChain::find_supported_format(const RendererContext &ctx, const vector<vk::Format> &candidates,
                                          const vk::ImageTiling tiling, const vk::FormatFeatureFlags features) {
    for (const vk::Format format: candidates) {
        const vk::FormatProperties props = ctx.physical_device->getFormatProperties(format);

        if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) {
            return format;
        }
        if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }

    Logger::error("failed to find supported format!");
    return {};
}
} // zrx
