#pragma once

#include "image.hpp"
#include "src/render/libs.hpp"

struct GLFWwindow;

namespace zrx {
/**
 * Helper structure holding details about supported features of the swap chain.
 */
struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> present_modes;

    SwapChainSupportDetails(const vk::raii::PhysicalDevice &physical_device, const vk::raii::SurfaceKHR &surface);
};

struct RendererContext;
struct QueueFamilyIndices;

struct SwapChainRenderTargets {
    RenderTarget color_target;
    RenderTarget depth_target;
};

/**
* Abstraction over a Vulkan swap chain, making it easier to manage by hiding all the Vulkan API calls.
*/
class SwapChain {
    unique_ptr<vk::raii::SwapchainKHR> swap_chain;
    std::vector<vk::Image> images;
    vk::Format image_format{};
    vk::Format depth_format{};
    vk::Extent2D extent{};

    unique_ptr<Image> color_image;
    unique_ptr<Image> depth_image;

    std::vector<shared_ptr<vk::raii::ImageView>> cached_views;

    uint32_t current_image_index = 0;

    vk::SampleCountFlagBits msaa_sample_count;

public:
    explicit SwapChain(const RendererContext &ctx, const vk::raii::SurfaceKHR &surface,
                       const QueueFamilyIndices &queue_families, GLFWwindow *window,
                       vk::SampleCountFlagBits sample_count = vk::SampleCountFlagBits::e1);

    SwapChain(const SwapChain &other) = delete;

    SwapChain &operator=(const SwapChain &other) = delete;

    [[nodiscard]] const vk::raii::SwapchainKHR &operator*() const { return *swap_chain; }

    [[nodiscard]] vk::Format get_image_format() const { return image_format; }

    [[nodiscard]] vk::Format get_depth_format() const { return depth_format; }

    [[nodiscard]] vk::Extent2D get_extent() const { return extent; }

    /**
     * Returns the index of the image that was most recently acquired and will be presented next.
     * @return Index of the current image.
     */
    [[nodiscard]] uint32_t get_current_image_index() const { return current_image_index; }

    /**
     * Wraps swapchain image views in `RenderTarget` objects and returns them.
     * When called the first time, these views are created and cached for later.
     */
    [[nodiscard]] std::vector<SwapChainRenderTargets> get_render_targets(const RendererContext &ctx);

    /**
     * Requests a new image from the swap chain and signals a given semaphore when the image is available.
     * @param semaphore Semaphore which should be signalled after completion.
     * @return Result code and index of the new image.
     */
    [[nodiscard]] std::pair<vk::Result, uint32_t> acquire_next_image(const vk::raii::Semaphore &semaphore);

    [[nodiscard]] static uint32_t get_image_count(const RendererContext &ctx, const vk::raii::SurfaceKHR &surface);

    /**
     * Records commands that transition the most newly acquired image to a layout
     * appropriate for having the image serve as a color attachment.
     */
    void transition_to_attachment_layout(const vk::raii::CommandBuffer &command_buffer) const;

    /**
     * Records commands that transition the most newly acquired image to a layout
     * appropriate for having the image be presented to the screen.
     */
    void transition_to_present_layout(const vk::raii::CommandBuffer &command_buffer) const;

private:
    void create_color_resources(const RendererContext &ctx);

    void create_depth_resources(const RendererContext &ctx);

    [[nodiscard]] static vk::Format find_depth_format(const RendererContext &ctx);

    [[nodiscard]] static vk::Format
    find_supported_format(const RendererContext &ctx, const std::vector<vk::Format> &candidates,
                        vk::ImageTiling tiling, vk::FormatFeatureFlags features);

    [[nodiscard]] static vk::Extent2D choose_extent(const vk::SurfaceCapabilitiesKHR &capabilities, GLFWwindow *window);

    static vk::SurfaceFormatKHR choose_surface_format(const std::vector<vk::SurfaceFormatKHR> &available_formats);

    static vk::PresentModeKHR choose_present_mode(const std::vector<vk::PresentModeKHR> &available_present_modes);
};
} // zrx
