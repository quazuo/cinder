module;

export module Cinder.Render.Vulkan:Swapchain;

import vulkan;
import std;
import glfw;

import :Image;
import :Context;

import Cinder.Globals;

export namespace zrx {
/**
 * Helper structure holding details about supported features of the swap chain.
 */
struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    vector<vk::SurfaceFormatKHR> formats;
    vector<vk::PresentModeKHR> present_modes;

    SwapChainSupportDetails(const vk::raii::PhysicalDevice &physical_device, const vk::raii::SurfaceKHR &surface);
};

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
    vector<vk::Image> images;
    vk::Format image_format{};
    vk::Format depth_format{};
    vk::Extent2D extent{};

    unique_ptr<Image> color_image;
    unique_ptr<Image> depth_image;

    vector<shared_ptr<vk::raii::ImageView>> cached_views;

    uint32_t current_image_index = 0;

    vk::SampleCountFlagBits msaa_sample_count;

public:
    explicit SwapChain(const RendererContext &ctx, const vk::raii::SurfaceKHR &surface,
                       const QueueFamilyIndices &queue_families,
                       vk::SampleCountFlagBits sample_count = vk::SampleCountFlagBits::e1);

    SwapChain(const SwapChain &other) = delete;

    SwapChain &operator=(const SwapChain &other) = delete;

    auto operator*() const -> const vk::raii::SwapchainKHR& { return *swap_chain; }

    auto get_image_format() const -> vk::Format { return image_format; }

    auto get_depth_format() const -> vk::Format { return depth_format; }

    auto get_extent() const -> vk::Extent2D { return extent; }

    /**
     * Returns the index of the image that was most recently acquired and will be presented next.
     * @return Index of the current image.
     */
    auto get_current_image_index() const -> uint32_t { return current_image_index; }

    /**
     * Returns the image that was most recently acquired and will be presented next.
     * @return Index of the current image.
     */
    auto get_current_image() const -> const vk::Image& { return images[current_image_index]; }

    /**
     * Returns the image that is being rendered to during the current frame.
     * @return Index of the current image.
     */
    auto get_current_rendered_image() const -> const vk::Image& {
        const bool is_msaa = msaa_sample_count != vk::SampleCountFlagBits::e1;
        return is_msaa ? **color_image : images[current_image_index];
    }

    /**
     * Wraps swapchain image views in `RenderTarget` objects and returns them.
     * When called the first time, these views are created and cached for later.
     */
    auto get_render_targets(const RendererContext &ctx) -> vector<SwapChainRenderTargets>;

    /**
     * Requests a new image from the swap chain and signals a given semaphore when the image is available.
     * @param semaphore Semaphore which should be signalled after completion.
     * @return Result code and index of the new image.
     */
    auto acquire_next_image(const vk::raii::Semaphore &semaphore) -> pair<vk::Result, uint32_t>;

    static auto get_image_count(const RendererContext &ctx, const vk::raii::SurfaceKHR &surface) -> uint32_t;

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

    static auto find_depth_format(const RendererContext &ctx) -> vk::Format;

    static auto find_supported_format(const RendererContext &ctx, const vector<vk::Format> &candidates,
                                      vk::ImageTiling tiling, vk::FormatFeatureFlags features) -> vk::Format;

    static auto choose_extent(const vk::SurfaceCapabilitiesKHR &capabilities, GLFWwindow *window) -> vk::Extent2D;

    static auto choose_surface_format(const vector<vk::SurfaceFormatKHR> &available_formats) -> vk::SurfaceFormatKHR;

    static auto choose_present_mode(const vector<vk::PresentModeKHR> &available_present_modes) -> vk::PresentModeKHR;
};
} // zrx
