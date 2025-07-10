module;

#include <vulkan/vulkan.hpp>

export module cvulkan;

export {
    using ::VkImage;
    using ::VkBuffer;
    using ::VkMemoryPropertyFlags;
    using ::VkBool32;
    using ::VkSurfaceKHR;
    using ::VkDescriptorPool;
    using ::VkSampleCountFlagBits;
    using ::VkFormat;
    using ::VkImageCreateInfo;
    using ::VkBufferCreateInfo;

    using ::VkDebugUtilsMessageSeverityFlagBitsEXT;
    using ::VkDebugUtilsMessageTypeFlagsEXT;
    using ::VkDebugUtilsMessengerCallbackDataEXT;

    using ::vkGetInstanceProcAddr;
    using ::vkGetDeviceProcAddr;

    using ::VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;

    using ::VK_SUCCESS;

    constexpr auto ZRX_VK_KHR_SWAPCHAIN_EXTENSION_NAME = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    constexpr auto ZRX_VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME = VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME;
    constexpr auto ZRX_VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME = VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME;
    constexpr auto ZRX_VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME = VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME;
    constexpr auto ZRX_VK_KHR_MULTIVIEW_EXTENSION_NAME = VK_KHR_MULTIVIEW_EXTENSION_NAME;
    constexpr auto ZRX_VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME = VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME;
    constexpr auto ZRX_VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME = VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME;
    constexpr auto ZRX_VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME = VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME;
    constexpr auto ZRX_VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME = VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME;
    constexpr auto ZRX_VK_EXT_DEBUG_UTILS_EXTENSION_NAME = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    constexpr auto ZRX_VK_EXT_DEBUG_MARKER_EXTENSION_NAME = VK_EXT_DEBUG_MARKER_EXTENSION_NAME;
}
