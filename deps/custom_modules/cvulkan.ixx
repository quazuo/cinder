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
}
