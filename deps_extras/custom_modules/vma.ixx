module;

#include "../VulkanMemoryAllocator/include/vk_mem_alloc.h"

export module vma;

export {
    using ::VmaAllocator_T;
    using ::VmaAllocator;
    using ::VmaAllocation_T;
    using ::VmaAllocation;

    using ::VmaVulkanFunctions;

    using ::VmaAllocatorCreateInfo;
    using ::VmaAllocationCreateInfo;

    using ::VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

    using ::VmaAllocationCreateFlags;
    using ::VmaAllocationCreateFlagBits;
    using ::VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

    using ::VmaMemoryUsage;
    using ::VMA_MEMORY_USAGE_AUTO;

    using ::vmaCreateAllocator;
    using ::vmaDestroyAllocator;
    using ::vmaCreateBuffer;
    using ::vmaDestroyBuffer;
    using ::vmaCreateImage;
    using ::vmaDestroyImage;
    using ::vmaMapMemory;
    using ::vmaUnmapMemory;
    using ::vmaFreeMemory;
}
