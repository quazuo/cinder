#include "ctx.hpp"

#include <vma/vk_mem_alloc.h>

namespace zrx {
VmaAllocatorWrapper::VmaAllocatorWrapper(const vk::PhysicalDevice physical_device, const vk::Device device,
                                         const vk::Instance instance) {
    static constexpr VmaVulkanFunctions funcs{
        .vkGetInstanceProcAddr = vkGetInstanceProcAddr,
        .vkGetDeviceProcAddr = vkGetDeviceProcAddr
    };

    const VmaAllocatorCreateInfo allocator_create_info{
        .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = physical_device,
        .device = device,
        .pVulkanFunctions = &funcs,
        .instance = instance,
    };

    vmaCreateAllocator(&allocator_create_info, &allocator);
}

VmaAllocatorWrapper::~VmaAllocatorWrapper() {
    vmaDestroyAllocator(allocator);
}
} // zrx
