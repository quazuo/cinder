#pragma once

#include "src/render/libs.hpp"
#include "src/render/globals.hpp"

struct VmaAllocator_T;

namespace zrx {
/**
 * Simple RAII-preserving wrapper class for the VMA allocator.
 */
class VmaAllocatorWrapper {
    VmaAllocator_T* allocator{};

public:
    VmaAllocatorWrapper(vk::PhysicalDevice physical_device, vk::Device device, vk::Instance instance);

    ~VmaAllocatorWrapper();

    VmaAllocatorWrapper(const VmaAllocatorWrapper &other) = delete;

    VmaAllocatorWrapper(VmaAllocatorWrapper &&other) = delete;

    VmaAllocatorWrapper &operator=(const VmaAllocatorWrapper &other) = delete;

    VmaAllocatorWrapper &operator=(VmaAllocatorWrapper &&other) = delete;

    [[nodiscard]] VmaAllocator_T* operator*() const { return allocator; }
};

/**
 * Helper structure used to pass handles to essential Vulkan objects which are used while interacting with the API.
 * Introduced so that we can preserve top-down data flow and no object needs to refer to a renderer object
 * to get access to these.
 */
struct RendererContext {
    unique_ptr<vk::raii::PhysicalDevice> physical_device;
    unique_ptr<vk::raii::Device> device;
    unique_ptr<vk::raii::CommandPool> command_pool;
    unique_ptr<vk::raii::Queue> graphics_queue;
    unique_ptr<VmaAllocatorWrapper> allocator;
};
} // zrx
