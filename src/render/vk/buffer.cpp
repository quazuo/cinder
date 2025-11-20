module;

module Cinder.Render.Vulkan;

import vulkan_hpp;
import cvulkan;

import Cinder.Render.Mesh;
import :Command;

namespace zrx {
Buffer::Buffer(const VmaAllocator _allocator, const vk::DeviceSize size, const vk::BufferUsageFlags usage,
               const vk::MemoryPropertyFlags properties)
    : allocator(_allocator), size(size) {
    const vk::BufferCreateInfo buffer_info{
        .size = size,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
    };

    VmaAllocationCreateFlags flags{};
    if (properties & vk::MemoryPropertyFlagBits::eHostVisible) {
        flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    }

    const VmaAllocationCreateInfo alloc_info{
        .flags = flags,
        .usage = VMA_MEMORY_USAGE_AUTO,
        .requiredFlags = static_cast<VkMemoryPropertyFlags>(properties)
    };

    const auto result = vmaCreateBuffer(
        allocator,
        reinterpret_cast<const VkBufferCreateInfo *>(&buffer_info),
        &alloc_info,
        reinterpret_cast<VkBuffer *>(&buffer),
        &allocation,
        nullptr
    );

    if (result != VK_SUCCESS) {
        Logger::error("failed to allocate buffer!");
    }
}

Buffer::~Buffer() {
    if (mapped) {
        unmap();
    }

    vmaDestroyBuffer(allocator, static_cast<VkBuffer>(buffer), allocation);
}

Buffer::Buffer(Buffer &&other) {
    allocator = other.allocator;
    buffer = other.buffer;
    allocation = other.allocation;
    size = other.size;

    other.buffer = nullptr;
    other.allocation = nullptr;
    other.size = 0;
}

Buffer& Buffer::operator=(Buffer &&other) {
    allocator = other.allocator;
    buffer = other.buffer;
    allocation = other.allocation;
    size = other.size;

    other.buffer = nullptr;
    other.allocation = nullptr;
    other.size = 0;

    return *this;
}

void *Buffer::map() {
    if (!mapped && vmaMapMemory(allocator, allocation, &mapped) != VK_SUCCESS) {
        Logger::error("failed to map buffer memory!");
    }

    return mapped;
}

void Buffer::unmap() {
    if (!mapped) {
        Logger::error("tried to unmap a buffer that wasn't mapped!");
    }

    vmaUnmapMemory(allocator, allocation);
    mapped = nullptr;
}

void Buffer::copy_from_buffer(const RendererContext &ctx, const Buffer &other_buffer,
                              const vk::DeviceSize size, const vk::DeviceSize src_offset,
                              const vk::DeviceSize dst_offset) const {
    utils::cmd::do_single_time_commands(ctx, [&](const vk::raii::CommandBuffer& command_buffer) {
        const vk::BufferCopy copy_region{
            .srcOffset = src_offset,
            .dstOffset = dst_offset,
            .size = size,
        };

        command_buffer.copyBuffer(*other_buffer, buffer, copy_region);
    });
}

namespace utils::buf {
    auto create_uniform_buffer(const RendererContext &ctx, const vk::DeviceSize size) -> Buffer {
        return Buffer {
            **ctx.allocator,
            size,
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        };
    }
}
} // zrx
