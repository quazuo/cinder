module;

module Cinder.Render.Vulkan;

import vulkan_hpp;
import cvulkan;

import Cinder.Render.Mesh;
import :Command;
#include "glm/gtx/scalar_relational.inl"

namespace zrx {
Buffer::Buffer(const RendererContext& ctx, const vk::DeviceSize size, const vk::BufferUsageFlags usage,
               const vk::MemoryPropertyFlags properties)
    : size(size), allocator(*ctx.allocator) {
    const vk::BufferCreateInfo buffer_info{
        .size = size,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
    };

    vma::AllocationCreateFlags flags{};
    if (properties & vk::MemoryPropertyFlagBits::eHostVisible) {
        flags |= vma::AllocationCreateFlagBits::eHostAccessRandom;
    }

    const vma::AllocationCreateInfo alloc_info{
        .flags = flags,
        .usage = vma::MemoryUsage::eAuto,
        .requiredFlags = static_cast<VkMemoryPropertyFlags>(properties)
    };

    auto [allocation, buffer] = ctx.allocator->createBuffer(buffer_info, alloc_info);
    this->allocation = allocation;
    this->buffer = make_unique<vk::raii::Buffer>(*ctx.device, buffer);
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

        command_buffer.copyBuffer(*other_buffer, *buffer, copy_region);
    });
}

void Buffer::copy_from_ptr(const void *ptr, vk::DeviceSize size, vk::DeviceSize dst_offset) const {
    allocation->copyFromMemory(ptr, dst_offset, size);
}

namespace utils::buf {
    auto create_uniform_buffer(const RendererContext &ctx, const vk::DeviceSize size) -> Buffer {
        return Buffer {
            ctx,
            size,
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        };
    }
}
} // zrx
