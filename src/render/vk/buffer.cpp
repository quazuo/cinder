module;

module Cinder.Render.Vulkan;

import vulkan;
import cvulkan;

import Cinder.Render.Mesh;
import :Command;

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
        .requiredFlags = properties
    };

    auto [allocation, buffer] = ctx.allocator->createBuffer(buffer_info, alloc_info).split();
    this->allocation = make_unique<vma::raii::Allocation>(std::move(allocation));
    this->buffer = make_unique<vk::raii::Buffer>(std::move(buffer));
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

void Buffer::copy_from_ptr(const void *ptr, const vk::DeviceSize size, const vk::DeviceSize dst_offset) const {
    allocation->copyFromMemory(ptr, dst_offset, size);
}

auto BufferBuilder::as_staging(const vk::DeviceSize size) -> const BufferBuilder& {
    this->size = size;
    usage |= vk::BufferUsageFlagBits::eTransferSrc;
    memory_properties |= vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    return *this;
}

auto BufferBuilder::as_local(const void *data, const vk::DeviceSize size) -> const BufferBuilder& {
    this->data = data;
    this->size = size;
    usage |= vk::BufferUsageFlagBits::eTransferDst;
    memory_properties |= vk::MemoryPropertyFlagBits::eDeviceLocal;
    return *this;
}

auto BufferBuilder::as_uniform(const vk::DeviceSize size) -> const BufferBuilder& {
    this->size = size;
    usage |= vk::BufferUsageFlagBits::eUniformBuffer;
    memory_properties |= vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    return *this;
}

auto BufferBuilder::as_uniform(const void *data, const vk::DeviceSize size) -> const BufferBuilder& {
    this->data = data;
    this->size = size;
    usage |= vk::BufferUsageFlagBits::eUniformBuffer;
    memory_properties |= vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    return *this;
}

auto BufferBuilder::as_uninitialized(const vk::DeviceSize size) -> const BufferBuilder& {
    this->size = size;
    return *this;
}

auto BufferBuilder::from_data(const void *data, const vk::DeviceSize size) -> const BufferBuilder& {
    if (this->size) {
        if (this->data) {
            Logger::error("cannot specify buffer data source: data already specified");
        } else {
            Logger::error("cannot specify buffer data source: buffer already specified as uninitialized");
        }
    }

    this->data = data;
    this->size = size;
    usage |= vk::BufferUsageFlagBits::eTransferDst;
    return *this;
}

auto BufferBuilder::with_usage(const vk::BufferUsageFlags u) -> BufferBuilder& {
    usage |= u;
    return *this;
}

auto BufferBuilder::with_memory_properties(const vk::MemoryPropertyFlags mp) -> BufferBuilder& {
    memory_properties |= mp;
    return *this;
}

Buffer BufferBuilder::create(const RendererContext &ctx) const {
    validate();

    Buffer result_buffer {
        ctx,
        size,
        usage,
        memory_properties
    };

    if (data) {
        const Buffer staging_buffer{
            ctx,
            size,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        };

        staging_buffer.copy_from_ptr(*data, size);
        result_buffer.copy_from_buffer(ctx, staging_buffer, size);
    }

    return result_buffer;
}

void BufferBuilder::validate() const {
    if (!usage) {
        Logger::error("cannot create buffer: no preset or explicit usage specified");
    }

    if (!memory_properties) {
        Logger::error("cannot create buffer: no preset or explicit memory properties specified");
    }
}
} // zrx
