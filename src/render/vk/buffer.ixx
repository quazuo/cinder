module;

export module Cinder.Render.Vulkan:Buffer;

import vk_mem_alloc;
import std;
import cvulkan;

import Cinder.Utils;
import :Context;

import Cinder.Globals;

export namespace zrx {
class BufferBuilder;

/**
 * Abstraction over a Vulkan buffer, making it easier to manage by hiding all the Vulkan API calls.
 * These buffers are allocated using VMA and are currently suited mostly for two scenarios: first,
 * when one needs a device-local buffer, and second, when one needs a host-visible and host-coherent
 * buffer, e.g. for use as a staging buffer.
 */
class Buffer {
    unique_ptr<vk::raii::Buffer> buffer;
    vk::DeviceSize size;
    void *mapped = nullptr;

    reference_wrapper<const vma::raii::Allocator> allocator;
    unique_ptr<vma::raii::Allocation> allocation;

public:
    explicit Buffer(const RendererContext& ctx, vk::DeviceSize size, vk::BufferUsageFlags usage,
                    vk::MemoryPropertyFlags properties);

    /**
     * Returns a raw handle to the actual Vulkan buffer.
     *
     * @return Handle to the buffer.
     */
    auto operator*() const -> const vk::raii::Buffer& { return *buffer; }

    auto get_size() const -> vk::DeviceSize { return size; }

    /**
     * Copies the contents of some other given buffer to this buffer and waits until completion.
     *
     * @param ctx Renderer context.
     * @param other_buffer Buffer from which to copy.
     * @param size Size of the data to copy.
     * @param src_offset Offset in the source buffer.
     * @param dst_offset Offset in this (destination) buffer.
     */
    void copy_from_buffer(const RendererContext &ctx, const Buffer &other_buffer, vk::DeviceSize size,
                          vk::DeviceSize src_offset = 0, vk::DeviceSize dst_offset = 0) const;

    /**
     * Copies the contents of some other pointer to this buffer and waits until completion.
     *
     * @param ptr Pointer from which to copy.
     * @param size Size of the data to copy.
     * @param dst_offset Offset in this (destination) buffer.
     */
    void copy_from_ptr(const void *ptr, vk::DeviceSize size, vk::DeviceSize dst_offset = 0) const;
};

struct BufferSlice {
    reference_wrapper<const Buffer> buffer;
    vk::DeviceSize size;
    vk::DeviceSize offset;

    BufferSlice(const Buffer &buffer, const vk::DeviceSize size, const vk::DeviceSize offset = 0)
        : buffer(buffer), size(size), offset(offset) {
        if (size + offset > buffer.get_size()) {
            Logger::error("buffer slice extent out of range");
        }
    }

    auto operator*() const -> const Buffer&{ return buffer.get(); }
};

class BufferBuilder {
    optional<const void*> data;
    vk::DeviceSize size = 0;
    vk::BufferUsageFlags usage;
    vk::MemoryPropertyFlags memory_properties;

public:
    // presets

    auto as_staging(vk::DeviceSize size)                    -> const BufferBuilder&;
    auto as_local(const void *data, vk::DeviceSize size)    -> const BufferBuilder&;
    auto as_uniform(vk::DeviceSize size)                    -> const BufferBuilder&;
    auto as_uniform(const void *data, vk::DeviceSize size)  -> const BufferBuilder&;

    // manual modifiers

    auto as_uninitialized(vk::DeviceSize size)              -> const BufferBuilder&;
    auto from_data(const void *data, vk::DeviceSize size)   -> const BufferBuilder&;
    auto with_usage(vk::BufferUsageFlags u)                 -> BufferBuilder&;
    auto with_memory_properties(vk::MemoryPropertyFlags mp) -> BufferBuilder&;

    Buffer create(const RendererContext& ctx) const;

private:
    void validate() const;
};
} // zrx
