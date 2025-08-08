module;

export module Cinder.Render.Vulkan:Buffer;

import vma;
import std;
import cvulkan;

import Cinder.Utils;
import :Context;

import Cinder.Globals;

export namespace zrx {
/**
 * Abstraction over a Vulkan buffer, making it easier to manage by hiding all the Vulkan API calls.
 * These buffers are allocated using VMA and are currently suited mostly for two scenarios: first,
 * when one needs a device-local buffer, and second, when one needs a host-visible and host-coherent
 * buffer, e.g. for use as a staging buffer.
 */
class Buffer {
    VmaAllocator allocator;
    vk::Buffer buffer;
    VmaAllocation allocation;
    vk::DeviceSize size;
    void *mapped = nullptr;

public:
    explicit Buffer(VmaAllocator _allocator, vk::DeviceSize size, vk::BufferUsageFlags usage,
                    vk::MemoryPropertyFlags properties);

    ~Buffer();

    Buffer(const Buffer &other) = delete;

    Buffer(Buffer &&other) = delete;

    Buffer &operator=(const Buffer &other) = delete;

    Buffer &operator=(Buffer &&other) = delete;

    /**
         * Returns a raw handle to the actual Vulkan buffer.
         *
         * @return Handle to the buffer.
         */
    auto operator*() const -> const vk::Buffer& { return buffer; }

    auto get_size() const -> vk::DeviceSize { return size; }

    /**
     * Maps the buffer's memory to host memory. This requires the buffer to *not* be created
     * with the vk::MemoryPropertyFlagBits::eDeviceLocal flag set in `properties` during object creation.
     * If already mapped, just returns the pointer to the previous mapping.
     *
     * @return Pointer to the mapped memory.
     */
    auto map() -> void*;

    /**
         * Unmaps the memory, after which the pointer returned by `map()` becomes invalidated.
         * Fails if `map()` wasn't called beforehand.
         */
    void unmap();

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

namespace utils::buf {
    template<typename ElemType>
    auto create_local_buffer(const RendererContext &ctx, const vector<ElemType> &contents,
                             const vk::BufferUsageFlags usage) -> unique_ptr<Buffer> {
        const vk::DeviceSize buffer_size = sizeof(contents[0]) * contents.size();

        Buffer staging_buffer{
            **ctx.allocator,
            buffer_size,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        };

        void *data = staging_buffer.map();
        std::memcpy(data, contents.data(), static_cast<size_t>(buffer_size));
        staging_buffer.unmap();

        auto result_buffer = make_unique<Buffer>(
            **ctx.allocator,
            buffer_size,
            vk::BufferUsageFlagBits::eTransferDst | usage,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );

        result_buffer->copy_from_buffer(ctx, staging_buffer, buffer_size);

        return result_buffer;
    }

    auto create_uniform_buffer(const RendererContext &ctx, vk::DeviceSize size) -> unique_ptr<Buffer>;
} // utils::buf
} // zrx
