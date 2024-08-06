#include "buffer.hpp"

#include "cmd.hpp"
#include "src/render/renderer.hpp"

Buffer::Buffer(const VmaAllocator _allocator, const vk::DeviceSize size, const vk::BufferUsageFlags usage,
               const vk::MemoryPropertyFlags properties)
    : allocator(_allocator) {
    const vk::BufferCreateInfo bufferInfo{
        .size = size,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
    };

    VmaAllocationCreateFlags flags{};
    if (properties & vk::MemoryPropertyFlagBits::eHostVisible) {
        flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    }

    const VmaAllocationCreateInfo allocInfo{
        .flags = flags,
        .usage = VMA_MEMORY_USAGE_AUTO,
        .requiredFlags = static_cast<VkMemoryPropertyFlags>(properties)
    };

    const auto result = vmaCreateBuffer(
        allocator,
        reinterpret_cast<const VkBufferCreateInfo *>(&bufferInfo),
        &allocInfo,
        reinterpret_cast<VkBuffer *>(&buffer),
        &allocation,
        nullptr
    );

    if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer!");
    }
}

Buffer::~Buffer() {
    if (mapped) {
        unmap();
    }

    vmaDestroyBuffer(allocator, static_cast<VkBuffer>(buffer), allocation);
}

void *Buffer::map() {
    if (!mapped && vmaMapMemory(allocator, allocation, &mapped) != VK_SUCCESS) {
        throw std::runtime_error("failed to map buffer memory!");
    }

    return mapped;
}

void Buffer::unmap() {
    if (!mapped) {
        throw std::runtime_error("tried to unmap a buffer that wasn't mapped!");
    }

    vmaUnmapMemory(allocator, allocation);
    mapped = nullptr;
}

void Buffer::copyFromBuffer(const RendererContext &ctx, const Buffer &otherBuffer,
                            const vk::DeviceSize size, const vk::DeviceSize srcOffset,
                            const vk::DeviceSize dstOffset) const {
    const vk::raii::CommandBuffer commandBuffer = vkutils::cmd::beginSingleTimeCommands(ctx);

    const vk::BufferCopy copyRegion{
        .srcOffset = srcOffset,
        .dstOffset = dstOffset,
        .size = size,
    };

    commandBuffer.copyBuffer(*otherBuffer, buffer, copyRegion);

    vkutils::cmd::endSingleTimeCommands(commandBuffer, *ctx.graphicsQueue);
}

namespace vkutils::buf {
    template<typename ElemType>
    unique_ptr<Buffer> createLocalBuffer(const RendererContext &ctx, const std::vector<ElemType> &contents,
                                         const vk::BufferUsageFlags usage) {
        const vk::DeviceSize bufferSize = sizeof(contents[0]) * contents.size();

        Buffer stagingBuffer{
            **ctx.allocator,
            bufferSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        };

        void *data = stagingBuffer.map();
        memcpy(data, contents.data(), static_cast<size_t>(bufferSize));
        stagingBuffer.unmap();

        auto resultBuffer = make_unique<Buffer>(
            **ctx.allocator,
            bufferSize,
            vk::BufferUsageFlagBits::eTransferDst | usage,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );

        resultBuffer->copyFromBuffer(ctx, stagingBuffer, bufferSize);

        return resultBuffer;
    }

#define CREATE_LOCAL_BUFFER_DECL(ELEM_TYPE) \
    template unique_ptr<Buffer> \
    createLocalBuffer<ELEM_TYPE>(const RendererContext &, const std::vector<ELEM_TYPE> &, vk::BufferUsageFlags);

    CREATE_LOCAL_BUFFER_DECL(ModelVertex)
    CREATE_LOCAL_BUFFER_DECL(SkyboxVertex)
    CREATE_LOCAL_BUFFER_DECL(ScreenSpaceQuadVertex)
    CREATE_LOCAL_BUFFER_DECL(uint32_t)
    CREATE_LOCAL_BUFFER_DECL(glm::mat4)
}
