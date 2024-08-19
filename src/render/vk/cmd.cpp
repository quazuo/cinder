#include "cmd.hpp"

#include "ctx.hpp"

namespace zrx {
namespace utils::cmd {
    vk::raii::CommandBuffer beginSingleTimeCommands(const RendererContext &ctx) {
        const vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = **ctx.commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1U,
        };

        vk::raii::CommandBuffers commandBuffers{*ctx.device, allocInfo};
        vk::raii::CommandBuffer buffer{std::move(commandBuffers[0])};

        constexpr vk::CommandBufferBeginInfo beginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        };

        buffer.begin(beginInfo);

        return buffer;
    }

    void endSingleTimeCommands(const vk::raii::CommandBuffer &commandBuffer, const vk::raii::Queue &queue) {
        commandBuffer.end();

        const vk::SubmitInfo submitInfo{
            .commandBufferCount = 1U,
            .pCommandBuffers = &*commandBuffer,
        };

        queue.submit(submitInfo);
        queue.waitIdle();
    }

    void doSingleTimeCommands(const RendererContext &ctx,
                              const std::function<void(const vk::raii::CommandBuffer &)> &func) {
        const vk::raii::CommandBuffer cmdBuffer = beginSingleTimeCommands(ctx);
        func(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, *ctx.graphicsQueue);
    }

    void setDynamicStates(const vk::raii::CommandBuffer &commandBuffer, const vk::Extent2D drawExtent) {
        const vk::Viewport viewport{
            .x = 0.0f,
            .y = static_cast<float>(drawExtent.height), // flip the y-axis
            .width = static_cast<float>(drawExtent.width),
            .height = -1 * static_cast<float>(drawExtent.height), // flip the y-axis
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };

        const vk::Rect2D scissor{
            .offset = {0, 0},
            .extent = drawExtent
        };

        commandBuffer.setViewport(0, viewport);
        commandBuffer.setScissor(0, scissor);
    }
} // utils::cmd
} // zrx
