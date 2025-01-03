#include "cmd.hpp"

#include "ctx.hpp"

namespace zrx {
namespace utils::cmd {
    vk::raii::CommandBuffer begin_single_time_commands(const RendererContext &ctx) {
        const vk::CommandBufferAllocateInfo alloc_info{
            .commandPool = **ctx.command_pool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1U,
        };

        vk::raii::CommandBuffers command_buffers{*ctx.device, alloc_info};
        vk::raii::CommandBuffer buffer{std::move(command_buffers[0])};

        constexpr vk::CommandBufferBeginInfo begin_info{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        };

        buffer.begin(begin_info);

        return buffer;
    }

    void end_single_time_commands(const vk::raii::CommandBuffer &command_buffer, const vk::raii::Queue &queue) {
        command_buffer.end();

        const vk::SubmitInfo submitInfo{
            .commandBufferCount = 1U,
            .pCommandBuffers = &*command_buffer,
        };

        queue.submit(submitInfo);
        queue.waitIdle();
    }

    void do_single_time_commands(const RendererContext &ctx,
                              const std::function<void(const vk::raii::CommandBuffer &)> &func) {
        const vk::raii::CommandBuffer cmd_buffer = begin_single_time_commands(ctx);
        func(cmd_buffer);
        end_single_time_commands(cmd_buffer, *ctx.graphics_queue);
    }

    void set_dynamic_states(const vk::raii::CommandBuffer &command_buffer, const vk::Extent2D draw_extent) {
        const vk::Viewport viewport{
            .x = 0.0f,
            .y = static_cast<float>(draw_extent.height), // flip the y-axis
            .width = static_cast<float>(draw_extent.width),
            .height = -1 * static_cast<float>(draw_extent.height), // flip the y-axis
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };

        const vk::Rect2D scissor{
            .offset = {0, 0},
            .extent = draw_extent
        };

        command_buffer.setViewport(0, viewport);
        command_buffer.setScissor(0, scissor);
    }
} // utils::cmd
} // zrx
