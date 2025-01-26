#include "cmd.hpp"

#include "ctx.hpp"

namespace zrx {
namespace utils::cmd {
    [[nodiscard]]
    vk::raii::CommandBuffer begin_single_time_commands(const RendererContext &ctx) {
        auto command_buffer = create_command_buffer(ctx, vk::CommandBufferLevel::ePrimary);

        command_buffer.begin(vk::CommandBufferBeginInfo {
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        });

        return command_buffer;
    }

    void end_single_time_commands(const vk::raii::CommandBuffer &command_buffer, const vk::raii::Queue &queue) {
        command_buffer.end();

        const vk::SubmitInfo submit_info{
            .commandBufferCount = 1U,
            .pCommandBuffers = &*command_buffer,
        };

        queue.submit(submit_info);
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

    vk::raii::CommandBuffers create_command_buffers(const RendererContext &ctx, const vk::CommandBufferLevel level,
                                                    const uint32_t count) {
        const vk::CommandBufferAllocateInfo alloc_info{
            .commandPool = **ctx.command_pool,
            .level = level,
            .commandBufferCount = count,
        };

        return {*ctx.device, alloc_info};
    }

    vk::raii::CommandBuffer create_command_buffer(const RendererContext &ctx, const vk::CommandBufferLevel level) {
        auto command_buffers = create_command_buffers(ctx, level, 1);
        return std::move(command_buffers[0]);
    }
} // utils::cmd
} // zrx
