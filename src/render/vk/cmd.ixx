module;

export module Cinder.Render.Vulkan:Command;

import vulkan;
import std;

import Cinder.Globals;

export namespace zrx {
struct RendererContext;

struct PipelineBarrierPack {
    vector<vk::MemoryBarrier2> memory_barriers;
    vector<vk::BufferMemoryBarrier2> buffer_memory_barriers;
    vector<vk::ImageMemoryBarrier2> image_memory_barriers;

    void insert(vk::MemoryBarrier2&& barrier)       { memory_barriers.emplace_back(barrier); }
    void insert(vk::BufferMemoryBarrier2&& barrier) { buffer_memory_barriers.emplace_back(barrier); }
    void insert(vk::ImageMemoryBarrier2&& barrier)  { image_memory_barriers.emplace_back(barrier); }

    void record_cmd(const vk::raii::CommandBuffer& command_buffer) const;
};

namespace utils::cmd {
    /**
    * Allocates and begins a new command buffer which is supposed to be recorded once
    * and destroyed after submission.
    *
    * @param ctx Renderer context.
    * @return The created single-use command buffer.
    */
    auto begin_single_time_commands(const RendererContext& ctx) -> vk::raii::CommandBuffer;

    /**
    * Ends a single-time command buffer created beforehand by `beginSingleTimeCommands`.
    * The buffer is then submitted and execution stops until the commands are fully processed.
    *
    * @param command_buffer The single-use command buffer which should be ended.
    * @param queue The queue to which the buffer should be submitted.
    */
    void end_single_time_commands(const vk::raii::CommandBuffer &command_buffer, const vk::raii::Queue &queue);

    /**
     * Convenience wrapper over `beginSingleTimeCommands` and `endSingleTimeCommands`.
     *
     * @param ctx Renderer context.
     * @param func Lambda containing commands with which the command buffer will be filled.
     */
    void do_single_time_commands(const RendererContext& ctx,
                                 const std::function<void(const vk::raii::CommandBuffer &)> &func);

    /**
     * Shorthand function to set all dynamic states used in rendering.
     * This currently includes only viewport and scissor, but might be extended later.
     */
    void set_dynamic_states(const vk::raii::CommandBuffer &command_buffer, vk::Extent2D draw_extent);

    auto create_command_buffers(const RendererContext& ctx, vk::CommandBufferLevel level, uint32_t count) -> vk::raii::CommandBuffers;

    auto create_command_buffer(const RendererContext &ctx, vk::CommandBufferLevel level) -> vk::raii::CommandBuffer;
}
} // zrx
