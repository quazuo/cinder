module;

module Cinder.Render.Vulkan;

import vulkan;

import :Context;

namespace zrx {
static queue_family_index_t select_present_queue_family(const RendererContext& ctx, const vk::raii::SurfaceKHR& surface) {
    queue_family_index_t queue_family_index = 0;

    const vector<vk::QueueFamilyProperties> queue_family_properties = ctx.physical_device->getQueueFamilyProperties();

    for (size_t i = 0; i < queue_family_properties.size(); i++) {
        if (ctx.physical_device->getSurfaceSupportKHR(i, *surface)) {
            queue_family_index = i;
            break;
        }

        if (i == queue_family_properties.size() - 1) {
            Logger::error("Failed to find a present queue family");
        }
    }

    return queue_family_index;
}

static queue_family_index_t select_graphics_queue_family(const RendererContext& ctx) {
    queue_family_index_t queue_family_index = 0;

    const vector<vk::QueueFamilyProperties> queue_family_properties = ctx.physical_device->getQueueFamilyProperties();

    for (size_t i = 0; i < queue_family_properties.size(); i++) {
        if (queue_family_properties[i].queueFlags & vk::QueueFlagBits::eGraphics) {
            queue_family_index = i;
            break;
        }

        if (i == queue_family_properties.size() - 1) {
            Logger::error("Failed to find a graphics queue family");
        }
    }

    return queue_family_index;
}

PresentQueue::PresentQueue(const RendererContext& ctx, const vk::raii::SurfaceKHR& surface)
    : family_index(select_present_queue_family(ctx, surface)), queue(*ctx.device, family_index, 0) {
}

auto PresentQueue::present(const SwapChain& swap_chain, const std::span<const BinarySemaphore>& wait_semaphores) const -> vk::Result {
    const vector<vk::Semaphore> wait_semaphores_raw = wait_semaphores
        | std::ranges::views::transform([](const BinarySemaphore& sem) -> vk::Semaphore { return **sem; })
        | std::ranges::to<vector<vk::Semaphore>>();
    return _present(swap_chain, std::span { wait_semaphores_raw });
}

auto PresentQueue::_present(const SwapChain &swap_chain, const std::span<const vk::Semaphore> &wait_semaphores) const -> vk::Result {
    const uint32_t image_index = swap_chain.get_current_image_index();

    const vk::PresentInfoKHR present_info{
        .waitSemaphoreCount = static_cast<uint32_t>(wait_semaphores.size()),
        .pWaitSemaphores = wait_semaphores.data(),
        .swapchainCount = 1u,
        .pSwapchains = &**swap_chain,
        .pImageIndices = &image_index,
    };

    auto present_result = vk::Result::eSuccess;

    try {
        present_result = queue.presentKHR(present_info);
    } catch (...) {
    }

    return present_result;
}

GraphicsQueue::GraphicsQueue(const RendererContext &ctx)
    : family_index(select_graphics_queue_family(ctx)), queue(*ctx.device, family_index, 0) {
}

void GraphicsQueue::submit(const QueueSubmission &submission) const {
    static constexpr vk::PipelineStageFlags wait_stages[] = {
        vk::PipelineStageFlagBits::eEarlyFragmentTests,
        vk::PipelineStageFlagBits::eVertexInput,
    };

    const vk::StructureChain<vk::SubmitInfo, vk::TimelineSemaphoreSubmitInfo> submit_info{
        {
            .waitSemaphoreCount = static_cast<uint32_t>(submission.wait_semaphores_raw.size()),
            .pWaitSemaphores = submission.wait_semaphores_raw.data(),
            .pWaitDstStageMask = wait_stages,
            .commandBufferCount = static_cast<uint32_t>(submission.command_buffers.size()),
            .pCommandBuffers = submission.command_buffers.data(),
            .signalSemaphoreCount = static_cast<uint32_t>(submission.signal_semaphores_raw.size()),
            .pSignalSemaphores = submission.signal_semaphores_raw.data(),
        },
        {
            .waitSemaphoreValueCount = static_cast<uint32_t>(submission.wait_semaphore_values.size()),
            .pWaitSemaphoreValues = submission.wait_semaphore_values.data(),
            .signalSemaphoreValueCount = static_cast<uint32_t>(submission.signal_semaphore_values.size()),
            .pSignalSemaphoreValues = submission.signal_semaphore_values.data(),
        }
    };

    queue.submit(submit_info.get<vk::SubmitInfo>());
}
} // zrx