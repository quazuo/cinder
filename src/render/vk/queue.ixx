module;

export module Cinder.Render.Vulkan:Queue;

import std;
import vulkan_hpp;

import Cinder.Globals;
import :Sync;

export namespace zrx {
struct RendererContext;
class SwapChain;

using queue_family_index_t = uint32_t;

struct QueueFamilyIndices {
    optional<uint32_t> graphics_compute_family;
    optional<uint32_t> present_family;

    auto is_complete() const -> bool {
        return graphics_compute_family.has_value() && present_family.has_value();
    }
};

class QueueSubmissionBuilder;

class QueueSubmission {
    vector<vk::Semaphore> wait_semaphores_raw;
    vector<vk::Semaphore> signal_semaphores_raw;
    vector<TimelineSemaphore::step_type_t> wait_semaphore_values;
    vector<TimelineSemaphore::step_type_t> signal_semaphore_values;
    vector<vk::CommandBuffer> command_buffers;

    QueueSubmission() = default;

    friend QueueSubmissionBuilder;
    friend GraphicsQueue;
};

class QueueSubmissionBuilder {
    QueueSubmission submission;

public:
    template<typename... Args>
        requires (is_one_of<Args, BinarySemaphore, TimelineSemaphore> && ...)
    auto with_wait_semaphores(const Args&... semaphores) -> QueueSubmissionBuilder& {
        auto [wait_semaphores, wait_semaphore_values] = utils::sync::make_semaphore_list_pair(semaphores...);
        submission.wait_semaphores_raw = std::move(wait_semaphores);
        submission.wait_semaphore_values = std::move(wait_semaphore_values);
        return *this;
    }

    template<typename... Args>
        requires (is_one_of<Args, BinarySemaphore, TimelineSemaphore> && ...)
    auto with_signal_semaphores(const Args&... semaphores) -> QueueSubmissionBuilder& {
        auto [signal_semaphores, signal_semaphore_values] = utils::sync::make_semaphore_list_pair(semaphores...);
        submission.signal_semaphores_raw = std::move(signal_semaphores);
        submission.signal_semaphore_values = std::move(signal_semaphore_values);
        return *this;
    }

    auto with_command_buffers(const std::span<vk::raii::CommandBuffer>& command_buffers) -> QueueSubmissionBuilder& {
        submission.command_buffers = command_buffers
            | std::ranges::views::transform([](const vk::raii::CommandBuffer& cb) { return *cb; })
            | std::ranges::to<vector<vk::CommandBuffer>>();
        return *this;
    }

    QueueSubmission create() const { return submission; }
};

class PresentQueue {
    queue_family_index_t family_index;
    vk::raii::Queue queue;

public:
    PresentQueue(const RendererContext& ctx, const vk::raii::SurfaceKHR& surface);

    const vk::raii::Queue& operator*() const { return queue; }

    queue_family_index_t get_family_index() const { return family_index; }

    auto present(const SwapChain& swap_chain, const std::span<const BinarySemaphore>& wait_semaphores) const -> vk::Result;

    template <typename... Ts>
        requires (std::same_as<Ts, BinarySemaphore> && ...)
    auto present(const SwapChain& swap_chain, const Ts&... wait_semaphores) const -> vk::Result {
        const array wait_semaphores_raw { (**wait_semaphores)... };
        return _present(swap_chain, wait_semaphores_raw);
    }

private:
    auto _present(const SwapChain& swap_chain, const std::span<const vk::Semaphore>& wait_semaphores) const -> vk::Result;
};

class GraphicsQueue {
    queue_family_index_t family_index;
    vk::raii::Queue queue;

public:
    explicit GraphicsQueue(const RendererContext& ctx);

    const vk::raii::Queue& operator*() const { return queue; }

    queue_family_index_t get_family_index() const { return family_index; }

    void submit(const QueueSubmission& submission) const;
};
} // zrx

