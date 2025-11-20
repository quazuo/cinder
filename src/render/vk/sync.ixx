module;

export module Cinder.Render.Vulkan:Sync;

import std;
import vulkan_hpp;

import Cinder.Globals;
import Cinder.Utils;
import :Context;

export namespace zrx {

class BinarySemaphore {
    vk::raii::Semaphore semaphore;

public:
    explicit BinarySemaphore(const RendererContext& ctx);

    const vk::raii::Semaphore& operator*() const { return semaphore; }
};

class TimelineSemaphore {
public:
    using step_type_t = uint64_t;

private:
    vk::raii::Semaphore semaphore;
    step_type_t timeline_step = 0;

public:
    explicit TimelineSemaphore(const RendererContext& ctx);

    const vk::raii::Semaphore& operator*() const { return semaphore; }

    TimelineSemaphore& operator++() { timeline_step++; return *this; }

    step_type_t get_step() const { return timeline_step; }
};

namespace utils::sync {
    template <typename T>
        requires zrx::is_one_of<T, BinarySemaphore, TimelineSemaphore>
    auto get_sem_value(const T& sem) -> uint64_t {
        if constexpr (std::same_as<T, TimelineSemaphore>) {
            return sem.get_step();
        }
        return 0;
    };

    template<typename... Args>
        requires (is_one_of<Args, BinarySemaphore, TimelineSemaphore> && ...)
    auto make_semaphore_list_pair(const Args&... semaphores)
        -> pair<vector<vk::Semaphore>, vector<TimelineSemaphore::step_type_t>> {
        return {
            { (*semaphores)... },
            { (get_sem_value(semaphores))... },
        };
    }

    template<typename... Args>
        requires (is_one_of<Args, BinarySemaphore, TimelineSemaphore> && ...)
    void wait(const RendererContext& ctx, const Args&... semaphores) {
        const auto& [wait_semaphores, wait_semaphore_values] = make_semaphore_list_pair(semaphores...);

        const vk::SemaphoreWaitInfo wait_info{
            .semaphoreCount = static_cast<uint32_t>(sizeof...(Args)),
            .pSemaphores = wait_semaphores.data(),
            .pValues = wait_semaphore_values.data(),
        };

        if (ctx.device->waitSemaphores(wait_info, numeric_limits<uint64_t>::max()) != vk::Result::eSuccess) {
            Logger::error("waitSemaphores on renderFinishedTimeline failed");
        }
    }
} // utils::sync

} // zrx
