module;

module Cinder.Render.Vulkan;

namespace zrx {
BinarySemaphore::BinarySemaphore(const RendererContext& ctx)
    : semaphore(*ctx.device, vk::SemaphoreCreateInfo {}) {
}

vk::raii::Semaphore create_default_timeline_semaphore(const RendererContext &ctx) {
    const vk::StructureChain<vk::SemaphoreCreateInfo, vk::SemaphoreTypeCreateInfo> timeline_semaphore_info{
        {},
        {
            .semaphoreType = vk::SemaphoreType::eTimeline,
            .initialValue = 0,
        }
    };

    return { *ctx.device, timeline_semaphore_info.get<vk::SemaphoreCreateInfo>() };
}

TimelineSemaphore::TimelineSemaphore(const RendererContext &ctx)
    : semaphore(create_default_timeline_semaphore(ctx)) {
}

}