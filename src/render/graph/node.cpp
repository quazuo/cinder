module;

module Cinder.Render.Graph;

import Cinder.Utils;
import Cinder.Render;
import Cinder.Render.Vulkan;

namespace zrx {
RenderNodeHandle::IDType RenderNodeHandle::next_free_handle_id = 0;

void RenderPassContext::bind_pipeline(const ResourceHandle pipeline_handle) {
    if (!resource_manager.get().contains<GraphicsPipeline>(pipeline_handle)) {
        Logger::error("Invalid pipeline handle in RenderPassContext::bind_pipeline!");
    }

    const auto& pipeline = resource_manager.get().get<GraphicsPipeline>(pipeline_handle);
    const auto& layout = pipeline.get_layout();
    const auto& bind_point = vk::PipelineBindPoint::eGraphics;

    command_buffer.get().bindPipeline(bind_point, **pipeline);
    command_buffer.get().bindDescriptorSets(bind_point, layout, 0, *bindless_set.get(), nullptr);

    last_bound_pipeline = pipeline_handle;
}

void RenderPassContext::bind_resources(const std::vector<ResourceHandle> &handles) {
    bound_resource_handles = handles;
    bound_resource_ids = {};

    for (auto& handle: handles) {
        if (handle == FINAL_IMAGE_HANDLE) {
            Logger::error("Cannot bind final image inside a render pass context");

        } else {
            bound_resource_ids.push_back(resource_manager.get().get_bindless_handle(handle));
        }
    }
}

void RenderPassContext::bind_vertex_buffers(const std::vector<ResourceHandle> &vb_handles) {
    std::vector<vk::Buffer> vertex_buffers;
    for (const auto& handle: vb_handles) {
        vertex_buffers.push_back(**resource_manager.get().get<Buffer>(handle));
    }

    const std::vector<vk::DeviceSize> offsets(vertex_buffers.size(), 0);

    command_buffer.get().bindVertexBuffers(0, vertex_buffers, offsets);
}

void RenderPassContext::bind_index_buffer(ResourceHandle indices_handle) {
    const Buffer &index_buffer = resource_manager.get().get<Buffer>(indices_handle);
    command_buffer.get().bindIndexBuffer(**index_buffer, 0, vk::IndexType::eUint32);
}

void RenderPassContext::draw(const ResourceHandle vertices_handle, const uint32_t vertex_count, const uint32_t instance_count,
                             const uint32_t first_vertex, const uint32_t first_instance) {
    if (!last_bound_pipeline) {
        Logger::error("no pipeline bound during draw!");
    }

    const Buffer &vertex_buffer = resource_manager.get().get<Buffer>(vertices_handle);
    command_buffer.get().bindVertexBuffers(0, **vertex_buffer, {0});
    push_bindless_constants();
    command_buffer.get().draw(vertex_count, instance_count, first_vertex, first_instance);
}

void RenderPassContext::draw(const uint32_t vertex_count, const uint32_t instance_count, const uint32_t first_vertex,
                             const uint32_t first_instance) {
    push_bindless_constants();
    command_buffer.get().draw(vertex_count, instance_count, first_vertex, first_instance);
}

void RenderPassContext::draw_indexed(const ResourceHandle vertices_handle, const ResourceHandle indices_handle,
                                     const uint32_t index_count, const uint32_t instance_count, const uint32_t first_index,
                                     const uint32_t vertex_offset, const uint32_t first_instance) {
    if (!last_bound_pipeline) {
        Logger::error("no pipeline bound during draw_indexed!");
    }

    const Buffer &vertex_buffer = resource_manager.get().get<Buffer>(vertices_handle);
    const Buffer &index_buffer = resource_manager.get().get<Buffer>(indices_handle);
    command_buffer.get().bindVertexBuffers(0, **vertex_buffer, {0});
    command_buffer.get().bindIndexBuffer(**index_buffer, 0, vk::IndexType::eUint32);
    push_bindless_constants();
    command_buffer.get().drawIndexed(index_count, instance_count, first_index, vertex_offset, first_instance);
}

void RenderPassContext::draw_indexed(const uint32_t index_count, const uint32_t instance_count, const uint32_t first_index,
                                     const uint32_t vertex_offset, const uint32_t first_instance) {
    push_bindless_constants();
    command_buffer.get().drawIndexed(index_count, instance_count, first_index, vertex_offset, first_instance);
}

void RenderPassContext::push_bindless_constants() {
    if (!last_bound_pipeline) {
        Logger::error("no pipeline bound during push_constants!");
    }

    if (!bound_resource_ids.empty()) {
        command_buffer.get().pushConstants<BindlessHandle>(
            *resource_manager.get().get<GraphicsPipeline>(*last_bound_pipeline).get_layout(),
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            0,
            bound_resource_ids
        );
    }
}

void ComputePassContext::bind_pipeline(ResourceHandle pipeline_handle) {
    if (!resource_manager.get().contains<ComputePipeline>(pipeline_handle)) {
        Logger::error("Invalid pipeline handle in ComputePassContext::bind_pipeline!");
    }

    const auto& pipeline = resource_manager.get().get<ComputePipeline>(pipeline_handle);
    const auto& layout = pipeline.get_layout();
    const auto& bind_point = vk::PipelineBindPoint::eCompute;

    command_buffer.get().bindPipeline(bind_point, **pipeline);
    command_buffer.get().bindDescriptorSets(bind_point, layout, 0, *bindless_set.get(), nullptr);

    last_bound_pipeline = pipeline_handle;
}

void ComputePassContext::bind_resources(const std::vector<ResourceHandle> &handles) {
    bound_resource_handles = handles;
    bound_resource_ids = {};

    for (auto& handle: handles) {
        if (handle == FINAL_IMAGE_HANDLE) {
            Logger::error("Cannot bind final image inside a compute render pass context");
        }

        bound_resource_ids.push_back(resource_manager.get().get_bindless_handle(handle));
    }
}

void ComputePassContext::dispatch(const uint32_t x, const uint32_t y, const uint32_t z) const {
    push_bindless_constants();
    command_buffer.get().dispatch(x, y, z);
}

void ComputePassContext::push_bindless_constants() const {
    if (!last_bound_pipeline) {
        Logger::error("no pipeline bound during ComputePassContext::push_constants!");
    }

    if (!bound_resource_ids.empty()) {
        command_buffer.get().pushConstants<BindlessHandle>(
            *resource_manager.get().get<ComputePipeline>(*last_bound_pipeline).get_layout(),
            vk::ShaderStageFlagBits::eCompute,
            0,
            bound_resource_ids
        );
    }
}

set<ResourceHandle> RenderNodeGraphics::get_all_targets_set() const {
    set result(color_targets.begin(), color_targets.end());
    if (depth_target) result.insert(*depth_target);
    return result;
}

const string& RenderNode::name() const {
    return visit([](const auto& n) -> const auto& { return n.name; });
}

const std::vector<RenderNodeHandle>& RenderNode::explicit_dependencies() const {
    return visit([](const auto& n) -> const auto& { return n.explicit_dependencies; });
}
} // zrx
