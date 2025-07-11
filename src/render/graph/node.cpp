module;

module Cinder.Render.Graph;

import Cinder.Utils;
import Cinder.Render;
import Cinder.Render.Vulkan;

namespace zrx {
void RenderPassContext::bind_pipeline(const ResourceHandle pipeline_handle) {
    vk::Pipeline pipeline {};
    vk::PipelineLayout layout {};
    vk::PipelineBindPoint bind_point {};

    if (graphics_pipelines.get().contains(pipeline_handle)) {
        pipeline = **graphics_pipelines.get().at(pipeline_handle);
        layout = *graphics_pipelines.get().at(pipeline_handle).get_layout();
        bind_point = vk::PipelineBindPoint::eGraphics;

    } else if (compute_pipelines.get().contains(pipeline_handle)) {
        pipeline = **compute_pipelines.get().at(pipeline_handle);
        layout = *compute_pipelines.get().at(pipeline_handle).get_layout();
        bind_point = vk::PipelineBindPoint::eCompute;

    } else {
        Logger::error("Invalid pipeline handle in RenderPassContext::bind_pipeline!");
    }

    command_buffer.get().bindPipeline(bind_point, pipeline);
    command_buffer.get().bindDescriptorSets(bind_point, layout, 0, *bindless_set.get(), nullptr);

    last_bound_pipeline = pipeline_handle;
}

void RenderPassContext::bind_resources(const std::vector<ResourceHandle> &handles) {
    bound_resource_ids = handles;

    for (auto& res_id: bound_resource_ids) {
        res_id = resource_manager.get().get_bindless_handle(res_id);
    }
}

void RenderPassContext::draw_model(const ResourceHandle model_handle) const {
    if (!last_bound_pipeline) {
        Logger::error("no pipeline bound during draw!");
    }

    uint32_t index_offset    = 0;
    int32_t vertex_offset    = 0;
    uint32_t instance_offset = 0;

    const Model &model = resource_manager.get().get_model(model_handle);
    model.bind_buffers(command_buffer);

    for (const auto &mesh: model.get_meshes()) {
        push_constants();

        command_buffer.get().drawIndexed(
            static_cast<uint32_t>(mesh.indices.size()),
            static_cast<uint32_t>(mesh.instances.size()),
            index_offset,
            vertex_offset,
            instance_offset
        );

        index_offset += static_cast<uint32_t>(mesh.indices.size());
        vertex_offset += static_cast<int32_t>(mesh.vertices.size());
        instance_offset += static_cast<uint32_t>(mesh.instances.size());
    }
}

void RenderPassContext::draw(const ResourceHandle vertices_handle,
                             const uint32_t vertex_count, const uint32_t instance_count,
                             const uint32_t first_vertex, const uint32_t first_instance) const {
    const Buffer &vertex_buffer = resource_manager.get().get_buffer(vertices_handle);
    command_buffer.get().bindVertexBuffers(0, *vertex_buffer, {0});
    push_constants();
    command_buffer.get().draw(vertex_count, instance_count, first_vertex, first_instance);
}

void RenderPassContext::dispatch(const uint32_t x, const uint32_t y, const uint32_t z) const {
    command_buffer.get().dispatch(x, y, z);
}

void RenderPassContext::push_constants() const {
    if (!last_bound_pipeline) {
        Logger::error("no pipeline bound during draw!");
    }

    if (!bound_resource_ids.empty()) {
        command_buffer.get().pushConstants<ResourceHandle>(
            *graphics_pipelines.get().at(*last_bound_pipeline).get_layout(),
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            0,
            bound_resource_ids
        );
    }
}

set<ResourceHandle> RenderNode::get_all_targets_set() const {
    set result(color_targets.begin(), color_targets.end());
    if (depth_target) result.insert(*depth_target);
    return result;
}
} // zrx
