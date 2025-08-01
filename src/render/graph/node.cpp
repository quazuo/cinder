module;

module Cinder.Render.Graph;

import Cinder.Utils;
import Cinder.Render;
import Cinder.Render.Vulkan;

namespace zrx {
void RenderPassContext::bind_pipeline(const ResourceHandle pipeline_handle) {
    if (!graphics_pipelines.get().contains(pipeline_handle)) {
        Logger::error("Invalid pipeline handle in RenderPassContext::bind_pipeline!");
    }

    const auto& pipeline = **graphics_pipelines.get().at(pipeline_handle);
    const auto& layout = *graphics_pipelines.get().at(pipeline_handle).get_layout();
    const auto& bind_point = vk::PipelineBindPoint::eGraphics;

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

void ComputePassContext::bind_pipeline(ResourceHandle pipeline_handle) {
    if (!compute_pipelines.get().contains(pipeline_handle)) {
        Logger::error("Invalid pipeline handle in ComputePassContext::bind_pipeline!");
    }

    const auto& pipeline = **compute_pipelines.get().at(pipeline_handle);
    const auto& layout = *compute_pipelines.get().at(pipeline_handle).get_layout();
    const auto& bind_point = vk::PipelineBindPoint::eCompute;

    command_buffer.get().bindPipeline(bind_point, pipeline);
    command_buffer.get().bindDescriptorSets(bind_point, layout, 0, *bindless_set.get(), nullptr);

    last_bound_pipeline = pipeline_handle;
}

void ComputePassContext::bind_resources(const std::vector<ResourceHandle> &handles) {
    bound_resource_ids = handles;

    for (auto& res_id: bound_resource_ids) {
        res_id = resource_manager.get().get_bindless_handle(res_id);
    }
}

void ComputePassContext::dispatch(const uint32_t x, const uint32_t y, const uint32_t z) const {
    command_buffer.get().dispatch(x, y, z);
}

void ComputePassContext::push_constants() const {
    if (!last_bound_pipeline) {
        Logger::error("no pipeline bound during ComputePassContext::push_constants!");
    }

    if (!bound_resource_ids.empty()) {
        command_buffer.get().pushConstants<ResourceHandle>(
            *compute_pipelines.get().at(*last_bound_pipeline).get_layout(),
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
    if (std::holds_alternative<RenderNodeGraphics>(node_)) {
        return std::get<RenderNodeGraphics>(node_).name;
    }
    if (std::holds_alternative<RenderNodeCompute>(node_)) {
        return std::get<RenderNodeCompute>(node_).name;
    }
    throw std::runtime_error("illegal type in RenderNode");
}

bool RenderNode::should_run() const {
    if (std::holds_alternative<RenderNodeGraphics>(node_)) {
        const auto& pred = std::get<RenderNodeGraphics>(node_).should_run_predicate;
        return !pred || (*pred)();
    }
    if (std::holds_alternative<RenderNodeCompute>(node_)) {
        const auto& pred = std::get<RenderNodeCompute>(node_).should_run_predicate;
        return !pred || (*pred)();
    }
    throw std::runtime_error("illegal type in RenderNode");
}

const std::vector<RenderNodeHandle>& RenderNode::explicit_dependencies() const {
    if (std::holds_alternative<RenderNodeGraphics>(node_)) {
        return std::get<RenderNodeGraphics>(node_).explicit_dependencies;
    }
    if (std::holds_alternative<RenderNodeCompute>(node_)) {
        return std::get<RenderNodeCompute>(node_).explicit_dependencies;
    }
    throw std::runtime_error("illegal type in RenderNode");
}
} // zrx
