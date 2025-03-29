#include "pipeline.hpp"

#include <fstream>

#include "src/render/mesh/vertex.hpp"
#include "ctx.hpp"
#include "buffer.hpp"

namespace zrx {
static vk::raii::ShaderModule create_shader_module(const RendererContext &ctx, const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        Logger::error("failed to open file!");
    }

    const size_t file_size = file.tellg();
    vector<char> buffer(file_size);
    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(file_size));

    const vk::ShaderModuleCreateInfo create_info{
        .codeSize = buffer.size(),
        .pCode = reinterpret_cast<const uint32_t *>(buffer.data()),
    };

    return vk::raii::ShaderModule{*ctx.device, create_info};
}

GraphicsPipelineBuilder &GraphicsPipelineBuilder::with_vertex_shader(const std::filesystem::path &path) {
    vertex_shader_path = path;
    return *this;
}

GraphicsPipelineBuilder &GraphicsPipelineBuilder::with_fragment_shader(const std::filesystem::path &path) {
    fragment_shader_path = path;
    return *this;
}

GraphicsPipelineBuilder &
GraphicsPipelineBuilder::with_vertices(vector<vk::VertexInputBindingDescription> bindings,
                                       vector<vk::VertexInputAttributeDescription> attributes) {
    vertex_bindings   = std::move(bindings);
    vertex_attributes = std::move(attributes);
    return *this;
}

GraphicsPipelineBuilder &GraphicsPipelineBuilder::with_descriptor_layouts(
    const vector<vk::DescriptorSetLayout> &layouts) {
    descriptor_set_layouts = layouts;
    return *this;
}

GraphicsPipelineBuilder &
GraphicsPipelineBuilder::with_push_constants(const vector<vk::PushConstantRange> &ranges) {
    push_constant_ranges = ranges;
    return *this;
}

GraphicsPipelineBuilder &GraphicsPipelineBuilder::with_rasterizer(
    const vk::PipelineRasterizationStateCreateInfo &rasterizer) {
    rasterizer_override = rasterizer;
    return *this;
}

GraphicsPipelineBuilder &GraphicsPipelineBuilder::with_multisampling(
    const vk::PipelineMultisampleStateCreateInfo &multisampling) {
    multisampling_override = multisampling;
    return *this;
}

GraphicsPipelineBuilder &GraphicsPipelineBuilder::with_depth_stencil(
    const vk::PipelineDepthStencilStateCreateInfo &depthStencil) {
    depth_stencil_override = depthStencil;
    return *this;
}

GraphicsPipelineBuilder &GraphicsPipelineBuilder::for_views(const uint32_t count) {
    multiview_count = count;
    return *this;
}

GraphicsPipelineBuilder &GraphicsPipelineBuilder::with_color_formats(const vector<vk::Format> &formats) {
    color_attachment_formats = formats;
    return *this;
}

GraphicsPipelineBuilder &GraphicsPipelineBuilder::with_depth_format(const vk::Format format) {
    depth_attachment_format = format;
    return *this;
}

GraphicsPipeline GraphicsPipelineBuilder::create(const RendererContext &ctx) const {
    GraphicsPipeline result;

    vk::raii::ShaderModule vert_shader_module = create_shader_module(ctx, vertex_shader_path);
    vk::raii::ShaderModule frag_shader_module = create_shader_module(ctx, fragment_shader_path);

    const vk::PipelineShaderStageCreateInfo vert_shader_stage_info{
        .stage = vk::ShaderStageFlagBits::eVertex,
        .module = *vert_shader_module,
        .pName = "main",
    };

    const vk::PipelineShaderStageCreateInfo frag_shader_stage_info{
        .stage = vk::ShaderStageFlagBits::eFragment,
        .module = *frag_shader_module,
        .pName = "main",
    };

    const vector shader_stages{
        vert_shader_stage_info,
        frag_shader_stage_info
    };

    const vk::PipelineVertexInputStateCreateInfo vertex_input_info{
        .vertexBindingDescriptionCount = static_cast<uint32_t>(vertex_bindings.size()),
        .pVertexBindingDescriptions = vertex_bindings.data(),
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(vertex_attributes.size()),
        .pVertexAttributeDescriptions = vertex_attributes.data()
    };

    constexpr vk::PipelineInputAssemblyStateCreateInfo input_assembly{
        .topology = vk::PrimitiveTopology::eTriangleList,
    };

    static constexpr std::array dynamic_states = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
    };

    static constexpr vk::PipelineDynamicStateCreateInfo dynamic_state{
        .dynamicStateCount = static_cast<uint32_t>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };

    constexpr vk::PipelineViewportStateCreateInfo viewport_state{
        .viewportCount = 1U,
        .scissorCount = 1U,
    };

    const auto rasterizer = rasterizer_override
                                ? *rasterizer_override
                                : vk::PipelineRasterizationStateCreateInfo{
                                    .polygonMode = vk::PolygonMode::eFill,
                                    .cullMode = vk::CullModeFlagBits::eBack,
                                    .frontFace = vk::FrontFace::eCounterClockwise,
                                    .lineWidth = 1.0f,
                                };

    const auto multisampling = multisampling_override
                                   ? *multisampling_override
                                   : vk::PipelineMultisampleStateCreateInfo{
                                       .rasterizationSamples = vk::SampleCountFlagBits::e1,
                                       .minSampleShading = 1.0f,
                                   };

    result.rasterization_samples = multisampling.rasterizationSamples;

    const vector<vk::PipelineColorBlendAttachmentState> color_blend_attachments(
        color_attachment_formats.size(),
        {
            .blendEnable = vk::False,
            .colorWriteMask = vk::ColorComponentFlagBits::eR
                              | vk::ColorComponentFlagBits::eG
                              | vk::ColorComponentFlagBits::eB
                              | vk::ColorComponentFlagBits::eA,
        }
    );

    const vk::PipelineColorBlendStateCreateInfo color_blending{
        .logicOpEnable = vk::False,
        .attachmentCount = static_cast<uint32_t>(color_blend_attachments.size()),
        .pAttachments = color_blend_attachments.data(),
    };

    const auto depth_stencil = depth_stencil_override
                                   ? *depth_stencil_override
                                   : vk::PipelineDepthStencilStateCreateInfo{
                                       .depthTestEnable = vk::True,
                                       .depthWriteEnable = vk::True,
                                       .depthCompareOp = vk::CompareOp::eLess,
                                   };

    const vk::PipelineLayoutCreateInfo pipeline_layout_info{
        .setLayoutCount = static_cast<uint32_t>(descriptor_set_layouts.size()),
        .pSetLayouts = descriptor_set_layouts.empty() ? nullptr : descriptor_set_layouts.data(),
        .pushConstantRangeCount = static_cast<uint32_t>(push_constant_ranges.size()),
        .pPushConstantRanges = push_constant_ranges.empty() ? nullptr : push_constant_ranges.data()
    };

    result.layout = make_unique<vk::raii::PipelineLayout>(*ctx.device, pipeline_layout_info);

    const vk::StructureChain<
        vk::GraphicsPipelineCreateInfo,
        vk::PipelineRenderingCreateInfo
    > pipeline_create_info{
        {
            .stageCount = static_cast<uint32_t>(shader_stages.size()),
            .pStages = shader_stages.data(),
            .pVertexInputState = &vertex_input_info,
            .pInputAssemblyState = &input_assembly,
            .pViewportState = &viewport_state,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depth_stencil,
            .pColorBlendState = &color_blending,
            .pDynamicState = &dynamic_state,
            .layout = **result.layout,
        },
        {
            .viewMask = multiview_count == 1 ? 0 : ((1u << multiview_count) - 1),
            .colorAttachmentCount = static_cast<uint32_t>(color_attachment_formats.size()),
            .pColorAttachmentFormats = color_attachment_formats.empty() ? nullptr : color_attachment_formats.data(),
            .depthAttachmentFormat = depth_attachment_format ? *depth_attachment_format : static_cast<vk::Format>(0)
        }
    };

    result.pipeline = make_unique<vk::raii::Pipeline>(
        *ctx.device,
        nullptr,
        pipeline_create_info.get<vk::GraphicsPipelineCreateInfo>()
    );

    return result;
}

void GraphicsPipelineBuilder::check_params() const {
    if (vertex_shader_path.empty()) {
        Logger::error("vertex shader must be specified during pipeline creation!");
    }

    if (fragment_shader_path.empty()) {
        Logger::error("fragment shader must be specified during pipeline creation!");
    }

    if (vertex_bindings.empty() && vertex_attributes.empty()) {
        Logger::error("vertex descriptions must be specified during pipeline creation!");
    }
}

RtPipelineBuilder &RtPipelineBuilder::with_ray_gen_shader(const std::filesystem::path &path) {
    raygen_shader_path = path;
    return *this;
}

RtPipelineBuilder &RtPipelineBuilder::with_closest_hit_shader(const std::filesystem::path &path) {
    closest_hit_shader_path = path;
    return *this;
}

RtPipelineBuilder &RtPipelineBuilder::with_miss_shader(const std::filesystem::path &path) {
    miss_shader_path = path;
    return *this;
}

RtPipelineBuilder &RtPipelineBuilder::with_descriptor_layouts(const vector<vk::DescriptorSetLayout> &layouts) {
    descriptor_set_layouts = layouts;
    return *this;
}

RtPipelineBuilder &RtPipelineBuilder::with_push_constants(const vector<vk::PushConstantRange> &ranges) {
    push_constant_ranges = ranges;
    return *this;
}

RtPipeline RtPipelineBuilder::create(const RendererContext &ctx) const {
    RtPipeline result;

    auto [pipeline, layout] = build_pipeline(ctx);
    result.pipeline         = make_unique<decltype(pipeline)>(std::move(pipeline));
    result.layout           = make_unique<decltype(layout)>(std::move(layout));

    result.sbt = build_sbt(ctx, *result.pipeline);

    return result;
}

void RtPipelineBuilder::check_params() const {
    if (raygen_shader_path.empty()) {
        Logger::error("ray generation shader must be specified during ray tracing pipeline creation!");
    }

    if (closest_hit_shader_path.empty()) {
        Logger::error("closest hit shader must be specified during ray tracing pipeline creation!");
    }

    if (miss_shader_path.empty()) {
        Logger::error("miss shader must be specified during ray tracing pipeline creation!");
    }
}

std::pair<vk::raii::Pipeline, vk::raii::PipelineLayout>
RtPipelineBuilder::build_pipeline(const RendererContext &ctx) const {
    enum StageIndices {
        eRaygen = 0,
        eMiss,
        eClosestHit,
        eShaderGroupCount
    };

    const vk::raii::ShaderModule raygen_shader_module      = create_shader_module(ctx, raygen_shader_path);
    const vk::raii::ShaderModule miss_shader_module        = create_shader_module(ctx, miss_shader_path);
    const vk::raii::ShaderModule closest_hit_shader_module = create_shader_module(ctx, closest_hit_shader_path);

    std::array<vk::PipelineShaderStageCreateInfo, eShaderGroupCount> shader_stages;

    shader_stages[eRaygen] = {
        .stage = vk::ShaderStageFlagBits::eRaygenKHR,
        .module = *raygen_shader_module,
        .pName = "main",
    };

    shader_stages[eMiss] = {
        .stage = vk::ShaderStageFlagBits::eMissKHR,
        .module = *miss_shader_module,
        .pName = "main",
    };

    shader_stages[eClosestHit] = {
        .stage = vk::ShaderStageFlagBits::eClosestHitKHR,
        .module = *closest_hit_shader_module,
        .pName = "main",
    };

    constexpr vk::RayTracingShaderGroupCreateInfoKHR shader_group_template{
        .generalShader = vk::ShaderUnusedKHR,
        .closestHitShader = vk::ShaderUnusedKHR,
        .anyHitShader = vk::ShaderUnusedKHR,
        .intersectionShader = vk::ShaderUnusedKHR,
    };

    vector shader_groups(eShaderGroupCount, shader_group_template);

    shader_groups[eRaygen].type          = vk::RayTracingShaderGroupTypeKHR::eGeneral;
    shader_groups[eRaygen].generalShader = eRaygen;

    shader_groups[eMiss].type          = vk::RayTracingShaderGroupTypeKHR::eGeneral;
    shader_groups[eMiss].generalShader = eMiss;

    shader_groups[eClosestHit].type             = vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup;
    shader_groups[eClosestHit].closestHitShader = eClosestHit;

    const vk::PipelineLayoutCreateInfo pipeline_layout_info{
        .setLayoutCount = static_cast<uint32_t>(descriptor_set_layouts.size()),
        .pSetLayouts = descriptor_set_layouts.empty() ? nullptr : descriptor_set_layouts.data(),
        .pushConstantRangeCount = static_cast<uint32_t>(push_constant_ranges.size()),
        .pPushConstantRanges = push_constant_ranges.empty() ? nullptr : push_constant_ranges.data()
    };

    vk::raii::PipelineLayout layout{*ctx.device, pipeline_layout_info};

    const vk::RayTracingPipelineCreateInfoKHR pipeline_create_info{
        .stageCount = static_cast<uint32_t>(shader_stages.size()),
        .pStages = shader_stages.data(),
        .groupCount = static_cast<uint32_t>(shader_groups.size()),
        .pGroups = shader_groups.data(),
        .maxPipelineRayRecursionDepth = 2u,
        .layout = *layout,
    };

    vk::raii::Pipeline pipeline{
        *ctx.device,
        nullptr,
        nullptr,
        pipeline_create_info
    };

    return std::make_pair(std::move(pipeline), std::move(layout));
}

static constexpr uint32_t align_up(const uint32_t size, const uint32_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

RtPipeline::ShaderBindingTable
RtPipelineBuilder::build_sbt(const RendererContext &ctx, const vk::raii::Pipeline &pipeline) const {
    const auto properties = ctx.physical_device->getProperties2<
        vk::PhysicalDeviceProperties2,
        vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
    const auto rt_properties = properties.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();

    constexpr uint32_t miss_count      = 1;
    constexpr uint32_t hit_count       = 1;
    constexpr uint32_t handle_count    = 1 + miss_count + hit_count; // 1 for raygen count (always 1)
    const uint32_t handle_size         = rt_properties.shaderGroupHandleSize;
    const uint32_t handle_size_aligned = align_up(
        handle_size,
        rt_properties.shaderGroupHandleAlignment);

    const auto rgen_stride = align_up(handle_size_aligned, rt_properties.shaderGroupBaseAlignment);
    vk::StridedDeviceAddressRegionKHR rgen_region = {
        .stride = rgen_stride,
        .size = rgen_stride
    };

    vk::StridedDeviceAddressRegionKHR miss_region{
        .stride = handle_size_aligned,
        .size = align_up(miss_count * handle_size_aligned, rt_properties.shaderGroupBaseAlignment)
    };

    vk::StridedDeviceAddressRegionKHR hit_region{
        .stride = handle_size_aligned,
        .size = align_up(hit_count * handle_size_aligned, rt_properties.shaderGroupBaseAlignment)
    };

    const uint32_t data_size = handle_count * handle_size;
    vector handles      = pipeline.getRayTracingShaderGroupHandlesKHR<uint8_t>(0, handle_count, data_size);

    const VkDeviceSize sbt_size = rgen_region.size + miss_region.size + hit_region.size;
    auto sbt_buffer             = make_unique<Buffer>(
        **ctx.allocator,
        sbt_size,
        vk::BufferUsageFlagBits::eShaderBindingTableKHR
        | vk::BufferUsageFlagBits::eShaderDeviceAddress
        | vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible
        | vk::MemoryPropertyFlagBits::eHostCoherent
    );

    const vk::DeviceAddress sbt_address = ctx.device->getBufferAddress({.buffer = **sbt_buffer});
    rgen_region.deviceAddress           = sbt_address;
    miss_region.deviceAddress           = rgen_region.deviceAddress + rgen_region.size;
    hit_region.deviceAddress            = miss_region.deviceAddress + miss_region.size;

    auto get_handle_ptr     = [&](const uint32_t i) { return handles.data() + i * handle_size; };
    auto *sbt_buffer_mapped = static_cast<uint8_t *>(sbt_buffer->map());

    uint32_t handle_index = 0;

    uint8_t *rgen_data = sbt_buffer_mapped;
    memcpy(rgen_data, get_handle_ptr(handle_index++), handle_size);

    uint8_t *miss_data = sbt_buffer_mapped + rgen_region.size;
    for (uint32_t i = 0; i < miss_count; i++) {
        memcpy(miss_data, get_handle_ptr(handle_index++), handle_size);
        miss_data += miss_region.stride;
    }

    uint8_t *hit_data = sbt_buffer_mapped + rgen_region.size + miss_region.size;
    for (uint32_t i = 0; i < hit_count; i++) {
        memcpy(hit_data, get_handle_ptr(handle_index++), handle_size);
        hit_data += hit_region.stride;
    }

    sbt_buffer->unmap();

    return {
        .backing_buffer = std::move(sbt_buffer),
        .rgen_region = rgen_region,
        .miss_region = miss_region,
        .hit_region = hit_region
    };
}
} // zrx
