#include "pipeline.hpp"

#include <fstream>

#include "src/render/renderer.hpp"
#include "src/render/mesh/vertex.hpp"
#include "buffer.hpp"

static vk::raii::ShaderModule createShaderModule(const RendererContext &ctx, const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    const size_t fileSize = file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(fileSize));

    const vk::ShaderModuleCreateInfo createInfo{
        .codeSize = buffer.size(),
        .pCode = reinterpret_cast<const uint32_t *>(buffer.data()),
    };

    return vk::raii::ShaderModule{*ctx.device, createInfo};
}

GraphicsPipelineBuilder &GraphicsPipelineBuilder::withVertexShader(const std::filesystem::path &path) {
    vertexShaderPath = path;
    return *this;
}

GraphicsPipelineBuilder &GraphicsPipelineBuilder::withFragmentShader(const std::filesystem::path &path) {
    fragmentShaderPath = path;
    return *this;
}

template<typename T>
GraphicsPipelineBuilder &GraphicsPipelineBuilder::withVertices() {
    vertexBindings = T::getBindingDescriptions();
    vertexAttributes = T::getAttributeDescriptions();
    return *this;
}

GraphicsPipelineBuilder &GraphicsPipelineBuilder::withDescriptorLayouts(
    const std::vector<vk::DescriptorSetLayout> &layouts) {
    descriptorSetLayouts = layouts;
    return *this;
}

GraphicsPipelineBuilder &GraphicsPipelineBuilder::withPushConstants(const std::vector<vk::PushConstantRange> &ranges) {
    pushConstantRanges = ranges;
    return *this;
}

GraphicsPipelineBuilder &GraphicsPipelineBuilder::withRasterizer(
    const vk::PipelineRasterizationStateCreateInfo &rasterizer) {
    rasterizerOverride = rasterizer;
    return *this;
}

GraphicsPipelineBuilder &GraphicsPipelineBuilder::withMultisampling(
    const vk::PipelineMultisampleStateCreateInfo &multisampling) {
    multisamplingOverride = multisampling;
    return *this;
}

GraphicsPipelineBuilder &GraphicsPipelineBuilder::withDepthStencil(
    const vk::PipelineDepthStencilStateCreateInfo &depthStencil) {
    depthStencilOverride = depthStencil;
    return *this;
}

GraphicsPipelineBuilder &GraphicsPipelineBuilder::forViews(const uint32_t count) {
    multiviewCount = count;
    return *this;
}

GraphicsPipelineBuilder &GraphicsPipelineBuilder::withColorFormats(const std::vector<vk::Format> &formats) {
    colorAttachmentFormats = formats;
    return *this;
}

GraphicsPipelineBuilder &GraphicsPipelineBuilder::withDepthFormat(const vk::Format format) {
    depthAttachmentFormat = format;
    return *this;
}

GraphicsPipeline GraphicsPipelineBuilder::create(const RendererContext &ctx) const {
    GraphicsPipeline result;

    vk::raii::ShaderModule vertShaderModule = createShaderModule(ctx, vertexShaderPath);
    vk::raii::ShaderModule fragShaderModule = createShaderModule(ctx, fragmentShaderPath);

    const vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eVertex,
        .module = *vertShaderModule,
        .pName = "main",
    };

    const vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eFragment,
        .module = *fragShaderModule,
        .pName = "main",
    };

    const std::vector shaderStages{
        vertShaderStageInfo,
        fragShaderStageInfo
    };

    const vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
        .vertexBindingDescriptionCount = static_cast<uint32_t>(vertexBindings.size()),
        .pVertexBindingDescriptions = vertexBindings.data(),
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttributes.size()),
        .pVertexAttributeDescriptions = vertexAttributes.data()
    };

    constexpr vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList,
    };

    static constexpr std::array dynamicStates = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
    };

    static constexpr vk::PipelineDynamicStateCreateInfo dynamicState{
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data(),
    };

    constexpr vk::PipelineViewportStateCreateInfo viewportState{
        .viewportCount = 1U,
        .scissorCount = 1U,
    };

    const auto rasterizer = rasterizerOverride
                                ? *rasterizerOverride
                                : vk::PipelineRasterizationStateCreateInfo{
                                    .polygonMode = vk::PolygonMode::eFill,
                                    .cullMode = vk::CullModeFlagBits::eBack,
                                    .frontFace = vk::FrontFace::eCounterClockwise,
                                    .lineWidth = 1.0f,
                                };

    const auto multisampling = multisamplingOverride
                                   ? *multisamplingOverride
                                   : vk::PipelineMultisampleStateCreateInfo{
                                       .rasterizationSamples = vk::SampleCountFlagBits::e1,
                                       .minSampleShading = 1.0f,
                                   };

    result.rasterizationSamples = multisampling.rasterizationSamples;

    const std::vector<vk::PipelineColorBlendAttachmentState> colorBlendAttachments(
        colorAttachmentFormats.size(),
        {
            .blendEnable = vk::False,
            .colorWriteMask = vk::ColorComponentFlagBits::eR
                              | vk::ColorComponentFlagBits::eG
                              | vk::ColorComponentFlagBits::eB
                              | vk::ColorComponentFlagBits::eA,
        }
    );

    const vk::PipelineColorBlendStateCreateInfo colorBlending{
        .logicOpEnable = vk::False,
        .attachmentCount = static_cast<uint32_t>(colorBlendAttachments.size()),
        .pAttachments = colorBlendAttachments.data(),
    };

    const auto depthStencil = depthStencilOverride
                                  ? *depthStencilOverride
                                  : vk::PipelineDepthStencilStateCreateInfo{
                                      .depthTestEnable = vk::True,
                                      .depthWriteEnable = vk::True,
                                      .depthCompareOp = vk::CompareOp::eLess,
                                  };

    const vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size()),
        .pSetLayouts = descriptorSetLayouts.empty() ? nullptr : descriptorSetLayouts.data(),
        .pushConstantRangeCount = static_cast<uint32_t>(pushConstantRanges.size()),
        .pPushConstantRanges = pushConstantRanges.empty() ? nullptr : pushConstantRanges.data()
    };

    result.layout = make_unique<vk::raii::PipelineLayout>(*ctx.device, pipelineLayoutInfo);

    const vk::StructureChain<
        vk::GraphicsPipelineCreateInfo,
        vk::PipelineRenderingCreateInfo
    > pipelineCreateInfo{
        {
            .stageCount = static_cast<uint32_t>(shaderStages.size()),
            .pStages = shaderStages.data(),
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depthStencil,
            .pColorBlendState = &colorBlending,
            .pDynamicState = &dynamicState,
            .layout = **result.layout,
        },
        {
            .viewMask = multiviewCount == 1 ? 0 : ((1u << multiviewCount) - 1),
            .colorAttachmentCount = static_cast<uint32_t>(colorAttachmentFormats.size()),
            .pColorAttachmentFormats = colorAttachmentFormats.empty() ? nullptr : colorAttachmentFormats.data(),
            .depthAttachmentFormat = depthAttachmentFormat ? *depthAttachmentFormat : static_cast<vk::Format>(0)
        }
    };

    result.pipeline = make_unique<vk::raii::Pipeline>(
        *ctx.device,
        nullptr,
        pipelineCreateInfo.get<vk::GraphicsPipelineCreateInfo>()
    );

    return result;
}

void GraphicsPipelineBuilder::checkParams() const {
    if (vertexShaderPath.empty()) {
        throw std::invalid_argument("vertex shader must be specified during pipeline creation!");
    }

    if (fragmentShaderPath.empty()) {
        throw std::invalid_argument("fragment shader must be specified during pipeline creation!");
    }

    if (vertexBindings.empty() && vertexAttributes.empty()) {
        throw std::invalid_argument("vertex descriptions must be specified during pipeline creation!");
    }
}

RtPipelineBuilder &RtPipelineBuilder::withRayGenShader(const std::filesystem::path &path) {
    raygenShaderPath = path;
    return *this;
}

RtPipelineBuilder &RtPipelineBuilder::withClosestHitShader(const std::filesystem::path &path) {
    closestHitShaderPath = path;
    return *this;
}

RtPipelineBuilder &RtPipelineBuilder::withMissShader(const std::filesystem::path &path) {
    missShaderPath = path;
    return *this;
}

RtPipelineBuilder &RtPipelineBuilder::withDescriptorLayouts(const std::vector<vk::DescriptorSetLayout> &layouts) {
    descriptorSetLayouts = layouts;
    return *this;
}

RtPipelineBuilder &RtPipelineBuilder::withPushConstants(const std::vector<vk::PushConstantRange> &ranges) {
    pushConstantRanges = ranges;
    return *this;
}

RtPipeline RtPipelineBuilder::create(const RendererContext &ctx) const {
    RtPipeline result;

    auto [pipeline, layout] = buildPipeline(ctx);
    result.pipeline = make_unique<decltype(pipeline)>(std::move(pipeline));
    result.layout = make_unique<decltype(layout)>(std::move(layout));

    result.sbt = buildSbt(ctx, *result.pipeline);

    return result;
}

void RtPipelineBuilder::checkParams() const {
    if (raygenShaderPath.empty()) {
        throw std::invalid_argument("ray generation shader must be specified during ray tracing pipeline creation!");
    }

    if (closestHitShaderPath.empty()) {
        throw std::invalid_argument("closest hit shader must be specified during ray tracing pipeline creation!");
    }

    if (missShaderPath.empty()) {
        throw std::invalid_argument("miss shader must be specified during ray tracing pipeline creation!");
    }
}

std::pair<vk::raii::Pipeline, vk::raii::PipelineLayout>
RtPipelineBuilder::buildPipeline(const RendererContext &ctx) const {
    enum StageIndices {
        eRaygen = 0,
        eMiss,
        eClosestHit,
        eShaderGroupCount
    };

    const vk::raii::ShaderModule raygenShaderModule = createShaderModule(ctx, raygenShaderPath);
    const vk::raii::ShaderModule missShaderModule = createShaderModule(ctx, missShaderPath);
    const vk::raii::ShaderModule closestHitShaderModule = createShaderModule(ctx, closestHitShaderPath);

    std::array<vk::PipelineShaderStageCreateInfo, eShaderGroupCount> shaderStages;

    shaderStages[eRaygen] = {
        .stage = vk::ShaderStageFlagBits::eRaygenKHR,
        .module = *raygenShaderModule,
        .pName = "main",
    };

    shaderStages[eMiss] = {
        .stage = vk::ShaderStageFlagBits::eMissKHR,
        .module = *missShaderModule,
        .pName = "main",
    };

    shaderStages[eClosestHit] = {
        .stage = vk::ShaderStageFlagBits::eClosestHitKHR,
        .module = *closestHitShaderModule,
        .pName = "main",
    };

    constexpr vk::RayTracingShaderGroupCreateInfoKHR shaderGroupTemplate{
        .generalShader = vk::ShaderUnusedKHR,
        .closestHitShader = vk::ShaderUnusedKHR,
        .anyHitShader = vk::ShaderUnusedKHR,
        .intersectionShader = vk::ShaderUnusedKHR,
    };

    std::vector shaderGroups(eShaderGroupCount, shaderGroupTemplate);

    shaderGroups[eRaygen].type = vk::RayTracingShaderGroupTypeKHR::eGeneral;
    shaderGroups[eRaygen].generalShader = eRaygen;

    shaderGroups[eMiss].type = vk::RayTracingShaderGroupTypeKHR::eGeneral;
    shaderGroups[eMiss].generalShader = eMiss;

    shaderGroups[eClosestHit].type = vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup;
    shaderGroups[eClosestHit].closestHitShader = eClosestHit;

    const vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size()),
        .pSetLayouts = descriptorSetLayouts.empty() ? nullptr : descriptorSetLayouts.data(),
        .pushConstantRangeCount = static_cast<uint32_t>(pushConstantRanges.size()),
        .pPushConstantRanges = pushConstantRanges.empty() ? nullptr : pushConstantRanges.data()
    };

    vk::raii::PipelineLayout layout{*ctx.device, pipelineLayoutInfo};

    const vk::RayTracingPipelineCreateInfoKHR pipelineCreateInfo{
        .stageCount = static_cast<uint32_t>(shaderStages.size()),
        .pStages = shaderStages.data(),
        .groupCount = static_cast<uint32_t>(shaderGroups.size()),
        .pGroups = shaderGroups.data(),
        .maxPipelineRayRecursionDepth = 1u,
        .layout = *layout,
    };

    vk::raii::Pipeline pipeline{
        *ctx.device,
        nullptr,
        nullptr,
        pipelineCreateInfo
    };

    return std::make_pair(std::move(pipeline), std::move(layout));
}

static constexpr uint32_t alignUp(const uint32_t size, const uint32_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

RtPipeline::ShaderBindingTable
RtPipelineBuilder::buildSbt(const RendererContext &ctx, const vk::raii::Pipeline &pipeline) const {
    const auto properties = ctx.physicalDevice->getProperties2<
        vk::PhysicalDeviceProperties2,
        vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
    const auto rtProperties = properties.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();

    constexpr uint32_t missCount = 1;
    constexpr uint32_t hitCount = 1;
    constexpr uint32_t handleCount = 1 + missCount + hitCount; // 1 for raygen count (always 1)
    const uint32_t handleSize = rtProperties.shaderGroupHandleSize;
    const uint32_t handleSizeAligned = alignUp(
        handleSize,
        rtProperties.shaderGroupHandleAlignment);

    const auto rgenStride = alignUp(handleSizeAligned, rtProperties.shaderGroupBaseAlignment);
    vk::StridedDeviceAddressRegionKHR rgenRegion = {
        .stride = rgenStride,
        .size = rgenStride
    };

    vk::StridedDeviceAddressRegionKHR missRegion{
        .stride = handleSizeAligned,
        .size = alignUp(missCount * handleSizeAligned, rtProperties.shaderGroupBaseAlignment)
    };

    vk::StridedDeviceAddressRegionKHR hitRegion{
        .stride = handleSizeAligned,
        .size = alignUp(hitCount * handleSizeAligned, rtProperties.shaderGroupBaseAlignment)
    };

    const uint32_t dataSize = handleCount * handleSize;
    std::vector handles = pipeline.getRayTracingShaderGroupHandlesKHR<uint8_t>(0, handleCount, dataSize);

    const VkDeviceSize sbtSize = rgenRegion.size + missRegion.size + hitRegion.size;
    auto sbtBuffer = make_unique<Buffer>(
        **ctx.allocator,
        sbtSize,
        vk::BufferUsageFlagBits::eShaderBindingTableKHR
        | vk::BufferUsageFlagBits::eShaderDeviceAddress
        | vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible
        | vk::MemoryPropertyFlagBits::eHostCoherent
    );

    const vk::DeviceAddress sbtAddress = ctx.device->getBufferAddress({.buffer = **sbtBuffer});
    rgenRegion.deviceAddress = sbtAddress;
    missRegion.deviceAddress = rgenRegion.deviceAddress + rgenRegion.size;
    hitRegion.deviceAddress = missRegion.deviceAddress + missRegion.size;

    auto getHandlePtr = [&](const uint32_t i) { return handles.data() + i * handleSize; };
    auto *sbtBufferMapped = static_cast<uint8_t *>(sbtBuffer->map());

    uint32_t handleIndex = 0;

    uint8_t *rgenData = sbtBufferMapped;
    memcpy(rgenData, getHandlePtr(handleIndex++), handleSize);

    uint8_t *missData = sbtBufferMapped + rgenRegion.size;
    for (uint32_t i = 0; i < missCount; i++) {
        memcpy(missData, getHandlePtr(handleIndex++), handleSize);
        missData += missRegion.stride;
    }

    uint8_t *hitData = sbtBufferMapped + rgenRegion.size + missRegion.size;
    for (uint32_t i = 0; i < hitCount; i++) {
        memcpy(hitData, getHandlePtr(handleIndex++), handleSize);
        hitData += hitRegion.stride;
    }

    sbtBuffer->unmap();

    return {
        .backingBuffer = std::move(sbtBuffer),
        .rgenRegion = rgenRegion,
        .missRegion = missRegion,
        .hitRegion = hitRegion
    };
}

template GraphicsPipelineBuilder &GraphicsPipelineBuilder::withVertices<ModelVertex>();

template GraphicsPipelineBuilder &GraphicsPipelineBuilder::withVertices<SkyboxVertex>();

template GraphicsPipelineBuilder &GraphicsPipelineBuilder::withVertices<ScreenSpaceQuadVertex>();
