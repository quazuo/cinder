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

GraphicsPipelineBuilder &GraphicsPipelineBuilder::withDescriptorLayouts(const std::vector<vk::DescriptorSetLayout> &layouts) {
    descriptorSetLayouts = layouts;
    return *this;
}

GraphicsPipelineBuilder &GraphicsPipelineBuilder::withPushConstants(const std::vector<vk::PushConstantRange> &ranges) {
    pushConstantRanges = ranges;
    return *this;
}

GraphicsPipelineBuilder &GraphicsPipelineBuilder::withRasterizer(const vk::PipelineRasterizationStateCreateInfo &rasterizer) {
    rasterizerOverride = rasterizer;
    return *this;
}

GraphicsPipelineBuilder &GraphicsPipelineBuilder::withMultisampling(const vk::PipelineMultisampleStateCreateInfo &multisampling) {
    multisamplingOverride = multisampling;
    return *this;
}

GraphicsPipelineBuilder &GraphicsPipelineBuilder::withDepthStencil(const vk::PipelineDepthStencilStateCreateInfo &depthStencil) {
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

    constexpr vk::RayTracingShaderGroupCreateInfoKHR shaderGroupTemplate {
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

    result.layout = make_unique<vk::raii::PipelineLayout>(*ctx.device, pipelineLayoutInfo);

    const vk::RayTracingPipelineCreateInfoKHR pipelineCreateInfo{
        .stageCount = static_cast<uint32_t>(shaderStages.size()),
        .pStages = shaderStages.data(),
        .groupCount = static_cast<uint32_t>(shaderGroups.size()),
        .pGroups = shaderGroups.data(),
        .maxPipelineRayRecursionDepth = 1u,
        .layout = **result.layout,
    };

    result.pipeline = make_unique<vk::raii::Pipeline>(
        *ctx.device,
        nullptr,
        nullptr,
        pipelineCreateInfo
    );

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

template GraphicsPipelineBuilder &GraphicsPipelineBuilder::withVertices<ModelVertex>();

template GraphicsPipelineBuilder &GraphicsPipelineBuilder::withVertices<SkyboxVertex>();

template GraphicsPipelineBuilder &GraphicsPipelineBuilder::withVertices<ScreenSpaceQuadVertex>();
