#pragma once

#include "src/render/libs.hpp"
#include "src/render/globals.hpp"

#include <filesystem>

struct RendererContext;
class Buffer;

/**
 * Convenience wrappers around Vulkan pipelines, mainly to pair them together with related layouts.
 * Might be extended in the future as it's very bare-bones at this moment.
 */
class Pipeline {
    unique_ptr<vk::raii::Pipeline> pipeline;
    unique_ptr<vk::raii::PipelineLayout> layout;

    friend class GraphicsPipelineBuilder;
    friend class RtPipelineBuilder;

protected:
    Pipeline() = default;

public:
    [[nodiscard]] const vk::raii::Pipeline &operator*() const { return *pipeline; }

    [[nodiscard]] const vk::raii::PipelineLayout &getLayout() const { return *layout; }
};

class GraphicsPipeline : public Pipeline {
    vk::SampleCountFlagBits rasterizationSamples{};

    friend class GraphicsPipelineBuilder;

    GraphicsPipeline() = default;

public:
    [[nodiscard]] vk::SampleCountFlagBits getSampleCount() const { return rasterizationSamples; }
};

class RtPipeline : public Pipeline {
    unique_ptr<Buffer> shaderBindingTableBuffer;

    friend class RtPipelineBuilder;

    RtPipeline() = default;
};

/**
 * Builder class streamlining graphics pipeline creation.
 */
class GraphicsPipelineBuilder {
    std::filesystem::path vertexShaderPath;
    std::filesystem::path fragmentShaderPath;

    std::vector<vk::VertexInputBindingDescription> vertexBindings;
    std::vector<vk::VertexInputAttributeDescription> vertexAttributes;

    std::vector<vk::DescriptorSetLayout> descriptorSetLayouts;
    std::vector<vk::PushConstantRange> pushConstantRanges;

    std::optional<vk::PipelineRasterizationStateCreateInfo> rasterizerOverride;
    std::optional<vk::PipelineMultisampleStateCreateInfo> multisamplingOverride;
    std::optional<vk::PipelineDepthStencilStateCreateInfo> depthStencilOverride;

    uint32_t multiviewCount = 1;
    std::vector<vk::Format> colorAttachmentFormats;
    std::optional<vk::Format> depthAttachmentFormat;

public:
    GraphicsPipelineBuilder &withVertexShader(const std::filesystem::path &path);

    GraphicsPipelineBuilder &withFragmentShader(const std::filesystem::path &path);

    template<typename T>
    GraphicsPipelineBuilder &withVertices();

    GraphicsPipelineBuilder &withDescriptorLayouts(const std::vector<vk::DescriptorSetLayout> &layouts);

    GraphicsPipelineBuilder &withPushConstants(const std::vector<vk::PushConstantRange> &ranges);

    GraphicsPipelineBuilder &withRasterizer(const vk::PipelineRasterizationStateCreateInfo &rasterizer);

    GraphicsPipelineBuilder &withMultisampling(const vk::PipelineMultisampleStateCreateInfo &multisampling);

    GraphicsPipelineBuilder &withDepthStencil(const vk::PipelineDepthStencilStateCreateInfo &depthStencil);

    /**
     * Sets the number of views used with the `VK_KHR_multiview` extension.
     */
    GraphicsPipelineBuilder &forViews(uint32_t count);

    GraphicsPipelineBuilder &withColorFormats(const std::vector<vk::Format> &formats);

    GraphicsPipelineBuilder &withDepthFormat(vk::Format format);

    [[nodiscard]] GraphicsPipeline create(const RendererContext &ctx) const;

private:
    void checkParams() const;
};

class RtPipelineBuilder {
    std::filesystem::path raygenShaderPath;
    std::filesystem::path closestHitShaderPath;
    std::filesystem::path missShaderPath;

    std::vector<vk::DescriptorSetLayout> descriptorSetLayouts;
    std::vector<vk::PushConstantRange> pushConstantRanges;

public:
    RtPipelineBuilder &withRayGenShader(const std::filesystem::path &path);

    RtPipelineBuilder &withClosestHitShader(const std::filesystem::path &path);

    RtPipelineBuilder &withMissShader(const std::filesystem::path &path);

    RtPipelineBuilder &withDescriptorLayouts(const std::vector<vk::DescriptorSetLayout> &layouts);

    RtPipelineBuilder &withPushConstants(const std::vector<vk::PushConstantRange> &ranges);

    [[nodiscard]] RtPipeline create(const RendererContext &ctx) const;

private:
    void checkParams() const;
};
