#pragma once

#include "src/render/libs.hpp"
#include "src/render/globals.hpp"

#include <filesystem>

struct RendererContext;

/**
 * Convenience wrapper around Vulkan pipelines, mainly to pair them together with related layouts.
 * Might be extended in the future as it's very bare-bones at this moment.
 */
class Pipeline {
    unique_ptr<vk::raii::Pipeline> pipeline;
    unique_ptr<vk::raii::PipelineLayout> layout;
    vk::SampleCountFlagBits rasterizationSamples{};

    friend class PipelineBuilder;
    friend class RtPipelineBuilder;

    Pipeline() = default;

public:
    [[nodiscard]] const vk::raii::Pipeline &operator*() const { return *pipeline; }

    [[nodiscard]] const vk::raii::PipelineLayout &getLayout() const { return *layout; }

    [[nodiscard]] vk::SampleCountFlagBits getSampleCount() const { return rasterizationSamples; }
};

/**
 * Builder class streamlining pipeline creation.
 */
class PipelineBuilder {
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
    PipelineBuilder &withVertexShader(const std::filesystem::path &path);

    PipelineBuilder &withFragmentShader(const std::filesystem::path &path);

    template<typename T>
    PipelineBuilder &withVertices();

    PipelineBuilder &withDescriptorLayouts(const std::vector<vk::DescriptorSetLayout> &layouts);

    PipelineBuilder &withPushConstants(const std::vector<vk::PushConstantRange> &ranges);

    PipelineBuilder &withRasterizer(const vk::PipelineRasterizationStateCreateInfo &rasterizer);

    PipelineBuilder &withMultisampling(const vk::PipelineMultisampleStateCreateInfo &multisampling);

    PipelineBuilder &withDepthStencil(const vk::PipelineDepthStencilStateCreateInfo &depthStencil);

    /**
     * Sets the number of views used with the `VK_KHR_multiview` extension.
     */
    PipelineBuilder &forViews(uint32_t count);

    PipelineBuilder &withColorFormats(const std::vector<vk::Format> &formats);

    PipelineBuilder &withDepthFormat(vk::Format format);

    [[nodiscard]] Pipeline create(const RendererContext &ctx) const;

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

    [[nodiscard]] Pipeline create(const RendererContext &ctx) const;

private:
    void checkParams() const;
};
