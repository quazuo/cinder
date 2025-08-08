module;

export module Cinder.Render.Vulkan:Pipeline;

import vulkan_hpp;
import std;

import Cinder.Utils;
import Cinder.Globals;
import :Buffer;
import :Image;
import :Context;

export namespace zrx {
class GraphicsPipelineBuilder;
class ComputePipelineBuilder;
class RtPipelineBuilder;

template<typename T>
concept VertexLike = requires {
    { T::get_binding_descriptions() } -> std::same_as<vector<vk::VertexInputBindingDescription>>;
    { T::get_attribute_descriptions() } -> std::same_as<vector<vk::VertexInputAttributeDescription>>;
};

/**
 * Convenience wrappers around Vulkan pipelines, mainly to pair them together with related layouts.
 * Might be extended in the future as it's very bare-bones at this moment.
 */
class Pipeline {
    vk::raii::Pipeline pipeline;
    vk::raii::PipelineLayout layout;

protected:
    Pipeline(vk::raii::Pipeline&& pipeline, vk::raii::PipelineLayout&& layout)
        : pipeline(std::move(pipeline)), layout(std::move(layout)) {}

public:
    auto operator*() const -> const vk::raii::Pipeline& { return pipeline; }

    auto get_layout() const -> const vk::raii::PipelineLayout& { return layout; }
};

class GraphicsPipeline : public Pipeline {
    vk::SampleCountFlagBits rasterization_samples;

    friend GraphicsPipelineBuilder;

    GraphicsPipeline(vk::raii::Pipeline&& pipeline, vk::raii::PipelineLayout&& layout,
                     const vk::SampleCountFlagBits samples = {})
        : Pipeline(std::move(pipeline), std::move(layout)), rasterization_samples(samples) {}

public:
    auto get_sample_count() const -> vk::SampleCountFlagBits { return rasterization_samples; }
};

class ComputePipeline : public Pipeline {
    friend ComputePipelineBuilder;

    ComputePipeline(vk::raii::Pipeline&& pipeline, vk::raii::PipelineLayout&& layout)
        : Pipeline(std::move(pipeline), std::move(layout)) {}
};

class RtPipeline : public Pipeline {
public:
    struct ShaderBindingTable {
        unique_ptr<Buffer> backing_buffer;
        vk::StridedDeviceAddressRegionKHR rgen_region;
        vk::StridedDeviceAddressRegionKHR miss_region;
        vk::StridedDeviceAddressRegionKHR hit_region;
        vk::StridedDeviceAddressRegionKHR call_region;
    };

private:
    ShaderBindingTable sbt;

    friend RtPipelineBuilder;

    RtPipeline(vk::raii::Pipeline&& pipeline, vk::raii::PipelineLayout&& layout, ShaderBindingTable&& sbt)
        : Pipeline(std::move(pipeline), std::move(layout)), sbt(std::move(sbt)) {}

public:
    auto get_sbt() const -> const ShaderBindingTable& { return sbt; }
};

/**
 * Builder class streamlining graphics pipeline creation.
 */
class GraphicsPipelineBuilder {
    std::filesystem::path vertex_shader_path;
    std::filesystem::path fragment_shader_path;

    vector<vk::VertexInputBindingDescription> vertex_bindings;
    vector<vk::VertexInputAttributeDescription> vertex_attributes;

    vector<vk::DescriptorSetLayout> descriptor_set_layouts;

    optional<vk::PipelineRasterizationStateCreateInfo> rasterizer_override;
    optional<vk::PipelineMultisampleStateCreateInfo> multisampling_override;
    optional<vk::PipelineDepthStencilStateCreateInfo> depth_stencil_override;

    uint32_t multiview_count = 1;
    vector<vk::Format> color_attachment_formats;
    optional<vk::Format> depth_attachment_format;

public:
    auto with_vertex_shader(const std::filesystem::path &path) -> GraphicsPipelineBuilder&;

    auto with_fragment_shader(const std::filesystem::path &path) -> GraphicsPipelineBuilder&;

    template<typename T>
        requires VertexLike<T>
    auto with_vertices() -> GraphicsPipelineBuilder& {
        vertex_bindings   = T::get_binding_descriptions();
        vertex_attributes = T::get_attribute_descriptions();
        return *this;
    }

    auto with_vertices(vector<vk::VertexInputBindingDescription> bindings,
                       vector<vk::VertexInputAttributeDescription> attributes) -> GraphicsPipelineBuilder&;

    auto with_descriptor_layouts(const vector<vk::DescriptorSetLayout> &layouts) -> GraphicsPipelineBuilder&;

    auto with_rasterizer(const vk::PipelineRasterizationStateCreateInfo &rasterizer) -> GraphicsPipelineBuilder&;

    auto with_multisampling(const vk::PipelineMultisampleStateCreateInfo &multisampling) -> GraphicsPipelineBuilder&;

    auto with_depth_stencil(const vk::PipelineDepthStencilStateCreateInfo &depth_stencil) -> GraphicsPipelineBuilder&;

    /**
     * Sets the number of views used with the `VK_KHR_multiview` extension.
     */
    auto for_views(uint32_t count) -> GraphicsPipelineBuilder&;

    auto with_color_formats(const vector<vk::Format> &formats) -> GraphicsPipelineBuilder&;

    auto with_depth_format(vk::Format format) -> GraphicsPipelineBuilder&;

    auto create(const RendererContext &ctx) const -> GraphicsPipeline;

private:
    void check_params() const;

    static auto eval_push_constant_ranges(
        const SpirvReflectModuleWrapper& vertex_spirv_reflect_module,
        const SpirvReflectModuleWrapper& fragment_spirv_reflect_module
    ) -> vector<vk::PushConstantRange>;
};

class ComputePipelineBuilder {
    std::filesystem::path shader_path;
    vector<vk::DescriptorSetLayout> descriptor_set_layouts;

public:
    auto with_shader(const std::filesystem::path &path) -> ComputePipelineBuilder&;

    auto with_descriptor_layouts(const vector<vk::DescriptorSetLayout> &layouts) -> ComputePipelineBuilder&;

    auto create(const RendererContext &ctx) const -> ComputePipeline;
};

class RtPipelineBuilder {
    std::filesystem::path raygen_shader_path;
    std::filesystem::path closest_hit_shader_path;
    std::filesystem::path miss_shader_path;

    vector<vk::DescriptorSetLayout> descriptor_set_layouts;
    vector<vk::PushConstantRange> push_constant_ranges;

public:
    auto with_ray_gen_shader(const std::filesystem::path &path) -> RtPipelineBuilder&;

    auto with_closest_hit_shader(const std::filesystem::path &path) -> RtPipelineBuilder&;

    auto with_miss_shader(const std::filesystem::path &path) -> RtPipelineBuilder&;

    auto with_descriptor_layouts(const vector<vk::DescriptorSetLayout> &layouts) -> RtPipelineBuilder&;

    auto with_push_constants(const vector<vk::PushConstantRange> &ranges) -> RtPipelineBuilder&;

    auto create(const RendererContext &ctx) const -> RtPipeline;

private:
    void check_params() const;

    auto build_pipeline(const RendererContext &ctx) const -> pair<vk::raii::Pipeline, vk::raii::PipelineLayout>;

    auto build_sbt(const RendererContext &ctx, const vk::raii::Pipeline &pipeline) const -> RtPipeline::ShaderBindingTable;
};
} // zrx
