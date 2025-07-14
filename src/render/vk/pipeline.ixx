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
    [[nodiscard]] const vk::raii::Pipeline &operator*() const { return pipeline; }

    [[nodiscard]] const vk::raii::PipelineLayout &get_layout() const { return layout; }
};

class GraphicsPipeline : public Pipeline {
    vk::SampleCountFlagBits rasterization_samples;

    friend GraphicsPipelineBuilder;

    GraphicsPipeline(vk::raii::Pipeline&& pipeline, vk::raii::PipelineLayout&& layout,
                     const vk::SampleCountFlagBits samples = {})
        : Pipeline(std::move(pipeline), std::move(layout)), rasterization_samples(samples) {}

public:
    [[nodiscard]] vk::SampleCountFlagBits get_sample_count() const { return rasterization_samples; }
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
    [[nodiscard]] const ShaderBindingTable &get_sbt() const { return sbt; }
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
    GraphicsPipelineBuilder &with_vertex_shader(const std::filesystem::path &path);

    GraphicsPipelineBuilder &with_fragment_shader(const std::filesystem::path &path);

    template<typename T>
        requires VertexLike<T>
    GraphicsPipelineBuilder &with_vertices() {
        vertex_bindings   = T::get_binding_descriptions();
        vertex_attributes = T::get_attribute_descriptions();
        return *this;
    }

    GraphicsPipelineBuilder &with_vertices(vector<vk::VertexInputBindingDescription> bindings,
                                           vector<vk::VertexInputAttributeDescription> attributes);

    GraphicsPipelineBuilder &with_descriptor_layouts(const vector<vk::DescriptorSetLayout> &layouts);

    GraphicsPipelineBuilder &with_rasterizer(const vk::PipelineRasterizationStateCreateInfo &rasterizer);

    GraphicsPipelineBuilder &with_multisampling(const vk::PipelineMultisampleStateCreateInfo &multisampling);

    GraphicsPipelineBuilder &with_depth_stencil(const vk::PipelineDepthStencilStateCreateInfo &depth_stencil);

    /**
     * Sets the number of views used with the `VK_KHR_multiview` extension.
     */
    GraphicsPipelineBuilder &for_views(uint32_t count);

    GraphicsPipelineBuilder &with_color_formats(const vector<vk::Format> &formats);

    GraphicsPipelineBuilder &with_depth_format(vk::Format format);

    [[nodiscard]] GraphicsPipeline create(const RendererContext &ctx) const;

private:
    void check_params() const;

    [[nodiscard]] static vector<vk::PushConstantRange>
    eval_push_constant_ranges(const SpirvReflectModuleWrapper& vertex_spirv_reflect_module,
                              const SpirvReflectModuleWrapper& fragment_spirv_reflect_module);
};

class ComputePipelineBuilder {
    std::filesystem::path shader_path;
    vector<vk::DescriptorSetLayout> descriptor_set_layouts;

public:
    ComputePipelineBuilder &with_shader(const std::filesystem::path &path);

    ComputePipelineBuilder &with_descriptor_layouts(const vector<vk::DescriptorSetLayout> &layouts);

    [[nodiscard]] ComputePipeline create(const RendererContext &ctx) const;
};

class RtPipelineBuilder {
    std::filesystem::path raygen_shader_path;
    std::filesystem::path closest_hit_shader_path;
    std::filesystem::path miss_shader_path;

    vector<vk::DescriptorSetLayout> descriptor_set_layouts;
    vector<vk::PushConstantRange> push_constant_ranges;

public:
    RtPipelineBuilder &with_ray_gen_shader(const std::filesystem::path &path);

    RtPipelineBuilder &with_closest_hit_shader(const std::filesystem::path &path);

    RtPipelineBuilder &with_miss_shader(const std::filesystem::path &path);

    RtPipelineBuilder &with_descriptor_layouts(const vector<vk::DescriptorSetLayout> &layouts);

    RtPipelineBuilder &with_push_constants(const vector<vk::PushConstantRange> &ranges);

    [[nodiscard]] RtPipeline create(const RendererContext &ctx) const;

private:
    void check_params() const;

    [[nodiscard]] pair<vk::raii::Pipeline, vk::raii::PipelineLayout>
    build_pipeline(const RendererContext &ctx) const;

    [[nodiscard]] RtPipeline::ShaderBindingTable
    build_sbt(const RendererContext &ctx, const vk::raii::Pipeline &pipeline) const;
};
} // zrx
