module;

module Cinder.Render;

import spirv_reflect;
import glfw;
import std;
import imgui;

import :GlfwStatics;
import Cinder.Globals;
import Cinder.Utils;
import Cinder.Render.Vulkan;
import Cinder.Render.Gui;
import Cinder.Render.Graph;
import Cinder.Render.Mesh;

namespace zrx {
VulkanRenderer::VulkanRenderer() {
    constexpr int INIT_WINDOW_WIDTH = 1200;
    constexpr int INIT_WINDOW_HEIGHT = 800;

    glfwWindowHint(ZRX_GLFW_CLIENT_API, ZRX_GLFW_NO_API);
    window = glfwCreateWindow(INIT_WINDOW_WIDTH, INIT_WINDOW_HEIGHT, "Cinder", nullptr, nullptr);

    init_glfw_user_pointer(window);
    auto *user_data = static_cast<GlfwStaticUserData *>(glfwGetWindowUserPointer(window));
    if (!user_data) Logger::error("unexpected null window user pointer");
    user_data->renderer = this;

    glfwSetFramebufferSizeCallback(window, framebuffer_resize_callback);

    const auto vkb_instance = create_instance();
    debug_messenger = make_unique<vk::raii::DebugUtilsMessengerEXT>(*instance, vkb_instance.debug_messenger);
    create_surface();
    const auto vkb_physical_device = pick_physical_device(vkb_instance);
    create_logical_device(vkb_physical_device);

    ctx.allocator = make_unique<VmaAllocatorWrapper>(**ctx.physical_device, **ctx.device, **instance);

    resource_manager = make_unique<ResourceManager>(BINDLESS_ARRAY_SIZE);

    swap_chain = make_unique<SwapChain>(
        ctx,
        *surface,
        queue_family_indices,
        window,
        get_msaa_sample_count()
    );

    create_command_pool();
    create_command_buffers();

    create_descriptor_pool();

    create_sync_objects();

    create_bindless_resources();

    init_imgui();
}

VulkanRenderer::~VulkanRenderer() {
    glfwDestroyWindow(window);
}

void VulkanRenderer::framebuffer_resize_callback(GLFWwindow *window, const int width, const int height) {
    (void) (width + height);
    const auto user_data = static_cast<GlfwStaticUserData *>(glfwGetWindowUserPointer(window));
    if (!user_data) Logger::error("unexpected null window user pointer");
    user_data->renderer->framebuffer_resized = true;
}

// ==================== instance creation ====================

vkb::Instance VulkanRenderer::create_instance() {
    const auto debug_callback = [](
            const VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
            const VkDebugUtilsMessageTypeFlagsEXT messageType,
            const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
            void *p_user_data) -> VkBool32 {
        const auto severity = vkb::to_string_message_severity(messageSeverity);
        const auto type = vkb::to_string_message_type(messageType);

        std::stringstream ss;
        ss << "[VALIDATION LAYER / " << severity << " / " << type << "]\n" << pCallbackData->pMessage << "\n";
        std::cout << ss.str() << std::endl;

        return vk::False;
    };

    auto instance_result = vkb::InstanceBuilder()
            .set_app_name("Cinder")
            .request_validation_layers()
            .enable_layer("VK_LAYER_KHRONOS_validation")
            .set_debug_callback(debug_callback)
            .require_api_version(1, 3)
            .set_minimum_instance_version(1, 3)
            .enable_extensions(get_required_extensions())
            .build();

    if (!instance_result) {
        Logger::error("failed to create instance: {}", instance_result.error().message());
    }

    instance = make_unique<vk::raii::Instance>(vk_ctx, instance_result.value().instance);

    return instance_result.value();
}

vector<const char *> VulkanRenderer::get_required_extensions() {
    uint32_t glfw_extension_count = 0;
    const char **glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

    vector extensions(glfw_extensions, glfw_extensions + glfw_extension_count);

    if (ENABLE_VALIDATION_LAYERS) {
        extensions.push_back(vk::EXTDebugUtilsExtensionName);
    }

    return extensions;
}

// ==================== startup ====================

void VulkanRenderer::create_surface() {
    VkSurfaceKHR _surface;

    if (glfwCreateWindowSurface(**instance, window, nullptr, &_surface) != VK_SUCCESS) {
        Logger::error("failed to create window surface!");
    }

    surface = make_unique<vk::raii::SurfaceKHR>(*instance, _surface);
}

vkb::PhysicalDevice VulkanRenderer::pick_physical_device(const vkb::Instance &vkb_instance) {
    const vector device_extensions{
        vk::EXTDescriptorIndexingExtensionName,
        // vk::EXTDebugMarkerExtensionName,
        vk::KHRAccelerationStructureExtensionName,
        vk::KHRDeferredHostOperationsExtensionName,
        vk::KHRDynamicRenderingExtensionName,
        vk::KHRMultiviewExtensionName,
        vk::KHRRayTracingPipelineExtensionName,
        vk::KHRSwapchainExtensionName,
        vk::KHRSynchronization2ExtensionName,
        vk::KHRTimelineSemaphoreExtensionName,
    };

    auto physical_device_result = vkb::PhysicalDeviceSelector(vkb_instance, **surface)
            .set_minimum_version(1, 3)
            .require_dedicated_transfer_queue()
            .prefer_gpu_device_type()
            .require_present()
            .add_required_extensions(device_extensions)
            .set_required_features(vk::PhysicalDeviceFeatures{
                .fillModeNonSolid = vk::True,
                .samplerAnisotropy = vk::True,
            })
            .set_required_features_12(vk::PhysicalDeviceVulkan12Features{
                .descriptorIndexing = vk::True,
                .shaderUniformBufferArrayNonUniformIndexing = vk::True,
                .shaderSampledImageArrayNonUniformIndexing = vk::True,
                .shaderStorageBufferArrayNonUniformIndexing = vk::True,
                .descriptorBindingUniformBufferUpdateAfterBind = vk::True,
                .descriptorBindingSampledImageUpdateAfterBind = vk::True,
                .descriptorBindingStorageImageUpdateAfterBind = vk::True,
                .descriptorBindingStorageBufferUpdateAfterBind = vk::True,
                .descriptorBindingPartiallyBound = vk::True,
                .runtimeDescriptorArray = vk::True,
                .timelineSemaphore = vk::True,
                .bufferDeviceAddress = vk::True,
            })
            .add_required_extension_features(vk::PhysicalDeviceDynamicRenderingFeatures{
                .dynamicRendering = vk::True,
            })
            .add_required_extension_features(vk::PhysicalDeviceSynchronization2FeaturesKHR{
                .synchronization2 = vk::True,
            })
            .add_required_extension_features(vk::PhysicalDeviceMultiviewFeatures{
                .multiview = vk::True,
            })
            .add_required_extension_features(vk::PhysicalDeviceAccelerationStructureFeaturesKHR{
                .accelerationStructure = vk::True,
            })
            .add_required_extension_features(vk::PhysicalDeviceRayTracingPipelineFeaturesKHR{
                .rayTracingPipeline = vk::True,
            })
            .select();

    if (!physical_device_result) {
        Logger::error("failed to select physical device: {}", physical_device_result.error().message());
    }

    ctx.physical_device = make_unique<vk::raii::PhysicalDevice>(
        *instance, physical_device_result.value().physical_device);
    msaa_sample_count = get_max_usable_sample_count();

    return physical_device_result.value();
}

void VulkanRenderer::create_logical_device(const vkb::PhysicalDevice &vkb_physical_device) {
    auto device_result = vkb::DeviceBuilder(vkb_physical_device).build();

    if (!device_result) {
        Logger::error("failed to select logical device: {}", device_result.error().message());
    }

    ctx.device = make_unique<vk::raii::Device>(*ctx.physical_device, device_result.value().device);

    auto graphics_queue_result = device_result.value().get_queue(vkb::QueueType::graphics);
    auto graphics_queue_index_result = device_result.value().get_queue_index(vkb::QueueType::graphics);
    if (!graphics_queue_result || !graphics_queue_index_result) {
        Logger::error("failed to get graphics queue: {}", device_result.error().message());
    }

    auto present_queue_result = device_result.value().get_queue(vkb::QueueType::present);
    auto present_queue_index_result = device_result.value().get_queue_index(vkb::QueueType::present);
    if (!present_queue_result || !present_queue_index_result) {
        Logger::error("failed to get present queue: {}", device_result.error().message());
    }

    ctx.graphics_queue = make_unique<vk::raii::Queue>(*ctx.device, graphics_queue_result.value());
    present_queue = make_unique<vk::raii::Queue>(*ctx.device, present_queue_result.value());

    queue_family_indices = {
        .graphics_compute_family = graphics_queue_index_result.value(),
        .present_family = present_queue_index_result.value()
    };
}

// ==================== swapchain ====================

void VulkanRenderer::recreate_swap_chain() {
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);

    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }

    wait_idle();

    swap_chain.reset();
    swap_chain = make_unique<SwapChain>(
        ctx,
        *surface,
        queue_family_indices,
        window,
        get_msaa_sample_count()
    );

    for (auto& [node_handle, resources]: node_resources) {
        if (render_graph->nodes().at(node_handle).is_graphics()) {
            resources.render_infos = create_node_render_infos(node_handle);
        }
    }
}

// ==================== descriptors ====================

void VulkanRenderer::create_descriptor_pool() {
    const vector<vk::DescriptorPoolSize> pool_sizes = {
        {
            .type = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 100u,
        },
        {
            .type = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1000u,
        },
        {
            .type = vk::DescriptorType::eStorageImage,
            .descriptorCount = 100u,
        },
        {
            .type = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 100u,
        },
        {
            .type = vk::DescriptorType::eAccelerationStructureKHR,
            .descriptorCount = 100u,
        },
    };

    const vk::DescriptorPoolCreateInfo pool_info{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet
                 | vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind,
        .maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * 6 + 5,
        .poolSizeCount = static_cast<uint32_t>(pool_sizes.size()),
        .pPoolSizes = pool_sizes.data(),
    };

    descriptor_pool = make_unique<vk::raii::DescriptorPool>(*ctx.device, pool_info);
}

void VulkanRenderer::create_bindless_resources() {
    constexpr vk::DescriptorBindingFlags binding_flags = vk::DescriptorBindingFlagBits::ePartiallyBound
                                                         | vk::DescriptorBindingFlagBits::eUpdateAfterBind;

    bindless_descriptor_set = make_unique<BindlessDescriptorSet>(
        ctx,
        *descriptor_pool,
        ResourcePack<Texture> {
            BINDLESS_ARRAY_SIZE,
            vk::ShaderStageFlagBits::eAllGraphics,
            vk::DescriptorType::eCombinedImageSampler,
            binding_flags
        },
        ResourcePack<Texture> {
            BINDLESS_ARRAY_SIZE,
            vk::ShaderStageFlagBits::eCompute,
            vk::DescriptorType::eStorageImage,
            binding_flags
        },
        ResourcePack<Buffer> {
            BINDLESS_ARRAY_SIZE,
            vk::ShaderStageFlagBits::eAll,
            vk::DescriptorType::eUniformBuffer,
            binding_flags
        }
    );
}

// ==================== render infos ====================

RenderInfo::RenderInfo(vector<RenderTarget> colors) : color_targets(std::move(colors)) {
    make_attachment_infos();
}

RenderInfo::RenderInfo(vector<RenderTarget> colors, RenderTarget depth)
    : color_targets(std::move(colors)), depth_target(std::move(depth)) {
    make_attachment_infos();
}

vk::RenderingInfo RenderInfo::get(const vk::Extent2D extent, const uint32_t views,
                                  const vk::RenderingFlags flags) const {
    return {
        .flags = flags,
        .renderArea = {
            .offset = {0, 0},
            .extent = extent
        },
        .layerCount = views == 1 ? 1u : 0u,
        .viewMask = views == 1 ? 0 : (1u << views) - 1,
        .colorAttachmentCount = static_cast<uint32_t>(color_attachments.size()),
        .pColorAttachments = color_attachments.data(),
        .pDepthAttachment = depth_attachment ? &depth_attachment.value() : nullptr
    };
}

void RenderInfo::make_attachment_infos() {
    for (const auto &target: color_targets) {
        color_attachments.emplace_back(target.get_attachment_info());
        cached_color_attachment_formats.push_back(target.get_format());
    }

    if (depth_target) {
        depth_attachment = depth_target->get_attachment_info();
    }
}

// ==================== multisampling ====================

vk::SampleCountFlagBits VulkanRenderer::get_max_usable_sample_count() const {
    const vk::PhysicalDeviceProperties physical_device_properties = ctx.physical_device->getProperties();

    const vk::SampleCountFlags counts = physical_device_properties.limits.framebufferColorSampleCounts
                                        & physical_device_properties.limits.framebufferDepthSampleCounts;

    if (counts & vk::SampleCountFlagBits::e64) { return vk::SampleCountFlagBits::e64; }
    if (counts & vk::SampleCountFlagBits::e32) { return vk::SampleCountFlagBits::e32; }
    if (counts & vk::SampleCountFlagBits::e16) { return vk::SampleCountFlagBits::e16; }
    if (counts & vk::SampleCountFlagBits::e8) { return vk::SampleCountFlagBits::e8; }
    if (counts & vk::SampleCountFlagBits::e4) { return vk::SampleCountFlagBits::e4; }
    if (counts & vk::SampleCountFlagBits::e2) { return vk::SampleCountFlagBits::e2; }

    return vk::SampleCountFlagBits::e1;
}

// ==================== buffers ====================

unique_ptr<Buffer> VulkanRenderer::create_local_buffer(const void *data, const vk::DeviceSize size,
                                                       const vk::BufferUsageFlags usage) const {
    Buffer staging_buffer{
        **ctx.allocator,
        size,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    };

    void *staging_data = staging_buffer.map();
    std::memcpy(staging_data, data, static_cast<size_t>(size));
    staging_buffer.unmap();

    auto result_buffer = make_unique<Buffer>(
        **ctx.allocator,
        size,
        vk::BufferUsageFlagBits::eTransferDst | usage,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    result_buffer->copy_from_buffer(ctx, staging_buffer, size);

    return result_buffer;
}

// ==================== commands ====================

void VulkanRenderer::create_command_pool() {
    const vk::CommandPoolCreateInfo pool_info{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = queue_family_indices.graphics_compute_family.value()
    };

    ctx.command_pool = make_unique<vk::raii::CommandPool>(*ctx.device, pool_info);
}

void VulkanRenderer::create_command_buffers() {
    const uint32_t n_buffers = frame_resources.size();

    auto graphics_command_buffers =
            utils::cmd::create_command_buffers(ctx, vk::CommandBufferLevel::ePrimary, n_buffers);

    for (size_t i = 0; i < graphics_command_buffers.size(); i++) {
        frame_resources[i].main_cmd_buffer =
                make_unique<vk::raii::CommandBuffer>(std::move(graphics_command_buffers[i]));
    }
}

// ==================== sync ====================

void VulkanRenderer::create_sync_objects() {
    const vk::StructureChain<vk::SemaphoreCreateInfo, vk::SemaphoreTypeCreateInfo> timeline_semaphore_info{
        {},
        {
            .semaphoreType = vk::SemaphoreType::eTimeline,
            .initialValue = 0,
        }
    };

    constexpr vk::SemaphoreCreateInfo binary_semaphore_info;

    for (auto &res: frame_resources) {
        res.sync = {
            .image_available_semaphore = make_unique<vk::raii::Semaphore>(*ctx.device, binary_semaphore_info),
            .ready_to_present_semaphore = make_unique<vk::raii::Semaphore>(*ctx.device, binary_semaphore_info),
            .render_finished_timeline = {
                make_unique<vk::raii::Semaphore>(*ctx.device, timeline_semaphore_info.get<vk::SemaphoreCreateInfo>())
            },
        };
    }
}

// ==================== gui ====================

void VulkanRenderer::init_imgui() {
    const vector<vk::DescriptorPoolSize> pool_sizes = {
        {vk::DescriptorType::eSampler, 1000},
        {vk::DescriptorType::eCombinedImageSampler, 1000},
        {vk::DescriptorType::eSampledImage, 1000},
        {vk::DescriptorType::eStorageImage, 1000},
        {vk::DescriptorType::eUniformTexelBuffer, 1000},
        {vk::DescriptorType::eStorageTexelBuffer, 1000},
        {vk::DescriptorType::eUniformBuffer, 1000},
        {vk::DescriptorType::eStorageBuffer, 1000},
        {vk::DescriptorType::eUniformBufferDynamic, 1000},
        {vk::DescriptorType::eStorageBufferDynamic, 1000},
        {vk::DescriptorType::eInputAttachment, 1000}
    };

    const vk::DescriptorPoolCreateInfo pool_info = {
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = 1000,
        .poolSizeCount = static_cast<uint32_t>(pool_sizes.size()),
        .pPoolSizes = pool_sizes.data(),
    };

    imgui_descriptor_pool = make_unique<vk::raii::DescriptorPool>(*ctx.device, pool_info);

    const uint32_t image_count = SwapChain::get_image_count(ctx, *surface);

    ImGui_ImplVulkan_InitInfo imgui_init_info = {
        .Instance = **instance,
        .PhysicalDevice = **ctx.physical_device,
        .Device = **ctx.device,
        .Queue = **ctx.graphics_queue,
        .DescriptorPool = static_cast<VkDescriptorPool>(**imgui_descriptor_pool),
        .MinImageCount = image_count,
        .ImageCount = image_count,
        .MSAASamples = static_cast<VkSampleCountFlagBits>(get_msaa_sample_count()),
        .UseDynamicRendering = true,
        .ColorAttachmentFormat = static_cast<VkFormat>(swap_chain->get_image_format()),
    };

    gui_renderer = make_unique<GuiRenderer>(window, imgui_init_info);
}

void VulkanRenderer::render_gui_section() {
    constexpr auto section_flags = ImGuiTreeNodeFlags_DefaultOpen;

    if (ImGui::CollapsingHeader("Renderer ", section_flags)) {
        static bool use_msaa_dummy = use_msaa;
        if (ImGui::Checkbox("MSAA", &use_msaa_dummy)) {
            queued_frame_begin_actions.emplace([this](const FrameBeginActionContext &fba_ctx) {
                use_msaa = use_msaa_dummy;

                wait_idle();
                recreate_swap_chain();

                gui_renderer.reset();
                init_imgui();
            });
        }
    }
}

// ==================== render graph ====================

void VulkanRenderer::register_render_graph(const RenderGraph &graph) {
    render_graph = make_unique<RenderGraph>(graph);
    create_render_graph_resources();
    repeated_frame_begin_actions = render_graph->frame_begin_callbacks();
}

void VulkanRenderer::create_render_graph_resources() {
    const auto compute_accessed_resources = gather_compute_accessed_resources();

    for (const auto &[handle, description]: render_graph->model_resources()) {
        auto model = make_unique<Model>(ctx, description.path, false);
        resource_manager->add(handle, std::move(model), description.name);
    }

    for (const auto &[handle, description]: render_graph->vertex_buffers()) {
        resource_manager->add(
            handle,
            create_local_buffer(description.data, description.size, vk::BufferUsageFlagBits::eVertexBuffer),
            description.name
        );
    }

    for (const auto &[handle, description]: render_graph->uniform_buffers()) {
        resource_manager->add(handle, utils::buf::create_uniform_buffer(ctx, description.size), description.name);

        const auto bindless_handle = resource_manager->get_bindless_handle(handle);
        const auto& buffer = resource_manager->get_buffer(handle);
        bindless_descriptor_set->update_binding<BINDLESS_UBO_BINDING>(buffer, bindless_handle);
    }

    for (const auto &[handle, description]: render_graph->external_tex_resources()) {
        auto usage_flags = vk::ImageUsageFlagBits::eTransferSrc
                           | vk::ImageUsageFlagBits::eTransferDst
                           | vk::ImageUsageFlagBits::eSampled
                           | utils::img::get_format_attachment_type(description.format);

        const bool is_compute_accessed = compute_accessed_resources.contains(handle);
        if (is_compute_accessed) {
            usage_flags |= vk::ImageUsageFlagBits::eStorage;
        }

        constexpr auto layout = vk::ImageLayout::eShaderReadOnlyOptimal;
        last_image_layouts.emplace(handle, layout);

        auto builder = TextureBuilder()
                .with_flags(description.flags)
                .with_name(description.name.c_str())
                .from_paths(description.paths)
                .use_format(description.format)
                .use_layout(layout)
                .use_usage(usage_flags);

        if (description.paths.size() > 1 && !(description.flags & vk::TextureFlagBitsZRX::CUBEMAP))
            builder.as_separate_channels();
        if (description.swizzle)
            builder.with_swizzle(*description.swizzle);

        resource_manager->add(handle, builder.create(ctx), description.name);

        const auto bindless_handle = resource_manager->get_bindless_handle(handle);
        const auto& texture = resource_manager->get_texture(handle);

        bindless_descriptor_set->update_binding<BINDLESS_SAMPLER_BINDING>(texture, bindless_handle);

        if (is_compute_accessed) {
            bindless_descriptor_set->update_binding<BINDLESS_STORAGE_TEXTURE_BINDING>(texture, bindless_handle);
        }
    }

    for (const auto &[handle, description]: render_graph->target_tex_resources()) {
        auto extent = description.extent;
        if (extent.width == 0 && extent.height == 0) {
            extent = swap_chain->get_extent();
        }

        auto usage_flags = vk::ImageUsageFlagBits::eTransferSrc
                           | vk::ImageUsageFlagBits::eTransferDst
                           | vk::ImageUsageFlagBits::eSampled
                           | utils::img::get_format_attachment_type(description.format);

        const bool is_compute_accessed = compute_accessed_resources.contains(handle);
        if (is_compute_accessed) {
            usage_flags |= vk::ImageUsageFlagBits::eStorage;
        }

        const auto layout = utils::img::is_depth_format(description.format)
                            ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                            : vk::ImageLayout::eColorAttachmentOptimal;
        last_image_layouts.emplace(handle, layout);

        auto builder = TextureBuilder()
                .with_flags(description.flags)
                .with_name(description.name.c_str())
                .as_uninitialized({extent.width, extent.height, 1u})
                .use_format(description.format)
                .use_layout(layout)
                .use_usage(usage_flags);

        resource_manager->add(handle, builder.create(ctx), description.name);

        const auto bindless_handle = resource_manager->get_bindless_handle(handle);
        const auto& texture = resource_manager->get_texture(handle);

        bindless_descriptor_set->update_binding<BINDLESS_SAMPLER_BINDING>(texture, bindless_handle);

        if (is_compute_accessed) {
            bindless_descriptor_set->update_binding<BINDLESS_STORAGE_TEXTURE_BINDING>(texture, bindless_handle);
        }
    }

    for (const auto &[handle, description]: render_graph->transient_tex_resources()) {
        auto extent = description.extent;
        if (extent.width == 0 && extent.height == 0) {
            extent = swap_chain->get_extent();
        }

        auto builder = TextureBuilder()
                .with_flags(description.flags)
                .with_name(description.name.c_str())
                .as_uninitialized({extent.width, extent.height, 1u})
                .use_format(description.format)
                .use_usage(vk::ImageUsageFlagBits::eTransientAttachment
                           | utils::img::get_format_attachment_type(description.format));

        resource_manager->add(handle, builder.create(ctx), description.name);
    }

    for (const auto &[handle, description]: render_graph->graphics_pipelines()) {
        auto builder = create_graph_gfx_pipeline_builder(handle);
        graphics_pipelines.emplace(handle, builder.create(ctx));
    }

    for (const auto &[handle, description]: render_graph->compute_pipelines()) {
        auto builder = create_graph_compute_pipeline_builder(handle);
        compute_pipelines.emplace(handle, builder.create(ctx));
    }
}

GraphicsPipelineBuilder
VulkanRenderer::create_graph_gfx_pipeline_builder(const ResourceHandle pipeline_handle) const {
    const auto &pipeline_info = render_graph->graphics_pipelines().at(pipeline_handle);

    vector<vk::Format> color_formats;
    for (const auto &format_variant: pipeline_info.color_formats) {
        const vk::Format format = std::holds_alternative<vk::Format>(format_variant)
                                  ? std::get<vk::Format>(format_variant)
                                  : swap_chain->get_image_format();
        color_formats.push_back(format);
    }

    vector<vk::DescriptorSetLayout> descriptor_set_layouts;
    descriptor_set_layouts.push_back(*bindless_descriptor_set->get_layout());

    auto builder = GraphicsPipelineBuilder()
            .with_vertex_shader(pipeline_info.vertex_path)
            .with_fragment_shader(pipeline_info.fragment_path)
            .with_vertices(
                pipeline_info.binding_descriptions,
                pipeline_info.attr_descriptions
            )
            .with_rasterizer({
                .polygonMode = vk::PolygonMode::eFill,
                .cullMode = pipeline_info.custom_properties.cull_mode,
                .frontFace = vk::FrontFace::eCounterClockwise,
                .lineWidth = 1.0f,
            })
            .with_depth_stencil({
                .depthTestEnable = !pipeline_info.custom_properties.disable_depth_test,
                .depthWriteEnable = !pipeline_info.custom_properties.disable_depth_write,
                .depthCompareOp = pipeline_info.custom_properties.depth_compare_op,
            })
            .with_multisampling({
                .rasterizationSamples = pipeline_info.custom_properties.use_msaa
                                        ? get_msaa_sample_count()
                                        : vk::SampleCountFlagBits::e1,
                .minSampleShading = 1.0f,
            })
            .with_descriptor_layouts(descriptor_set_layouts)
            .with_color_formats(color_formats);

    if (pipeline_info.depth_format) {
        const vk::Format format = std::holds_alternative<vk::Format>(*pipeline_info.depth_format)
                                  ? std::get<vk::Format>(*pipeline_info.depth_format)
                                  : swap_chain->get_depth_format();
        builder.with_depth_format(format);
    } else {
        builder.with_depth_stencil({
            .depthTestEnable = vk::False,
            .depthWriteEnable = vk::False,
        });
    }

    if (pipeline_info.custom_properties.multiview_count > 1) {
        builder.for_views(pipeline_info.custom_properties.multiview_count);
    }

    return builder;
}

ComputePipelineBuilder
VulkanRenderer::create_graph_compute_pipeline_builder(const ResourceHandle pipeline_handle) const {
    const auto &pipeline_info = render_graph->compute_pipelines().at(pipeline_handle);

    vector<vk::DescriptorSetLayout> descriptor_set_layouts;
    descriptor_set_layouts.push_back(*bindless_descriptor_set->get_layout());

    auto builder = ComputePipelineBuilder()
            .with_shader(pipeline_info.path)
            .with_descriptor_layouts(descriptor_set_layouts);

    return builder;
}

void VulkanRenderer::queue_set_update_with_handle(DescriptorSet &descriptor_set, const ResourceHandle res_handle,
                                                  const uint32_t binding, const uint32_t array_element) const {
    if (resource_manager->contains_buffer(res_handle)) {
        const auto &buffer = resource_manager->get_buffer(res_handle);
        descriptor_set.queue_update(
            binding,
            buffer,
            vk::DescriptorType::eUniformBuffer,
            buffer.get_size(),
            0,
            array_element
        );
    } else if (resource_manager->contains_texture(res_handle)) {
        const auto &texture = resource_manager->get_texture(res_handle);
        descriptor_set.queue_update(
            ctx,
            binding,
            texture,
            vk::DescriptorType::eCombinedImageSampler,
            array_element
        );
    }
}

vector<RenderInfo> VulkanRenderer::create_node_render_infos(const RenderNodeHandle node_handle) const {
    const auto &node_info = render_graph->nodes().at(node_handle).get_graphics();

    vector<RenderInfo> render_infos;

    if (has_swapchain_target(node_handle)) {
        bool is_first_with_final_target = is_first_node_targetting_final_image(node_handle);

        for (auto &swap_chain_targets: swap_chain->get_render_targets(ctx)) {
            vector<RenderTarget> color_targets;

            if (!is_first_with_final_target) {
                // has to be overridden, otherwise this render pass will clear the swapchain image
                // todo - maybe also add this functionality to targets other than swapchain? like postprocessing
                swap_chain_targets.color_target.override_attachment_config(vk::AttachmentLoadOp::eLoad);
            }

            for (auto color_target_handle: node_info.color_targets) {
                if (color_target_handle == FINAL_IMAGE_HANDLE) {
                    color_targets.emplace_back(std::move(swap_chain_targets.color_target));
                } else {
                    const auto &target_texture = resource_manager->get_texture(color_target_handle);
                    color_targets.emplace_back(target_texture.get_image().get_view(ctx), target_texture.get_format());
                }
            }

            if (node_info.depth_target) {
                render_infos.emplace_back(std::move(color_targets), std::move(swap_chain_targets.depth_target));
            } else {
                render_infos.emplace_back(std::move(color_targets));
            }
        }
    } else {
        vector<RenderTarget> color_targets;
        optional<RenderTarget> depth_target;

        for (auto color_target_handle: node_info.color_targets) {
            const auto &target_texture = resource_manager->get_texture(color_target_handle);
            color_targets.emplace_back(target_texture.get_image().get_mip_view(ctx, 0),
                                       target_texture.get_format());
        }

        if (node_info.depth_target) {
            const auto &target_texture = resource_manager->get_texture(*node_info.depth_target);
            depth_target = RenderTarget(target_texture.get_image().get_layer_mip_view(ctx, 0, 0),
                                        target_texture.get_format());
        }

        if (depth_target) {
            render_infos.emplace_back(std::move(color_targets), std::move(*depth_target));
        } else {
            render_infos.emplace_back(std::move(color_targets));
        }
    }

    return render_infos;
}

void VulkanRenderer::run_render_graph() {
    if (start_frame()) {
        Logger::debug("starting frame");

        partitioned_nodes = render_graph->get_partitioned();

        node_resources.clear();
        for (const auto& [node_handle, _]: render_graph->nodes()) {
            const auto& node = render_graph->nodes().at(node_handle);

            if (node.is_graphics()) {
                node_resources.emplace(node_handle, RenderNodeResources{
                    .render_infos = create_node_render_infos(node_handle),
                });
            }
        }

        record_graph_commands();
        end_frame();
    }
}

void VulkanRenderer::record_graph_commands() {
    const auto &command_buffer = *frame_resources[current_frame_idx].main_cmd_buffer;

    command_buffer.begin({});

    swap_chain->transition_to_attachment_layout(command_buffer);

    for (size_t i = 0; i < partitioned_nodes.size(); i++) {
        const auto& curr_partition = partitioned_nodes[i];

        record_pre_partition_commands(curr_partition);

        for (const auto &node_handle: curr_partition) {
            const RenderNode& node = render_graph->nodes().at(node_handle);

            if (node.is_graphics()) {
                record_graphics_node_commands(node_handle);

                for (const auto &target: get_node_target_handles(node_handle)) {
                    unbarriered_targets.insert(target);
                }
            } else if (node.is_compute()) {
                record_compute_node_commands(node_handle);

                for (const auto &target: node.get_compute().bound_write_resources) {
                    unbarriered_targets.insert(target);
                }
            }
        }
    }

    swap_chain->transition_to_present_layout(command_buffer);

    command_buffer.end();
}

void VulkanRenderer::record_graphics_node_commands(const RenderNodeHandle node_handle) {
    const auto &command_buffer = *frame_resources[current_frame_idx].main_cmd_buffer;
    const auto &node = render_graph->nodes().at(node_handle).get_graphics();

    Logger::debug("recording gfx node: {}", node.name);

    // if size > 1, then this means that this pass (node) draws to the swapchain image
    // and thus benefits from double or triple buffering
    const auto &[render_infos] = node_resources.at(node_handle);
    const size_t subresource_index = render_infos.size() == 1 ? 0 : current_frame_idx;
    const auto &node_render_info = render_infos[subresource_index];

    // command_buffer.debugMarkerBeginEXT(vk::DebugMarkerMarkerInfoEXT { .pMarkerName = node.name.c_str(), });

    command_buffer.beginRendering(node_render_info.get(
            get_node_target_extent(node_handle),
            node.custom_properties.multiview_count)
    );
    record_node_rendering_commands(node_handle);
    command_buffer.endRendering();

    // regenerate mipmaps for each target that had them
    record_regenerate_mipmaps_commands(node_handle);

    // command_buffer.debugMarkerEndEXT();
}

void VulkanRenderer::record_node_rendering_commands(const RenderNodeHandle node_handle) const {
    const auto &command_buffer = *frame_resources[current_frame_idx].main_cmd_buffer;
    const auto &node_info = render_graph->nodes().at(node_handle).get_graphics();

    utils::cmd::set_dynamic_states(command_buffer, get_node_target_extent(node_handle));

    RenderPassContext ctx{
        command_buffer,
        *resource_manager,
        graphics_pipelines,
        **bindless_descriptor_set
    };
    node_info.body(ctx);
}

void VulkanRenderer::record_regenerate_mipmaps_commands(const RenderNodeHandle node_handle) {
    const auto &command_buffer = *frame_resources[current_frame_idx].main_cmd_buffer;
    const auto &node = render_graph->nodes().at(node_handle).get_graphics();

    for (const auto color_target: node.color_targets) {
        if (color_target == FINAL_IMAGE_HANDLE) continue;

        const auto &target_texture = resource_manager->get_texture(color_target);
        if (target_texture.get_mip_levels() == 1) continue;

        Logger::debug("recording mipmap regeneration commands for texture: {}", resource_manager->get_name(color_target));

        target_texture.get_image().transition_layout(
            vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::ImageLayout::eTransferDstOptimal,
            command_buffer
        );

        target_texture.generate_mipmaps(ctx, vk::ImageLayout::eShaderReadOnlyOptimal);

        last_image_layouts[color_target] = vk::ImageLayout::eShaderReadOnlyOptimal;
    }
}

void VulkanRenderer::record_pre_partition_commands(const vector<RenderNodeHandle> &partition) {
    const auto &command_buffer = *frame_resources[current_frame_idx].main_cmd_buffer;

    cached_barriers.emplace_back();
    auto& barriers = cached_barriers.back();

    vector<ResourceHandle> sampled_resources;
    vector<ResourceHandle> target_resources;

    for (const auto& node_handle: partition) {
        const auto& node = render_graph->nodes().at(node_handle);
        if (!node.is_graphics()) continue;
        const auto& node_info = node.get_graphics();

        for (const auto color_target: node_info.bound_resources) {
            sampled_resources.emplace_back(color_target);
        }

        for (const auto color_target: node_info.color_targets) {
            target_resources.emplace_back(color_target);
        }
        if (node_info.depth_target) {
            target_resources.emplace_back(*node_info.depth_target);
        }
    }

    for (const auto& sampled_resource: sampled_resources) {
        if (!unbarriered_targets.contains(sampled_resource)) continue;

        Logger::debug("inserting pre-partition (sampled) barrier for texture: {}", resource_manager->get_name(sampled_resource));

        const Texture& sampled_texture = resource_manager->get_texture(sampled_resource);
        const bool is_depth_texture = utils::img::is_depth_format(sampled_texture.get_image().get_format());
        constexpr auto new_layout = vk::ImageLayout::eShaderReadOnlyOptimal;

        barriers.insert(vk::ImageMemoryBarrier2 {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllGraphics,
            .srcAccessMask = is_depth_texture
                             ? vk::AccessFlagBits2::eDepthStencilAttachmentWrite
                             : vk::AccessFlagBits2::eColorAttachmentWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllGraphics,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = is_depth_texture ? vk::ImageLayout::eDepthAttachmentOptimal : vk::ImageLayout::eColorAttachmentOptimal,
            .newLayout = new_layout,
            .image = *sampled_texture.get_image(),
            .subresourceRange = {
                .aspectMask = is_depth_texture ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor,
                .levelCount = 1,
                .layerCount = 1,
            }
        });

        unbarriered_targets.erase(sampled_resource);
        last_image_layouts[sampled_resource] = new_layout;
    }

    for (const auto& target_resource: target_resources) {
        if (!resource_manager->contains_texture(target_resource)) continue;

        Logger::debug("inserting pre-partition (target) barrier for texture: {}", resource_manager->get_name(target_resource));

        const Texture& target_texture = resource_manager->get_texture(target_resource);
        const bool is_depth_texture = utils::img::is_depth_format(target_texture.get_image().get_format());
        const auto expected_layout = is_depth_texture
                                     ? vk::ImageLayout::eDepthAttachmentOptimal
                                     : vk::ImageLayout::eColorAttachmentOptimal;
        const auto current_layout = last_image_layouts.at(target_resource);

        if (current_layout == expected_layout) continue;

        barriers.insert(vk::ImageMemoryBarrier2 {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllGraphics,
            .srcAccessMask = is_depth_texture
                             ? vk::AccessFlagBits2::eDepthStencilAttachmentWrite
                             : vk::AccessFlagBits2::eColorAttachmentWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllGraphics,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = current_layout,
            .newLayout = expected_layout,
            .image = *target_texture.get_image(),
            .subresourceRange = {
                .aspectMask = is_depth_texture ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor,
                .levelCount = 1,
                .layerCount = 1,
            }
        });

        last_image_layouts[target_resource] = expected_layout;
    }

    barriers.record_cmd(command_buffer);
}

void VulkanRenderer::record_compute_node_commands(const RenderNodeHandle node_handle) const {
    const auto &command_buffer = *frame_resources[current_frame_idx].main_cmd_buffer;
    const auto &node = render_graph->nodes().at(node_handle).get_compute();

    Logger::debug("recording compute node: {}", node.name);

    // todo
    //
    // if size > 1, then this means that this pass (node) draws to the swapchain image
    // and thus benefits from double or triple buffering
    //
    // const auto &[render_infos] = node_resources.at(node_handle);
    // const size_t subresource_index = render_infos.size() == 1 ? 0 : current_frame_idx;
    // const auto &node_render_info = render_infos[subresource_index];

    // command_buffer.debugMarkerBeginEXT(vk::DebugMarkerMarkerInfoEXT { .pMarkerName = node.name.c_str(), });

    ComputePassContext ctx{
        command_buffer,
        *resource_manager,
        compute_pipelines,
        **bindless_descriptor_set
    };
    node.body(ctx);

    // todo
    // regenerate mipmaps for each target that had them
    // record_regenerate_mipmaps_commands(node_handle);

    // command_buffer.debugMarkerEndEXT();
}

set<ResourceHandle> VulkanRenderer::gather_compute_accessed_resources() const {
    set<ResourceHandle> result;

    for (const auto& [node_handle, node_info] : render_graph->nodes()) {
        if (!node_info.is_compute()) continue;

        result.insert_range(node_info.get_compute().bound_read_resources);
        result.insert_range(node_info.get_compute().bound_write_resources);
    }

    return result;
}

bool VulkanRenderer::has_swapchain_target(const RenderNodeHandle node_handle) const {
    return render_graph->nodes().at(node_handle).get_graphics()
            .get_all_targets_set()
            .contains(FINAL_IMAGE_HANDLE);
}

bool VulkanRenderer::is_first_node_targetting_final_image(const RenderNodeHandle node_handle) const {
    if (!has_swapchain_target(node_handle)) return false;

    const auto flattened = std::ranges::join_view(partitioned_nodes);

    auto first_it = std::ranges::find_if(flattened, [&](const RenderNodeHandle &handle) {
        return has_swapchain_target(handle);
    });

    return first_it == flattened.end() || *first_it == node_handle;
}

bool VulkanRenderer::should_run_node_pass(const RenderNodeHandle node_handle) const {
    return render_graph->nodes().at(node_handle).should_run();
}

vk::Extent2D VulkanRenderer::get_node_target_extent(const RenderNodeHandle node_handle) const {
    const auto &node_info = render_graph->nodes().at(node_handle);

    return has_swapchain_target(node_handle)
           ? swap_chain->get_extent()
           : resource_manager->get_texture(node_info.get_graphics().color_targets[0])
           .get_image()
           .get_extent_2d();
}

vk::Format VulkanRenderer::get_target_color_format(const ResourceHandle resource_handle) const {
    if (resource_handle == FINAL_IMAGE_HANDLE) {
        return swap_chain->get_image_format();
    }
    return resource_manager->get_texture(resource_handle).get_format();
}

vk::Format VulkanRenderer::get_target_depth_format(const ResourceHandle resource_handle) const {
    if (resource_handle == FINAL_IMAGE_HANDLE) {
        return swap_chain->get_depth_format();
    }
    return resource_manager->get_texture(resource_handle).get_format();
}

vector<ResourceHandle> VulkanRenderer::get_node_target_handles(const RenderNodeHandle node_handle) const {
    const auto& node = render_graph->nodes().at(node_handle).get_graphics();
    vector<ResourceHandle> handles = node.color_targets;
    if (node.depth_target) handles.emplace_back(*node.depth_target);
    return handles;
}

// ==================== render loop ====================

void VulkanRenderer::tick(const float delta_time) {
    (void) delta_time;
    // unused rn
}

void VulkanRenderer::do_frame_begin_actions() {
    const FrameBeginActionContext fba_ctx{*resource_manager};

    for (const auto &action: repeated_frame_begin_actions) {
        action(fba_ctx);
    }

    while (!queued_frame_begin_actions.empty()) {
        queued_frame_begin_actions.front()(fba_ctx);
        queued_frame_begin_actions.pop();
    }
}

bool VulkanRenderer::start_frame() {
    const auto &sync = frame_resources[current_frame_idx].sync;

    const vector wait_semaphores = {
        **sync.render_finished_timeline.semaphore,
    };

    const vector wait_semaphore_values = {
        sync.render_finished_timeline.timeline,
    };

    const vk::SemaphoreWaitInfo wait_info{
        .semaphoreCount = static_cast<uint32_t>(wait_semaphores.size()),
        .pSemaphores = wait_semaphores.data(),
        .pValues = wait_semaphore_values.data(),
    };

    if (ctx.device->waitSemaphores(wait_info, numeric_limits<uint64_t>::max()) != vk::Result::eSuccess) {
        Logger::error("waitSemaphores on renderFinishedTimeline failed");
    }

    do_frame_begin_actions();

    const auto &[result, image_index] = swap_chain->acquire_next_image(*sync.image_available_semaphore);

    if (result == vk::Result::eErrorOutOfDateKHR) {
        recreate_swap_chain();
        return false;
    }
    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
        Logger::error("failed to acquire swap chain image!");
    }

    unbarriered_targets.clear();
    cached_barriers.clear();

    return true;
}

void VulkanRenderer::end_frame() {
    auto &sync = frame_resources[current_frame_idx].sync;

    const vector wait_semaphores = {
        **sync.image_available_semaphore
    };

    const vector<TimelineSemValueType> wait_semaphore_values = {
        0
    };

    static constexpr vk::PipelineStageFlags wait_stages[] = {
        vk::PipelineStageFlagBits::eEarlyFragmentTests,
        vk::PipelineStageFlagBits::eVertexInput,
    };

    const array signal_semaphores = {
        **sync.render_finished_timeline.semaphore,
        **sync.ready_to_present_semaphore
    };

    sync.render_finished_timeline.timeline++;
    const vector<TimelineSemValueType> signal_semaphore_values{
        sync.render_finished_timeline.timeline,
        0
    };

    const vk::StructureChain<vk::SubmitInfo, vk::TimelineSemaphoreSubmitInfo> submit_info{
        {
            .waitSemaphoreCount = static_cast<uint32_t>(wait_semaphores.size()),
            .pWaitSemaphores = wait_semaphores.data(),
            .pWaitDstStageMask = wait_stages,
            .commandBufferCount = 1,
            .pCommandBuffers = &**frame_resources[current_frame_idx].main_cmd_buffer,
            .signalSemaphoreCount = signal_semaphores.size(),
            .pSignalSemaphores = signal_semaphores.data(),
        },
        {
            .waitSemaphoreValueCount = static_cast<uint32_t>(wait_semaphore_values.size()),
            .pWaitSemaphoreValues = wait_semaphore_values.data(),
            .signalSemaphoreValueCount = static_cast<uint32_t>(signal_semaphore_values.size()),
            .pSignalSemaphoreValues = signal_semaphore_values.data(),
        }
    };

    try {
        ctx.graphics_queue->submit(submit_info.get<vk::SubmitInfo>());
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        throw;
    }

    const array present_wait_semaphores = {**sync.ready_to_present_semaphore};

    const array image_indices = {swap_chain->get_current_image_index()};

    const vk::PresentInfoKHR present_info{
        .waitSemaphoreCount = present_wait_semaphores.size(),
        .pWaitSemaphores = present_wait_semaphores.data(),
        .swapchainCount = 1U,
        .pSwapchains = &***swap_chain,
        .pImageIndices = image_indices.data(),
    };

    auto present_result = vk::Result::eSuccess;

    try {
        present_result = present_queue->presentKHR(present_info);
    } catch (...) {
    }

    const bool did_resize = present_result == vk::Result::eErrorOutOfDateKHR
                            || present_result == vk::Result::eSuboptimalKHR
                            || framebuffer_resized;
    if (did_resize) {
        framebuffer_resized = false;
        recreate_swap_chain();
    } else if (present_result != vk::Result::eSuccess) {
        Logger::error("failed to present swap chain image!");
    }

    current_frame_idx = (current_frame_idx + 1) % MAX_FRAMES_IN_FLIGHT;
}
} // zrx
