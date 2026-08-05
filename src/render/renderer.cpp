module;

module Cinder.Render;

import spirv_reflect;
import glfw;
import std;
import imgui;
import vk_mem_alloc;

import :GlfwStatics;
import Cinder.Globals;
import Cinder.Utils;
import Cinder.Render.Vulkan;
import Cinder.Render.Gui;
import Cinder.Render.Graph;
import Cinder.Render.Mesh;

namespace zrx {
// ==================== RenderInfo ====================

RenderInfo::RenderInfo(vector<RenderTarget> colors) : color_targets(std::move(colors)) {
    make_attachment_infos();
}

RenderInfo::RenderInfo(vector<RenderTarget> colors, RenderTarget depth)
    : color_targets(std::move(colors)), depth_target(std::move(depth)) {
    make_attachment_infos();
}

auto RenderInfo::get(const vk::Extent2D extent, const uint32_t views, const vk::RenderingFlags flags) const -> vk::RenderingInfo {
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

// ==================== VulkanRenderer ====================

VulkanRenderer::VulkanRenderer() {
    constexpr int INIT_WINDOW_WIDTH = 1600;
    constexpr int INIT_WINDOW_HEIGHT = 1200;

    glfwWindowHint(glfw::CLIENT_API, glfw::NO_API);
    ctx.window = glfwCreateWindow(INIT_WINDOW_WIDTH, INIT_WINDOW_HEIGHT, "Cinder", nullptr, nullptr);

    init_glfw_user_pointer(ctx.window);
    auto *user_data = static_cast<GlfwStaticUserData *>(glfwGetWindowUserPointer(ctx.window));
    if (!user_data) Logger::error("unexpected null window user pointer");
    user_data->renderer = this;

    glfwSetFramebufferSizeCallback(ctx.window, framebuffer_resize_callback);

    const auto vkb_instance = create_instance();
    debug_messenger = make_unique<vk::raii::DebugUtilsMessengerEXT>(*instance, vkb_instance.debug_messenger);
    create_surface();
    const auto vkb_physical_device = pick_physical_device(vkb_instance);
    create_logical_device(vkb_physical_device);
    create_queues();

    // create allocator
    {
        static constexpr vma::VulkanFunctions funcs{
            .vkGetInstanceProcAddr = vkGetInstanceProcAddr,
            .vkGetDeviceProcAddr = vkGetDeviceProcAddr
        };

        const vma::AllocatorCreateInfo allocator_create_info{
            .flags = vma::AllocatorCreateFlagBits::eBufferDeviceAddress,
            .physicalDevice = **ctx.physical_device,
        };

        ctx.allocator = make_unique<vma::raii::Allocator>(*instance, *ctx.device, allocator_create_info);
    }

    resource_manager = make_unique<ResourceManager>(ctx, BINDLESS_ARRAY_SIZE, MAX_FRAMES_IN_FLIGHT);
    repeated_frame_begin_actions.emplace_back([&](const FrameBeginActionContext& fba_ctx) {
        resource_manager->clear_removal_queue();
    });

    create_command_pool();
    create_command_buffers();

    swap_chain = make_unique<SwapChain>(
        ctx,
        *surface,
        queue_family_indices,
        get_msaa_sample_count()
    );

    create_descriptor_pool();

    create_sync_objects();

    create_bindless_resources();

    init_imgui();
}

VulkanRenderer::~VulkanRenderer() {
    glfwDestroyWindow(ctx.window);
}

void VulkanRenderer::framebuffer_resize_callback(GLFWwindow *window, const int width, const int height) {
    (void) (width + height);
    const auto user_data = static_cast<GlfwStaticUserData *>(glfwGetWindowUserPointer(window));
    if (!user_data) Logger::error("unexpected null window user pointer");
    user_data->renderer->framebuffer_resized = true;
}

// ==================== instance creation ====================

auto VulkanRenderer::create_instance() -> vkb::Instance {
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

    constexpr array<uint32_t, 1> layer_setting_values = { 100 };

    auto instance_result = vkb::InstanceBuilder()
            .set_app_name("Cinder")
            .request_validation_layers()
            .enable_layer("VK_LAYER_KHRONOS_validation")
            .set_debug_callback(debug_callback)
            .require_api_version(1, 3)
            .set_minimum_instance_version(1, 3)
            .enable_extensions(get_required_extensions())
            .add_layer_setting(vk::LayerSettingEXT {
                .pLayerName = "VK_LAYER_KHRONOS_validation",
                .pSettingName = "duplicate_message_limit",
                .type = vk::LayerSettingTypeEXT::eUint32,
                .valueCount = static_cast<uint32_t>(layer_setting_values.size()),
                .pValues = layer_setting_values.data(),
            })
            .build();

    if (!instance_result) {
        Logger::error("failed to create instance: {}", instance_result.error().message());
    }

    instance = make_unique<vk::raii::Instance>(vk_ctx, instance_result.value().instance);

    return instance_result.value();
}

auto VulkanRenderer::get_required_extensions() -> vector<const char *> {
    uint32_t glfw_extension_count = 0;
    const char **glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

    vector extensions(glfw_extensions, glfw_extensions + glfw_extension_count);
    extensions.push_back(vk::EXTDebugUtilsExtensionName);

    return extensions;
}

// ==================== startup ====================

void VulkanRenderer::create_surface() {
    VkSurfaceKHR _surface;

    if (glfwCreateWindowSurface(**instance, ctx.window, nullptr, &_surface) != VK_SUCCESS) {
        Logger::error("failed to create window surface!");
    }

    surface = make_unique<vk::raii::SurfaceKHR>(*instance, _surface);
}

auto VulkanRenderer::pick_physical_device(const vkb::Instance &vkb_instance) -> vkb::PhysicalDevice {
    const vector device_extensions{
        vk::EXTDescriptorIndexingExtensionName,
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
                .hostQueryReset = vk::True,
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
    timestamp_period = ctx.physical_device->getProperties().limits.timestampPeriod;

    return physical_device_result.value();
}

void VulkanRenderer::create_logical_device(const vkb::PhysicalDevice &vkb_physical_device) {
    auto device_result = vkb::DeviceBuilder(vkb_physical_device).build();
    if (!device_result) {
        Logger::error("failed to select logical device: {}", device_result.error().message());
    }

    ctx.device = make_unique<vk::raii::Device>(*ctx.physical_device, device_result.value().device);
}

void VulkanRenderer::create_queues() {
    present_queue = make_unique<PresentQueue>(ctx, *surface);
    ctx.graphics_queue = make_unique<GraphicsQueue>(ctx);

    queue_family_indices = {
        .graphics_compute_family = ctx.graphics_queue->get_family_index(),
        .present_family = present_queue->get_family_index()
    };
}

// ==================== swapchain ====================

void VulkanRenderer::recreate_swap_chain() {
    int width = 0, height = 0;
    glfwGetFramebufferSize(ctx.window, &width, &height);

    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(ctx.window, &width, &height);
        glfwWaitEvents();
    }

    wait_idle();

    Logger::info("recreating swap chain in frame #{}", ctx.current_frame_idx);

    swap_chain.reset();
    swap_chain = make_unique<SwapChain>(
        ctx,
        *surface,
        queue_family_indices,
        get_msaa_sample_count()
    );

    for (const auto& handle: render_graph->get_all_used_resources()) {
        if (handle == FINAL_IMAGE_HANDLE) continue;
        const auto& desc_variant = resource_manager->get_desc_variant(handle);
        if (!std::holds_alternative<TargetTextureResourceDesc>(desc_variant)) continue;
        const auto& description = std::get<TargetTextureResourceDesc>(desc_variant);

        auto extent = description.extent;
        if (extent.width != 0 || extent.height != 0) continue;

        if (!resource_manager->contains<ImageBuilder>(handle)) {
            Logger::error("missing texture builder for window-sized texture ({}) during window resize",
                          resource_manager->get_name(handle));
        }

        auto& image_builder = resource_manager->get<ImageBuilder>(handle);
        image_builder.invalidate_loaded_texture_data();
        image_builder.with_layout(last_image_layouts.at(handle));
        resource_manager->recreate(handle);

        // update bindless handles

        const bool is_compute_accessed = compute_accessed_resources.contains(handle);
        const auto& image = resource_manager->get<Image>(handle);
        const auto bindless_handle = resource_manager->get_bindless_handle(handle);

        bindless_descriptor_set->update_binding<BINDLESS_SAMPLER_BINDING>(image, static_cast<uint32_t>(bindless_handle));

        if (is_compute_accessed) {
            bindless_descriptor_set->update_binding<BINDLESS_STORAGE_TEXTURE_BINDING>(image, static_cast<uint32_t>(bindless_handle));
        }
    }

    for (auto& [handle, resources]: node_resources) {
        if (render_graph->nodes().at(handle).is_graphics()) {
            resources.render_infos = create_node_render_infos(handle);
        }
    }
}

// ==================== descriptors ====================

void VulkanRenderer::create_descriptor_pool() {
    const vector<vk::DescriptorPoolSize> pool_sizes = {
        {
            .type = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = BINDLESS_ARRAY_SIZE,
        },
        {
            .type = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 2 * BINDLESS_ARRAY_SIZE,
        },
        {
            .type = vk::DescriptorType::eStorageImage,
            .descriptorCount = BINDLESS_ARRAY_SIZE,
        },
        {
            .type = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = BINDLESS_ARRAY_SIZE,
        },
        {
            .type = vk::DescriptorType::eAccelerationStructureKHR,
            .descriptorCount = BINDLESS_ARRAY_SIZE,
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
        ResourcePack<Image> {
            BINDLESS_ARRAY_SIZE,
            vk::ShaderStageFlagBits::eAllGraphics,
            vk::DescriptorType::eCombinedImageSampler,
            binding_flags
        },
        ResourcePack<Image> {
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

// ==================== multisampling ====================

auto VulkanRenderer::get_max_usable_sample_count() const -> vk::SampleCountFlagBits {
    const vk::PhysicalDeviceProperties physical_device_properties = ctx.physical_device->getProperties();

    const vk::SampleCountFlags counts = physical_device_properties.limits.framebufferColorSampleCounts
                                        & physical_device_properties.limits.framebufferDepthSampleCounts;

    // if (counts & vk::SampleCountFlagBits::e64) return vk::SampleCountFlagBits::e64;
    // if (counts & vk::SampleCountFlagBits::e32) return vk::SampleCountFlagBits::e32;
    if (counts & vk::SampleCountFlagBits::e16) return vk::SampleCountFlagBits::e16;
    if (counts & vk::SampleCountFlagBits::e8)  return vk::SampleCountFlagBits::e8;
    if (counts & vk::SampleCountFlagBits::e4)  return vk::SampleCountFlagBits::e4;
    if (counts & vk::SampleCountFlagBits::e2)  return vk::SampleCountFlagBits::e2;

    return vk::SampleCountFlagBits::e1;
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

    auto graphics_command_buffers = utils::cmd::create_command_buffers(ctx, vk::CommandBufferLevel::ePrimary, n_buffers);

    for (size_t i = 0; i < graphics_command_buffers.size(); i++) {
        frame_resources[i].main_cmd_buffer = make_unique<vk::raii::CommandBuffer>(std::move(graphics_command_buffers[i]));

        const std::string name = "graphics_command_buffer#" + std::to_string(i);

        ctx.device->setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT {
            .objectType = vk::ObjectType::eCommandBuffer,
            .objectHandle = reinterpret_cast<uint64_t>(static_cast<VkCommandBuffer>(**frame_resources[i].main_cmd_buffer)),
            .pObjectName = name.c_str(),
        });
    }
}

// ==================== sync ====================

void VulkanRenderer::create_sync_objects() {
    for (auto &res: frame_resources) {
        res.sync = {
            .image_available_sem = make_unique<BinarySemaphore>(ctx),
            .ready_to_present_sem = make_unique<BinarySemaphore>(ctx),
            .render_finished_timeline_sem = make_unique<TimelineSemaphore>(ctx),
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

    const vector color_attachment_formats { swap_chain->get_image_format() };

    ImGui_ImplVulkan_InitInfo imgui_init_info = {
        .Instance = **instance,
        .PhysicalDevice = **ctx.physical_device,
        .Device = **ctx.device,
        .Queue = ***ctx.graphics_queue,
        .DescriptorPool = static_cast<VkDescriptorPool>(**imgui_descriptor_pool),
        .MinImageCount = image_count,
        .ImageCount = image_count,
        .PipelineInfoMain = {
            .MSAASamples = static_cast<VkSampleCountFlagBits>(get_msaa_sample_count()),
            .PipelineRenderingCreateInfo = vk::PipelineRenderingCreateInfo {
                .colorAttachmentCount = static_cast<uint32_t>(color_attachment_formats.size()),
                .pColorAttachmentFormats = color_attachment_formats.data(),
            },
        },
        .UseDynamicRendering = true,
    };

    gui_renderer = make_unique<GuiRenderer>(ctx.window, imgui_init_info);
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

        const auto& frame = frame_resources[ctx.current_frame_idx];

        vector<GuiRenderer::ProfilerNodeInfo> profiler_frame_infos;
        const uint32_t timestamp_count = frame.prev_time_query_results.size();
        const uint64_t frame_start_timestamp = frame.prev_time_query_results.empty() ? 0 : frame.prev_time_query_results[0];
        float last_node_time = 0.0f;

        for (uint32_t i = 0; i < timestamp_count; i += 2) {
            const uint64_t start_timestamp = frame.prev_time_query_results[i];
            const uint64_t end_timestamp = frame.prev_time_query_results[i + 1];
            const string& name = frame.prev_node_names[i / 2];

            ImVec4 color{};
            ImGui::ColorConvertHSVtoRGB(
                static_cast<float>(i % 24) / 24.0f, 1.0f, 0.7f,
                color.x, color.y, color.z
            );
            color.w = 1.0f;

            const std::chrono::nanoseconds start_ns {
                static_cast<long long>(timestamp_period * static_cast<float>(start_timestamp - frame_start_timestamp)) };
            const std::chrono::nanoseconds end_ns {
                static_cast<long long>(timestamp_period * static_cast<float>(end_timestamp - frame_start_timestamp)) };
            const std::chrono::duration<float> node_time = end_ns - start_ns;

            profiler_frame_infos.emplace_back(GuiRenderer::ProfilerNodeInfo {
                .start_time = last_node_time,
                .end_time = last_node_time + node_time.count(),
                .name = name,
                .color = color
            });

            last_node_time += node_time.count();
        }

        gui_renderer->render_profiler(profiler_frame_infos);
    }
}

// ==================== render graph ====================

void VulkanRenderer::register_render_graph(const RenderGraph &graph) {
    render_graph = make_unique<RenderGraph>(graph);
    create_render_graph_resources();
}

void VulkanRenderer::reload_all_pipelines() {
    wait_idle();

    queued_frame_begin_actions.emplace([&](const FrameBeginActionContext& fba_ctx) {
        resource_manager->reload_all_pipelines();
    });
}

void VulkanRenderer::create_render_graph_resources() {
    const set<ResourceHandle> prev_compute_accessed_resources = compute_accessed_resources;
    compute_accessed_resources = gather_compute_accessed_resources();

    for (const auto& res_handle : resource_manager->get_all_resource_handles_range()) {
        if (res_handle == FINAL_IMAGE_HANDLE) continue;

        const bool requires_recreate = compute_accessed_resources.contains(res_handle)
                                       != prev_compute_accessed_resources.contains(res_handle);

        if (!requires_recreate && resource_manager->has_attached_resource(res_handle)) continue;

        std::visit(
            [&](const auto& arg) { create_resource(res_handle, arg); },
            resource_manager->get_desc_variant(res_handle)
        );
    }

    bindless_descriptor_set->commit_updates();
}

void VulkanRenderer::create_resource(const ResourceHandle handle, const VertexBufferResourceDesc &description) {
    Buffer buffer = utils::buf::create_local_buffer(ctx, description.data, description.size, vk::BufferUsageFlagBits::eVertexBuffer);
    resource_manager->attach_raw(handle, std::move(buffer));
}

void VulkanRenderer::create_resource(const ResourceHandle handle, const IndexBufferResourceDesc &description) {
    Buffer buffer = utils::buf::create_local_buffer(ctx, description.data, description.size, vk::BufferUsageFlagBits::eIndexBuffer);
    resource_manager->attach_raw(handle, std::move(buffer));
}

void VulkanRenderer::create_resource(const ResourceHandle handle, const UniformBufferResourceDesc &description) {
    if (description.data) {
        Buffer buffer = utils::buf::create_local_buffer(ctx, *description.data, description.size, vk::BufferUsageFlagBits::eUniformBuffer);
        resource_manager->attach_raw(handle, std::move(buffer));
    } else {
        resource_manager->attach_raw(handle, utils::buf::create_uniform_buffer(ctx, description.size));
    }

    const auto bindless_handle = resource_manager->get_bindless_handle(handle);
    auto& buffer = resource_manager->get<Buffer>(handle);
    bindless_descriptor_set->queue_update<BINDLESS_UBO_BINDING>(buffer, static_cast<uint32_t>(bindless_handle));
}

void VulkanRenderer::create_resource(const ResourceHandle handle, const ExternalTextureResourceDesc &description) {
    auto usage_flags = vk::ImageUsageFlagBits::eTransferSrc
                       | vk::ImageUsageFlagBits::eTransferDst
                       | vk::ImageUsageFlagBits::eSampled;

    const auto attachment_resources = gather_attachment_resources(); // todo: optimize this
    if (attachment_resources.contains(handle)) {
        usage_flags |= utils::img::get_format_attachment_type(description.format);
    }

    const bool is_compute_accessed = compute_accessed_resources.contains(handle);
    if (is_compute_accessed) {
        usage_flags |= vk::ImageUsageFlagBits::eStorage;
    }

    constexpr auto layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    last_image_layouts.emplace(handle, layout);

    auto builder = ImageBuilder()
            .with_flags(description.flags)
            .with_name(description.name.c_str())
            .from_paths(description.paths)
            .with_format(description.format)
            .with_layout(layout)
            .with_usage(usage_flags);

    if (description.paths.size() > 1 && !(description.flags & ImageFlags::CUBEMAP))
        builder.as_separate_channels();
    if (description.swizzle)
        builder.with_swizzle(*description.swizzle);

    resource_manager->attach_builder(ctx, handle, std::move(builder));

    const auto bindless_handle = resource_manager->get_bindless_handle(handle);
    auto& image = resource_manager->get<Image>(handle);

    bindless_descriptor_set->queue_update<BINDLESS_SAMPLER_BINDING>(image, static_cast<uint32_t>(bindless_handle));

    if (is_compute_accessed) {
        bindless_descriptor_set->queue_update<BINDLESS_STORAGE_TEXTURE_BINDING>(image, static_cast<uint32_t>(bindless_handle));
    }
}

void VulkanRenderer::create_resource(const ResourceHandle handle, const TargetTextureResourceDesc &description) {
    vk::ImageLayout layout;

    auto usage_flags = vk::ImageUsageFlagBits::eTransferSrc
                       | vk::ImageUsageFlagBits::eTransferDst
                       | vk::ImageUsageFlagBits::eSampled;

    const bool is_compute_accessed = compute_accessed_resources.contains(handle);
    if (is_compute_accessed) {
        usage_flags |= vk::ImageUsageFlagBits::eStorage;
    }

    const auto attachment_resources = gather_attachment_resources(); // todo: optimize this
    if (attachment_resources.contains(handle)) {
        usage_flags |= utils::img::get_format_attachment_type(description.format);
        layout = vk::hasDepthComponent(description.format)
                 ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                 : vk::ImageLayout::eColorAttachmentOptimal;
    } else if (is_compute_accessed) {
        layout = vk::ImageLayout::eGeneral;
    } else {
        layout = vk::ImageLayout::eGeneral;
    }

    last_image_layouts.insert_or_assign(handle, layout);

    auto builder = ImageBuilder()
            .with_flags(description.flags)
            .with_name(description.name.c_str())
            .as_uninitialized()
            .with_format(description.format)
            .with_layout(layout)
            .with_usage(usage_flags);

    if (description.extent.width == 0 && description.extent.height == 0) {
        builder.with_window_size();
    } else {
        builder.with_extent({description.extent.width, description.extent.height, 1u});
    }

    resource_manager->attach_builder<ImageBuilder>(ctx, handle, std::move(builder));
    const auto bindless_handle = resource_manager->get_bindless_handle(handle);
    auto& texture = resource_manager->get<Image>(handle);

    bindless_descriptor_set->queue_update<BINDLESS_SAMPLER_BINDING>(texture, static_cast<uint32_t>(bindless_handle));

    if (is_compute_accessed) {
        bindless_descriptor_set->queue_update<BINDLESS_STORAGE_TEXTURE_BINDING>(texture, static_cast<uint32_t>(bindless_handle));
    }
}

void VulkanRenderer::create_resource(const ResourceHandle handle, const PersistentTextureResourceDesc &description) {
    // todo
}

void VulkanRenderer::create_resource(const ResourceHandle handle, const TransientTextureResourceDesc &description) {
    auto extent = description.extent;
    if (extent.width == 0 && extent.height == 0) {
        extent = swap_chain->get_extent();
    }

    auto builder = ImageBuilder()
            .with_flags(description.flags)
            .with_name(description.name.c_str())
            .as_uninitialized()
            .with_format(description.format)
            .with_usage(vk::ImageUsageFlagBits::eTransientAttachment
                        | utils::img::get_format_attachment_type(description.format));

    if (description.extent.width == 0 && description.extent.height == 0) {
        builder.with_window_size();
    } else {
        builder.with_extent({description.extent.width, description.extent.height, 1u});
    }

    resource_manager->attach_builder<ImageBuilder>(ctx, handle, std::move(builder));
}

void VulkanRenderer::create_resource(const ResourceHandle handle, const GraphicsPipelineResourceDesc &description) {
    vector<vk::Format> color_formats;
    for (const auto &format_variant: description.color_formats) {
        const vk::Format format = std::holds_alternative<vk::Format>(format_variant)
                                  ? std::get<vk::Format>(format_variant)
                                  : swap_chain->get_image_format();
        color_formats.push_back(format);
    }

    vector<vk::DescriptorSetLayout> descriptor_set_layouts;
    descriptor_set_layouts.push_back(*bindless_descriptor_set->get_layout());

    auto builder = GraphicsPipelineBuilder()
            .with_vertex_shader(shader_base_path / description.vertex_path)
            .with_fragment_shader(shader_base_path / description.fragment_path)
            .with_vertices(
                description.vertex_bindings,
                description.vertex_attributes
            )
            .with_rasterizer({
                .polygonMode = vk::PolygonMode::eFill,
                .cullMode = description.custom_properties.cull_mode,
                .frontFace = vk::FrontFace::eCounterClockwise,
                .lineWidth = 1.0f,
            })
            .with_depth_stencil({
                .depthTestEnable = !description.custom_properties.disable_depth_test,
                .depthWriteEnable = !description.custom_properties.disable_depth_write,
                .depthCompareOp = description.custom_properties.depth_compare_op,
            })
            .with_multisampling({
                .rasterizationSamples = description.custom_properties.use_msaa
                                        ? get_msaa_sample_count()
                                        : vk::SampleCountFlagBits::e1,
                .minSampleShading = 1.0f,
            })
            .with_descriptor_layouts(descriptor_set_layouts)
            .with_color_formats(color_formats);

    if (description.depth_format) {
        const vk::Format format = std::holds_alternative<vk::Format>(*description.depth_format)
                                  ? std::get<vk::Format>(*description.depth_format)
                                  : swap_chain->get_depth_format();
        builder.with_depth_format(format);
    } else {
        builder.with_depth_stencil({
            .depthTestEnable = vk::False,
            .depthWriteEnable = vk::False,
        });
    }

    if (description.custom_properties.multiview_count > 1) {
        builder.for_views(description.custom_properties.multiview_count);
    }

    resource_manager->attach_builder(ctx, handle, std::move(builder));
}

void VulkanRenderer::create_resource(const ResourceHandle handle, const ComputePipelineResourceDesc &description) {
    vector<vk::DescriptorSetLayout> descriptor_set_layouts;
    descriptor_set_layouts.push_back(*bindless_descriptor_set->get_layout());

    auto builder = ComputePipelineBuilder()
            .with_shader(shader_base_path / description.path)
            .with_descriptor_layouts(descriptor_set_layouts);

    resource_manager->attach_builder(ctx, handle, std::move(builder));
}

void VulkanRenderer::queue_set_update_with_handle(DescriptorSet &descriptor_set, const ResourceHandle res_handle,
                                                  const uint32_t binding, const uint32_t array_element) const {
    if (resource_manager->contains<Buffer>(res_handle)) {
        const auto &buffer = resource_manager->get<Buffer>(res_handle);
        descriptor_set.queue_update(
            binding,
            buffer,
            vk::DescriptorType::eUniformBuffer,
            buffer.get_size(),
            0,
            array_element
        );
    } else if (resource_manager->contains<Image>(res_handle)) {
        auto &texture = resource_manager->get<Image>(res_handle);
        descriptor_set.queue_update(
            ctx,
            binding,
            texture,
            vk::DescriptorType::eCombinedImageSampler,
            array_element
        );
    }
}

auto VulkanRenderer::create_node_render_infos(const RenderNodeHandle node_handle) const -> vector<RenderInfo> {
    const auto &node_info = render_graph->nodes().at(node_handle).get_graphics();

    vector<RenderInfo> render_infos;

    if (has_swapchain_target(node_handle)) {
        bool is_first_with_final_target = is_first_node_targetting_final_image(node_handle);

        for (auto &swap_chain_targets: swap_chain->get_render_targets(ctx)) {
            vector<RenderTarget> color_targets;

            if (!is_first_with_final_target) {
                // has to be overridden, otherwise this render pass will clear the swapchain image
                swap_chain_targets.color_target.override_attachment_config(vk::AttachmentLoadOp::eLoad);
            }

            for (auto color_target_handle: node_info.color_targets) {
                if (color_target_handle == FINAL_IMAGE_HANDLE) {
                    color_targets.emplace_back(std::move(swap_chain_targets.color_target));
                } else {
                    const auto &target_texture = resource_manager->get<Image>(color_target_handle);
                    color_targets.emplace_back(target_texture.get_full_view(ctx), target_texture.get_format());
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
            const auto &target_texture = resource_manager->get<Image>(color_target_handle);
            color_targets.emplace_back(target_texture.get_mip_view(ctx, 0),
                                       target_texture.get_format());
        }

        if (node_info.depth_target) {
            const auto &target_texture = resource_manager->get<Image>(*node_info.depth_target);
            depth_target = RenderTarget(target_texture.get_layer_mip_view(ctx, 0, 0),
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
    partitioned_nodes = render_graph->get_partitioned();

    if (start_frame()) {
        Logger::debug("starting frame #{}", ctx.current_frame_idx);

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
    const auto &command_buffer = *frame_resources[ctx.current_frame_idx].main_cmd_buffer;

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
                    unbarriered_gfx_written_resources.insert(target);
                }
            } else if (node.is_compute()) {
                record_compute_node_commands(node_handle);

                for (const auto &target: node.get_compute().bound_write_resources) {
                    unbarriered_compute_written_resources.insert(target);
                }
            }
        }
    }

    swap_chain->transition_to_present_layout(command_buffer);

    command_buffer.end();
}

void VulkanRenderer::record_graphics_node_commands(const RenderNodeHandle node_handle) {
    const auto &command_buffer = *frame_resources[ctx.current_frame_idx].main_cmd_buffer;
    const auto &time_query_pool = *frame_resources[ctx.current_frame_idx].time_query_pool;
    const auto &node = render_graph->nodes().at(node_handle).get_graphics();

    Logger::debug("recording gfx node: {}", node.name);

    // if size > 1, then this means that this pass (node) draws to the swapchain image
    // and thus benefits from double or triple buffering
    const auto &render_infos = node_resources.at(node_handle).render_infos;
    const size_t subresource_index = render_infos.size() == 1 ? 0 : swap_chain->get_current_image_index();
    const auto &node_render_info = render_infos[subresource_index];

    command_buffer.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT { .pLabelName = node.name.c_str() });

    command_buffer.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe, *time_query_pool, current_query_idx++);

    command_buffer.beginRendering(node_render_info.get(
            get_node_target_extent(node_handle),
            node.custom_properties.multiview_count)
    );
    record_node_rendering_commands(node_handle);
    command_buffer.endRendering();

    record_regenerate_mipmaps_commands(node_handle);

    command_buffer.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, *time_query_pool, current_query_idx++);

    command_buffer.endDebugUtilsLabelEXT();
}

void VulkanRenderer::record_node_rendering_commands(const RenderNodeHandle node_handle) const {
    const auto &command_buffer = *frame_resources[ctx.current_frame_idx].main_cmd_buffer;
    const auto &node_info = render_graph->nodes().at(node_handle).get_graphics();

    utils::cmd::set_dynamic_states(command_buffer, get_node_target_extent(node_handle));

    RenderPassContext ctx{
        command_buffer,
        *resource_manager,
        **bindless_descriptor_set
    };
    node_info.body(ctx);
}

void VulkanRenderer::record_regenerate_mipmaps_commands(const RenderNodeHandle node_handle) {
    const auto &command_buffer = *frame_resources[ctx.current_frame_idx].main_cmd_buffer;
    const auto &node = render_graph->nodes().at(node_handle).get_graphics();

    for (const auto color_target: node.color_targets) {
        if (color_target == FINAL_IMAGE_HANDLE) continue;

        auto &target_texture = resource_manager->get<Image>(color_target);
        if (target_texture.get_mip_levels() == 1) continue;

        Logger::debug("recording mipmap regeneration commands for texture: {}", resource_manager->get_name(color_target));

        const vk::ImageLayout last_layout = last_image_layouts.at(color_target);

        target_texture.transition_layout(
            last_layout,
            vk::ImageLayout::eTransferDstOptimal,
            command_buffer
        );
        target_texture.generate_mipmaps(ctx, last_layout, command_buffer);
    }
}

void VulkanRenderer::record_pre_partition_commands(const vector<RenderNodeHandle> &partition) {
    const auto &command_buffer = *frame_resources[ctx.current_frame_idx].main_cmd_buffer;

    PipelineBarrierPack barriers;

    vector<ResourceHandle> sampled_resources;
    vector<ResourceHandle> target_resources;
    vector<ResourceHandle> compute_read_resources;
    vector<ResourceHandle> compute_write_resources;

    for (const auto& node_handle: partition) {
        const auto& node = render_graph->nodes().at(node_handle);

        if (node.is_graphics()) {
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
        } else if (node.is_compute()) {
            const auto& node_info = node.get_compute();

            for (const auto read_resource : node_info.bound_read_resources) {
                compute_read_resources.emplace_back(read_resource);
            }

            for (const auto write_resource : node_info.bound_write_resources) {
                compute_write_resources.emplace_back(write_resource);
            }
        }
    }

    for (const auto& handle: sampled_resources) {
        const bool is_unbarriered_gfx = unbarriered_gfx_written_resources.contains(handle);
        const bool is_unbarriered_compute = unbarriered_compute_written_resources.contains(handle);
        if (!is_unbarriered_gfx && !is_unbarriered_compute) continue;

        Logger::debug("inserting pre-partition (sampled) barrier for texture: {}", resource_manager->get_name(handle));

        const auto& texture = resource_manager->get<Image>(handle);
        const bool is_depth_texture = vk::hasDepthComponent(texture.get_format());
        const auto old_layout = last_image_layouts[handle];
        constexpr auto new_layout = vk::ImageLayout::eShaderReadOnlyOptimal;

        barriers.insert(vk::ImageMemoryBarrier2 {
            .srcStageMask = is_unbarriered_gfx
                            ? (is_depth_texture
                                ? vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests
                                : vk::PipelineStageFlagBits2::eColorAttachmentOutput)
                            : vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = is_unbarriered_gfx
                             ? (is_depth_texture
                                 ? vk::AccessFlagBits2::eDepthStencilAttachmentWrite
                                 : vk::AccessFlagBits2::eColorAttachmentWrite)
                             : vk::AccessFlagBits2::eShaderWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
            .dstAccessMask = is_depth_texture
                             ? vk::AccessFlagBits2::eDepthStencilAttachmentRead
                             : vk::AccessFlagBits2::eColorAttachmentRead,
            .oldLayout = old_layout,
            .newLayout = new_layout,
            .image = *texture,
            .subresourceRange = {
                .aspectMask = is_depth_texture ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = texture.get_mip_levels(),
                .baseArrayLayer = 0,
                .layerCount = 1,
            }
        });

        if (is_unbarriered_gfx) unbarriered_gfx_written_resources.erase(handle);
        else if (is_unbarriered_compute) unbarriered_compute_written_resources.erase(handle);

        last_image_layouts[handle] = new_layout;
    }

    for (const auto& handle: target_resources) {
        Logger::debug("inserting pre-partition (target) barrier for texture: {}", resource_manager->get_name(handle));

        if (handle == FINAL_IMAGE_HANDLE) {
            barriers.insert(vk::ImageMemoryBarrier2 {
                .srcStageMask = vk::PipelineStageFlagBits2::eAllGraphics,
                .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eAllGraphics,
                .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .image = swap_chain->get_current_rendered_image(),
                .subresourceRange = {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .levelCount = 1,
                    .layerCount = 1,
                }
            });

            continue;
        }

        const auto& texture = resource_manager->get<Image>(handle);
        const bool is_depth_texture = vk::hasDepthComponent(texture.get_format());
        const auto expected_layout = is_depth_texture
                                     ? vk::ImageLayout::eDepthAttachmentOptimal
                                     : vk::ImageLayout::eColorAttachmentOptimal;
        const auto current_layout = last_image_layouts.at(handle);

        if (current_layout == expected_layout) continue;

        barriers.insert(vk::ImageMemoryBarrier2 {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = is_depth_texture
                             ? vk::AccessFlagBits2::eDepthStencilAttachmentWrite
                             : vk::AccessFlagBits2::eColorAttachmentWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllGraphics,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = current_layout,
            .newLayout = expected_layout,
            .image = *texture,
            .subresourceRange = {
                .aspectMask = is_depth_texture ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = texture.get_mip_levels(),
                .baseArrayLayer = 0,
                .layerCount = 1,
            }
        });

        last_image_layouts[handle] = expected_layout;
    }

    for (const auto& handle: compute_read_resources) {
        const bool is_unbarriered_gfx = unbarriered_gfx_written_resources.contains(handle);
        const bool is_unbarriered_compute = unbarriered_compute_written_resources.contains(handle);
        if (!is_unbarriered_gfx && !is_unbarriered_compute) continue;

        Logger::debug("inserting pre-partition (compute-read) barrier for texture: {}", resource_manager->get_name(handle));

        const auto& texture = resource_manager->get<Image>(handle);
        const bool is_depth_texture = vk::hasDepthComponent(texture.get_format());
        const auto old_layout = last_image_layouts[handle];
        constexpr auto new_layout = vk::ImageLayout::eShaderReadOnlyOptimal;

        barriers.insert(vk::ImageMemoryBarrier2 {
            .srcStageMask = is_unbarriered_gfx
                            ? (is_depth_texture
                                ? vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests
                                : vk::PipelineStageFlagBits2::eColorAttachmentOutput)
                            : vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = is_unbarriered_gfx
                             ? (is_depth_texture
                                 ? vk::AccessFlagBits2::eDepthStencilAttachmentWrite
                                 : vk::AccessFlagBits2::eColorAttachmentWrite)
                             : vk::AccessFlagBits2::eShaderWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = old_layout,
            .newLayout = new_layout,
            .image = *texture,
            .subresourceRange = {
                .aspectMask = is_depth_texture ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = texture.get_mip_levels(),
                .baseArrayLayer = 0,
                .layerCount = 1,
            }
        });

        if (is_unbarriered_gfx) unbarriered_gfx_written_resources.erase(handle);
        else if (is_unbarriered_compute) unbarriered_compute_written_resources.erase(handle);

        last_image_layouts[handle] = new_layout;
    }

    for (const auto& handle: compute_write_resources) {
        Logger::debug("inserting pre-partition (compute-write) barrier for texture: {}", resource_manager->get_name(handle));

        const auto& texture = resource_manager->get<Image>(handle);
        const bool is_depth_texture = vk::hasDepthComponent(texture.get_format());
        constexpr auto expected_layout = vk::ImageLayout::eGeneral;
        const auto current_layout = last_image_layouts.at(handle);

        if (current_layout == expected_layout) continue;

        barriers.insert(vk::ImageMemoryBarrier2 {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = is_depth_texture
                             ? vk::AccessFlagBits2::eDepthStencilAttachmentWrite
                             : vk::AccessFlagBits2::eColorAttachmentWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
            .oldLayout = current_layout,
            .newLayout = expected_layout,
            .image = *texture,
            .subresourceRange = {
                .aspectMask = is_depth_texture ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = texture.get_mip_levels(),
                .baseArrayLayer = 0,
                .layerCount = 1,
            }
        });

        last_image_layouts[handle] = expected_layout;
    }

    barriers.record_cmd(command_buffer);
}

void VulkanRenderer::record_compute_node_commands(const RenderNodeHandle node_handle) {
    const auto &command_buffer = *frame_resources[ctx.current_frame_idx].main_cmd_buffer;
    const auto &time_query_pool = *frame_resources[ctx.current_frame_idx].time_query_pool;
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

    command_buffer.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT { .pLabelName = node.name.c_str(), });

    command_buffer.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe, *time_query_pool, current_query_idx++);

    ComputePassContext ctx{
        command_buffer,
        *resource_manager,
        **bindless_descriptor_set
    };
    node.body(ctx);

    command_buffer.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, *time_query_pool, current_query_idx++);

    // todo
    // regenerate mipmaps for each target that had them
    // record_regenerate_mipmaps_commands(node_handle);

    command_buffer.endDebugUtilsLabelEXT();
}

auto VulkanRenderer::gather_attachment_resources() const -> set<ResourceHandle> {
    set<ResourceHandle> result;

    for (const auto& [node_handle, node_info] : render_graph->nodes()) {
        if (!node_info.is_graphics()) continue;

        const auto& node_gfx = node_info.get_graphics();
        result.insert_range(node_gfx.color_targets);
        if (node_gfx.depth_target) result.insert(*node_gfx.depth_target);
    }

    return result;
}

auto VulkanRenderer::gather_compute_accessed_resources() const -> set<ResourceHandle> {
    set<ResourceHandle> result;

    for (const auto& [node_handle, node_info] : render_graph->nodes()) {
        if (!node_info.is_compute()) continue;

        result.insert_range(node_info.get_compute().bound_read_resources);
        result.insert_range(node_info.get_compute().bound_write_resources);
    }

    return result;
}

auto VulkanRenderer::has_swapchain_target(const RenderNodeHandle node_handle) const -> bool {
    return render_graph->nodes().at(node_handle).get_graphics()
            .get_all_targets_set()
            .contains(FINAL_IMAGE_HANDLE);
}

auto VulkanRenderer::is_first_node_targetting_final_image(const RenderNodeHandle node_handle) const -> bool {
    if (!has_swapchain_target(node_handle)) return false;

    const auto flattened = std::ranges::join_view(partitioned_nodes);

    auto first_it = std::ranges::find_if(flattened, [&](const RenderNodeHandle &handle) {
        return has_swapchain_target(handle);
    });

    return first_it == flattened.end() || *first_it == node_handle;
}

auto VulkanRenderer::get_node_target_extent(const RenderNodeHandle node_handle) const -> vk::Extent2D {
    const auto &gfx_node_info = render_graph->nodes().at(node_handle).get_graphics();

    if (has_swapchain_target(node_handle)) {
        return swap_chain->get_extent();
    }

    if (gfx_node_info.color_targets.empty()) {
        return resource_manager->get<Image>(*gfx_node_info.depth_target).get_extent_2d();
    }

    return resource_manager->get<Image>(gfx_node_info.color_targets[0]).get_extent_2d();
}

auto VulkanRenderer::get_target_color_format(const ResourceHandle resource_handle) const -> vk::Format {
    if (resource_handle == FINAL_IMAGE_HANDLE) {
        return swap_chain->get_image_format();
    }
    return resource_manager->get<Image>(resource_handle).get_format();
}

auto VulkanRenderer::get_target_depth_format(const ResourceHandle resource_handle) const -> vk::Format {
    if (resource_handle == FINAL_IMAGE_HANDLE) {
        return swap_chain->get_depth_format();
    }
    return resource_manager->get<Image>(resource_handle).get_format();
}

auto VulkanRenderer::get_node_target_handles(const RenderNodeHandle node_handle) const -> vector<ResourceHandle> {
    const auto& node = render_graph->nodes().at(node_handle).get_graphics();
    vector<ResourceHandle> handles = node.color_targets;
    if (node.depth_target) handles.emplace_back(*node.depth_target);
    return handles;
}

// ==================== render loop ====================

void VulkanRenderer::tick(const float delta_time) {
    (void) delta_time;
}

void VulkanRenderer::add_frame_begin_action(FrameBeginCallback &&callback) {
    repeated_frame_begin_actions.emplace_back(std::move(callback));
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
    auto& frame = frame_resources[ctx.current_frame_idx];
    const auto &sync = frame.sync;

    utils::sync::wait(ctx, *sync.render_finished_timeline_sem);

    // handle timing queries for profiling. these are behind by MAX_FRAMES_IN_FLIGHT frames
    {
        if (frame.time_query_pool) {
            frame.prev_time_query_results = frame.time_query_pool->get_results();
        }

        const auto node_count = std::ranges::distance(partitioned_nodes | std::ranges::views::join);
        frame_resources[ctx.current_frame_idx].time_query_pool = make_unique<QueryPool>(
                ctx,
                vk::QueryType::eTimestamp,
                2 * node_count
            );
        current_query_idx = 0;
    }

    do_frame_begin_actions();

    const auto &[result, image_index] = swap_chain->acquire_next_image(**sync.image_available_sem);

    if (result == vk::Result::eErrorOutOfDateKHR) {
        recreate_swap_chain();
        return false;
    }
    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
        Logger::error("failed to acquire swap chain image!");
    }

    unbarriered_gfx_written_resources.clear();
    unbarriered_compute_written_resources.clear();

    return true;
}

void VulkanRenderer::end_frame() {
    auto& frame = frame_resources[ctx.current_frame_idx];
    auto& sync = frame.sync;

    ++(*sync.render_finished_timeline_sem);

    QueueSubmission submission = QueueSubmissionBuilder()
        .with_wait_semaphores(*sync.image_available_sem)
        .with_signal_semaphores(*sync.render_finished_timeline_sem, *sync.ready_to_present_sem)
        .with_command_buffers(std::span { &*frame.main_cmd_buffer, 1 })
        .create();
    ctx.graphics_queue->submit(std::move(submission));

    const vk::Result present_result = present_queue->present(*swap_chain, *sync.ready_to_present_sem);

    const bool did_resize = present_result == vk::Result::eErrorOutOfDateKHR
                            || present_result == vk::Result::eSuboptimalKHR
                            || framebuffer_resized;
    if (did_resize) {
        framebuffer_resized = false;
        recreate_swap_chain();
    } else if (present_result != vk::Result::eSuccess) {
        Logger::error("failed to present swap chain image!");
    }

    frame.prev_node_names.clear();
    for (const auto& node : partitioned_nodes | std::ranges::views::join) {
        frame.prev_node_names.emplace_back(render_graph->nodes().at(node).name());
    }

    ctx.current_frame_idx = (ctx.current_frame_idx + 1) % MAX_FRAMES_IN_FLIGHT;
}
} // zrx
