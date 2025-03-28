#include "renderer.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <optional>
#include <vector>
#include <filesystem>
#include <array>
#include <random>

#include "camera.hpp"
#include "gui/gui.hpp"
#include "mesh/model.hpp"
#include "mesh/vertex.hpp"
#include "src/utils/glfw-statics.hpp"
#include "vk/image.hpp"
#include "vk/buffer.hpp"
#include "vk/swapchain.hpp"
#include "vk/cmd.hpp"
#include "vk/descriptor.hpp"
#include "vk/pipeline.hpp"
#include "vk/accel-struct.hpp"
#include "vk/ctx.hpp"

#include <vk-bootstrap/VkBootstrap.h>

/**
 * Information held in the fragment shader's uniform buffer.
 * This (obviously) has to exactly match the corresponding definition in the fragment shader.
 */
struct GraphicsUBO {
    struct WindowRes {
        uint32_t window_width;
        uint32_t window_height;
    };

    struct Matrices {
        glm::mat4 model;
        glm::mat4 view;
        glm::mat4 proj;
        glm::mat4 view_inverse;
        glm::mat4 proj_inverse;
        glm::mat4 vp_inverse;
        glm::mat4 static_view;
        glm::mat4 cubemap_capture_views[6];
        glm::mat4 cubemap_capture_proj;
    };

    struct MiscData {
        float debug_number;
        float z_near;
        float z_far;
        uint32_t use_ssao;
        float light_intensity;
        glm::vec3 light_dir;
        glm::vec3 light_color;
        glm::vec3 camera_pos;
    };

    alignas(16) WindowRes window{};
    alignas(16) Matrices matrices{};
    alignas(16) MiscData misc{};
};

namespace zrx {
VulkanRenderer::VulkanRenderer() {
    constexpr int INIT_WINDOW_WIDTH = 1200;
    constexpr int INIT_WINDOW_HEIGHT = 800;

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(INIT_WINDOW_WIDTH, INIT_WINDOW_HEIGHT, "Rayzor", nullptr, nullptr);

    init_glfw_user_pointer(window);
    auto *user_data = static_cast<GlfwStaticUserData *>(glfwGetWindowUserPointer(window));
    if (!user_data) throw std::runtime_error("unexpected null window user pointer");
    user_data->renderer = this;

    glfwSetFramebufferSizeCallback(window, framebuffer_resize_callback);

    const auto vkb_instance = create_instance();
    create_surface();
    const auto vkb_physical_device = pick_physical_device(vkb_instance);
    create_logical_device(vkb_physical_device);

    ctx.allocator = make_unique<VmaAllocatorWrapper>(**ctx.physical_device, **ctx.device, **instance);

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

    create_screen_space_quad_vertex_buffer();
    create_skybox_vertex_buffer();

    create_sync_objects();

    init_imgui();
}

VulkanRenderer::~VulkanRenderer() {
    glfwDestroyWindow(window);
}

void VulkanRenderer::framebuffer_resize_callback(GLFWwindow *window, const int width, const int height) {
    (void) (width + height);
    const auto user_data = static_cast<GlfwStaticUserData *>(glfwGetWindowUserPointer(window));
    if (!user_data) throw std::runtime_error("unexpected null window user pointer");
    user_data->renderer->framebuffer_resized = true;
}

// ==================== instance creation ====================

vkb::Instance VulkanRenderer::create_instance() {
    auto instance_result = vkb::InstanceBuilder().set_app_name("Rayzor")
            .request_validation_layers()
            .enable_layer("VK_LAYER_KHRONOS_validation")
            .set_debug_callback([](
            const VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
            const VkDebugUtilsMessageTypeFlagsEXT messageType,
            const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
            void *p_user_data) -> VkBool32 {
                    const auto severity = vkb::to_string_message_severity(messageSeverity);
                    const auto type = vkb::to_string_message_type(messageType);

                    std::stringstream ss;
                    ss << "[" << severity << ": " << type << "]\n" << pCallbackData->pMessage << "\n";

                    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
                        std::cerr << ss.str() << std::endl;
                    } else {
                        std::cout << ss.str() << std::endl;
                    }

                    return VK_FALSE;
                })
            .require_api_version(1, 3)
            .set_minimum_instance_version(1, 3)
            .enable_extensions(get_required_extensions())
            .build();

    if (!instance_result) {
        throw std::runtime_error("failed to create instance: " + instance_result.error().message());
    }

    instance = make_unique<vk::raii::Instance>(vk_ctx, instance_result.value().instance);

    return instance_result.value();
}

vector<const char *> VulkanRenderer::get_required_extensions() {
    uint32_t glfw_extension_count = 0;
    const char **glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

    vector extensions(glfw_extensions, glfw_extensions + glfw_extension_count);

    if (ENABLE_VALIDATION_LAYERS) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

// ==================== startup ====================

void VulkanRenderer::create_surface() {
    VkSurfaceKHR _surface;

    if (glfwCreateWindowSurface(**instance, window, nullptr, &_surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }

    surface = make_unique<vk::raii::SurfaceKHR>(*instance, _surface);
}

vkb::PhysicalDevice VulkanRenderer::pick_physical_device(const vkb::Instance &vkb_instance) {
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
                // .descriptorBindingUniformBufferUpdateAfterBind = vk::True,
                .descriptorBindingSampledImageUpdateAfterBind = vk::True,
                .descriptorBindingStorageBufferUpdateAfterBind = vk::True,
                .descriptorBindingPartiallyBound = vk::True,
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
        throw std::runtime_error("failed to select physical device: " + physical_device_result.error().message());
    }

    ctx.physical_device = make_unique<vk::raii::PhysicalDevice>(
        *instance, physical_device_result.value().physical_device);
    msaa_sample_count = get_max_usable_sample_count();

    return physical_device_result.value();
}

void VulkanRenderer::create_logical_device(const vkb::PhysicalDevice &vkb_physical_device) {
    auto device_result = vkb::DeviceBuilder(vkb_physical_device).build();

    if (!device_result) {
        throw std::runtime_error("failed to select logical device: " + device_result.error().message());
    }

    ctx.device = make_unique<vk::raii::Device>(*ctx.physical_device, device_result.value().device);

    auto graphics_queue_result = device_result.value().get_queue(vkb::QueueType::graphics);
    auto graphics_queue_index_result = device_result.value().get_queue_index(vkb::QueueType::graphics);
    if (!graphics_queue_result || !graphics_queue_index_result) {
        throw std::runtime_error("failed to get graphics queue: " + device_result.error().message());
    }

    auto present_queue_result = device_result.value().get_queue(vkb::QueueType::present);
    auto present_queue_index_result = device_result.value().get_queue_index(vkb::QueueType::present);
    if (!present_queue_result || !present_queue_index_result) {
        throw std::runtime_error("failed to get present queue: " + device_result.error().message());
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

void VulkanRenderer::create_skybox_vertex_buffer() {
    skybox_vertex_buffer = create_local_buffer<SkyboxVertex>(skybox_vertices, vk::BufferUsageFlagBits::eVertexBuffer);
}

void VulkanRenderer::create_screen_space_quad_vertex_buffer() {
    screen_space_quad_vertex_buffer = create_local_buffer<ScreenSpaceQuadVertex>(
        screen_space_quad_vertices,
        vk::BufferUsageFlagBits::eVertexBuffer
    );
}

template<typename ElemType>
unique_ptr<Buffer>
VulkanRenderer::create_local_buffer(const vector<ElemType> &contents, const vk::BufferUsageFlags usage) {
    const vk::DeviceSize buffer_size = sizeof(contents[0]) * contents.size();

    Buffer staging_buffer{
        **ctx.allocator,
        buffer_size,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    };

    void *data = staging_buffer.map();
    memcpy(data, contents.data(), static_cast<size_t>(buffer_size));
    staging_buffer.unmap();

    auto result_buffer = make_unique<Buffer>(
        **ctx.allocator,
        buffer_size,
        vk::BufferUsageFlagBits::eTransferDst | usage,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    result_buffer->copy_from_buffer(ctx, staging_buffer, buffer_size);

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
        frame_resources[i].graphics_cmd_buffer =
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
    render_graph_info.render_graph = make_unique<RenderGraph>(graph);

    create_render_graph_resources();

    const auto topo_sorted_handles = render_graph_info.render_graph->get_topo_sorted();
    const uint32_t n_nodes = topo_sorted_handles.size();

    for (uint32_t i = 0; i < n_nodes; i++) {
        const auto node_handle = topo_sorted_handles[i];
        auto render_infos = create_node_render_infos(node_handle);

        render_graph_info.topo_sorted_nodes.emplace_back(RenderNodeResources{
            .handle = node_handle,
            .render_infos = std::move(render_infos),
        });
    }

    repeated_frame_begin_actions = render_graph_info.render_graph->frame_begin_callbacks;
}

void VulkanRenderer::create_render_graph_resources() {
    for (const auto &[handle, description]: render_graph_info.render_graph->uniform_buffers) {
        render_graph_ubos.emplace(handle, utils::buf::create_uniform_buffer(ctx, description.size));
    }

    for (const auto &[handle, description]: render_graph_info.render_graph->external_tex_resources) {
        auto builder = TextureBuilder()
                .with_flags(description.tex_flags)
                .from_paths(description.paths)
                .use_format(description.format)
                .use_usage(vk::ImageUsageFlagBits::eTransferSrc
                           | vk::ImageUsageFlagBits::eTransferDst
                           | vk::ImageUsageFlagBits::eSampled
                           | utils::img::get_format_attachment_type(description.format));

        if (description.paths.size() > 1 && !(description.tex_flags & vk::TextureFlagBitsZRX::CUBEMAP))
            builder.as_separate_channels();
        if (description.swizzle)
            builder.with_swizzle(*description.swizzle);

        render_graph_textures.emplace(handle, builder.create(ctx));
    }

    for (const auto &[handle, description]: render_graph_info.render_graph->empty_tex_resources) {
        auto extent = description.extent;

        if (extent.width == 0 && extent.height == 0) {
            extent = swap_chain->get_extent();
        }

        auto builder = TextureBuilder()
                .with_flags(description.tex_flags)
                .as_uninitialized({extent.width, extent.height, 1u})
                .use_format(description.format)
                .use_usage(vk::ImageUsageFlagBits::eTransferSrc
                           | vk::ImageUsageFlagBits::eTransferDst
                           | vk::ImageUsageFlagBits::eSampled
                           | utils::img::get_format_attachment_type(description.format));

        render_graph_textures.emplace(handle, builder.create(ctx));
    }

    for (const auto &[handle, description]: render_graph_info.render_graph->transient_tex_resources) {
        auto extent = description.extent;

        if (extent.width == 0 && extent.height == 0) {
            extent = swap_chain->get_extent();
        }

        auto builder = TextureBuilder()
                .with_flags(description.tex_flags)
                .as_uninitialized({extent.width, extent.height, 1u})
                .use_format(description.format)
                .use_usage(vk::ImageUsageFlagBits::eTransientAttachment
                           | utils::img::get_format_attachment_type(description.format));

        render_graph_textures.emplace(handle, builder.create(ctx));
    }
}

vector<shared_ptr<DescriptorSet>>
VulkanRenderer::create_graph_descriptor_sets(const ResourceHandle pipeline_handle) const {
    const auto &pipeline_info = render_graph_info.render_graph->pipelines.at(pipeline_handle);
    const auto &set_descs = pipeline_info.descriptor_set_descs;
    vector<shared_ptr<DescriptorSet> > descriptor_sets;

    for (size_t i = 0; i < set_descs.size(); i++) {
        const auto &set_desc = set_descs[i];
        DescriptorLayoutBuilder builder;

        for (size_t j = 0; j < set_desc.size(); j++) {
            if (std::holds_alternative<std::monostate>(set_desc[j])) continue;

            bool is_ubo_descriptor = false;
            bool is_tex_descriptor = false;
            vk::DescriptorType type{};
            vk::ShaderStageFlags stages{};
            uint32_t descriptor_count = 1;

            if (std::holds_alternative<ResourceHandle>(set_desc[j])) {
                const auto res_handle = std::get<ResourceHandle>(set_desc[j]);
                is_ubo_descriptor = render_graph_ubos.contains(res_handle);
                is_tex_descriptor = render_graph_textures.contains(res_handle);
            } else if (std::holds_alternative<ResourceHandleArray>(set_desc[j])) {
                const auto &res_handles = std::get<ResourceHandleArray>(set_desc[j]);
                is_ubo_descriptor = std::ranges::any_of(res_handles, [&](auto res_handle) {
                    return render_graph_ubos.contains(res_handle);
                });
                is_tex_descriptor = std::ranges::any_of(res_handles, [&](auto res_handle) {
                    return render_graph_textures.contains(res_handle);
                });
                descriptor_count = res_handles.size();
            }

            if (is_ubo_descriptor && is_tex_descriptor) {
                throw std::runtime_error("ambiguous resource handle type");
            }
            if (is_ubo_descriptor) {
                type = vk::DescriptorType::eUniformBuffer;
            } else if (is_tex_descriptor) {
                type = vk::DescriptorType::eCombinedImageSampler;
            } else {
                throw std::runtime_error("unknown resource handle");
            }

            if (
                vert_set_descs.size() >= i + 1
                && vert_set_descs[i].size() >= j + 1
                && !std::holds_alternative<std::monostate>(vert_set_descs[i][j])
            ) {
                stages |= vk::ShaderStageFlagBits::eVertex;
            }

            if (
                frag_set_descs.size() >= i + 1
                && frag_set_descs[i].size() >= j + 1
                && !std::holds_alternative<std::monostate>(frag_set_descs[i][j])
            ) {
                stages |= vk::ShaderStageFlagBits::eFragment;
            }

            builder.add_binding(type, stages, descriptor_count);
        }

        auto layout = std::make_shared<vk::raii::DescriptorSetLayout>(builder.create(ctx));
        auto descriptor_set = std::make_shared<DescriptorSet>(
            utils::desc::create_descriptor_set(ctx, *descriptor_pool, layout));
        descriptor_sets.emplace_back(descriptor_set);
    }

    for (size_t i = 0; i < set_descs.size(); i++) {
        const auto &set_desc = set_descs[i];
        const auto &descriptor_set = descriptor_sets[i];

        for (uint32_t binding = 0; binding < set_desc.size(); binding++) {
            if (std::holds_alternative<ResourceHandle>(set_desc[binding])) {
                const auto res_handle = std::get<ResourceHandle>(set_desc[binding]);
                queue_set_update_with_handle(*descriptor_set, res_handle, binding);
            } else if (std::holds_alternative<ResourceHandleArray>(set_desc[binding])) {
                const auto &res_handles = std::get<ResourceHandleArray>(set_desc[binding]);
                for (uint32_t array_element = 0; array_element < res_handles.size(); array_element++) {
                    queue_set_update_with_handle(*descriptor_set, res_handles[array_element], binding, array_element);
                }
            }
        }

        descriptor_set->commit_updates(ctx);
    }

    return descriptor_sets;
}

GraphicsPipelineBuilder
VulkanRenderer::create_graph_pipeline_builder(const ResourceHandle pipeline_handle,
                                              const vector<shared_ptr<DescriptorSet>> &descriptor_sets) const {
    const auto &pipeline_info = render_graph_info.render_graph->pipelines.at(pipeline_handle);

    vector<vk::Format> color_formats;
    for (const auto &format_variant: pipeline_info.color_formats) {
        const vk::Format format = std::holds_alternative<vk::Format>(format_variant)
                                    ? std::get<vk::Format>(format_variant)
                                    : swap_chain->get_image_format();
        color_formats.push_back(format);
    }

    vector<vk::DescriptorSetLayout> descriptor_set_layouts;
    for (const auto &set: descriptor_sets) {
        descriptor_set_layouts.emplace_back(*set->get_layout());
    }

    auto builder = GraphicsPipelineBuilder()
            .with_vertex_shader(pipeline_info.vertex_path)
            .with_fragment_shader(pipeline_info.fragment_path)
            .with_vertices(
                pipeline_info.binding_descriptions,
                pipeline_info.attribute_descriptions
            )
            .with_rasterizer({
                .polygonMode = vk::PolygonMode::eFill,
                .cullMode = pipeline_info.custom_properties.cull_mode,
                .frontFace = vk::FrontFace::eCounterClockwise,
                .lineWidth = 1.0f,
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

/*
vector<shared_ptr<DescriptorSet> >
VulkanRenderer::create_node_descriptor_sets(const RenderNodeHandle node_handle) const {
    const auto &node_info = render_graph_info.render_graph->nodes.at(node_handle);
    const auto &vert_set_descs = node_info.vertex_shader->descriptor_set_descs;
    const auto &frag_set_descs = node_info.fragment_shader->descriptor_set_descs;

    auto merged_set_descs = vert_set_descs;
    if (merged_set_descs.size() < frag_set_descs.size()) {
        merged_set_descs.resize(frag_set_descs.size());
    }

    for (size_t i = 0; i < frag_set_descs.size(); i++) {
        const auto &frag_set_desc = frag_set_descs[i];

        if (merged_set_descs[i].size() < frag_set_desc.size()) {
            merged_set_descs[i].resize(frag_set_desc.size());
        }

        for (size_t j = 0; j < frag_set_desc.size(); j++) {
            if (!std::holds_alternative<std::monostate>(frag_set_desc[j])) {
                if (
                    !std::holds_alternative<std::monostate>(merged_set_descs[i][j])
                    && merged_set_descs[i][j] != frag_set_desc[j]
                ) {
                    throw std::runtime_error("incompatible shader descriptor set bindings for node " + node_info.name);
                }

                merged_set_descs[i][j] = frag_set_desc[j];
            }
        }
    }

    vector<shared_ptr<DescriptorSet> > descriptor_sets;
    for (size_t i = 0; i < merged_set_descs.size(); i++) {
        const auto &set_desc = merged_set_descs[i];
        DescriptorLayoutBuilder builder;

        for (size_t j = 0; j < set_desc.size(); j++) {
            if (std::holds_alternative<std::monostate>(set_desc[j])) continue;

            bool is_ubo_descriptor = false;
            bool is_tex_descriptor = false;
            vk::DescriptorType type{};
            vk::ShaderStageFlags stages{};
            uint32_t descriptor_count = 1;

            if (std::holds_alternative<ResourceHandle>(set_desc[j])) {
                const auto res_handle = std::get<ResourceHandle>(set_desc[j]);
                is_ubo_descriptor = render_graph_ubos.contains(res_handle);
                is_tex_descriptor = render_graph_textures.contains(res_handle);
            } else if (std::holds_alternative<ResourceHandleArray>(set_desc[j])) {
                const auto &res_handles = std::get<ResourceHandleArray>(set_desc[j]);
                is_ubo_descriptor = std::ranges::any_of(res_handles, [&](auto res_handle) {
                    return render_graph_ubos.contains(res_handle);
                });
                is_tex_descriptor = std::ranges::any_of(res_handles, [&](auto res_handle) {
                    return render_graph_textures.contains(res_handle);
                });
                descriptor_count = res_handles.size();
            }

            if (is_ubo_descriptor && is_tex_descriptor) {
                throw std::runtime_error("ambiguous resource handle type");
            }
            if (is_ubo_descriptor) {
                type = vk::DescriptorType::eUniformBuffer;
            } else if (is_tex_descriptor) {
                type = vk::DescriptorType::eCombinedImageSampler;
            } else {
                throw std::runtime_error("unknown resource handle");
            }

            if (
                vert_set_descs.size() >= i + 1
                && vert_set_descs[i].size() >= j + 1
                && !std::holds_alternative<std::monostate>(vert_set_descs[i][j])
            ) {
                stages |= vk::ShaderStageFlagBits::eVertex;
            }

            if (
                frag_set_descs.size() >= i + 1
                && frag_set_descs[i].size() >= j + 1
                && !std::holds_alternative<std::monostate>(frag_set_descs[i][j])
            ) {
                stages |= vk::ShaderStageFlagBits::eFragment;
            }

            builder.add_binding(type, stages, descriptor_count);
        }

        auto layout = std::make_shared<vk::raii::DescriptorSetLayout>(builder.create(ctx));
        auto descriptor_set = std::make_shared<DescriptorSet>(
            utils::desc::create_descriptor_set(ctx, *descriptor_pool, layout));
        descriptor_sets.emplace_back(descriptor_set);
    }

    for (size_t i = 0; i < merged_set_descs.size(); i++) {
        const auto &set_desc = merged_set_descs[i];
        const auto &descriptor_set = descriptor_sets[i];

        for (uint32_t binding = 0; binding < set_desc.size(); binding++) {
            if (std::holds_alternative<ResourceHandle>(set_desc[binding])) {
                const auto res_handle = std::get<ResourceHandle>(set_desc[binding]);
                queue_set_update_with_handle(*descriptor_set, res_handle, binding);
            } else if (std::holds_alternative<ResourceHandleArray>(set_desc[binding])) {
                const auto &res_handles = std::get<ResourceHandleArray>(set_desc[binding]);
                for (uint32_t array_element = 0; array_element < res_handles.size(); array_element++) {
                    queue_set_update_with_handle(*descriptor_set, res_handles[array_element], binding, array_element);
                }
            }
        }

        descriptor_set->commit_updates(ctx);
    }

    return descriptor_sets;
}
*/

void VulkanRenderer::queue_set_update_with_handle(DescriptorSet &descriptor_set, const ResourceHandle res_handle,
                                                  const uint32_t binding, const uint32_t array_element) const {
    if (render_graph_ubos.contains(res_handle)) {
        const auto &buffer = render_graph_ubos.at(res_handle);
        descriptor_set.queue_update(
            binding,
            *buffer,
            vk::DescriptorType::eUniformBuffer,
            buffer->get_size(),
            0,
            array_element
        );
    } else if (render_graph_textures.contains(res_handle)) {
        const auto &texture = render_graph_textures.at(res_handle);
        descriptor_set.queue_update(ctx, binding, *texture, vk::DescriptorType::eCombinedImageSampler, array_element);
    }
}

/*
GraphicsPipelineBuilder VulkanRenderer::create_node_pipeline_builder(
    const RenderNodeHandle node_handle,
    const vector<shared_ptr<DescriptorSet> > &descriptor_sets
) const {
    const auto &node_info = render_graph_info.render_graph->nodes.at(node_handle);

    vector<vk::Format> color_formats;
    for (const auto &target_handle: node_info.color_targets) {
        color_formats.push_back(get_target_color_format(target_handle));
    }

    vector<vk::DescriptorSetLayout> descriptor_set_layouts;
    for (const auto &set: descriptor_sets) {
        descriptor_set_layouts.emplace_back(*set->get_layout());
    }

    auto builder = GraphicsPipelineBuilder()
            .with_vertex_shader(node_info.vertex_shader->path)
            .with_fragment_shader(node_info.fragment_shader->path)
            .with_vertices(
                node_info.vertex_shader->binding_descriptions,
                node_info.vertex_shader->attribute_descriptions
            )
            .with_rasterizer({
                .polygonMode = vk::PolygonMode::eFill,
                .cullMode = node_info.custom_properties.cull_mode,
                .frontFace = vk::FrontFace::eCounterClockwise,
                .lineWidth = 1.0f,
            })
            .with_multisampling({
                .rasterizationSamples = node_info.custom_properties.use_msaa
                                        ? get_msaa_sample_count()
                                        : vk::SampleCountFlagBits::e1,
                .minSampleShading = 1.0f,
            })
            .with_descriptor_layouts(descriptor_set_layouts)
            .with_color_formats(color_formats);

    if (node_info.depth_target) {
        builder.with_depth_format(get_target_depth_format(*node_info.depth_target));
    } else {
        builder.with_depth_stencil({
            .depthTestEnable = vk::False,
            .depthWriteEnable = vk::False,
        });
    }

    if (node_info.custom_properties.multiview_count > 1) {
        builder.for_views(node_info.custom_properties.multiview_count);
    }

    return builder;
}
*/

vector<RenderInfo> VulkanRenderer::create_node_render_infos(const RenderNodeHandle node_handle) const {
    const auto &node_info = render_graph_info.render_graph->nodes.at(node_handle);
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
                if (color_target_handle == FINAL_IMAGE_RESOURCE_HANDLE) {
                    color_targets.emplace_back(std::move(swap_chain_targets.color_target));
                } else {
                    const auto &target_texture = render_graph_textures.at(color_target_handle);
                    color_targets.emplace_back(target_texture->get_image().get_view(ctx), target_texture->get_format());
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
        std::optional<RenderTarget> depth_target;

        for (auto color_target_handle: node_info.color_targets) {
            const auto &target_texture = render_graph_textures.at(color_target_handle);
            color_targets.emplace_back(target_texture->get_image().get_mip_view(ctx, 0),
                                       target_texture->get_format());
        }

        if (node_info.depth_target) {
            const auto &target_texture = render_graph_textures.at(*node_info.depth_target);
            depth_target = RenderTarget(target_texture->get_image().get_layer_mip_view(ctx, 0, 0),
                                        target_texture->get_format());
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
        record_graph_commands();
        end_frame();
    }
}

void VulkanRenderer::record_graph_commands() const {
    const auto &command_buffer = *frame_resources[current_frame_idx].graphics_cmd_buffer;

    command_buffer.begin({});

    swap_chain->transition_to_attachment_layout(command_buffer);

    for (const auto &node_resources: render_graph_info.topo_sorted_nodes) {
        if (should_run_node_pass(node_resources.handle)) {
            record_node_commands(node_resources);
        }
    }

    swap_chain->transition_to_present_layout(command_buffer);

    command_buffer.end();
}

void VulkanRenderer::record_node_commands(const RenderNodeResources &node_resources) const {
    const auto &command_buffer = *frame_resources[current_frame_idx].graphics_cmd_buffer;
    const auto &node = render_graph_info.render_graph->nodes.at(node_resources.handle);

    std::cout << "recording node: " << node.name << "\n";

    // if size > 1, then this means that this pass (node) draws to the swapchain image
    // and thus benefits from double or triple buffering
    const size_t subresource_index = node_resources.render_infos.size() == 1 ? 0 : current_frame_idx;
    const auto &node_render_info = node_resources.render_infos[subresource_index];

    command_buffer.beginRendering(node_render_info.get(
        get_node_target_extent(node_resources),
        node.custom_properties.multiview_count)
    );
    record_node_rendering_commands(node_resources);
    command_buffer.endRendering();

    // regenerate mipmaps for each target that had them
    record_regenerate_mipmaps_commands(node_resources);

    // todo - this should have more refined logic
    // add barrier to the target image if it will be sampled
    // record_pre_sample_commands(node_resources);
}

void VulkanRenderer::record_node_rendering_commands(const RenderNodeResources &node_resources) const {
    const auto &command_buffer = *frame_resources[current_frame_idx].graphics_cmd_buffer;
    auto &[handle, render_infos] = node_resources;
    const auto &node_info = render_graph_info.render_graph->nodes.at(handle);

    utils::cmd::set_dynamic_states(command_buffer, get_node_target_extent(node_resources));

    RenderPassContext ctx{
        command_buffer,
        render_graph_models,
        render_graph_pipelines,
        *screen_space_quad_vertex_buffer,
        *skybox_vertex_buffer
    };
    node_info.body(ctx);
}

void VulkanRenderer::record_regenerate_mipmaps_commands(const RenderNodeResources &node_resources) const {
    const auto &command_buffer = *frame_resources[current_frame_idx].graphics_cmd_buffer;
    const auto &node = render_graph_info.render_graph->nodes.at(node_resources.handle);

    for (const auto color_target: node.color_targets) {
        if (color_target == FINAL_IMAGE_RESOURCE_HANDLE) continue;

        const auto &target_texture = render_graph_textures.at(color_target);
        if (target_texture->get_mip_levels() == 1) continue;

        target_texture->get_image().transition_layout(
            vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::ImageLayout::eTransferDstOptimal,
            command_buffer
        );

        target_texture->generate_mipmaps(ctx, vk::ImageLayout::eShaderReadOnlyOptimal);
    }
}

void VulkanRenderer::record_pre_sample_commands(const RenderNodeResources &node_resources) const {
    const auto &command_buffer = *frame_resources[current_frame_idx].graphics_cmd_buffer;
    const auto &node = render_graph_info.render_graph->nodes.at(node_resources.handle);

    for (const auto color_target: node.color_targets) {
        if (color_target == FINAL_IMAGE_RESOURCE_HANDLE) continue;

        const auto &target_texture = render_graph_textures.at(color_target);

        const vk::ImageMemoryBarrier2 image_memory_barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .image = **target_texture->get_image(),
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .levelCount = 1,
                .layerCount = 1,
            }
        };

        command_buffer.pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &image_memory_barrier,
        });
    }
}

bool VulkanRenderer::has_swapchain_target(const RenderNodeHandle handle) const {
    return render_graph_info.render_graph->nodes.at(handle)
            .get_all_targets_set()
            .contains(FINAL_IMAGE_RESOURCE_HANDLE);
}

bool VulkanRenderer::is_first_node_targetting_final_image(const RenderNodeHandle handle) const {
    if (!has_swapchain_target(handle)) return false;

    auto first_it = std::ranges::find_if(render_graph_info.topo_sorted_nodes, [&](const RenderNodeResources &res) {
        return has_swapchain_target(res.handle);
    });

    return first_it == render_graph_info.topo_sorted_nodes.end() || first_it->handle == handle;
}

bool VulkanRenderer::should_run_node_pass(const RenderNodeHandle handle) const {
    const auto &node = render_graph_info.render_graph->nodes.at(handle);
    return node.should_run_predicate ? (*node.should_run_predicate)() : true;
}

vk::Extent2D VulkanRenderer::get_node_target_extent(const RenderNodeResources &node_resources) const {
    const auto &node_info = render_graph_info.render_graph->nodes.at(node_resources.handle);

    return has_swapchain_target(node_resources.handle)
                        ? swap_chain->get_extent()
                        : render_graph_textures.at(node_info.color_targets[0])->get_image().get_extent_2d();
}

vk::Format VulkanRenderer::get_target_color_format(const ResourceHandle handle) const {
    if (handle == FINAL_IMAGE_RESOURCE_HANDLE) {
        return swap_chain->get_image_format();
    }
    return render_graph_textures.at(handle)->get_format();
}

vk::Format VulkanRenderer::get_target_depth_format(const ResourceHandle handle) const {
    if (handle == FINAL_IMAGE_RESOURCE_HANDLE) {
        return swap_chain->get_depth_format();
    }
    return render_graph_textures.at(handle)->get_format();
}

// ==================== render loop ====================

void VulkanRenderer::tick(const float delta_time) {
    (void) delta_time;
    // unused rn
}

void VulkanRenderer::do_frame_begin_actions() {
    const FrameBeginActionContext fba_ctx{render_graph_ubos, render_graph_textures, render_graph_models};

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

    if (ctx.device->waitSemaphores(wait_info, UINT64_MAX) != vk::Result::eSuccess) {
        throw std::runtime_error("waitSemaphores on renderFinishedTimeline failed");
    }

    do_frame_begin_actions();

    const auto &[result, image_index] = swap_chain->acquire_next_image(*sync.image_available_semaphore);

    if (result == vk::Result::eErrorOutOfDateKHR) {
        recreate_swap_chain();
        return false;
    }
    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

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

    const std::array signal_semaphores = {
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
            .pCommandBuffers = &**frame_resources[current_frame_idx].graphics_cmd_buffer,
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

    const std::array present_wait_semaphores = {**sync.ready_to_present_semaphore};

    const std::array image_indices = {swap_chain->get_current_image_index()};

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
        throw std::runtime_error("failed to present swap chain image!");
    }

    current_frame_idx = (current_frame_idx + 1) % MAX_FRAMES_IN_FLIGHT;
}
} // zrx
