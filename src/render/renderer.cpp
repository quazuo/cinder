#include "renderer.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <optional>
#include <vector>
#include <filesystem>
#include <array>
#include <mutex>
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

    // create_meshes_descriptor_set();
    // create_rt_target_texture();
    // create_rt_descriptor_sets();
    // create_rt_pipeline();

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
                    const auto severity = vkb::to_string_message_severity(
                        messageSeverity);
                    const auto type = vkb::to_string_message_type(messageType);

                    std::stringstream ss;
                    ss << "[" << severity << ": " << type << "]\n" << pCallbackData->pMessage << "\n\n";

                    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
                        std::cerr << ss.str();
                    } else {
                        std::cout << ss.str();
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

std::vector<const char *> VulkanRenderer::get_required_extensions() {
    uint32_t glfw_extension_count = 0;
    const char **glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

    std::vector extensions(glfw_extensions, glfw_extensions + glfw_extension_count);

    if (enable_validation_layers) {
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

// ==================== models ====================

void VulkanRenderer::load_model_with_materials(const std::filesystem::path &path) {
    wait_idle();

    model.reset();
    model = make_unique<Model>(ctx, path, true);

    const auto &materials = model->get_materials();

    for (uint32_t i = 0; i < materials.size(); i++) {
        const auto &material = materials[i];

        if (material.base_color) {
            materials_descriptor_set->queue_update<0>(*material.base_color, i);
        }

        if (material.normal) {
            materials_descriptor_set->queue_update<1>(*material.normal, i);
        }

        if (material.orm) {
            materials_descriptor_set->queue_update<2>(*material.orm, i);
        }
    }

    materials_descriptor_set->commit_updates();
}

void VulkanRenderer::load_model(const std::filesystem::path &path) {
    wait_idle();

    model.reset();
    model = make_unique<Model>(ctx, path, false);
}

// ==================== assets ====================

void VulkanRenderer::load_base_color_texture(const std::filesystem::path &path) {
    wait_idle();

    separate_material.base_color.reset();
    separate_material.base_color = TextureBuilder()
            .from_paths({path})
            .with_flags(vk::TextureFlagBitsZRX::MIPMAPS)
            .create(ctx);

    materials_descriptor_set->update_binding<0>(*separate_material.base_color);
}

void VulkanRenderer::load_normal_map(const std::filesystem::path &path) {
    wait_idle();

    separate_material.normal.reset();
    separate_material.normal = TextureBuilder()
            .use_format(vk::Format::eR8G8B8A8Unorm)
            .from_paths({path})
            .create(ctx);

    materials_descriptor_set->update_binding<1>(*separate_material.normal);
}

void VulkanRenderer::load_orm_map(const std::filesystem::path &path) {
    wait_idle();

    separate_material.orm.reset();
    separate_material.orm = TextureBuilder()
            .use_format(vk::Format::eR8G8B8A8Unorm)
            .from_paths({path})
            .create(ctx);

    materials_descriptor_set->update_binding<2>(*separate_material.orm);
}

void VulkanRenderer::load_orm_map(const std::filesystem::path &ao_path, const std::filesystem::path &roughness_path,
                                  const std::filesystem::path &metallic_path) {
    wait_idle();

    separate_material.orm.reset();
    separate_material.orm = TextureBuilder()
            .use_format(vk::Format::eR8G8B8A8Unorm)
            .as_separate_channels()
            .from_paths({ao_path, roughness_path, metallic_path})
            .with_swizzle({
                ao_path.empty() ? SwizzleComponent::MAX : SwizzleComponent::R,
                SwizzleComponent::G,
                metallic_path.empty() ? SwizzleComponent::ZERO : SwizzleComponent::B,
                SwizzleComponent::A
            })
            .with_flags(vk::TextureFlagBitsZRX::MIPMAPS)
            .create(ctx);

    materials_descriptor_set->update_binding<2>(*separate_material.orm);
}

void VulkanRenderer::load_rma_map(const std::filesystem::path &path) {
    wait_idle();

    separate_material.orm.reset();
    separate_material.orm = TextureBuilder()
            .with_swizzle({
                SwizzleComponent::B, SwizzleComponent::R, SwizzleComponent::G, SwizzleComponent::A
            })
            .use_format(vk::Format::eR8G8B8A8Unorm)
            .from_paths({path})
            .create(ctx);

    materials_descriptor_set->update_binding<2>(*separate_material.orm);
}

void VulkanRenderer::load_environment_map(const std::filesystem::path &path) {
    wait_idle();

    envmap_texture = TextureBuilder()
            .with_flags(vk::TextureFlagBitsZRX::MIPMAPS | vk::TextureFlagBitsZRX::HDR)
            .use_format(hdr_envmap_format)
            .from_paths({path})
            .with_sampler_address_mode(vk::SamplerAddressMode::eClampToEdge)
            .create(ctx);

    cubemap_capture_descriptor_set->update_binding<1>(*envmap_texture);

    capture_cubemap();
}

void VulkanRenderer::create_prepass_textures() {
    const auto &[width, height] = swap_chain->get_extent();

    const vk::Extent3D extent{
        .width = width,
        .height = height,
        .depth = 1
    };

    g_buffer_textures.pos = TextureBuilder()
            .as_uninitialized(extent)
            .use_format(prepass_color_format)
            .use_usage(vk::ImageUsageFlagBits::eTransferSrc
                       | vk::ImageUsageFlagBits::eTransferDst
                       | vk::ImageUsageFlagBits::eSampled
                       | vk::ImageUsageFlagBits::eColorAttachment)
            .create(ctx);

    g_buffer_textures.normal = TextureBuilder()
            .as_uninitialized(extent)
            .use_format(prepass_color_format)
            .use_usage(vk::ImageUsageFlagBits::eTransferSrc
                       | vk::ImageUsageFlagBits::eTransferDst
                       | vk::ImageUsageFlagBits::eSampled
                       | vk::ImageUsageFlagBits::eColorAttachment)
            .create(ctx);

    g_buffer_textures.depth = TextureBuilder()
            .as_uninitialized(extent)
            .use_format(swap_chain->get_depth_format())
            .use_usage(vk::ImageUsageFlagBits::eTransferSrc
                       | vk::ImageUsageFlagBits::eTransferDst
                       | vk::ImageUsageFlagBits::eSampled
                       | vk::ImageUsageFlagBits::eDepthStencilAttachment)
            .create(ctx);

    for (auto &res: frame_resources) {
        if (res.ssao_descriptor_set) {
            res.ssao_descriptor_set->queue_update<1>(*g_buffer_textures.depth)
                    .queue_update<2>(*g_buffer_textures.normal)
                    .queue_update<3>(*g_buffer_textures.pos)
                    .commit_updates();
        }
    }
}

void VulkanRenderer::create_skybox_texture() {
    skybox_texture = TextureBuilder()
            .with_flags(vk::TextureFlagBitsZRX::MIPMAPS | vk::TextureFlagBitsZRX::HDR | vk::TextureFlagBitsZRX::CUBEMAP)
            .as_uninitialized({2048, 2048, 1})
            .use_format(hdr_envmap_format)
            .use_usage(vk::ImageUsageFlagBits::eTransferSrc
                       | vk::ImageUsageFlagBits::eTransferDst
                       | vk::ImageUsageFlagBits::eSampled
                       | vk::ImageUsageFlagBits::eColorAttachment)
            .create(ctx);
}

static std::vector<glm::vec4> make_ssao_noise() {
    std::uniform_real_distribution<float> random_floats(0.0, 1.0); // random floats between [0.0, 1.0]
    std::default_random_engine generator;

    std::vector<glm::vec4> ssao_noise;
    for (unsigned int i = 0; i < 16; i++) {
        glm::vec4 noise(
            random_floats(generator) * 2.0 - 1.0,
            random_floats(generator) * 2.0 - 1.0,
            0.0f,
            0.0f
        );
        ssao_noise.push_back(noise);
    }

    return ssao_noise;
}

void VulkanRenderer::create_ssao_textures() {
    const auto attachment_usage_flags = vk::ImageUsageFlagBits::eTransferSrc
                                        | vk::ImageUsageFlagBits::eTransferDst
                                        | vk::ImageUsageFlagBits::eSampled
                                        | vk::ImageUsageFlagBits::eColorAttachment;

    const auto &[width, height] = swap_chain->get_extent();

    const vk::Extent3D extent{
        .width = width,
        .height = height,
        .depth = 1
    };

    ssao_texture = TextureBuilder()
            .as_uninitialized(extent)
            .use_format(vk::Format::eR8G8B8A8Unorm)
            .use_usage(attachment_usage_flags)
            .create(ctx);

    auto noise = make_ssao_noise();

    ssao_noise_texture = TextureBuilder()
            .from_memory(noise.data(), {4, 4, 1})
            .use_format(vk::Format::eR32G32B32A32Sfloat)
            .use_usage(attachment_usage_flags)
            .with_sampler_address_mode(vk::SamplerAddressMode::eRepeat)
            .create(ctx);

    for (auto &res: frame_resources) {
        if (res.scene_descriptor_set) {
            res.scene_descriptor_set->update_binding<1>(*ssao_texture);
        }

        if (res.ssao_descriptor_set) {
            res.ssao_descriptor_set->update_binding<4>(*ssao_noise_texture);
        }
    }
}

void VulkanRenderer::create_rt_target_texture() {
    const auto &[width, height] = swap_chain->get_extent();

    const vk::Extent3D extent{
        .width = width,
        .height = height,
        .depth = 1
    };

    rt_target_texture = TextureBuilder()
            .as_uninitialized(extent)
            .use_format(vk::Format::eR32G32B32A32Sfloat)
            .use_usage(vk::ImageUsageFlagBits::eStorage
                       | vk::ImageUsageFlagBits::eSampled
                       | vk::ImageUsageFlagBits::eTransferSrc
                       | vk::ImageUsageFlagBits::eTransferDst)
            .use_layout(vk::ImageLayout::eGeneral)
            .create(ctx);
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

    // todo - this shouldn't recreate pipelines
    create_scene_render_infos();
    create_skybox_render_infos();
    create_gui_render_infos();
    create_debug_quad_render_infos();

    create_prepass_textures();
    create_prepass_render_info();

    create_ssao_textures();
    create_ssao_render_info();
}

// ==================== descriptors ====================

void VulkanRenderer::create_descriptor_pool() {
    const std::vector<vk::DescriptorPoolSize> pool_sizes = {
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

void VulkanRenderer::create_scene_descriptor_sets() {
    for (auto &res: frame_resources) {
        res.scene_descriptor_set = make_unique<SceneDescriptorSet>(
            ctx,
            *descriptor_pool,
            ResourcePack{
                *res.graphics_uniform_buffer,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            },
            ResourcePack{*ssao_texture, vk::ShaderStageFlagBits::eFragment}
        );
    }
}

void VulkanRenderer::create_materials_descriptor_set() {
    constexpr auto scope = vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eClosestHitKHR;
    constexpr auto type = vk::DescriptorType::eCombinedImageSampler;
    constexpr auto descriptor_count = MATERIAL_TEX_ARRAY_SIZE;

    materials_descriptor_set = make_unique<MaterialsDescriptorSet>(
        ctx,
        *descriptor_pool,
        ResourcePack<Texture>{descriptor_count, scope, type}, // base colors
        ResourcePack<Texture>{descriptor_count, scope, type}, // normals
        ResourcePack<Texture>{descriptor_count, scope, type}, // orms
        ResourcePack{
            // skybox
            *skybox_texture,
            vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eMissKHR,
            type
        }
    );
}

void VulkanRenderer::create_skybox_descriptor_sets() {
    for (auto &res: frame_resources) {
        res.skybox_descriptor_set = make_unique<SkyboxDescriptorSet>(
            ctx,
            *descriptor_pool,
            ResourcePack{
                *res.graphics_uniform_buffer,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            },
            ResourcePack{*skybox_texture, vk::ShaderStageFlagBits::eFragment}
        );
    }
}

void VulkanRenderer::create_prepass_descriptor_sets() {
    for (auto &res: frame_resources) {
        res.prepass_descriptor_set = make_unique<PrepassDescriptorSet>(
            ctx,
            *descriptor_pool,
            ResourcePack{
                *res.graphics_uniform_buffer,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            }
        );
    }
}

void VulkanRenderer::create_ssao_descriptor_sets() {
    for (auto &res: frame_resources) {
        res.ssao_descriptor_set = make_unique<SsaoDescriptorSet>(
            ctx,
            *descriptor_pool,
            ResourcePack{
                *res.graphics_uniform_buffer,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            },
            ResourcePack{*g_buffer_textures.depth, vk::ShaderStageFlagBits::eFragment},
            ResourcePack{*g_buffer_textures.normal, vk::ShaderStageFlagBits::eFragment},
            ResourcePack{*g_buffer_textures.pos, vk::ShaderStageFlagBits::eFragment},
            ResourcePack{*ssao_noise_texture, vk::ShaderStageFlagBits::eFragment}
        );
    }
}

void VulkanRenderer::create_cubemap_capture_descriptor_set() {
    cubemap_capture_descriptor_set = make_unique<CubemapCaptureDescriptorSet>(
        ctx,
        *descriptor_pool,
        ResourcePack{
            *frame_resources[0].graphics_uniform_buffer,
            vk::ShaderStageFlagBits::eVertex,
        },
        envmap_texture
        ? ResourcePack{*envmap_texture, vk::ShaderStageFlagBits::eFragment}
        : ResourcePack<Texture>{1, vk::ShaderStageFlagBits::eFragment}
    );
}

void VulkanRenderer::create_debug_quad_descriptor_set() {
    debug_quad_descriptor_set = make_unique<DebugQuadDescriptorSet>(
        ctx,
        *descriptor_pool,
        ResourcePack{*rt_target_texture, vk::ShaderStageFlagBits::eFragment}
    );
}

void VulkanRenderer::create_rt_descriptor_sets() {
    for (auto &res: frame_resources) {
        res.rt_descriptor_set = make_unique<RtDescriptorSet>(
            ctx,
            *descriptor_pool,
            ResourcePack{
                *res.graphics_uniform_buffer,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eRaygenKHR,
            },
            ResourcePack{
                *tlas, vk::ShaderStageFlagBits::eRaygenKHR
            },
            ResourcePack{
                *rt_target_texture,
                vk::ShaderStageFlagBits::eRaygenKHR,
                vk::DescriptorType::eStorageImage
            }
        );
    }
}

void VulkanRenderer::create_meshes_descriptor_set() {
    meshes_descriptor_set = make_unique<MeshesDescriptorSet>(
        ctx,
        *descriptor_pool,
        ResourcePack{
            model->get_mesh_descriptions_buffer(),
            vk::ShaderStageFlagBits::eClosestHitKHR,
            vk::DescriptorType::eStorageBuffer
        },
        ResourcePack{
            model->get_vertex_buffer(),
            vk::ShaderStageFlagBits::eClosestHitKHR,
            vk::DescriptorType::eStorageBuffer
        },
        ResourcePack{
            model->get_index_buffer(),
            vk::ShaderStageFlagBits::eClosestHitKHR,
            vk::DescriptorType::eStorageBuffer
        }
    );
}

// ==================== render infos ====================

RenderInfo::RenderInfo(GraphicsPipelineBuilder builder, shared_ptr<GraphicsPipeline> pipeline,
                       std::vector<RenderTarget> colors)
    : cached_pipeline_builder(std::move(builder)), pipeline(std::move(pipeline)), color_targets(std::move(colors)) {
    make_attachment_infos();
}

RenderInfo::RenderInfo(GraphicsPipelineBuilder builder, shared_ptr<GraphicsPipeline> pipeline,
                       std::vector<RenderTarget> colors, RenderTarget depth)
    : cached_pipeline_builder(std::move(builder)), pipeline(std::move(pipeline)),
      color_targets(std::move(colors)), depth_target(std::move(depth)) {
    make_attachment_infos();
}

RenderInfo::RenderInfo(std::vector<RenderTarget> colors) : color_targets(std::move(colors)) {
    make_attachment_infos();
}

RenderInfo::RenderInfo(std::vector<RenderTarget> colors, RenderTarget depth)
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

vk::CommandBufferInheritanceRenderingInfo RenderInfo::get_inheritance_rendering_info() const {
    return vk::CommandBufferInheritanceRenderingInfo{
        .colorAttachmentCount = static_cast<uint32_t>(cached_color_attachment_formats.size()),
        .pColorAttachmentFormats = cached_color_attachment_formats.data(),
        .depthAttachmentFormat = depth_target ? depth_target->get_format() : static_cast<vk::Format>(0),
        .rasterizationSamples = pipeline->get_sample_count(),
    };
}

void RenderInfo::reload_shaders(const RendererContext &ctx) const {
    *pipeline = cached_pipeline_builder.create(ctx);
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

void VulkanRenderer::create_scene_render_infos() {
    scene_render_infos.clear();

    auto builder = GraphicsPipelineBuilder()
            .with_vertex_shader("../shaders/obj/main-vert.spv")
            .with_fragment_shader("../shaders/obj/main-frag.spv")
            .with_vertices<ModelVertex>()
            .with_rasterizer({
                .polygonMode = wireframe_mode ? vk::PolygonMode::eLine : vk::PolygonMode::eFill,
                .cullMode = cull_back_faces ? vk::CullModeFlagBits::eBack : vk::CullModeFlagBits::eNone,
                .frontFace = vk::FrontFace::eCounterClockwise,
                .lineWidth = 1.0f,
            })
            .with_multisampling({
                .rasterizationSamples = get_msaa_sample_count(),
                .minSampleShading = 1.0f,
            })
            .with_descriptor_layouts({
                *frame_resources[0].scene_descriptor_set->get_layout(),
                *materials_descriptor_set->get_layout(),
            })
            .with_push_constants({
                vk::PushConstantRange{
                    .stageFlags = vk::ShaderStageFlagBits::eFragment,
                    .offset = 0,
                    .size = sizeof(ScenePushConstants),
                }
            })
            .with_color_formats({swap_chain->get_image_format()})
            .with_depth_format(swap_chain->get_depth_format());

    auto pipeline = make_shared<GraphicsPipeline>(builder.create(ctx));

    for (auto &target: swap_chain->get_render_targets(ctx)) {
        std::vector<RenderTarget> color_targets;
        color_targets.emplace_back(std::move(target.color_target));

        target.depth_target.override_attachment_config(vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare);

        scene_render_infos.emplace_back(
            builder,
            pipeline,
            std::move(color_targets),
            std::move(target.depth_target)
        );
    }
}

void VulkanRenderer::create_skybox_render_infos() {
    skybox_render_infos.clear();

    auto builder = GraphicsPipelineBuilder()
            .with_vertex_shader("../shaders/obj/skybox-vert.spv")
            .with_fragment_shader("../shaders/obj/skybox-frag.spv")
            .with_vertices<SkyboxVertex>()
            .with_rasterizer({
                .polygonMode = vk::PolygonMode::eFill,
                .cullMode = vk::CullModeFlagBits::eNone,
                .frontFace = vk::FrontFace::eCounterClockwise,
                .lineWidth = 1.0f,
            })
            .with_multisampling({
                .rasterizationSamples = get_msaa_sample_count(),
                .minSampleShading = 1.0f,
            })
            .with_depth_stencil({
                .depthTestEnable = vk::False,
                .depthWriteEnable = vk::False,
            })
            .with_descriptor_layouts({
                *frame_resources[0].skybox_descriptor_set->get_layout(),
            })
            .with_color_formats({swap_chain->get_image_format()})
            .with_depth_format(swap_chain->get_depth_format());

    auto pipeline = make_shared<GraphicsPipeline>(builder.create(ctx));

    for (auto &target: swap_chain->get_render_targets(ctx)) {
        std::vector<RenderTarget> color_targets;
        color_targets.emplace_back(std::move(target.color_target));

        skybox_render_infos.emplace_back(
            builder,
            pipeline,
            std::move(color_targets),
            std::move(target.depth_target)
        );
    }
}

void VulkanRenderer::create_gui_render_infos() {
    gui_render_infos.clear();

    for (auto &target: swap_chain->get_render_targets(ctx)) {
        target.color_target.override_attachment_config(vk::AttachmentLoadOp::eLoad);

        std::vector<RenderTarget> color_targets;
        color_targets.emplace_back(std::move(target.color_target));

        gui_render_infos.emplace_back(std::move(color_targets));
    }
}

void VulkanRenderer::create_prepass_render_info() {
    std::vector<RenderTarget> color_targets;
    color_targets.emplace_back(ctx, *g_buffer_textures.normal);
    color_targets.emplace_back(ctx, *g_buffer_textures.pos);

    RenderTarget depth_target{ctx, *g_buffer_textures.depth};

    std::vector<vk::Format> color_formats;
    for (const auto &target: color_targets) color_formats.emplace_back(target.get_format());

    auto builder = GraphicsPipelineBuilder()
            .with_vertex_shader("../shaders/obj/prepass-vert.spv")
            .with_fragment_shader("../shaders/obj/prepass-frag.spv")
            .with_vertices<ModelVertex>()
            .with_rasterizer({
                .polygonMode = vk::PolygonMode::eFill,
                .cullMode = vk::CullModeFlagBits::eNone,
                .frontFace = vk::FrontFace::eCounterClockwise,
                .lineWidth = 1.0f,
            })
            .with_descriptor_layouts({
                *frame_resources[0].prepass_descriptor_set->get_layout(),
            })
            .with_color_formats(color_formats)
            .with_depth_format(depth_target.get_format());

    auto pipeline = make_shared<GraphicsPipeline>(builder.create(ctx));

    prepass_render_info = make_unique<RenderInfo>(
        builder,
        pipeline,
        std::move(color_targets),
        std::move(depth_target)
    );
}

void VulkanRenderer::create_ssao_render_info() {
    RenderTarget target{ctx, *ssao_texture};

    auto builder = GraphicsPipelineBuilder()
            .with_vertex_shader("../shaders/obj/ssao-vert.spv")
            .with_fragment_shader("../shaders/obj/ssao-frag.spv")
            .with_vertices<ScreenSpaceQuadVertex>()
            .with_rasterizer({
                .polygonMode = vk::PolygonMode::eFill,
                .cullMode = vk::CullModeFlagBits::eNone,
                .frontFace = vk::FrontFace::eCounterClockwise,
                .lineWidth = 1.0f,
            })
            .with_descriptor_layouts({
                *frame_resources[0].ssao_descriptor_set->get_layout(),
            })
            .with_color_formats({target.get_format()});

    auto pipeline = make_shared<GraphicsPipeline>(builder.create(ctx));

    std::vector<RenderTarget> targets;
    targets.emplace_back(std::move(target));

    ssao_render_info = make_unique<RenderInfo>(
        builder,
        pipeline,
        std::move(targets)
    );
}

void VulkanRenderer::create_cubemap_capture_render_info() {
    RenderTarget target{
        skybox_texture->get_image().get_mip_view(ctx, 0),
        skybox_texture->get_format()
    };

    auto builder = GraphicsPipelineBuilder()
            .with_vertex_shader("../shaders/obj/sphere-cube-vert.spv")
            .with_fragment_shader("../shaders/obj/sphere-cube-frag.spv")
            .with_vertices<SkyboxVertex>()
            .with_rasterizer({
                .polygonMode = vk::PolygonMode::eFill,
                .cullMode = vk::CullModeFlagBits::eNone,
                .frontFace = vk::FrontFace::eCounterClockwise,
                .lineWidth = 1.0f,
            })
            .with_depth_stencil({
                .depthTestEnable = vk::False,
                .depthWriteEnable = vk::False,
            })
            .with_descriptor_layouts({
                *cubemap_capture_descriptor_set->get_layout(),
            })
            .for_views(6)
            .with_color_formats({target.get_format()});

    auto pipeline = make_shared<GraphicsPipeline>(builder.create(ctx));

    std::vector<RenderTarget> targets;
    targets.emplace_back(std::move(target));

    cubemap_capture_render_info = make_unique<RenderInfo>(
        builder,
        pipeline,
        std::move(targets)
    );
}

void VulkanRenderer::create_debug_quad_render_infos() {
    debug_quad_render_infos.clear();

    auto builder = GraphicsPipelineBuilder()
            .with_vertex_shader("../shaders/obj/ss-quad-vert.spv")
            .with_fragment_shader("../shaders/obj/ss-quad-frag.spv")
            .with_vertices<ScreenSpaceQuadVertex>()
            .with_rasterizer({
                .polygonMode = vk::PolygonMode::eFill,
                .cullMode = vk::CullModeFlagBits::eNone,
                .frontFace = vk::FrontFace::eCounterClockwise,
                .lineWidth = 1.0f,
            })
            .with_multisampling({
                .rasterizationSamples = get_msaa_sample_count(),
                .minSampleShading = 1.0f,
            })
            .with_depth_stencil({
                .depthTestEnable = vk::False,
                .depthWriteEnable = vk::False,
            })
            .with_descriptor_layouts({
                *debug_quad_descriptor_set->get_layout(),
            })
            .with_color_formats({swap_chain->get_image_format()})
            .with_depth_format(swap_chain->get_depth_format());

    auto pipeline = make_shared<GraphicsPipeline>(builder.create(ctx));

    for (auto &target: swap_chain->get_render_targets(ctx)) {
        std::vector<RenderTarget> color_targets;
        color_targets.emplace_back(std::move(target.color_target));

        debug_quad_render_infos.emplace_back(
            builder,
            pipeline,
            std::move(color_targets),
            std::move(target.depth_target)
        );
    }
}

// ==================== pipelines ====================

void VulkanRenderer::reload_shaders() const {
    wait_idle();

    scene_render_infos[0].reload_shaders(ctx);
    skybox_render_infos[0].reload_shaders(ctx);
    prepass_render_info->reload_shaders(ctx);
    ssao_render_info->reload_shaders(ctx);
    cubemap_capture_render_info->reload_shaders(ctx);
    debug_quad_render_infos[0].reload_shaders(ctx);
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
VulkanRenderer::create_local_buffer(const std::vector<ElemType> &contents, const vk::BufferUsageFlags usage) {
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

void VulkanRenderer::create_uniform_buffers() {
    for (auto &res: frame_resources) {
        res.graphics_uniform_buffer = utils::buf::create_uniform_buffer(ctx, sizeof(GraphicsUBO));
        res.graphics_ubo_mapped = res.graphics_uniform_buffer->map();
    }
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

    auto scene_command_buffers =
            utils::cmd::create_command_buffers(ctx, vk::CommandBufferLevel::eSecondary, n_buffers);
    auto rt_command_buffers =
            utils::cmd::create_command_buffers(ctx, vk::CommandBufferLevel::eSecondary, n_buffers);
    auto gui_command_buffers =
            utils::cmd::create_command_buffers(ctx, vk::CommandBufferLevel::eSecondary, n_buffers);
    auto prepass_command_buffers =
            utils::cmd::create_command_buffers(ctx, vk::CommandBufferLevel::eSecondary, n_buffers);
    auto debug_command_buffers =
            utils::cmd::create_command_buffers(ctx, vk::CommandBufferLevel::eSecondary, n_buffers);
    auto ssao_command_buffers =
            utils::cmd::create_command_buffers(ctx, vk::CommandBufferLevel::eSecondary, n_buffers);

    for (size_t i = 0; i < graphics_command_buffers.size(); i++) {
        frame_resources[i].graphics_cmd_buffer =
                make_unique<vk::raii::CommandBuffer>(std::move(graphics_command_buffers[i]));
        frame_resources[i].rt_cmd_buffer =
                {make_unique<vk::raii::CommandBuffer>(std::move(rt_command_buffers[i]))};
        frame_resources[i].scene_cmd_buffer =
                {make_unique<vk::raii::CommandBuffer>(std::move(scene_command_buffers[i]))};
        frame_resources[i].gui_cmd_buffer =
                {make_unique<vk::raii::CommandBuffer>(std::move(gui_command_buffers[i]))};
        frame_resources[i].prepass_cmd_buffer =
                {make_unique<vk::raii::CommandBuffer>(std::move(prepass_command_buffers[i]))};
        frame_resources[i].debug_cmd_buffer =
                {make_unique<vk::raii::CommandBuffer>(std::move(debug_command_buffers[i]))};
        frame_resources[i].ssao_cmd_buffer =
                {make_unique<vk::raii::CommandBuffer>(std::move(ssao_command_buffers[i]))};
    }
}

void VulkanRenderer::record_graphics_command_buffer() {
    const auto &command_buffer = *frame_resources[current_frame_idx].graphics_cmd_buffer;
    constexpr auto rendering_flags = vk::RenderingFlagBits::eContentsSecondaryCommandBuffers;

    command_buffer.begin({});

    swap_chain->transition_to_attachment_layout(command_buffer);

    for (const auto &node_resources: render_graph_info.topo_sorted_nodes) {
        // if size > 1, then this means that this pass (node) draws to the swapchain image
        // and thus benefits from double or triple buffering
        const auto &node_render_info = node_resources.render_infos.size() == 1
                                       ? node_resources.render_infos[0]
                                       : node_resources.render_infos[current_frame_idx];
        const auto &node_command_buffer = node_resources.command_buffers.size() == 1
                                          ? node_resources.command_buffers[0]
                                          : node_resources.command_buffers[current_frame_idx];

        command_buffer.beginRendering(node_render_info.get(swap_chain->get_extent(), 1, rendering_flags));
        command_buffer.executeCommands(*node_command_buffer);
        command_buffer.endRendering();
    }

    swap_chain->transition_to_present_layout(command_buffer);

    command_buffer.end();
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

// ==================== ray tracing ====================

void VulkanRenderer::create_tlas() {
    const uint32_t instance_count = 1; // todo
    std::vector<vk::AccelerationStructureInstanceKHR> instances;

    decltype(vk::TransformMatrixKHR::matrix) matrix;
    matrix[0] = {{1.0f, 0.0f, 0.0f, 0.0f}};
    matrix[1] = {{0.0f, 1.0f, 0.0f, 0.0f}};
    matrix[2] = {{0.0f, 0.0f, 1.0f, 0.0f}};
    const vk::TransformMatrixKHR transform_matrix{matrix};

    const vk::AccelerationStructureDeviceAddressInfoKHR blas_address_info{.accelerationStructure = *model->get_blas()};
    const vk::DeviceAddress blas_reference = ctx.device->getAccelerationStructureAddressKHR(blas_address_info);
    constexpr auto flags = vk::GeometryInstanceFlagBitsKHR::eTriangleCullDisable; // todo

    instances.emplace_back(vk::AccelerationStructureInstanceKHR{
        .transform = transform_matrix,
        .instanceCustomIndex = 0, // todo
        .mask = 0xFFu,
        .instanceShaderBindingTableRecordOffset = 0,
        .flags = static_cast<VkGeometryInstanceFlagsKHR>(flags),
        .accelerationStructureReference = blas_reference,
    });

    const size_t instances_buffer_size = instances.size() * sizeof(vk::AccelerationStructureInstanceKHR);

    Buffer instances_buffer{
        **ctx.allocator,
        instances_buffer_size,
        vk::BufferUsageFlagBits::eShaderDeviceAddress
        | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR
        | vk::BufferUsageFlagBits::eTransferSrc
        | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostCoherent
        | vk::MemoryPropertyFlagBits::eHostVisible
    };

    void *instances_buffer_mapped = instances_buffer.map();
    memcpy(instances_buffer_mapped, instances.data(), instances_buffer_size);
    instances_buffer.unmap();

    const vk::AccelerationStructureGeometryInstancesDataKHR geometry_instances_data{
        .data = ctx.device->getBufferAddress({.buffer = *instances_buffer}),
    };

    const vk::AccelerationStructureGeometryKHR geometry{
        .geometryType = vk::GeometryTypeKHR::eInstances,
        .geometry = geometry_instances_data,
    };

    vk::AccelerationStructureBuildGeometryInfoKHR build_info{
        .type = vk::AccelerationStructureTypeKHR::eTopLevel,
        .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
        .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
        .geometryCount = 1u,
        .pGeometries = &geometry,
    };

    const vk::AccelerationStructureBuildSizesInfoKHR size_info = ctx.device->getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice,
        build_info,
        instance_count
    );

    const vk::DeviceSize tlas_size = size_info.accelerationStructureSize;

    auto tlas_buffer = make_unique<Buffer>(
        **ctx.allocator,
        tlas_size,
        vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    const vk::AccelerationStructureCreateInfoKHR create_info{
        .buffer = **tlas_buffer,
        .size = tlas_size,
        .type = vk::AccelerationStructureTypeKHR::eTopLevel,
    };

    tlas = make_unique<AccelerationStructure>(
        make_unique<vk::raii::AccelerationStructureKHR>(*ctx.device, create_info),
        std::move(tlas_buffer)
    );

    const Buffer scratch_buffer{
        **ctx.allocator,
        size_info.buildScratchSize,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    };

    build_info.srcAccelerationStructure = nullptr;
    build_info.dstAccelerationStructure = ***tlas;
    build_info.scratchData = ctx.device->getBufferAddress({.buffer = *scratch_buffer});

    static constexpr vk::AccelerationStructureBuildRangeInfoKHR range_info{
        .primitiveCount = instance_count,
        .primitiveOffset = 0,
        .firstVertex = 0,
        .transformOffset = 0,
    };

    static constexpr vk::MemoryBarrier2 memory_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
        .dstAccessMask = vk::AccessFlagBits2::eAccelerationStructureWriteKHR
    };

    utils::cmd::do_single_time_commands(ctx, [&](const vk::raii::CommandBuffer &command_buffer) {
        command_buffer.pipelineBarrier2({
            .memoryBarrierCount = 1u,
            .pMemoryBarriers = &memory_barrier,
        });

        command_buffer.buildAccelerationStructuresKHR(build_info, &range_info);
    });
}

void VulkanRenderer::create_rt_pipeline() {
    const auto builder = RtPipelineBuilder()
            .with_ray_gen_shader("../shaders/obj/raytrace-rgen.spv")
            .with_miss_shader("../shaders/obj/raytrace-rmiss.spv")
            .with_closest_hit_shader("../shaders/obj/raytrace-rchit.spv")
            .with_descriptor_layouts({
                *frame_resources[0].rt_descriptor_set->get_layout(),
                *materials_descriptor_set->get_layout(),
                *meshes_descriptor_set->get_layout(),
            });

    rt_pipeline = make_unique<RtPipeline>(builder.create(ctx));
}

// ==================== gui ====================

void VulkanRenderer::init_imgui() {
    const std::vector<vk::DescriptorPoolSize> pool_sizes = {
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
        // todo - convert these 2 to dynamic states
        if (ImGui::Checkbox("Cull backfaces", &cull_back_faces)) {
            queued_frame_begin_actions.emplace([&](const FrameBeginActionContext& fba_ctx) {
                wait_idle();
                scene_render_infos[0].reload_shaders(ctx);
            });
        }

        if (ImGui::Checkbox("Wireframe mode", &wireframe_mode)) {
            queued_frame_begin_actions.emplace([&](const FrameBeginActionContext& fba_ctx) {
                wait_idle();
                scene_render_infos[0].reload_shaders(ctx);
            });
        }

        static bool use_msaa_dummy = use_msaa;
        if (ImGui::Checkbox("MSAA", &use_msaa_dummy)) {
            queued_frame_begin_actions.emplace([this](const FrameBeginActionContext& fba_ctx) {
                use_msaa = use_msaa_dummy;

                wait_idle();
                recreate_swap_chain();

                create_scene_render_infos();
                create_skybox_render_infos();
                create_debug_quad_render_infos();

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
        auto descriptor_sets = create_node_descriptor_sets(node_handle);
        auto render_infos = create_node_render_infos(node_handle, descriptor_sets);

        render_graph_info.topo_sorted_nodes.emplace_back(RenderNodeResources{
            .handle = node_handle,
            .command_buffers = utils::cmd::create_command_buffers(
                ctx,
                vk::CommandBufferLevel::eSecondary,
                MAX_FRAMES_IN_FLIGHT
            ),
            .descriptor_sets = std::move(descriptor_sets),
            .render_infos = std::move(render_infos),
        });
    }

    repeated_frame_begin_actions = render_graph_info.render_graph->frame_begin_callbacks;
}

void VulkanRenderer::create_render_graph_resources() {
    const auto &uniform_buffers = render_graph_info.render_graph->uniform_buffers;
    const auto &external_resources = render_graph_info.render_graph->external_resources;
    const auto &transient_resources = render_graph_info.render_graph->transient_resources;
    const auto &model_resources = render_graph_info.render_graph->model_resources;

    for (const auto &[handle, description]: uniform_buffers) {
        render_graph_ubos.emplace(handle, utils::buf::create_uniform_buffer(ctx, description.size));
    }

    for (const auto &[handle, description]: external_resources) {
        const auto attachment_type = utils::img::is_depth_format(description.format)
                                     ? vk::ImageUsageFlagBits::eDepthStencilAttachment
                                     : vk::ImageUsageFlagBits::eColorAttachment;

        auto builder = TextureBuilder()
                .with_flags(description.tex_flags)
                .from_paths(description.paths)
                .use_format(description.format)
                .use_usage(vk::ImageUsageFlagBits::eTransferSrc
                           | vk::ImageUsageFlagBits::eTransferDst
                           | vk::ImageUsageFlagBits::eSampled
                           | attachment_type);

        if (description.paths.size() > 1) builder.as_separate_channels();
        if (description.swizzle) builder.with_swizzle(*description.swizzle);

        render_graph_textures.emplace(handle, builder.create(ctx));
    }

    for (const auto &[handle, description]: transient_resources) {
        const auto attachment_type = utils::img::is_depth_format(description.format)
                                     ? vk::ImageUsageFlagBits::eDepthStencilAttachment
                                     : vk::ImageUsageFlagBits::eColorAttachment;

        auto extent = description.extent;

        if (extent.width == 0 || extent.height == 0) {
            extent = swap_chain->get_extent();
        }

        auto builder = TextureBuilder()
                .with_flags(description.tex_flags)
                .as_uninitialized({extent.width, extent.height, 1u})
                .use_format(description.format)
                .use_usage(vk::ImageUsageFlagBits::eTransferSrc
                           | vk::ImageUsageFlagBits::eTransferDst
                           | vk::ImageUsageFlagBits::eSampled
                           | attachment_type);

        render_graph_textures.emplace(handle, builder.create(ctx));
    }

    for (const auto &[handle, description]: model_resources) {
        model = make_unique<Model>(ctx, description.path, false);
        render_graph_models.emplace(handle, std::move(model));
    }
}

std::vector<shared_ptr<DescriptorSet> >
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

    std::vector<shared_ptr<DescriptorSet> > descriptor_sets;
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

GraphicsPipelineBuilder VulkanRenderer::create_node_pipeline_builder(
    const RenderNodeHandle node_handle,
    const std::vector<shared_ptr<DescriptorSet> > &descriptor_sets
) const {
    const auto &node_info = render_graph_info.render_graph->nodes.at(node_handle);

    std::vector<vk::Format> color_formats;
    for (const auto &target_handle: node_info.color_targets) {
        if (target_handle == FINAL_IMAGE_RESOURCE_HANDLE) {
            color_formats.push_back(swap_chain->get_image_format());
        } else {
            color_formats.push_back(render_graph_info.render_graph->transient_resources.at(target_handle).format);
        }
    }

    std::vector<vk::DescriptorSetLayout> descriptor_set_layouts;
    for (const auto &set: descriptor_sets) {
        descriptor_set_layouts.emplace_back(*set->get_layout());
    }

    auto builder = GraphicsPipelineBuilder()
            .with_vertex_shader(node_info.vertex_shader->path)
            .with_fragment_shader(node_info.fragment_shader->path)
            .with_vertices<ModelVertex>()
            .with_rasterizer({
                .polygonMode = vk::PolygonMode::eFill,
                .cullMode = node_info.custom_config.cull_mode,
                .frontFace = vk::FrontFace::eCounterClockwise,
                .lineWidth = 1.0f,
            })
            .with_multisampling({
                .rasterizationSamples = node_info.custom_config.use_msaa
                                        ? get_msaa_sample_count()
                                        : vk::SampleCountFlagBits::e1,
                .minSampleShading = 1.0f,
            })
            .with_descriptor_layouts(descriptor_set_layouts)
            .with_color_formats(color_formats);

    if (node_info.depth_target) {
        builder.with_depth_format(
            render_graph_info.render_graph->transient_resources.at(*node_info.depth_target).format);
    } else if (has_swapchain_target(node_handle)) {
        builder.with_depth_format(swap_chain->get_depth_format());
    } else {
        builder.with_depth_stencil({
            .depthTestEnable = vk::False,
            .depthWriteEnable = vk::False,
        });
    }

    return builder;
}

std::vector<RenderInfo> VulkanRenderer::create_node_render_infos(
    const RenderNodeHandle node_handle,
    const std::vector<shared_ptr<DescriptorSet> > &descriptor_sets
) const {
    const auto &node_info = render_graph_info.render_graph->nodes.at(node_handle);
    auto pipeline_builder = create_node_pipeline_builder(node_handle, descriptor_sets);
    const auto pipeline = make_shared<GraphicsPipeline>(pipeline_builder.create(ctx));
    std::vector<RenderInfo> render_infos;

    if (has_swapchain_target(node_handle)) {
        for (auto &swap_chain_targets: swap_chain->get_render_targets(ctx)) {
            std::vector<RenderTarget> color_targets;

            for (auto color_target_handle: node_info.color_targets) {
                if (color_target_handle == FINAL_IMAGE_RESOURCE_HANDLE) {
                    color_targets.emplace_back(std::move(swap_chain_targets.color_target));
                } else {
                    const auto& target_texture = render_graph_textures.at(color_target_handle);
                    color_targets.emplace_back(target_texture->get_image().get_view(ctx), target_texture->get_format());
                }
            }

            if (node_info.depth_target) {
                throw std::runtime_error("depth target must be left unspecified for nodes targeting the final image");
            }

            render_infos.emplace_back(
                pipeline_builder,
                pipeline,
                std::move(color_targets),
                std::move(swap_chain_targets.depth_target)
            );
        }
    } else {
        std::vector<RenderTarget> color_targets;
        std::optional<RenderTarget> depth_target;

        for (auto color_target_handle: node_info.color_targets) {
            const auto& target_texture = render_graph_textures.at(color_target_handle);
            color_targets.emplace_back(target_texture->get_image().get_view(ctx), target_texture->get_format());
        }

        if (node_info.depth_target) {
            const auto& target_texture = render_graph_textures.at(*node_info.depth_target);
            depth_target = RenderTarget(target_texture->get_image().get_view(ctx), target_texture->get_format());
        }

        if (depth_target) {
            render_infos.emplace_back(pipeline_builder, pipeline, std::move(color_targets), std::move(*depth_target));
        } else {
            render_infos.emplace_back(pipeline_builder, pipeline, std::move(color_targets));
        }
    }

    return render_infos;
}

void VulkanRenderer::run_render_graph() {
    if (start_frame()) {
        for (const auto &node_resources: render_graph_info.topo_sorted_nodes) {
            record_render_graph_node_commands(node_resources);
        }

        record_graphics_command_buffer();

        end_frame();
    }
}

void VulkanRenderer::record_render_graph_node_commands(const RenderNodeResources &node_resources) {
    const auto &[handle, command_buffers, descriptor_sets, render_infos] = node_resources;
    const auto &command_buffer = command_buffers[current_frame_idx];
    const auto &render_info = render_infos[has_swapchain_target(node_resources.handle) ? current_frame_idx : 0];

    std::vector<vk::DescriptorSet> raw_descriptor_sets;
    for (const auto &descriptor_set: descriptor_sets) {
        raw_descriptor_sets.push_back(***descriptor_set);
    }

    const auto &node_info = render_graph_info.render_graph->nodes.at(handle);

    std::vector<vk::Format> color_formats;
    for (const auto &target_handle: node_info.color_targets) {
        if (target_handle == FINAL_IMAGE_RESOURCE_HANDLE) {
            color_formats.push_back(swap_chain->get_image_format());
        } else {
            color_formats.push_back(render_graph_info.render_graph->transient_resources.at(target_handle).format);
        }
    }

    auto depth_format = static_cast<vk::Format>(0);
    if (node_info.depth_target) {
        depth_format = render_graph_info.render_graph->transient_resources.at(*node_info.depth_target).format;
    } else if (has_swapchain_target(handle)) {
        depth_format = swap_chain->get_depth_format();
    }

    const vk::StructureChain inheritance_info{
        vk::CommandBufferInheritanceInfo{},
        vk::CommandBufferInheritanceRenderingInfo{
            .colorAttachmentCount = static_cast<uint32_t>(node_info.color_targets.size()),
            .pColorAttachmentFormats = color_formats.data(),
            .depthAttachmentFormat = depth_format,
            .rasterizationSamples = node_info.custom_config.use_msaa
                                    ? get_msaa_sample_count()
                                    : vk::SampleCountFlagBits::e1,
        }
    };

    command_buffer.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eRenderPassContinue,
        .pInheritanceInfo = &inheritance_info.get<vk::CommandBufferInheritanceInfo>(),
    });

    utils::cmd::set_dynamic_states(command_buffer, swap_chain->get_extent());

    const auto &pipeline = render_info.get_pipeline();
    command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, **pipeline);

    command_buffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *pipeline.get_layout(),
        0,
        raw_descriptor_sets,
        nullptr
    );

    RenderPassContext pass_ctx{command_buffer, render_graph_models};
    node_info.body(pass_ctx);

    command_buffer.end();
}

bool VulkanRenderer::has_swapchain_target(const RenderNodeHandle handle) const {
    return render_graph_info.render_graph->nodes.at(handle)
            .get_all_targets_set()
            .contains(FINAL_IMAGE_RESOURCE_HANDLE);
}

// ==================== render loop ====================

void VulkanRenderer::tick(const float delta_time) {
    (void) delta_time;
    // unused rn
}

void VulkanRenderer::do_frame_begin_actions() {
    const FrameBeginActionContext fba_ctx { render_graph_ubos, render_graph_textures, render_graph_models };

    for (const auto& action: repeated_frame_begin_actions) {
        action(fba_ctx);
    }

    while (!queued_frame_begin_actions.empty()) {
        queued_frame_begin_actions.front()(fba_ctx);
        queued_frame_begin_actions.pop();
    }
}

void VulkanRenderer::render_gui(const std::function<void()> &render_commands) {
    const auto &command_buffer = *frame_resources[current_frame_idx].gui_cmd_buffer.buffer;

    const std::vector color_attachment_formats{swap_chain->get_image_format()};

    const vk::StructureChain<
        vk::CommandBufferInheritanceInfo,
        vk::CommandBufferInheritanceRenderingInfo
    > inheritance_info{
        {},
        {
            .colorAttachmentCount = static_cast<uint32_t>(color_attachment_formats.size()),
            .pColorAttachmentFormats = color_attachment_formats.data(),
            .rasterizationSamples = get_msaa_sample_count(),
        }
    };

    command_buffer.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eRenderPassContinue,
        .pInheritanceInfo = &inheritance_info.get<vk::CommandBufferInheritanceInfo>(),
    });

    gui_renderer->begin_rendering();
    render_commands();
    gui_renderer->end_rendering(command_buffer);

    command_buffer.end();

    frame_resources[current_frame_idx].gui_cmd_buffer.was_recorded_this_frame = true;
}

bool VulkanRenderer::start_frame() {
    const auto &sync = frame_resources[current_frame_idx].sync;

    const std::vector wait_semaphores = {
        **sync.render_finished_timeline.semaphore,
    };

    const std::vector wait_semaphore_values = {
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

    const std::vector wait_semaphores = {
        **sync.image_available_semaphore
    };

    const std::vector<TimelineSemValueType> wait_semaphore_values = {
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
    const std::vector<TimelineSemValueType> signal_semaphore_values{
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

void VulkanRenderer::run_prepass() {
    if (!model) {
        return;
    }

    const auto &command_buffer = *frame_resources[current_frame_idx].prepass_cmd_buffer.buffer;

    const vk::StructureChain inheritance_info{
        vk::CommandBufferInheritanceInfo{},
        prepass_render_info->get_inheritance_rendering_info()
    };

    command_buffer.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eRenderPassContinue,
        .pInheritanceInfo = &inheritance_info.get<vk::CommandBufferInheritanceInfo>(),
    });

    utils::cmd::set_dynamic_states(command_buffer, swap_chain->get_extent());

    auto &pipeline = prepass_render_info->get_pipeline();
    command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, **pipeline);

    command_buffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *pipeline.get_layout(),
        0,
        ***frame_resources[current_frame_idx].prepass_descriptor_set,
        nullptr
    );

    draw_model(command_buffer, false, pipeline);

    command_buffer.end();

    frame_resources[current_frame_idx].prepass_cmd_buffer.was_recorded_this_frame = true;
}

void VulkanRenderer::run_ssao_pass() {
    if (!model) {
        return;
    }

    const auto &command_buffer = *frame_resources[current_frame_idx].ssao_cmd_buffer.buffer;

    const vk::StructureChain inheritance_info{
        vk::CommandBufferInheritanceInfo{},
        ssao_render_info->get_inheritance_rendering_info()
    };

    command_buffer.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eRenderPassContinue,
        .pInheritanceInfo = &inheritance_info.get<vk::CommandBufferInheritanceInfo>(),
    });

    utils::cmd::set_dynamic_states(command_buffer, swap_chain->get_extent());

    auto &pipeline = ssao_render_info->get_pipeline();
    command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, **pipeline);

    command_buffer.bindVertexBuffers(0, **screen_space_quad_vertex_buffer, {0});

    command_buffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *pipeline.get_layout(),
        0,
        ***frame_resources[current_frame_idx].ssao_descriptor_set,
        nullptr
    );

    command_buffer.draw(screen_space_quad_vertices.size(), 1, 0, 0);

    command_buffer.end();

    frame_resources[current_frame_idx].ssao_cmd_buffer.was_recorded_this_frame = true;
}

void VulkanRenderer::raytrace() {
    const auto &command_buffer = *frame_resources[current_frame_idx].rt_cmd_buffer.buffer;

    static constexpr vk::CommandBufferInheritanceInfo inheritance_info{};

    command_buffer.begin(vk::CommandBufferBeginInfo{
        .pInheritanceInfo = &inheritance_info,
    });

    command_buffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, ***rt_pipeline);

    command_buffer.bindDescriptorSets(
        vk::PipelineBindPoint::eRayTracingKHR,
        *rt_pipeline->get_layout(),
        0,
        {
            ***frame_resources[current_frame_idx].rt_descriptor_set,
            ***materials_descriptor_set,
            ***meshes_descriptor_set,
        },
        nullptr
    );

    const auto &sbt = rt_pipeline->get_sbt();
    const auto &extent = rt_target_texture->get_image().get_extent();

    command_buffer.traceRaysKHR(
        sbt.rgen_region,
        sbt.miss_region,
        sbt.hit_region,
        sbt.call_region,
        extent.width,
        extent.height,
        extent.depth
    );

    command_buffer.end();

    frame_resources[current_frame_idx].rt_cmd_buffer.was_recorded_this_frame = true;
}

void VulkanRenderer::draw_scene() {
    if (!model) {
        return;
    }

    const auto &command_buffer = *frame_resources[current_frame_idx].scene_cmd_buffer.buffer;

    const vk::StructureChain inheritance_info{
        vk::CommandBufferInheritanceInfo{},
        scene_render_infos[0].get_inheritance_rendering_info()
    };

    command_buffer.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eRenderPassContinue,
        .pInheritanceInfo = &inheritance_info.get<vk::CommandBufferInheritanceInfo>(),
    });

    utils::cmd::set_dynamic_states(command_buffer, swap_chain->get_extent());

    // skybox

    const auto &skybox_pipeline = skybox_render_infos[swap_chain->get_current_image_index()].get_pipeline();
    command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, **skybox_pipeline);

    command_buffer.bindVertexBuffers(0, **skybox_vertex_buffer, {0});

    command_buffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *skybox_pipeline.get_layout(),
        0,
        ***frame_resources[current_frame_idx].skybox_descriptor_set,
        nullptr
    );

    command_buffer.draw(static_cast<uint32_t>(skybox_vertices.size()), 1, 0, 0);

    // scene

    const auto &scene_pipeline = scene_render_infos[swap_chain->get_current_image_index()].get_pipeline();
    command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, **scene_pipeline);

    command_buffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *scene_pipeline.get_layout(),
        0,
        {
            ***frame_resources[current_frame_idx].scene_descriptor_set,
            ***materials_descriptor_set,
        },
        nullptr
    );

    draw_model(command_buffer, true, scene_pipeline);

    command_buffer.end();

    frame_resources[current_frame_idx].scene_cmd_buffer.was_recorded_this_frame = true;
}

void VulkanRenderer::draw_debug_quad() {
    const auto &command_buffer = *frame_resources[current_frame_idx].debug_cmd_buffer.buffer;

    const vk::StructureChain inheritance_info{
        vk::CommandBufferInheritanceInfo{},
        debug_quad_render_infos[0].get_inheritance_rendering_info()
    };

    command_buffer.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eRenderPassContinue,
        .pInheritanceInfo = &inheritance_info.get<vk::CommandBufferInheritanceInfo>(),
    });

    utils::cmd::set_dynamic_states(command_buffer, swap_chain->get_extent());

    auto &pipeline = debug_quad_render_infos[swap_chain->get_current_image_index()].get_pipeline();
    command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, **pipeline);

    command_buffer.bindVertexBuffers(0, **screen_space_quad_vertex_buffer, {0});

    command_buffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *pipeline.get_layout(),
        0,
        ***debug_quad_descriptor_set,
        nullptr
    );

    command_buffer.draw(screen_space_quad_vertices.size(), 1, 0, 0);

    command_buffer.end();

    frame_resources[current_frame_idx].debug_cmd_buffer.was_recorded_this_frame = true;
}

void VulkanRenderer::draw_model(const vk::raii::CommandBuffer &command_buffer, const bool do_push_constants,
                                const GraphicsPipeline &pipeline) const {
    uint32_t index_offset = 0;
    int32_t vertex_offset = 0;
    uint32_t instance_offset = 0;

    model->bind_buffers(command_buffer);

    for (const auto &mesh: model->get_meshes()) {
        // todo - make this a bit nicer (without the ugly bool)
        if (do_push_constants) {
            command_buffer.pushConstants<ScenePushConstants>(
                *pipeline.get_layout(),
                vk::ShaderStageFlagBits::eFragment,
                0,
                ScenePushConstants{
                    .material_id = mesh.material_id
                }
            );
        }

        command_buffer.drawIndexed(
            static_cast<uint32_t>(mesh.indices.size()),
            static_cast<uint32_t>(mesh.instances.size()),
            index_offset,
            vertex_offset,
            instance_offset
        );

        index_offset += static_cast<uint32_t>(mesh.indices.size());
        vertex_offset += static_cast<int32_t>(mesh.vertices.size());
        instance_offset += static_cast<uint32_t>(mesh.instances.size());
    }
}

void VulkanRenderer::capture_cubemap() const {
    const vk::Extent2D extent = skybox_texture->get_image().get_extent_2d();

    const auto command_buffer = utils::cmd::begin_single_time_commands(ctx);

    utils::cmd::set_dynamic_states(command_buffer, extent);

    command_buffer.beginRendering(cubemap_capture_render_info->get(extent, 6));

    command_buffer.bindVertexBuffers(0, **skybox_vertex_buffer, {0});

    const auto &pipeline = cubemap_capture_render_info->get_pipeline();
    command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, **pipeline);

    command_buffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *pipeline.get_layout(),
        0,
        ***cubemap_capture_descriptor_set,
        nullptr
    );

    command_buffer.draw(skybox_vertices.size(), 1, 0, 0);

    command_buffer.endRendering();

    skybox_texture->get_image().transition_layout(
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::ImageLayout::eTransferDstOptimal,
        command_buffer
    );

    utils::cmd::end_single_time_commands(command_buffer, *ctx.graphics_queue);

    skybox_texture->generate_mipmaps(ctx, vk::ImageLayout::eShaderReadOnlyOptimal);
}
} // zrx
