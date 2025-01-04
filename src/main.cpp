#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLFW_EXPOSE_NATIVE_WIN32
#define NOMINMAX 1
#include <iostream>
#include <random>
#include <GLFW/glfw3native.h>

#include "render/graph.hpp"
#include "render/renderer.hpp"
#include "render/gui/gui.hpp"
#include "utils/input-manager.hpp"
#include "utils/file-type.hpp"

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
class Engine {
    GLFWwindow *window = nullptr;
    VulkanRenderer renderer;
    std::unique_ptr<InputManager> input_manager;

    float last_time = 0.0f;

    bool is_gui_enabled = false;
    bool show_debug_quad = false;

    ImGui::FileBrowser file_browser;
    std::optional<FileType> current_type_being_chosen;
    std::unordered_map<FileType, std::filesystem::path> chosen_paths{};
    uint32_t load_scheme_idx = 0;

    std::string curr_error_message;

public:
    Engine() {
        window = renderer.get_window();

        input_manager = std::make_unique<InputManager>(window);
        bind_key_actions();

        build_render_graph();
    }

    [[nodiscard]] GLFWwindow *get_window() const { return window; }

    void run() {
        while (!glfwWindowShouldClose(window)) {
            tick();
        }

        renderer.wait_idle();
    }

private:
    void tick() {
        const auto current_time = static_cast<float>(glfwGetTime());
        const float delta_time = current_time - last_time;
        last_time = current_time;

        input_manager->tick(delta_time);
        renderer.tick(delta_time);

        renderer.run_render_graph();

        // if (renderer.start_frame()) {
        //     if (is_gui_enabled) {
        //         renderer.render_gui([&] {
        //             render_gui_section(delta_time);
        //             renderer.render_gui_section();
        //         });
        //     }
        //
        //     renderer.run_prepass();
        //     renderer.run_ssao_pass();
        //     renderer.draw_scene();
        //     renderer.raytrace();
        //
        //     if (show_debug_quad) {
        //         renderer.draw_debug_quad();
        //     }
        //
        //     renderer.end_frame();
        // }

        if (file_browser.HasSelected()) {
            const std::filesystem::path path = file_browser.GetSelected().string();

            if (*current_type_being_chosen == FileType::ENVMAP_HDR) {
                renderer.load_environment_map(path);
            } else {
                chosen_paths[*current_type_being_chosen] = path;
            }

            file_browser.ClearSelected();
            current_type_being_chosen = {};
        }
    }

    void build_render_graph() {
        RenderGraph render_graph;

        constexpr auto depth_format = vk::Format::eD32Sfloat;

        // ================== models ==================

        const auto scene_model = render_graph.add_model_resource({
            "scene-model",
            "../assets/example models/kettle/kettle.obj"
        });

        // ================== uniform buffers ==================

        const auto uniform_buffer = render_graph.add_uniform_buffer({
            "general-ubo",
            sizeof(GraphicsUBO)
        });

        // ================== external resources ==================

        const auto base_color_texture = render_graph.add_external_resource(ExternalTextureResource{
            "base-color-texture",
            {"../assets/example models/kettle/kettle-albedo.png"},
            vk::Format::eR8G8B8A8Srgb
        });

        const auto normal_texture = render_graph.add_external_resource(ExternalTextureResource{
            "normal-texture",
            {"../assets/example models/kettle/kettle-normal.png"},
            vk::Format::eR8G8B8A8Unorm,
        });

        const auto orm_texture = render_graph.add_external_resource(ExternalTextureResource{
            "orm-texture",
            {"../assets/example models/kettle/kettle-orm.png"},
            vk::Format::eR8G8B8A8Unorm,
        });

        // ================== transient resources ==================

        const auto g_buffer_normal = render_graph.add_transient_resource(TransientTextureResource{
            "g-buffer-normal",
            vk::Format::eR8G8B8A8Unorm,
        });

        const auto g_buffer_pos = render_graph.add_transient_resource(TransientTextureResource{
            "g-buffer-pos",
            vk::Format::eR8G8B8A8Unorm,
        });

        const auto g_buffer_depth = render_graph.add_transient_resource(TransientTextureResource{
            "g-buffer-depth",
            depth_format,
        });

        const auto ssao_texture = render_graph.add_transient_resource(TransientTextureResource{
            "ssao-texture",
            vk::Format::eR8G8B8A8Unorm,
        });

        // ================== prepass ==================

        const auto prepass_vertex_shader = std::make_shared<Shader>(Shader{
            "../shaders/obj/prepass-vert.spv",
            {
                {uniform_buffer}
            }
        });

        const auto prepass_fragment_shader = std::make_shared<Shader>(Shader{
            "../shaders/obj/prepass-frag.spv",
            {
                {uniform_buffer}
            }
        });

        render_graph.add_node({
            "prepass",
            prepass_vertex_shader,
            prepass_fragment_shader,
            {g_buffer_normal, g_buffer_pos},
            g_buffer_depth,
            [scene_model](RenderPassContext &ctx) {
                std::cout << "prepass\n";
                ctx.draw_model(scene_model);
            }
        });

        // ================== main pass ==================

        const auto main_vertex_shader = std::make_shared<Shader>(Shader{
            "../shaders/obj/main-vert.spv",
            {
                {uniform_buffer}
            }
        });

        const auto main_fragment_shader = std::make_shared<Shader>(Shader{
            "../shaders/obj/main-frag.spv",
            {
                {uniform_buffer, ssao_texture},
                {
                    ResourceHandleArray{base_color_texture},
                    ResourceHandleArray{normal_texture},
                    ResourceHandleArray{orm_texture}
                }
            }
        });

        render_graph.add_node({
            "main",
            main_vertex_shader,
            main_fragment_shader,
            {FINAL_IMAGE_RESOURCE_HANDLE},
            {},
            [scene_model](RenderPassContext &ctx) {
                std::cout << "main\n";
                ctx.draw_model(scene_model);
            }
        });

        renderer.register_render_graph(render_graph);
    }

    void bind_key_actions() {
        input_manager->bind_callback(GLFW_KEY_GRAVE_ACCENT, EActivationType::PRESS_ONCE, [&](const float delta_time) {
            (void) delta_time;
            is_gui_enabled = !is_gui_enabled;
        });
    }

    // ========================== gui ==========================

    void render_gui_section(const float delta_time) {
        static float fps = 1 / delta_time;

        constexpr float smoothing = 0.95f;
        fps = fps * smoothing + (1 / delta_time) * (1.0f - smoothing);

        constexpr auto section_flags = ImGuiTreeNodeFlags_DefaultOpen;

        if (ImGui::CollapsingHeader("Engine ", section_flags)) {
            ImGui::Text("FPS: %.2f", fps);

            ImGui::Checkbox("Debug quad", &show_debug_quad);
            ImGui::Separator();

            if (ImGui::Button("Reload shaders")) {
                renderer.reload_shaders();
            }
            ImGui::Separator();

            render_load_model_popup();

            if (!curr_error_message.empty()) {
                ImGui::OpenPopup("Model load error");
            }

            render_model_load_error_popup();
        }

        if (ImGui::CollapsingHeader("Environment ", section_flags)) {
            render_tex_load_button("Choose environment map...", FileType::ENVMAP_HDR, {".hdr"});

            file_browser.Display();
        }
    }

    void render_tex_load_button(const std::string &label, const FileType file_type,
                             const std::vector<std::string> &type_filters) {
        if (ImGui::Button(label.c_str(), ImVec2(180, 0))) {
            current_type_being_chosen = file_type;
            file_browser.SetTypeFilters(type_filters);
            file_browser.Open();
        }

        if (chosen_paths.contains(file_type)) {
            ImGui::SameLine();
            ImGui::Text(chosen_paths.at(file_type).filename().string().c_str());
        }
    }

    void render_load_model_popup() {
        constexpr auto combo_flags = ImGuiComboFlags_WidthFitPreview;

        if (ImGui::BeginPopupModal("Load model", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Load scheme:");

            if (ImGui::BeginCombo("##scheme", file_load_schemes[load_scheme_idx].name.c_str(),
                                  combo_flags)) {
                for (uint32_t i = 0; i < file_load_schemes.size(); i++) {
                    const bool is_selected = load_scheme_idx == i;

                    if (ImGui::Selectable(file_load_schemes[i].name.c_str(), is_selected)) {
                        load_scheme_idx = i;
                    }

                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }

            ImGui::Separator();

            for (const auto &type: file_load_schemes[load_scheme_idx].requirements) {
                render_tex_load_button(
                    get_file_type_load_label(type),
                    type,
                    get_file_type_extensions(type)
                );
            }

            ImGui::Separator();

            const bool can_submit = std::ranges::all_of(file_load_schemes[load_scheme_idx].requirements, [&](const auto &t) {
                return is_file_type_optional(t) || chosen_paths.contains(t);
            });

            if (!can_submit) {
                ImGui::BeginDisabled();
            }

            if (ImGui::Button("OK", ImVec2(120, 0))) {
                load_model();
                chosen_paths.clear();
                ImGui::CloseCurrentPopup();
            }

            if (!can_submit) {
                ImGui::EndDisabled();
            }

            ImGui::SameLine();

            if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                chosen_paths.clear();
                ImGui::CloseCurrentPopup();
            }

            file_browser.Display();

            ImGui::EndPopup();
        }
    }

    void load_model() {
        const auto &reqs = file_load_schemes[load_scheme_idx].requirements;

        try {
            if (reqs.contains(FileType::BASE_COLOR_PNG)) {
                renderer.load_model(chosen_paths.at(FileType::MODEL));
                renderer.load_base_color_texture(chosen_paths.at(FileType::BASE_COLOR_PNG));
            } else {
                renderer.load_model_with_materials(chosen_paths.at(FileType::MODEL));
            }

            if (reqs.contains(FileType::NORMAL_PNG)) {
                renderer.load_normal_map(chosen_paths.at(FileType::NORMAL_PNG));
            }

            if (reqs.contains(FileType::ORM_PNG)) {
                renderer.load_orm_map(chosen_paths.at(FileType::ORM_PNG));
            } else if (reqs.contains(FileType::RMA_PNG)) {
                renderer.load_rma_map(chosen_paths.at(FileType::RMA_PNG));
            } else if (reqs.contains(FileType::ROUGHNESS_PNG)) {
                const auto roughness_path = chosen_paths.at(FileType::ROUGHNESS_PNG);
                const auto ao_path = chosen_paths.contains(FileType::AO_PNG)
                                    ? chosen_paths.at(FileType::AO_PNG)
                                    : "";
                const auto metallic_path = chosen_paths.contains(FileType::METALLIC_PNG)
                                          ? chosen_paths.at(FileType::METALLIC_PNG)
                                          : "";

                renderer.load_orm_map(ao_path, roughness_path, metallic_path);
            }
        } catch (std::exception &e) {
            curr_error_message = e.what();
        }
    }

    void render_model_load_error_popup() {
        if (ImGui::BeginPopupModal("Model load error", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("An error occurred while loading the model:");
            ImGui::Text(curr_error_message.c_str());

            ImGui::Separator();

            if (ImGui::Button("OK", ImVec2(120, 0))) {
                ImGui::CloseCurrentPopup();
                curr_error_message = "";
            }

            ImGui::EndPopup();
        }
    }
};
}

static void show_error_box(const std::string &message) {
    MessageBox(
        nullptr,
        static_cast<LPCSTR>(message.c_str()),
        static_cast<LPCSTR>("Error"),
        MB_OK
    );
}

void generate_ssao_kernel_samples() {
    std::uniform_real_distribution<float> random_floats(0.0, 1.0);
    std::default_random_engine generator;
    std::vector<glm::vec3> ssao_kernel;
    for (int i = 0; i < 64; ++i) {
        glm::vec3 sample(
            random_floats(generator) * 2.0 - 1.0,
            random_floats(generator) * 2.0 - 1.0,
            random_floats(generator)
        );
        sample = glm::normalize(sample);
        sample *= random_floats(generator);

        float scale = (float) i / 64.0;
        scale = glm::mix(0.1f, 1.0f, scale * scale);
        sample *= scale;

        ssao_kernel.push_back(sample);
    }

    for (auto &v: ssao_kernel) {
        std::cout << "vec3(" << v.x << ", " << v.y << ", " << v.z << "),\n";
    }
}

int main() {
    if (!glfwInit()) {
        show_error_box("Fatal error: GLFW initialization failed.");
        return EXIT_FAILURE;
    }

#ifdef NDEBUG
    try {
        zrx::Engine engine;
        engine.run();
    } catch (std::exception &e) {
        show_error_box(std::string("Fatal error: ") + e.what());
        glfw_terminate();
        return EXIT_FAILURE;
    }
#else
    zrx::Engine engine;
    engine.run();
#endif

    glfwTerminate();

    return EXIT_SUCCESS;
}
