#include "camera.hpp"

#include "gui/gui.hpp"
#include "src/utils/glfw-statics.hpp"

#define GLFW_INCLUDE_VULKAN
#include <iostream>
#include <GLFW/glfw3.h>

namespace zrx {
Rotator &Rotator::operator=(const glm::vec2 other) {
    rot = other;
    return *this;
}

Rotator &Rotator::operator+=(const glm::vec2 other) {
    static constexpr float y_angle_limit = glm::pi<float>() / 2 - 0.1f;

    rot.x += other.x;
    rot.y = std::clamp(
        rot.y + other.y,
        -y_angle_limit,
        y_angle_limit
    );

    return *this;
}

Rotator &Rotator::operator-=(const glm::vec2 other) {
    *this += -other;
    return *this;
}

Rotator::ViewVectors Rotator::get_view_vectors() const {
    const glm::vec3 front = {
        std::cos(rot.y) * std::sin(rot.x),
        std::sin(rot.y),
        std::cos(rot.y) * std::cos(rot.x)
    };

    const glm::vec3 right = {
        std::sin(rot.x - glm::pi<float>() / 2.0f),
        0,
        std::cos(rot.x - glm::pi<float>() / 2.0f)
    };

    return {
        .front = front,
        .right = right,
        .up = glm::cross(right, front)
    };
}

Camera::Camera(GLFWwindow *w) : window(w), input_manager(make_unique<InputManager>(w)) {
    bind_camera_lock_key();
    bind_freecam_movement_keys();
    bind_freecam_rotation_keys();
    bind_mouse_drag_callback();

    init_glfw_user_pointer(window);
    auto *user_data = static_cast<GlfwStaticUserData *>(glfwGetWindowUserPointer(window));
    if (!user_data) throw std::runtime_error("unexpected null window user pointer");
    user_data->camera = this;

    glfwSetScrollCallback(window, &scroll_callback);
}

void Camera::tick(const float delta_time) {
    if (
        !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)
        && !ImGui::IsAnyItemActive()
        && !ImGui::IsAnyItemFocused()
    ) {
        input_manager->tick(delta_time);
    }

    if (is_locked_cam) {
        tick_locked_mode();
    } else if (is_locked_cursor) {
        tick_mouse_movement(delta_time);
    }

    update_aspect_ratio();
    update_vecs();
}

glm::mat4 Camera::get_view_matrix() const {
    return glm::lookAt(pos, pos + front, glm::vec3(0, 1, 0));
}

glm::mat4 Camera::get_static_view_matrix() const {
    return glm::lookAt(glm::vec3(0), front, glm::vec3(0, 1, 0));
}

glm::mat4 Camera::get_projection_matrix() const {
    return glm::perspective(glm::radians(field_of_view), aspect_ratio, z_near, z_far);
}

void Camera::render_gui_section() {
    ImDrawList *draw_list = ImGui::GetWindowDrawList();

    constexpr auto section_flags = ImGuiTreeNodeFlags_DefaultOpen;

    if (ImGui::CollapsingHeader("Camera ", section_flags)) {
        ImGui::Text("Position: (%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z);
        ImGui::Text("Rotation: (%.2f, %.2f)", (*rotator).x, (*rotator).y);

        ImGui::Separator();

        ImGui::Text("Axes:");
        if (ImGui::BeginChild("Axes", ImVec2(50, 50))) {
            draw_list->AddRectFilled(
                ImGui::GetWindowPos(),
                ImGui::GetWindowPos() + ImVec2(50, 50),
                IM_COL32(0, 0, 0, 255)
            );

            const ImVec2 offset        = ImGui::GetWindowPos() + ImVec2(25, 25);
            constexpr float scale      = 20;
            const glm::mat4 view       = get_static_view_matrix();
            constexpr auto projection_x = glm::vec3(1, 0, 0);
            constexpr auto projection_y = glm::vec3(0, 1, 0);

            const glm::vec3 x = view * glm::vec4(1, 0, 0, 0);
            const float tx1   = scale * glm::dot(projection_x, x);
            const float tx2   = scale * glm::dot(projection_y, x);
            draw_list->AddLine(offset, offset + ImVec2(tx1, -tx2), IM_COL32(255, 0, 0, 255));

            const glm::vec3 y = view * glm::vec4(0, 1, 0, 0);
            const float ty1   = scale * glm::dot(projection_x, y);
            const float ty2   = scale * glm::dot(projection_y, y);
            draw_list->AddLine(offset, offset + ImVec2(ty1, -ty2), IM_COL32(0, 255, 0, 255));

            const glm::vec3 z = view * glm::vec4(0, 0, 1, 0);
            const float tz1   = scale * glm::dot(projection_x, z);
            const float tz2   = scale * glm::dot(projection_y, z);
            draw_list->AddLine(offset, offset + ImVec2(tz1, -tz2), IM_COL32(0, 0, 255, 255));
        }
        ImGui::EndChild();

        ImGui::Separator();

        if (ImGui::RadioButton("Free camera", !is_locked_cam)) {
            is_locked_cam = false;
        }

        ImGui::SameLine();

        if (ImGui::RadioButton("Locked camera", is_locked_cam)) {
            is_locked_cam = true;

            if (is_locked_cursor) {
                center_cursor();
            }
        }

        ImGui::Separator();

        ImGui::SliderFloat("Field of view", &field_of_view, 20.0f, 160.0f, "%.0f");

        if (!is_locked_cam) {
            ImGui::DragFloat("Rotation speed", &rotation_speed, 0.01f, 0.0f, FLT_MAX, "%.2f");
            ImGui::DragFloat("Movement speed", &movement_speed, 0.01f, 0.0f, FLT_MAX, "%.2f");
        }
    }
}

void Camera::scroll_callback(GLFWwindow *window, const double dx, const double dy) {
    const auto user_data = static_cast<GlfwStaticUserData *>(glfwGetWindowUserPointer(window));
    if (!user_data) throw std::runtime_error("unexpected null window user pointer");

    if (
        !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)
        && !ImGui::IsAnyItemActive()
        && !ImGui::IsAnyItemFocused()
    ) {
        user_data->camera->locked_radius /= static_cast<float>(1 + dy * 0.05);
    }
}

void Camera::bind_camera_lock_key() {
    input_manager->bind_callback(GLFW_KEY_F1, EActivationType::PRESS_ONCE, [&](const float deltaTime) {
        (void) deltaTime;
        if (is_locked_cam) {
            return;
        }

        is_locked_cursor = !is_locked_cursor;

        if (is_locked_cursor) {
            center_cursor();
        }
    });
}

void Camera::bind_mouse_drag_callback() {
    input_manager->bind_mouse_drag_callback(GLFW_MOUSE_BUTTON_LEFT, [&](const double dx, const double dy) {
        if (is_locked_cam) {
            static constexpr float speed = 0.003;

            locked_rotator += {
                -speed * static_cast<float>(dx),
                -speed * static_cast<float>(dy)
            };
        }
    });
}

void Camera::bind_freecam_rotation_keys() {
    input_manager->bind_callback(GLFW_KEY_UP, EActivationType::PRESS_ANY, [&](const float delta_time) {
        if (!is_locked_cam) {
            rotator += glm::vec2(0, delta_time * rotation_speed);
        }
    });

    input_manager->bind_callback(GLFW_KEY_DOWN, EActivationType::PRESS_ANY, [&](const float delta_time) {
        if (!is_locked_cam) {
            rotator -= glm::vec2(0, delta_time * rotation_speed);
        }
    });

    input_manager->bind_callback(GLFW_KEY_RIGHT, EActivationType::PRESS_ANY, [&](const float delta_time) {
        if (!is_locked_cam) {
            rotator -= glm::vec2(delta_time * rotation_speed, 0);
        }
    });

    input_manager->bind_callback(GLFW_KEY_LEFT, EActivationType::PRESS_ANY, [&](const float delta_time) {
        if (!is_locked_cam) {
            rotator += glm::vec2(delta_time * rotation_speed, 0);
        }
    });
}

void Camera::bind_freecam_movement_keys() {
    input_manager->bind_callback(GLFW_KEY_W, EActivationType::PRESS_ANY, [&](const float delta_time) {
        if (!is_locked_cam) {
            pos += front * delta_time * movement_speed; // Move forward
        }
    });

    input_manager->bind_callback(GLFW_KEY_S, EActivationType::PRESS_ANY, [&](const float delta_time) {
        if (!is_locked_cam) {
            pos -= front * delta_time * movement_speed; // Move backward
        }
    });

    input_manager->bind_callback(GLFW_KEY_D, EActivationType::PRESS_ANY, [&](const float delta_time) {
        if (!is_locked_cam) {
            pos += right * delta_time * movement_speed; // Strafe right
        }
    });

    input_manager->bind_callback(GLFW_KEY_A, EActivationType::PRESS_ANY, [&](const float delta_time) {
        if (!is_locked_cam) {
            pos -= right * delta_time * movement_speed; // Strafe left
        }
    });

    input_manager->bind_callback(GLFW_KEY_SPACE, EActivationType::PRESS_ANY, [&](const float delta_time) {
        if (!is_locked_cam) {
            pos += glm::vec3(0, 1, 0) * delta_time * movement_speed; // Fly upwards
        }
    });

    input_manager->bind_callback(GLFW_KEY_LEFT_SHIFT, EActivationType::PRESS_ANY, [&](const float delta_time) {
        if (!is_locked_cam) {
            pos -= glm::vec3(0, 1, 0) * delta_time * movement_speed; // Fly downwards
        }
    });
}

void Camera::tick_mouse_movement(const float delta_time) {
    (void) delta_time;

    glm::vec<2, double> cursor_pos{};
    glfwGetCursorPos(window, &cursor_pos.x, &cursor_pos.y);

    glm::ivec2 window_size{};
    glfwGetWindowSize(window, &window_size.x, &window_size.y);

    const float mouse_speed = 0.002f * rotation_speed;

    rotator += {
        mouse_speed * (static_cast<float>(window_size.x) / 2 - static_cast<float>(std::floor(cursor_pos.x))),
        mouse_speed * (static_cast<float>(window_size.y) / 2 - static_cast<float>(std::floor(cursor_pos.y)))
    };

    center_cursor();
}

void Camera::tick_locked_mode() {
    const glm::vec2 rot = *locked_rotator;

    pos = {
        glm::cos(rot.y) * locked_radius * glm::sin(rot.x),
        glm::sin(rot.y) * locked_radius * -1.0f,
        glm::cos(rot.y) * locked_radius * glm::cos(rot.x)
    };

    rotator = {
        rot.x - glm::pi<float>(),
        rot.y
    };
}

void Camera::update_vecs() {
    const Rotator::ViewVectors view_vectors = rotator.get_view_vectors();

    front = view_vectors.front;
    right = view_vectors.right;
    up    = view_vectors.up;
}

void Camera::update_aspect_ratio() {
    glm::vec<2, int> window_size{};
    glfwGetWindowSize(window, &window_size.x, &window_size.y);

    if (window_size.y == 0) {
        aspect_ratio = 1;
    } else {
        aspect_ratio = static_cast<float>(window_size.x) / static_cast<float>(window_size.y);
    }
}

void Camera::center_cursor() const {
    glm::ivec2 window_size{};
    glfwGetWindowSize(window, &window_size.x, &window_size.y);

    glfwSetCursorPos(
        window,
        static_cast<float>(window_size.x) / 2,
        static_cast<float>(window_size.y) / 2
    );
}
} // zrx
