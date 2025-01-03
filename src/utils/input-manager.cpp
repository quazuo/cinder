#include "input-manager.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace zrx {
void InputManager::bind_callback(const EKey k, const EActivationType type, const EInputCallback& f) {
    callback_map.emplace(k, std::make_pair(type, f));
    key_state_map.emplace(k, KeyState::RELEASED);
}

void InputManager::bind_mouse_drag_callback(const EMouseButton button, const EMouseDragCallback &f) {
    mouse_drag_callback_map.emplace(button, f);
    mouse_button_state_map.emplace(button, KeyState::RELEASED);
}

void InputManager::tick(const float deltaTime) {
    for (const auto &[key, callback_info]: callback_map) {
        const auto &[activation_type, callback] = callback_info;

        if (check_key(key, activation_type)) {
            callback(deltaTime);
        }
    }

    glm::dvec2 mouse_pos;
    glfwGetCursorPos(window, &mouse_pos.x, &mouse_pos.y);

    for (const auto &[button, callback]: mouse_drag_callback_map) {
        if (glfwGetMouseButton(window, button) == GLFW_PRESS) {
            if (mouse_button_state_map.at(button) == KeyState::PRESSED) {
                const glm::dvec2 mouse_pos_delta = mouse_pos - last_mouse_pos;
                callback(mouse_pos_delta.x, mouse_pos_delta.y);

            } else {
                mouse_button_state_map[button] = KeyState::PRESSED;
            }
        } else {
            mouse_button_state_map[button] = KeyState::RELEASED;
        }
    }

    last_mouse_pos = mouse_pos;
}

static bool is_pressed(GLFWwindow* window, const EKey key) {
    return glfwGetKey(window, key) == GLFW_PRESS || glfwGetMouseButton(window, key) == GLFW_PRESS;
}

static bool is_released(GLFWwindow* window, const EKey key) {
    return glfwGetKey(window, key) == GLFW_RELEASE || glfwGetMouseButton(window, key) == GLFW_RELEASE;
}

bool InputManager::check_key(const EKey key, const EActivationType type) {
    if (type == EActivationType::PRESS_ANY) {
        return is_pressed(window, key);
    }

    if (type == EActivationType::RELEASE_ONCE) {
        return is_released(window, key);
    }

    if (type == EActivationType::PRESS_ONCE) {
        if (is_pressed(window, key)) {
            const bool is_ok = key_state_map[key] == KeyState::RELEASED;
            key_state_map[key] = KeyState::PRESSED;
            return is_ok;
        }

        key_state_map[key] = KeyState::RELEASED;
    }

    return false;
}
} // zrx
