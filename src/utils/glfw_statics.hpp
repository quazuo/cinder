#pragma once

import glfw;

namespace zrx {
struct GlfwStaticUserData {
    class VulkanRenderer* renderer;
    class Camera* camera;
};

inline void init_glfw_user_pointer(GLFWwindow* window) {
    auto* user_data_ptr = static_cast<GlfwStaticUserData *>(glfwGetWindowUserPointer(window));
    if (!user_data_ptr) {
        glfwSetWindowUserPointer(window, new GlfwStaticUserData);
    }
}
} // zrx
