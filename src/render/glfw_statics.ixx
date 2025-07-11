module;

export module Cinder.Render:GlfwStatics;

import glm;
import std;
import glfw;

import :Camera;
import :Renderer;

import Cinder.Utils;
import Cinder.Globals;

export namespace zrx {
struct GlfwStaticUserData {
    VulkanRenderer* renderer;
    Camera* camera;
};

inline void init_glfw_user_pointer(GLFWwindow* window) {
    auto* user_data_ptr = static_cast<GlfwStaticUserData *>(glfwGetWindowUserPointer(window));
    if (!user_data_ptr) {
        glfwSetWindowUserPointer(window, new GlfwStaticUserData);
    }
}
} // zrx
