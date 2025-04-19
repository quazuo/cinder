#pragma once

import glfw;

namespace zrx {
struct GlfwStaticUserData {
    class VulkanRenderer* renderer;
    class Camera* camera;
};

void init_glfw_user_pointer(GLFWwindow* window);
} // zrx
