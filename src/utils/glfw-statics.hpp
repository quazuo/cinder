#pragma once

struct GLFWwindow;

namespace zrx {
struct GlfwStaticUserData {
    class VulkanRenderer* renderer;
    class Camera* camera;
};

void initGlfwUserPointer(GLFWwindow* window);
} // zrx
