#include "glfw-statics.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace zrx {
void init_glfw_user_pointer(GLFWwindow* window) {
    auto* user_data_ptr = static_cast<GlfwStaticUserData *>(glfwGetWindowUserPointer(window));
    if (!user_data_ptr) {
        glfwSetWindowUserPointer(window, new GlfwStaticUserData);
    }
}
} // zrx
