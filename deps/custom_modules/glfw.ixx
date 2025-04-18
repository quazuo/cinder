module;

#define GLFW_INCLUDE_VULKAN
#include <../glfw/include/GLFW/glfw3.h>

#define GLFW_EXPOSE_NATIVE_WIN32
#define NOMINMAX 1
#include <../glfw/include/GLFW/glfw3native.h>

export module glfw;

export {
    using ::GLFWwindow;
    using ::glfwWindowShouldClose;
    using ::glfwTerminate;
    using ::glfwGetWindowSize;
    using ::glfwGetWindowPos;
    using ::glfwGetMouseButton;
    using ::glfwGetCursorPos;
    using ::glfwGetKey;
    using ::glfwSetInputMode;
    using ::glfwInit;
    using ::glfwGetWindowUserPointer;
    using ::glfwSetWindowUserPointer;
    using ::glfwSetScrollCallback;
    using ::glfwPollEvents;
    using ::glfwSetCursorPos;
    using ::glfwGetTime;
    using ::glfwGetFramebufferSize;
}
