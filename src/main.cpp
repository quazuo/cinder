import std;
import glfw;

import Cinder.Globals;
import Cinder.Engine;

#include <vulkan/vulkan_hpp_macros.hpp>

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC == 1
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#endif

static void show_error_box(const string &message) {
    // MessageBox(
    //     nullptr,
    //     static_cast<LPCSTR>(message.c_str()),
    //     static_cast<LPCSTR>("Error"),
    //     MB_OK
    // );
}

int main() {
    if (!glfwInit()) {
        show_error_box("Fatal error: GLFW initialization failed.");
        return 1; // EXIT_FAILURE;
    }

#ifdef NDEBUG
    try {
        zrx::Engine engine;
        engine.run();
    } catch (std::exception &e) {
        show_error_box(string("Fatal error: ") + e.what());
        glfwTerminate();
        return 1; // EXIT_FAILURE;
    }
#else
    zrx::Engine engine;
    engine.run();
#endif

    glfwTerminate();

    return 0; // EXIT_SUCCESS;
}
