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
    using ::glfwWindowHint;
    using ::glfwCreateWindow;
    using ::glfwSetFramebufferSizeCallback;
    using ::glfwDestroyWindow;
    using ::glfwGetRequiredInstanceExtensions;
    using ::glfwCreateWindowSurface;
    using ::glfwWaitEvents;

    constexpr int ZRX_GLFW_CLIENT_API = GLFW_CLIENT_API;
    constexpr int ZRX_GLFW_NO_API = GLFW_NO_API;

    /* The unknown key */
    constexpr uint32_t ZRX_GLFW_KEY_UNKNOWN = GLFW_KEY_UNKNOWN;

    /* Printable keys */
    constexpr uint32_t ZRX_GLFW_KEY_SPACE = GLFW_KEY_SPACE;
    constexpr uint32_t ZRX_GLFW_KEY_APOSTROPHE = GLFW_KEY_APOSTROPHE;
    constexpr uint32_t ZRX_GLFW_KEY_COMMA = GLFW_KEY_COMMA;
    constexpr uint32_t ZRX_GLFW_KEY_MINUS = GLFW_KEY_MINUS;
    constexpr uint32_t ZRX_GLFW_KEY_PERIOD = GLFW_KEY_PERIOD;
    constexpr uint32_t ZRX_GLFW_KEY_SLASH = GLFW_KEY_SLASH;
    constexpr uint32_t ZRX_GLFW_KEY_0 = GLFW_KEY_0;
    constexpr uint32_t ZRX_GLFW_KEY_1 = GLFW_KEY_1;
    constexpr uint32_t ZRX_GLFW_KEY_2 = GLFW_KEY_2;
    constexpr uint32_t ZRX_GLFW_KEY_3 = GLFW_KEY_3;
    constexpr uint32_t ZRX_GLFW_KEY_4 = GLFW_KEY_4;
    constexpr uint32_t ZRX_GLFW_KEY_5 = GLFW_KEY_5;
    constexpr uint32_t ZRX_GLFW_KEY_6 = GLFW_KEY_6;
    constexpr uint32_t ZRX_GLFW_KEY_7 = GLFW_KEY_7;
    constexpr uint32_t ZRX_GLFW_KEY_8 = GLFW_KEY_8;
    constexpr uint32_t ZRX_GLFW_KEY_9 = GLFW_KEY_9;
    constexpr uint32_t ZRX_GLFW_KEY_SEMICOLON = GLFW_KEY_SEMICOLON;
    constexpr uint32_t ZRX_GLFW_KEY_EQUAL = GLFW_KEY_EQUAL;
    constexpr uint32_t ZRX_GLFW_KEY_A = GLFW_KEY_A;
    constexpr uint32_t ZRX_GLFW_KEY_B = GLFW_KEY_B;
    constexpr uint32_t ZRX_GLFW_KEY_C = GLFW_KEY_C;
    constexpr uint32_t ZRX_GLFW_KEY_D = GLFW_KEY_D;
    constexpr uint32_t ZRX_GLFW_KEY_E = GLFW_KEY_E;
    constexpr uint32_t ZRX_GLFW_KEY_F = GLFW_KEY_F;
    constexpr uint32_t ZRX_GLFW_KEY_G = GLFW_KEY_G;
    constexpr uint32_t ZRX_GLFW_KEY_H = GLFW_KEY_H;
    constexpr uint32_t ZRX_GLFW_KEY_I = GLFW_KEY_I;
    constexpr uint32_t ZRX_GLFW_KEY_J = GLFW_KEY_J;
    constexpr uint32_t ZRX_GLFW_KEY_K = GLFW_KEY_K;
    constexpr uint32_t ZRX_GLFW_KEY_L = GLFW_KEY_L;
    constexpr uint32_t ZRX_GLFW_KEY_M = GLFW_KEY_M;
    constexpr uint32_t ZRX_GLFW_KEY_N = GLFW_KEY_N;
    constexpr uint32_t ZRX_GLFW_KEY_O = GLFW_KEY_O;
    constexpr uint32_t ZRX_GLFW_KEY_P = GLFW_KEY_P;
    constexpr uint32_t ZRX_GLFW_KEY_Q = GLFW_KEY_Q;
    constexpr uint32_t ZRX_GLFW_KEY_R = GLFW_KEY_R;
    constexpr uint32_t ZRX_GLFW_KEY_S = GLFW_KEY_S;
    constexpr uint32_t ZRX_GLFW_KEY_T = GLFW_KEY_T;
    constexpr uint32_t ZRX_GLFW_KEY_U = GLFW_KEY_U;
    constexpr uint32_t ZRX_GLFW_KEY_V = GLFW_KEY_V;
    constexpr uint32_t ZRX_GLFW_KEY_W = GLFW_KEY_W;
    constexpr uint32_t ZRX_GLFW_KEY_X = GLFW_KEY_X;
    constexpr uint32_t ZRX_GLFW_KEY_Y = GLFW_KEY_Y;
    constexpr uint32_t ZRX_GLFW_KEY_Z = GLFW_KEY_Z;
    constexpr uint32_t ZRX_GLFW_KEY_LEFT_BRACKET = GLFW_KEY_LEFT_BRACKET;
    constexpr uint32_t ZRX_GLFW_KEY_BACKSLASH = GLFW_KEY_BACKSLASH;
    constexpr uint32_t ZRX_GLFW_KEY_RIGHT_BRACKET = GLFW_KEY_RIGHT_BRACKET;
    constexpr uint32_t ZRX_GLFW_KEY_GRAVE_ACCENT = GLFW_KEY_GRAVE_ACCENT;
    constexpr uint32_t ZRX_GLFW_KEY_WORLD_1 = GLFW_KEY_WORLD_1;
    constexpr uint32_t ZRX_GLFW_KEY_WORLD_2 = GLFW_KEY_WORLD_2;

    /* Function keys */
    constexpr uint32_t ZRX_GLFW_KEY_ESCAPE = GLFW_KEY_ESCAPE;
    constexpr uint32_t ZRX_GLFW_KEY_ENTER = GLFW_KEY_ENTER;
    constexpr uint32_t ZRX_GLFW_KEY_TAB = GLFW_KEY_TAB;
    constexpr uint32_t ZRX_GLFW_KEY_BACKSPACE = GLFW_KEY_BACKSPACE;
    constexpr uint32_t ZRX_GLFW_KEY_INSERT = GLFW_KEY_INSERT;
    constexpr uint32_t ZRX_GLFW_KEY_DELETE = GLFW_KEY_DELETE;
    constexpr uint32_t ZRX_GLFW_KEY_RIGHT = GLFW_KEY_RIGHT;
    constexpr uint32_t ZRX_GLFW_KEY_LEFT = GLFW_KEY_LEFT;
    constexpr uint32_t ZRX_GLFW_KEY_DOWN = GLFW_KEY_DOWN;
    constexpr uint32_t ZRX_GLFW_KEY_UP = GLFW_KEY_UP;
    constexpr uint32_t ZRX_GLFW_KEY_PAGE_UP = GLFW_KEY_PAGE_UP;
    constexpr uint32_t ZRX_GLFW_KEY_PAGE_DOWN = GLFW_KEY_PAGE_DOWN;
    constexpr uint32_t ZRX_GLFW_KEY_HOME = GLFW_KEY_HOME;
    constexpr uint32_t ZRX_GLFW_KEY_END = GLFW_KEY_END;
    constexpr uint32_t ZRX_GLFW_KEY_CAPS_LOCK = GLFW_KEY_CAPS_LOCK;
    constexpr uint32_t ZRX_GLFW_KEY_SCROLL_LOCK = GLFW_KEY_SCROLL_LOCK;
    constexpr uint32_t ZRX_GLFW_KEY_NUM_LOCK = GLFW_KEY_NUM_LOCK;
    constexpr uint32_t ZRX_GLFW_KEY_PRINT_SCREEN = GLFW_KEY_PRINT_SCREEN;
    constexpr uint32_t ZRX_GLFW_KEY_PAUSE = GLFW_KEY_PAUSE;
    constexpr uint32_t ZRX_GLFW_KEY_F1 = GLFW_KEY_F1;
    constexpr uint32_t ZRX_GLFW_KEY_F2 = GLFW_KEY_F2;
    constexpr uint32_t ZRX_GLFW_KEY_F3 = GLFW_KEY_F3;
    constexpr uint32_t ZRX_GLFW_KEY_F4 = GLFW_KEY_F4;
    constexpr uint32_t ZRX_GLFW_KEY_F5 = GLFW_KEY_F5;
    constexpr uint32_t ZRX_GLFW_KEY_F6 = GLFW_KEY_F6;
    constexpr uint32_t ZRX_GLFW_KEY_F7 = GLFW_KEY_F7;
    constexpr uint32_t ZRX_GLFW_KEY_F8 = GLFW_KEY_F8;
    constexpr uint32_t ZRX_GLFW_KEY_F9 = GLFW_KEY_F9;
    constexpr uint32_t ZRX_GLFW_KEY_F10 = GLFW_KEY_F10;
    constexpr uint32_t ZRX_GLFW_KEY_F11 = GLFW_KEY_F11;
    constexpr uint32_t ZRX_GLFW_KEY_F12 = GLFW_KEY_F12;
    constexpr uint32_t ZRX_GLFW_KEY_F13 = GLFW_KEY_F13;
    constexpr uint32_t ZRX_GLFW_KEY_F14 = GLFW_KEY_F14;
    constexpr uint32_t ZRX_GLFW_KEY_F15 = GLFW_KEY_F15;
    constexpr uint32_t ZRX_GLFW_KEY_F16 = GLFW_KEY_F16;
    constexpr uint32_t ZRX_GLFW_KEY_F17 = GLFW_KEY_F17;
    constexpr uint32_t ZRX_GLFW_KEY_F18 = GLFW_KEY_F18;
    constexpr uint32_t ZRX_GLFW_KEY_F19 = GLFW_KEY_F19;
    constexpr uint32_t ZRX_GLFW_KEY_F20 = GLFW_KEY_F20;
    constexpr uint32_t ZRX_GLFW_KEY_F21 = GLFW_KEY_F21;
    constexpr uint32_t ZRX_GLFW_KEY_F22 = GLFW_KEY_F22;
    constexpr uint32_t ZRX_GLFW_KEY_F23 = GLFW_KEY_F23;
    constexpr uint32_t ZRX_GLFW_KEY_F24 = GLFW_KEY_F24;
    constexpr uint32_t ZRX_GLFW_KEY_F25 = GLFW_KEY_F25;
    constexpr uint32_t ZRX_GLFW_KEY_KP_0 = GLFW_KEY_KP_0;
    constexpr uint32_t ZRX_GLFW_KEY_KP_1 = GLFW_KEY_KP_1;
    constexpr uint32_t ZRX_GLFW_KEY_KP_2 = GLFW_KEY_KP_2;
    constexpr uint32_t ZRX_GLFW_KEY_KP_3 = GLFW_KEY_KP_3;
    constexpr uint32_t ZRX_GLFW_KEY_KP_4 = GLFW_KEY_KP_4;
    constexpr uint32_t ZRX_GLFW_KEY_KP_5 = GLFW_KEY_KP_5;
    constexpr uint32_t ZRX_GLFW_KEY_KP_6 = GLFW_KEY_KP_6;
    constexpr uint32_t ZRX_GLFW_KEY_KP_7 = GLFW_KEY_KP_7;
    constexpr uint32_t ZRX_GLFW_KEY_KP_8 = GLFW_KEY_KP_8;
    constexpr uint32_t ZRX_GLFW_KEY_KP_9 = GLFW_KEY_KP_9;
    constexpr uint32_t ZRX_GLFW_KEY_KP_DECIMAL = GLFW_KEY_KP_DECIMAL;
    constexpr uint32_t ZRX_GLFW_KEY_KP_DIVIDE = GLFW_KEY_KP_DIVIDE;
    constexpr uint32_t ZRX_GLFW_KEY_KP_MULTIPLY = GLFW_KEY_KP_MULTIPLY;
    constexpr uint32_t ZRX_GLFW_KEY_KP_SUBTRACT = GLFW_KEY_KP_SUBTRACT;
    constexpr uint32_t ZRX_GLFW_KEY_KP_ADD = GLFW_KEY_KP_ADD;
    constexpr uint32_t ZRX_GLFW_KEY_KP_ENTER = GLFW_KEY_KP_ENTER;
    constexpr uint32_t ZRX_GLFW_KEY_KP_EQUAL = GLFW_KEY_KP_EQUAL;
    constexpr uint32_t ZRX_GLFW_KEY_LEFT_SHIFT = GLFW_KEY_LEFT_SHIFT;
    constexpr uint32_t ZRX_GLFW_KEY_LEFT_CONTROL = GLFW_KEY_LEFT_CONTROL;
    constexpr uint32_t ZRX_GLFW_KEY_LEFT_ALT = GLFW_KEY_LEFT_ALT;
    constexpr uint32_t ZRX_GLFW_KEY_LEFT_SUPER = GLFW_KEY_LEFT_SUPER;
    constexpr uint32_t ZRX_GLFW_KEY_RIGHT_SHIFT = GLFW_KEY_RIGHT_SHIFT;
    constexpr uint32_t ZRX_GLFW_KEY_RIGHT_CONTROL = GLFW_KEY_RIGHT_CONTROL;
    constexpr uint32_t ZRX_GLFW_KEY_RIGHT_ALT = GLFW_KEY_RIGHT_ALT;
    constexpr uint32_t ZRX_GLFW_KEY_RIGHT_SUPER = GLFW_KEY_RIGHT_SUPER;
    constexpr uint32_t ZRX_GLFW_KEY_MENU = GLFW_KEY_MENU;

    constexpr uint32_t ZRX_GLFW_KEY_LAST = GLFW_KEY_LAST;

    /* Mouse buttons */
    constexpr uint32_t ZRX_GLFW_MOUSE_BUTTON_1 = GLFW_MOUSE_BUTTON_1;
    constexpr uint32_t ZRX_GLFW_MOUSE_BUTTON_2 = GLFW_MOUSE_BUTTON_2;
    constexpr uint32_t ZRX_GLFW_MOUSE_BUTTON_3 = GLFW_MOUSE_BUTTON_3;
    constexpr uint32_t ZRX_GLFW_MOUSE_BUTTON_4 = GLFW_MOUSE_BUTTON_4;
    constexpr uint32_t ZRX_GLFW_MOUSE_BUTTON_5 = GLFW_MOUSE_BUTTON_5;
    constexpr uint32_t ZRX_GLFW_MOUSE_BUTTON_6 = GLFW_MOUSE_BUTTON_6;
    constexpr uint32_t ZRX_GLFW_MOUSE_BUTTON_7 = GLFW_MOUSE_BUTTON_7;
    constexpr uint32_t ZRX_GLFW_MOUSE_BUTTON_8 = GLFW_MOUSE_BUTTON_8;
    constexpr uint32_t ZRX_GLFW_MOUSE_BUTTON_LAST = GLFW_MOUSE_BUTTON_LAST;
    constexpr uint32_t ZRX_GLFW_MOUSE_BUTTON_LEFT = GLFW_MOUSE_BUTTON_LEFT;
    constexpr uint32_t ZRX_GLFW_MOUSE_BUTTON_RIGHT = GLFW_MOUSE_BUTTON_RIGHT;
    constexpr uint32_t ZRX_GLFW_MOUSE_BUTTON_MIDDLE = GLFW_MOUSE_BUTTON_MIDDLE;

    /* Windows */

    using ::LPCSTR;
    using ::MessageBoxA;

    /* Other macros */
    constexpr uint32_t WIN_MB_OK = MB_OK;
    constexpr uint32_t WIN_EXIT_FAILURE = EXIT_FAILURE;
    constexpr uint32_t WIN_EXIT_SUCCESS = EXIT_SUCCESS;
}
