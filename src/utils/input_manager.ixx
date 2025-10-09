module;

export module Cinder.Utils:InputManager;

import glm;
import std;
import glfw;

import Cinder.Globals;

export namespace zrx {
enum class EActivationType {
    PRESS_ANY,
    PRESS_ONCE,
    RELEASE_ONCE,
};

using EInputCallback = std::function<void(float)>;

using EMouseDragCallback = std::function<void(double, double)>;

/**
 * Class managing keyboard and mouse events, detecting them and calling certain callbacks when they occur.
 * This can safely be instantiated multiple times, handling different events across different instances.
 */
class InputManager {
    GLFWwindow *window = nullptr;

    using KeyCallbackInfo = std::pair<EActivationType, EInputCallback>;
    std::unordered_map<glfw::Key, KeyCallbackInfo> callback_map;

    enum class KeyState {
        PRESSED,
        RELEASED
    };

    std::unordered_map<glfw::Key, KeyState> key_state_map;

    std::unordered_map<glfw::MouseButton, EMouseDragCallback> mouse_drag_callback_map;
    std::unordered_map<glfw::MouseButton, KeyState> mouse_button_state_map;
    glm::dvec2 last_mouse_pos{};

public:
    explicit InputManager(GLFWwindow *w) : window(w) {}

    /**
     * Binds a given callback to a keyboard event. Only one callback can be bound at a time,
     * so this will overwrite an earlier bound callback if there was any.
     *
     * @param key Key which on press should fire the callback.
     * @param type The way the key should be managed.
     * @param fn The callback.
     */
    void bind_callback(glfw::Key key, EActivationType type, const EInputCallback& fn);

    /**
     * Binds a given callback to a mouse drag event. Only one callback can be bound at a time,
     * so this will overwrite an earlier bound callback if there was any.
     *
     * @param button Mouse button which on drag should fire the callback.
     * @param f The callback.
     */
    void bind_mouse_drag_callback(glfw::MouseButton button, const EMouseDragCallback& fn);

    void tick(float deltaTime);

private:
    /**
     * Checks if a given keyboard event has occured.
     *
     * @param key Key to check.
     * @param type Type of event the caller is interested in.
     * @return Did the event occur?
     */
    auto check_key(glfw::Key key, EActivationType type) -> bool;
};
} // zrx
