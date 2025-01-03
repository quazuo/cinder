#pragma once

#include <functional>
#include <optional>

#include "src/render/libs.hpp"

struct GLFWwindow;

namespace zrx {
enum class EActivationType {
    PRESS_ANY,
    PRESS_ONCE,
    RELEASE_ONCE,
};

using EKey = int;
using EInputCallback = std::function<void(float)>;

using EMouseButton = int;
using EMouseDragCallback = std::function<void(double, double)>;

/**
 * Class managing keyboard and mouse events, detecting them and calling certain callbacks when they occur.
 * This can safely be instantiated multiple times, handling different events across different instances.
 */
class InputManager {
    GLFWwindow *window = nullptr;

    using KeyCallbackInfo = std::pair<EActivationType, EInputCallback>;
    std::unordered_map<EKey, KeyCallbackInfo> callback_map;

    enum class KeyState {
        PRESSED,
        RELEASED
    };

    std::unordered_map<EKey, KeyState> key_state_map;

    std::unordered_map<EMouseButton, EMouseDragCallback> mouse_drag_callback_map;
    std::unordered_map<EMouseButton, KeyState> mouse_button_state_map;
    glm::dvec2 last_mouse_pos{};

public:
    explicit InputManager(GLFWwindow *w) : window(w) {}

    /**
     * Binds a given callback to a keyboard event. Only one callback can be bound at a time,
     * so this will overwrite an earlier bound callback if there was any.
     *
     * @param k Key which on press should fire the callback.
     * @param type The way the key should be managed.
     * @param f The callback.
     */
    void bind_callback(EKey k, EActivationType type, const EInputCallback& f);

    /**
     * Binds a given callback to a mouse drag event. Only one callback can be bound at a time,
     * so this will overwrite an earlier bound callback if there was any.
     *
     * @param button Mouse button which on drag should fire the callback.
     * @param f The callback.
     */
    void bind_mouse_drag_callback(EMouseButton button, const EMouseDragCallback& f);

    void tick(float deltaTime);

private:
    /**
     * Checks if a given keyboard event has occured.
     *
     * @param key Key to check.
     * @param type Type of event the caller is interested in.
     * @return Did the event occur?
     */
    bool check_key(EKey key, EActivationType type);
};
} // zrx
