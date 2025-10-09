module;

export module Cinder.Render:Camera;

import glm;
import std;
import glfw;

import Cinder.Globals;
import Cinder.Utils;

export namespace zrx {
class Rotator {
    glm::vec2 rot = {0, 0};

public:
    auto operator*() const -> glm::vec2 { return rot; }

    auto operator=(glm::vec2 other)  -> Rotator&;

    auto operator+=(glm::vec2 other) -> Rotator&;

    auto operator-=(glm::vec2 other) -> Rotator&;

    struct ViewVectors {
        glm::vec3 front, right, up;
    };

    auto get_view_vectors() const -> ViewVectors;
};

class Camera {
    GLFWwindow *window = nullptr;

    float aspect_ratio = 4.0f / 3.0f;
    float field_of_view = 80.0f;
    float z_near = 0.01f;
    float z_far = 500.0f;

    glm::vec3 pos = {0.0f, 0.0f, -2.0f};
    Rotator rotator;
    glm::vec3 front{}, right{}, up{};

    bool is_locked_cursor = false;
    bool is_locked_cam = true;
    float locked_radius = 20.0f;
    Rotator locked_rotator;

    float rotation_speed = 2.5f;
    float movement_speed = 5.0f;

    unique_ptr<InputManager> input_manager;

public:
    explicit Camera(GLFWwindow *w);

    void tick(float delta_time);

    auto get_pos() const -> glm::vec3 { return pos; }

    auto get_view_matrix() const -> glm::mat4;

    auto get_static_view_matrix() const -> glm::mat4;

    auto get_projection_matrix() const -> glm::mat4;

    auto get_view_vectors() const -> Rotator::ViewVectors { return rotator.get_view_vectors(); }

    auto get_clipping_planes() const -> std::pair<float, float> { return {z_near, z_far}; }

    void render_gui_section();

private:
    static void scroll_callback(GLFWwindow *window, double dx, double dy);

    void bind_cursor_lock_key();

    /**
     * Binds keys used to rotate the camera.
     */
    void bind_mouse_drag_callback();

    /**
     * Binds keys used to rotate the camera in freecam mode.
     */
    void bind_freecam_rotation_keys();

    /**
     * Binds keys used to move the camera in freecam mode.
     */
    void bind_freecam_movement_keys();

    void tick_mouse_movement(float delta_time);

    void tick_locked_mode();

    void update_aspect_ratio();

    void update_vecs();

    void center_cursor() const;
};
} // zrx
