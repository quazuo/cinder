#pragma once

#include "libs.hpp"
#include "globals.hpp"
#include "src/utils/input-manager.hpp"

struct GLFWwindow;

namespace zrx {
struct Rotator {
    glm::vec2 rot = {0, 0};

    [[nodiscard]] glm::vec2 operator*() const { return rot; }

    Rotator& operator=(glm::vec2 other);

    Rotator& operator+=(glm::vec2 other);

    Rotator& operator-=(glm::vec2 other);

    struct ViewVectors {
        glm::vec3 front, right, up;
    };

    [[nodiscard]] ViewVectors get_view_vectors() const;
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

    [[nodiscard]] glm::vec3 get_pos() const { return pos; }

    [[nodiscard]] glm::mat4 get_view_matrix() const;

    [[nodiscard]] glm::mat4 get_static_view_matrix() const;

    [[nodiscard]] glm::mat4 get_projection_matrix() const;

    [[nodiscard]] Rotator::ViewVectors get_view_vectors() const { return rotator.get_view_vectors(); }

    [[nodiscard]] std::pair<float, float> get_clipping_planes() const { return {z_near, z_far}; }

    void render_gui_section();

private:
    static void scroll_callback(GLFWwindow *window, double dx, double dy);

    void bind_camera_lock_key();

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
