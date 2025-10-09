struct WindowRes {
    uint width;
    uint height;
};

struct Matrices {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 view_inverse;
    mat4 proj_inverse;
    mat4 vp_inverse;
    mat4 static_view;
    mat4 cubemap_capture_views[6];
    mat4 cubemap_capture_proj;
};

struct LightData {
    vec3 direction;
    vec3 color;
    float intensity;
    mat4 proj_x_view;
};

struct MiscData {
    float debug_number;
    float z_near;
    float z_far;
    uint use_ssao;
    vec3 camera_pos;
};
