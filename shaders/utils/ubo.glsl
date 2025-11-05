#ifdef __cplusplus
#define GLSL_ALIGN4  alignas(4)
#define GLSL_ALIGN8  alignas(8)
#define GLSL_ALIGN16 alignas(16)
#else
#define GLSL_ALIGN4
#define GLSL_ALIGN8
#define GLSL_ALIGN16
#endif

struct WindowRes {
    GLSL_ALIGN4 uint width;
    GLSL_ALIGN4 uint height;
};

struct Matrices {
    GLSL_ALIGN16 mat4 model;
    GLSL_ALIGN16 mat4 view;
    GLSL_ALIGN16 mat4 proj;
    GLSL_ALIGN16 mat4 view_inverse;
    GLSL_ALIGN16 mat4 proj_inverse;
    GLSL_ALIGN16 mat4 vp_inverse;
    GLSL_ALIGN16 mat4 static_view;
    GLSL_ALIGN16 mat4 cubemap_capture_views[6];
    GLSL_ALIGN16 mat4 cubemap_capture_proj;
};

struct LightData {
    GLSL_ALIGN16 vec3 direction;
    GLSL_ALIGN16 vec3 color;
    GLSL_ALIGN4  float intensity;
    GLSL_ALIGN16 mat4 proj_x_view;
};

struct MiscData {
    GLSL_ALIGN4  float debug_number;
    GLSL_ALIGN4  float z_near;
    GLSL_ALIGN4  float z_far;
    GLSL_ALIGN4  uint use_ssao;
    GLSL_ALIGN16 vec3 camera_pos;
    GLSL_ALIGN4  float bias_weight_1;
    GLSL_ALIGN4  float bias_weight_2;
};
