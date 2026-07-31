#define MAX_MODEL_MESH_COUNT 256

#ifndef __cplusplus
#define BindlessHandle uint
#endif

struct MeshDescription {
    uint vertex_offset;
    uint index_offset;
    BindlessHandle base_color_id;
    BindlessHandle normal_id;
    BindlessHandle orm_id;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};
