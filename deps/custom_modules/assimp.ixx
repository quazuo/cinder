module;

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

export module Assimp;

export {
    using ::aiMaterial;
    using ::aiScene;
    using ::aiMesh;
    using ::aiNode;
    using ::aiVector3D;
    using ::aiMatrix4x4;
    using ::aiString;

    using ::aiReturn;
    using ::aiReturn_SUCCESS;

    using ::aiTextureType_BASE_COLOR;
    using ::aiTextureType_NORMALS;

    using ::aiProcess_RemoveRedundantMaterials;
    using ::aiProcess_FindInstances;
    using ::aiProcess_OptimizeMeshes;
    using ::aiProcess_OptimizeGraph;
    using ::aiProcess_FixInfacingNormals;
    using ::aiProcess_Triangulate;
    using ::aiProcess_JoinIdenticalVertices;
    using ::aiProcess_CalcTangentSpace;
    using ::aiProcess_SortByPType;
    using ::aiProcess_ImproveCacheLocality;
    using ::aiProcess_ValidateDataStructure;
}

export namespace Assimp {
    using Assimp::Importer;
}
