#pragma once

namespace zrx {
enum class FileType {
    MODEL,
    BASE_COLOR_PNG,
    NORMAL_PNG,
    ORM_PNG,
    RMA_PNG,
    AO_PNG,
    ROUGHNESS_PNG,
    METALLIC_PNG,
    ENVMAP_HDR,
};

[[nodiscard]] static vector<std::string> get_file_type_extensions(const FileType type) {
    switch (type) {
        case FileType::MODEL:
            return {".obj", ".fbx", ".gltf"};
        case FileType::BASE_COLOR_PNG:
        case FileType::NORMAL_PNG:
        case FileType::ORM_PNG:
        case FileType::RMA_PNG:
        case FileType::AO_PNG:
        case FileType::ROUGHNESS_PNG:
        case FileType::METALLIC_PNG:
            return {".png"};
        case FileType::ENVMAP_HDR:
            return {".hdr"};
        default:
            throw std::runtime_error("unexpected filetype in get_file_type_extensions");
    }
}

[[nodiscard]] static bool is_file_type_optional(const FileType type) {
    switch (type) {
        case FileType::AO_PNG:
        case FileType::METALLIC_PNG:
            return true;
        default:
            return false;
    }
}

[[nodiscard]] static std::string get_file_type_load_label(const FileType type) {
    switch (type) {
        case FileType::MODEL:
            return "Load model...";
        case FileType::BASE_COLOR_PNG:
            return "Load base color texture...";
        case FileType::NORMAL_PNG:
            return "Load normal map...";
        case FileType::ORM_PNG:
            return "Load ORM map...";
        case FileType::RMA_PNG:
            return "Load RMA map...";
        case FileType::AO_PNG:
            return "Load AO map...";
        case FileType::ROUGHNESS_PNG:
            return "Load roughness map...";
        case FileType::METALLIC_PNG:
            return "Load metallic map...";
        case FileType::ENVMAP_HDR:
            return "Load environment map...";
        default:
            throw std::runtime_error("unexpected filetype in get_file_type_load_label");
    }
}

struct FileLoadScheme {
    std::string name;
    std::set<FileType> requirements;
};

static const vector<FileLoadScheme> file_load_schemes{
    {
        "Default (model packed with materials)",
        {
            FileType::MODEL,
        }
    },
    {
        "One material: Base color + Normal + ORM",
        {
            FileType::MODEL,
            FileType::BASE_COLOR_PNG,
            FileType::NORMAL_PNG,
            FileType::ORM_PNG,
        }
    },
    {
        "One material: Base color + Normal + RMA",
        {
            FileType::MODEL,
            FileType::BASE_COLOR_PNG,
            FileType::NORMAL_PNG,
            FileType::RMA_PNG,
        }
    },
    {
        "One material: Base color + Normal + AO + Roughness + Metallic",
        {
            FileType::MODEL,
            FileType::BASE_COLOR_PNG,
            FileType::NORMAL_PNG,
            FileType::AO_PNG,
            FileType::ROUGHNESS_PNG,
            FileType::METALLIC_PNG
        }
    },
};
} // zrx
