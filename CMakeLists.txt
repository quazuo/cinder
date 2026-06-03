cmake_minimum_required(VERSION 3.30.2 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_MODULE_STD 1)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "d0edc3af-4c50-42ea-a356-e2862fe7a444")

include(FetchContent)

project(cinder LANGUAGES CXX)

if(MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /permissive- /Zc:wchar_t /Zc:forScope /Zc:inline")
endif()

# add_compile_options(-fsanitize=address)
# add_link_options(-fsanitize=address)

# ========================= Vulkan =========================

find_package(Vulkan REQUIRED)

if(${Vulkan_VERSION} VERSION_LESS "1.3.256")
        message(FATAL_ERROR
                "Minimum required Vulkan version for C++ modules is 1.3.256. "
                "Found ${Vulkan_VERSION}."
        )
endif()

add_library(VulkanHppModule)
target_sources(VulkanHppModule PUBLIC
        FILE_SET CXX_MODULES
        BASE_DIRS ${Vulkan_INCLUDE_DIR}
        FILES ${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm
)
target_compile_definitions(VulkanHppModule PUBLIC
        VULKAN_HPP_NO_SPACESHIP_OPERATOR
        VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
)
target_compile_features(VulkanHppModule PUBLIC cxx_std_20)
target_link_libraries(VulkanHppModule PUBLIC Vulkan::Vulkan)

# ========================= VMA =========================

FetchContent_Declare(
        vmahpp
        URL      https://github.com/YaaZ/VulkanMemoryAllocator-Hpp/releases/download/v3.3.0+3/VulkanMemoryAllocator-Hpp-3.3.0.tar.gz
        URL_HASH SHA256=8c2a0573babe1f86f3241d12b4c9df40bb944649b7f4239c1c611797f8af6cd5
)
FetchContent_MakeAvailable(vmahpp)

# ========================= GLFW =========================

set(GLFW_BUILD_DOCS OFF CACHE BOOL "GLFW build documentation" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "GLFW build documentation" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "GLFW build documentation" FORCE)
add_subdirectory(deps/glfw)

# ========================= ImGUI =========================

file(GLOB IMGUI_SRCS
        "deps/imgui/*.h"
        "deps/imgui/*.cpp"
)

set(IMGUI_IMPL_SRCS
        deps/imgui/backends/imgui_impl_glfw.h
        deps/imgui/backends/imgui_impl_glfw.cpp
        deps/imgui/backends/imgui_impl_vulkan.h
        deps/imgui/backends/imgui_impl_vulkan.cpp
)

set_source_files_properties(
        ${IMGUI_SRCS}
        ${IMGUI_IMPL_SRCS}
        PROPERTIES COMPILE_FLAGS "-w"
)

# ========================= Assimp =========================

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "Assimp build tests" FORCE)
set(ASSIMP_INSTALL OFF CACHE BOOL "Assimp install" FORCE)
set(ASSIMP_INSTALL_PDB OFF CACHE BOOL "Assimp install PDB" FORCE)
add_subdirectory(deps/assimp)

# ========================= enkiTS =========================

FetchContent_Declare(
        enki
        GIT_REPOSITORY  https://github.com/dougbinks/enkits
        GIT_TAG         0289cf6
)
FetchContent_MakeAvailable(enki)

# ========================= imGuIZMO.quat =========================

FetchContent_Declare(
        imguizmo
        GIT_REPOSITORY  https://github.com/BrutPitt/imGuIZMO.quat
        GIT_TAG         23c8e68
)
FetchContent_MakeAvailable(imguizmo)

add_library(imguizmo STATIC
        ${imguizmo_SOURCE_DIR}/imguizmo_quat/imguizmo_quat.cpp
        ${imguizmo_SOURCE_DIR}/imguizmo_quat/imGuIZMOquat.cpp
)

target_include_directories(imguizmo PUBLIC
        ${imguizmo_SOURCE_DIR}/imguizmo_quat
)

target_compile_options(imguizmo PRIVATE -DIMGUIZMO_USES_GLM)

# ========================= vk-bootstrap =========================

FetchContent_Declare(
        vkb
        GIT_REPOSITORY  https://github.com/charles-lunarg/vk-bootstrap
        GIT_TAG         da41b78
)
FetchContent_MakeAvailable(vkb)

# ========================= SPIRV-Reflect =========================

set(SPIRV_REFLECT_STATIC_LIB ON)

FetchContent_Declare(
        spirv-refl
        GIT_REPOSITORY  https://github.com/KhronosGroup/SPIRV-Reflect
        GIT_TAG         c90b7b7
)
FetchContent_MakeAvailable(spirv-refl)

# ========================= glm =========================

set(GLM_ENABLE_CXX_20 ON)
set(GLM_FORCE_DEPTH_ZERO_TO_ONE ON)
set(GLM_ENABLE_EXPERIMENTAL ON)
add_subdirectory(deps/glm)

target_compile_definitions(glm INTERFACE
        GLM_ENABLE_CXX_20
        GLM_FORCE_DEPTH_ZERO_TO_ONE
        GLM_ENABLE_EXPERIMENTAL
)

# ========================= General =========================

include_directories(
        deps/
        deps/assimp/include/
        deps/glfw/include/
        deps/glfw/include/GLFW
        deps/glm/
        deps/imgui/
        deps/LegitProfiler/
        deps/pix/Include/
        .
)

set(ALL_LIBS
        assimp
        enkiTS
        glfw
        glm::glm
        imguizmo
        spirv-reflect-static
        vk-bootstrap
        Vulkan::Vulkan
        VulkanHppModule
        VulkanMemoryAllocator-Hpp::VulkanMemoryAllocator-Hpp
)

file(GLOB HEADER_ONLY_DEPS_IMPL_SRCS
        "deps_extras/impls/stb_image.cpp"
        "deps_extras/impls/vma.cpp"
)

file(GLOB DEPS_CUSTOM_MODULES
        "deps_extras/custom_modules/*.ixx"
)

file(GLOB_RECURSE CINDER_SRCS "src/*.cpp")
file(GLOB_RECURSE CINDER_MODULES "src/*.ixx")

add_executable(cinder)

target_sources(cinder
        PUBLIC
        FILE_SET CXX_MODULES
        FILES
                ${CINDER_MODULES}
                ${DEPS_CUSTOM_MODULES}
                "deps/glm/glm/glm.cppm"
                "${vmahpp_SOURCE_DIR}/vk_mem_alloc.cppm"
        PRIVATE
                ${CINDER_SRCS}
                ${IMGUI_SRCS}
                ${IMGUI_IMPL_SRCS}
                ${HEADER_ONLY_DEPS_IMPL_SRCS})

target_link_libraries(cinder PRIVATE ${ALL_LIBS})

#find_package(glm CONFIG REQUIRED)
#target_link_libraries(main PRIVATE glm::glm)
