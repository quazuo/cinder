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

set(GLFW_BUILD_DOCS     OFF CACHE BOOL "GLFW build documentation" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "GLFW build documentation" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "GLFW build documentation" FORCE)

FetchContent_Declare(
        glfw
        GIT_REPOSITORY  https://github.com/glfw/glfw
        GIT_TAG         b00e6a8
)
FetchContent_MakeAvailable(glfw)

# ========================= ImGUI =========================

FetchContent_Declare(
        imgui
        GIT_REPOSITORY  https://github.com/ocornut/imgui
        GIT_TAG         6acba3b
)
FetchContent_MakeAvailable(imgui)

file(GLOB IMGUI_SRCS
        "${imgui_SOURCE_DIR}/*.h"
        "${imgui_SOURCE_DIR}/*.cpp"
)

set(IMGUI_IMPL_SRCS
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.h
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.h
        ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)

add_library(imgui STATIC ${IMGUI_SRCS} ${IMGUI_IMPL_SRCS})

target_link_libraries(imgui PUBLIC Vulkan::Vulkan glfw)

target_compile_options(imgui PRIVATE -w)

target_include_directories(imgui PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
)

# ========================= Assimp =========================

set(BUILD_SHARED_LIBS   OFF CACHE BOOL "Build shared libraries" FORCE)
set(ASSIMP_BUILD_TESTS  OFF CACHE BOOL "Assimp build tests"     FORCE)
set(ASSIMP_INSTALL      OFF CACHE BOOL "Assimp install"         FORCE)
set(ASSIMP_INSTALL_PDB  OFF CACHE BOOL "Assimp install PDB"     FORCE)

FetchContent_Declare(
        assimp
        GIT_REPOSITORY  https://github.com/assimp/assimp
        GIT_TAG         392a658
)
FetchContent_MakeAvailable(assimp)

# ========================= enkiTS =========================

FetchContent_Declare(
        enki
        GIT_REPOSITORY  https://github.com/dougbinks/enkits
        GIT_TAG         0289cf6
)
FetchContent_MakeAvailable(enki)

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
        spirv-reflect
        GIT_REPOSITORY  https://github.com/KhronosGroup/SPIRV-Reflect
        GIT_TAG         c90b7b7
)
FetchContent_MakeAvailable(spirv-reflect)

# ========================= glm =========================

add_compile_definitions(
        GLM_FORCE_DEPTH_ZERO_TO_ONE
        GLM_ENABLE_CXX_20
        GLM_ENABLE_EXPERIMENTAL
        GLM_FORCE_RADIANS
)

FetchContent_Declare(
        glm
        GIT_REPOSITORY  https://github.com/g-truc/glm
        GIT_TAG         6f14f47
)
FetchContent_MakeAvailable(glm)

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

target_link_libraries(imguizmo PUBLIC glm imgui)

target_include_directories(imguizmo PUBLIC
        ${imguizmo_SOURCE_DIR}/imguizmo_quat
)

target_compile_options(imguizmo PRIVATE -DIMGUIZMO_USES_GLM -DIMGUIZMO_IMGUI_FOLDER= )

# ========================= LegitProfiler =========================
# TODO - implement this as my own version... the #includes are annoying

FetchContent_Declare(
        legit-profiler
        GIT_REPOSITORY  https://github.com/Raikiri/LegitProfiler
        GIT_TAG         4c647dc
)
FetchContent_MakeAvailable(legit-profiler)

add_library(legit-profiler INTERFACE)

target_include_directories(legit-profiler INTERFACE
        ${legit-profiler_SOURCE_DIR}
)

# ========================= imgui-filebrowser =========================

FetchContent_Declare(
        imgui-filebrowser
        GIT_REPOSITORY  https://github.com/AirGuanZ/imgui-filebrowser
        GIT_TAG         47a1884
)
FetchContent_MakeAvailable(imgui-filebrowser)

add_library(imgui-filebrowser STATIC
        ${imgui-filebrowser_SOURCE_DIR}/imfilebrowser.h
)

target_link_libraries(imgui-filebrowser PUBLIC imgui)

target_include_directories(imgui-filebrowser PUBLIC
        ${imgui-filebrowser_SOURCE_DIR}
)

# ========================= General =========================

include_directories(
        deps/
        .
)

set(ALL_LIBS
        assimp
        enkiTS
        glfw
        glm::glm
        imgui
        imguizmo
        imgui-filebrowser
        legit-profiler
        spirv-reflect-static
        vk-bootstrap
        Vulkan::Vulkan
        VulkanHppModule
        VulkanMemoryAllocator-Hpp
)

file(GLOB HEADER_ONLY_DEPS_IMPL_SRCS
        "deps_extras/impls/stb_image.cpp"
)

file(GLOB DEPS_CUSTOM_MODULES "deps_extras/custom_modules/*.ixx")

file(GLOB_RECURSE CINDER_SRCS "src/*.cpp")
file(GLOB_RECURSE CINDER_MODULES "src/*.ixx")

add_executable(cinder)

target_sources(cinder
        PUBLIC FILE_SET CXX_MODULES FILES
                ${CINDER_MODULES}
                ${DEPS_CUSTOM_MODULES}
                "${glm_SOURCE_DIR}/glm/glm.cppm"
                "${vmahpp_SOURCE_DIR}/vk_mem_alloc.cppm"
        PRIVATE
                ${CINDER_SRCS}
                ${HEADER_ONLY_DEPS_IMPL_SRCS})

target_link_libraries(cinder PRIVATE ${ALL_LIBS})
