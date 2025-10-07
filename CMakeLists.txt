cmake_minimum_required(VERSION 3.30.2 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_MODULE_STD 1)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "a9e1cf81-9932-4810-974b-6eccaf14e457")

include(FetchContent)

project(cinder LANGUAGES CXX)

# add_compile_options(-fsanitize=address)
# add_link_options(-fsanitize=address)

# ========================= Vulkan =========================

find_package(Vulkan REQUIRED)

if( ${Vulkan_VERSION} VERSION_LESS "1.3.256" )
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

# ========================= VulkanMemoryAllocator =========================

add_subdirectory(deps/VulkanMemoryAllocator)
# find_package(VulkanMemoryAllocator CONFIG REQUIRED)

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

# ========================= imGuIZMO.quat =========================

set(IMGUIZMO_QUAT_SRCS
        deps/imGuIZMO.quat/imguizmo_quat/imGuIZMOquat.cpp
)

# ========================= vk-bootstrap =========================

add_subdirectory(deps/vk-bootstrap)

# ========================= SPIRV-Reflect =========================

set(SPIRV_REFLECT_STATIC_LIB ON)
add_subdirectory(deps/SPIRV-Reflect)

# ========================= glm =========================

add_subdirectory(deps/glm)

# ========================= General =========================

include_directories(
        cmake-build-debug/deps/assimp/include/
        deps/
        deps/glfw/include/GLFW
        deps/imgui/
        deps/assimp/include/
        deps/glfw/include/
        deps/glm/
        deps/imGuIZMO.quat/
        deps/pix/Include/
        deps/LegitProfiler/
        .
)

set(ALL_LIBS
        assimp
        glfw
        glm::glm
        Vulkan::Vulkan
        VulkanHppModule
        GPUOpen::VulkanMemoryAllocator
        vk-bootstrap::vk-bootstrap
        spirv-reflect-static
)

file(GLOB HEADER_ONLY_DEPS_IMPL_SRCS
        "deps_extras/impls/stb_image.cpp"
        "deps_extras/impls/vma.cpp"
)

file(GLOB DEPS_CUSTOM_MODULES
        "deps_extras/custom_modules/*.ixx"
)

file(GLOB CINDER_SRCS
        "src/*.cpp"
        "src/render/*.cpp"
        "src/render/vk/*.cpp"
        "src/render/gui/*.cpp"
        "src/render/mesh/*.cpp"
        "src/render/graph/*.cpp"
        "src/utils/*.cpp"
)

file(GLOB CINDER_MODULES
        "src/*.ixx"
        "src/render/*.ixx"
        "src/render/vk/*.ixx"
        "src/render/gui/*.ixx"
        "src/render/mesh/*.ixx"
        "src/render/graph/*.ixx"
        "src/utils/*.ixx"
)

add_executable(cinder)

target_sources(cinder
        PUBLIC
        FILE_SET CXX_MODULES
        FILES
                ${CINDER_MODULES}
                ${DEPS_CUSTOM_MODULES}
                "deps/glm/glm/glm.cppm"
        PRIVATE
                ${CINDER_SRCS}
                ${IMGUI_SRCS}
                ${IMGUI_IMPL_SRCS}
                ${IMGUIZMO_QUAT_SRCS}
                ${HEADER_ONLY_DEPS_IMPL_SRCS})

target_link_libraries(cinder PRIVATE ${ALL_LIBS})

target_compile_options(cinder PRIVATE -DIMGUIZMO_USES_GLM)

#find_package(glm CONFIG REQUIRED)
#target_link_libraries(main PRIVATE glm::glm)
