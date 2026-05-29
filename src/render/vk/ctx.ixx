module;

export module Cinder.Render.Vulkan:Context;

import vk_mem_alloc;
import vulkan;
import std;
import glfw;

import Cinder.Globals;

export namespace zrx {
class GraphicsQueue;

/**
 * Helper structure used to pass handles to essential Vulkan objects which are used while interacting with the API.
 * Introduced so that we can preserve top-down data flow and no object needs to refer to a renderer object
 * to get access to these.
 */
struct RendererContext {
    unique_ptr<vk::raii::PhysicalDevice> physical_device;
    unique_ptr<vk::raii::Device> device;
    unique_ptr<vk::raii::CommandPool> command_pool;
    unique_ptr<GraphicsQueue> graphics_queue;
    unique_ptr<vma::raii::Allocator> allocator;
    GLFWwindow *window;
    uint32_t current_frame_idx = 0;
};
} // zrx
