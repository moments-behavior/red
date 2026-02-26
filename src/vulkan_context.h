#ifndef VULKAN_CONTEXT_H
#define VULKAN_CONTEXT_H

#ifdef __APPLE__

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_vulkan.h"

#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

struct VulkanTexture {
    VkImage         image          = VK_NULL_HANDLE;
    VkDeviceMemory  image_memory   = VK_NULL_HANDLE;
    VkImageView     image_view     = VK_NULL_HANDLE;
    VkBuffer        staging_buffer = VK_NULL_HANDLE;   // persistently mapped
    VkDeviceMemory  staging_memory = VK_NULL_HANDLE;
    void           *staging_mapped = nullptr;          // CPU-writable pointer
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;   // used as ImTextureID
};

struct VulkanContext {
    VkInstance           instance        = VK_NULL_HANDLE;
    VkPhysicalDevice     physical_device = VK_NULL_HANDLE;
    VkDevice             device          = VK_NULL_HANDLE;
    uint32_t             queue_family    = 0;
    VkQueue              queue           = VK_NULL_HANDLE;
    VkSurfaceKHR         surface         = VK_NULL_HANDLE;
    VkSwapchainKHR       swapchain       = VK_NULL_HANDLE;
    VkFormat             swapchain_format = VK_FORMAT_UNDEFINED;
    VkExtent2D           swapchain_extent = {0, 0};
    std::vector<VkImage>     swapchain_images;
    std::vector<VkImageView> swapchain_image_views;
    VkDescriptorPool     descriptor_pool = VK_NULL_HANDLE;
    VkCommandPool        command_pool    = VK_NULL_HANDLE;

    static const int FRAMES_IN_FLIGHT = 1;
    VkCommandBuffer  command_buffers[FRAMES_IN_FLIGHT] = {};
    VkSemaphore      image_available[FRAMES_IN_FLIGHT] = {};
    VkSemaphore      render_finished[FRAMES_IN_FLIGHT] = {};
    VkFence          in_flight_fences[FRAMES_IN_FLIGHT] = {};

    uint32_t current_frame = 0;
    uint32_t image_index   = 0;

    // Shared sampler for camera textures
    VkSampler texture_sampler = VK_NULL_HANDLE;

    // Loaded extension function pointers (KHR dynamic rendering)
    PFN_vkCmdBeginRenderingKHR fn_begin_rendering = nullptr;
    PFN_vkCmdEndRenderingKHR   fn_end_rendering   = nullptr;

    // Per-camera textures
    std::vector<VulkanTexture> textures;
};

extern VulkanContext *g_vk;

// ---------------------------------------------------------------------------
// Internal helpers (defined in this header — single translation unit)
// ---------------------------------------------------------------------------

static uint32_t vk_find_memory_type(uint32_t type_filter,
                                    VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(g_vk->physical_device, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }
    fprintf(stderr, "VK: failed to find suitable memory type\n");
    exit(EXIT_FAILURE);
}

static void vk_transition_image_layout(VkCommandBuffer cmd, VkImage image,
                                       VkImageLayout old_layout,
                                       VkImageLayout new_layout,
                                       VkAccessFlags src_access,
                                       VkAccessFlags dst_access,
                                       VkPipelineStageFlags src_stage,
                                       VkPipelineStageFlags dst_stage) {
    VkImageMemoryBarrier barrier = {};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout           = old_layout;
    barrier.newLayout           = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image               = image;
    barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier.srcAccessMask       = src_access;
    barrier.dstAccessMask       = dst_access;
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0,
                         0, nullptr, 0, nullptr, 1, &barrier);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void vk_init(GLFWwindow *window);
void vk_init_imgui();
void vk_create_texture(VulkanTexture &tex, uint32_t w, uint32_t h);
void vk_destroy_texture(VulkanTexture &tex);
void vk_allocate_textures(int num_cams, uint32_t *widths, uint32_t *heights);
bool vk_begin_frame();
void vk_upload_texture(int cam_idx, const uint8_t *rgba, uint32_t w, uint32_t h);
void vk_begin_rendering();
void vk_end_frame();
void vk_cleanup();

// ---------------------------------------------------------------------------
// Implementations
// ---------------------------------------------------------------------------

#define VK_CHECK(x) do { \
    VkResult _r = (x); \
    if (_r != VK_SUCCESS) { \
        fprintf(stderr, "VK error %d at %s:%d\n", _r, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

inline void vk_init(GLFWwindow *window) {
    g_vk = new VulkanContext();

    // --- Instance ---
    uint32_t glfw_ext_count = 0;
    const char **glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

    std::vector<const char *> extensions(glfw_exts, glfw_exts + glfw_ext_count);
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    VkApplicationInfo app_info = {};
    app_info.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName   = "RED";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName        = "No Engine";
    app_info.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion         = VK_API_VERSION_1_2;

    VkInstanceCreateInfo inst_info = {};
    inst_info.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_info.pApplicationInfo        = &app_info;
    inst_info.enabledExtensionCount   = (uint32_t)extensions.size();
    inst_info.ppEnabledExtensionNames = extensions.data();
    inst_info.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;

    VK_CHECK(vkCreateInstance(&inst_info, nullptr, &g_vk->instance));

    // --- Surface ---
    VK_CHECK(glfwCreateWindowSurface(g_vk->instance, window, nullptr, &g_vk->surface));

    // --- Physical device ---
    uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(g_vk->instance, &dev_count, nullptr);
    if (dev_count == 0) {
        fprintf(stderr, "VK: no Vulkan physical devices found\n");
        exit(EXIT_FAILURE);
    }
    std::vector<VkPhysicalDevice> devs(dev_count);
    vkEnumeratePhysicalDevices(g_vk->instance, &dev_count, devs.data());
    g_vk->physical_device = devs[0]; // pick first (MoltenVK exposes one)

    // --- Queue family ---
    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_vk->physical_device, &qf_count, nullptr);
    std::vector<VkQueueFamilyProperties> qf_props(qf_count);
    vkGetPhysicalDeviceQueueFamilyProperties(g_vk->physical_device, &qf_count, qf_props.data());
    g_vk->queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < qf_count; i++) {
        VkBool32 present_support = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(g_vk->physical_device, i,
                                             g_vk->surface, &present_support);
        if ((qf_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present_support) {
            g_vk->queue_family = i;
            break;
        }
    }
    if (g_vk->queue_family == UINT32_MAX) {
        fprintf(stderr, "VK: no suitable queue family\n");
        exit(EXIT_FAILURE);
    }

    // --- Logical device ---
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_ci = {};
    queue_ci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_ci.queueFamilyIndex = g_vk->queue_family;
    queue_ci.queueCount       = 1;
    queue_ci.pQueuePriorities = &queue_priority;

    const char *dev_extensions[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
        "VK_KHR_portability_subset",
    };

    VkPhysicalDeviceDynamicRenderingFeaturesKHR dyn_render_feat = {};
    dyn_render_feat.sType            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR;
    dyn_render_feat.dynamicRendering = VK_TRUE;

    VkDeviceCreateInfo dev_ci = {};
    dev_ci.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dev_ci.pNext                   = &dyn_render_feat;
    dev_ci.queueCreateInfoCount    = 1;
    dev_ci.pQueueCreateInfos       = &queue_ci;
    dev_ci.enabledExtensionCount   = (uint32_t)(sizeof(dev_extensions)/sizeof(dev_extensions[0]));
    dev_ci.ppEnabledExtensionNames = dev_extensions;

    VK_CHECK(vkCreateDevice(g_vk->physical_device, &dev_ci, nullptr, &g_vk->device));
    vkGetDeviceQueue(g_vk->device, g_vk->queue_family, 0, &g_vk->queue);

    // --- Swapchain ---
    VkSurfaceCapabilitiesKHR surf_caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_vk->physical_device, g_vk->surface, &surf_caps);

    uint32_t fmt_count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(g_vk->physical_device, g_vk->surface, &fmt_count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(fmt_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(g_vk->physical_device, g_vk->surface, &fmt_count, formats.data());

    VkSurfaceFormatKHR chosen_fmt = formats[0];
    for (auto &f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosen_fmt = f;
            break;
        }
    }
    g_vk->swapchain_format = chosen_fmt.format;
    g_vk->swapchain_extent = surf_caps.currentExtent;

    uint32_t image_count = surf_caps.minImageCount + 1;
    if (surf_caps.maxImageCount > 0 && image_count > surf_caps.maxImageCount)
        image_count = surf_caps.maxImageCount;

    VkSwapchainCreateInfoKHR sc_ci = {};
    sc_ci.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    sc_ci.surface          = g_vk->surface;
    sc_ci.minImageCount    = image_count;
    sc_ci.imageFormat      = chosen_fmt.format;
    sc_ci.imageColorSpace  = chosen_fmt.colorSpace;
    sc_ci.imageExtent      = g_vk->swapchain_extent;
    sc_ci.imageArrayLayers = 1;
    sc_ci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    sc_ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sc_ci.preTransform     = surf_caps.currentTransform;
    sc_ci.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sc_ci.presentMode      = VK_PRESENT_MODE_FIFO_KHR;
    sc_ci.clipped          = VK_TRUE;

    VK_CHECK(vkCreateSwapchainKHR(g_vk->device, &sc_ci, nullptr, &g_vk->swapchain));

    uint32_t sc_img_count = 0;
    vkGetSwapchainImagesKHR(g_vk->device, g_vk->swapchain, &sc_img_count, nullptr);
    g_vk->swapchain_images.resize(sc_img_count);
    vkGetSwapchainImagesKHR(g_vk->device, g_vk->swapchain, &sc_img_count,
                            g_vk->swapchain_images.data());

    g_vk->swapchain_image_views.resize(sc_img_count);
    for (uint32_t i = 0; i < sc_img_count; i++) {
        VkImageViewCreateInfo iv_ci = {};
        iv_ci.sType        = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        iv_ci.image        = g_vk->swapchain_images[i];
        iv_ci.viewType     = VK_IMAGE_VIEW_TYPE_2D;
        iv_ci.format       = g_vk->swapchain_format;
        iv_ci.components   = {VK_COMPONENT_SWIZZLE_IDENTITY,
                              VK_COMPONENT_SWIZZLE_IDENTITY,
                              VK_COMPONENT_SWIZZLE_IDENTITY,
                              VK_COMPONENT_SWIZZLE_IDENTITY};
        iv_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(g_vk->device, &iv_ci, nullptr,
                                   &g_vk->swapchain_image_views[i]));
    }

    // --- Command pool & buffers ---
    VkCommandPoolCreateInfo cp_ci = {};
    cp_ci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cp_ci.queueFamilyIndex = g_vk->queue_family;
    cp_ci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(g_vk->device, &cp_ci, nullptr, &g_vk->command_pool));

    VkCommandBufferAllocateInfo cb_ai = {};
    cb_ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cb_ai.commandPool        = g_vk->command_pool;
    cb_ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cb_ai.commandBufferCount = VulkanContext::FRAMES_IN_FLIGHT;
    VK_CHECK(vkAllocateCommandBuffers(g_vk->device, &cb_ai, g_vk->command_buffers));

    // --- Sync objects ---
    VkSemaphoreCreateInfo sem_ci = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fence_ci   = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                                    nullptr,
                                    VK_FENCE_CREATE_SIGNALED_BIT};
    for (int i = 0; i < VulkanContext::FRAMES_IN_FLIGHT; i++) {
        VK_CHECK(vkCreateSemaphore(g_vk->device, &sem_ci, nullptr, &g_vk->image_available[i]));
        VK_CHECK(vkCreateSemaphore(g_vk->device, &sem_ci, nullptr, &g_vk->render_finished[i]));
        VK_CHECK(vkCreateFence(g_vk->device, &fence_ci, nullptr, &g_vk->in_flight_fences[i]));
    }

    // --- Descriptor pool (ImGui + camera textures) ---
    VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1024},
    };
    VkDescriptorPoolCreateInfo dp_ci = {};
    dp_ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp_ci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dp_ci.maxSets       = 1024;
    dp_ci.poolSizeCount = 1;
    dp_ci.pPoolSizes    = pool_sizes;
    VK_CHECK(vkCreateDescriptorPool(g_vk->device, &dp_ci, nullptr, &g_vk->descriptor_pool));

    // Shared linear sampler for camera textures
    VkSamplerCreateInfo sampler_ci = {};
    sampler_ci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_ci.magFilter    = VK_FILTER_LINEAR;
    sampler_ci.minFilter    = VK_FILTER_LINEAR;
    sampler_ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_ci.maxLod       = VK_LOD_CLAMP_NONE;
    VK_CHECK(vkCreateSampler(g_vk->device, &sampler_ci, nullptr, &g_vk->texture_sampler));

    // Load KHR dynamic rendering extension function pointers
    g_vk->fn_begin_rendering = (PFN_vkCmdBeginRenderingKHR)
        vkGetDeviceProcAddr(g_vk->device, "vkCmdBeginRenderingKHR");
    g_vk->fn_end_rendering = (PFN_vkCmdEndRenderingKHR)
        vkGetDeviceProcAddr(g_vk->device, "vkCmdEndRenderingKHR");
    if (!g_vk->fn_begin_rendering || !g_vk->fn_end_rendering) {
        fprintf(stderr, "VK: vkCmdBeginRenderingKHR not available\n");
        exit(EXIT_FAILURE);
    }
}

inline void vk_init_imgui() {
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.ApiVersion      = VK_API_VERSION_1_2;
    init_info.Instance        = g_vk->instance;
    init_info.PhysicalDevice  = g_vk->physical_device;
    init_info.Device          = g_vk->device;
    init_info.QueueFamily     = g_vk->queue_family;
    init_info.Queue           = g_vk->queue;
    init_info.DescriptorPool  = g_vk->descriptor_pool;
    init_info.RenderPass      = VK_NULL_HANDLE;  // ignored with dynamic rendering
    init_info.MinImageCount   = (uint32_t)g_vk->swapchain_images.size();
    init_info.ImageCount      = (uint32_t)g_vk->swapchain_images.size();
    init_info.MSAASamples     = VK_SAMPLE_COUNT_1_BIT;
    init_info.UseDynamicRendering = true;

#ifdef IMGUI_IMPL_VULKAN_HAS_DYNAMIC_RENDERING
    VkPipelineRenderingCreateInfoKHR pipe_render_ci = {};
    pipe_render_ci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
    pipe_render_ci.colorAttachmentCount    = 1;
    pipe_render_ci.pColorAttachmentFormats = &g_vk->swapchain_format;
    init_info.PipelineRenderingCreateInfo  = pipe_render_ci;
#endif

    ImGui_ImplVulkan_Init(&init_info);
    // Font texture is uploaded automatically on the first NewFrame() call
}

inline void vk_create_texture(VulkanTexture &tex, uint32_t w, uint32_t h) {
    // --- VkImage ---
    VkImageCreateInfo img_ci = {};
    img_ci.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_ci.imageType     = VK_IMAGE_TYPE_2D;
    img_ci.format        = VK_FORMAT_R8G8B8A8_UNORM;
    img_ci.extent        = {w, h, 1};
    img_ci.mipLevels     = 1;
    img_ci.arrayLayers   = 1;
    img_ci.samples       = VK_SAMPLE_COUNT_1_BIT;
    img_ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
    img_ci.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    img_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(g_vk->device, &img_ci, nullptr, &tex.image));

    VkMemoryRequirements img_mem_req;
    vkGetImageMemoryRequirements(g_vk->device, tex.image, &img_mem_req);
    VkMemoryAllocateInfo img_alloc = {};
    img_alloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    img_alloc.allocationSize  = img_mem_req.size;
    img_alloc.memoryTypeIndex = vk_find_memory_type(img_mem_req.memoryTypeBits,
                                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(g_vk->device, &img_alloc, nullptr, &tex.image_memory));
    VK_CHECK(vkBindImageMemory(g_vk->device, tex.image, tex.image_memory, 0));

    // --- Image view ---
    VkImageViewCreateInfo iv_ci = {};
    iv_ci.sType        = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    iv_ci.image        = tex.image;
    iv_ci.viewType     = VK_IMAGE_VIEW_TYPE_2D;
    iv_ci.format       = VK_FORMAT_R8G8B8A8_UNORM;
    iv_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(g_vk->device, &iv_ci, nullptr, &tex.image_view));

    // --- Staging buffer (persistently mapped) ---
    VkDeviceSize buf_size = (VkDeviceSize)w * h * 4;
    VkBufferCreateInfo buf_ci = {};
    buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_ci.size  = buf_size;
    buf_ci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    buf_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(g_vk->device, &buf_ci, nullptr, &tex.staging_buffer));

    VkMemoryRequirements buf_mem_req;
    vkGetBufferMemoryRequirements(g_vk->device, tex.staging_buffer, &buf_mem_req);
    VkMemoryAllocateInfo buf_alloc = {};
    buf_alloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    buf_alloc.allocationSize  = buf_mem_req.size;
    buf_alloc.memoryTypeIndex = vk_find_memory_type(
        buf_mem_req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(g_vk->device, &buf_alloc, nullptr, &tex.staging_memory));
    VK_CHECK(vkBindBufferMemory(g_vk->device, tex.staging_buffer, tex.staging_memory, 0));
    VK_CHECK(vkMapMemory(g_vk->device, tex.staging_memory, 0, buf_size, 0, &tex.staging_mapped));

    // Transition to SHADER_READ_ONLY_OPTIMAL via a one-shot command buffer
    VkCommandBufferAllocateInfo alloc_info = {};
    alloc_info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool        = g_vk->command_pool;
    alloc_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;
    VkCommandBuffer tmp_cmd;
    VK_CHECK(vkAllocateCommandBuffers(g_vk->device, &alloc_info, &tmp_cmd));

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(tmp_cmd, &begin_info));

    vk_transition_image_layout(tmp_cmd, tex.image,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        0, VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    VK_CHECK(vkEndCommandBuffer(tmp_cmd));

    VkSubmitInfo submit = {};
    submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers    = &tmp_cmd;
    VK_CHECK(vkQueueSubmit(g_vk->queue, 1, &submit, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(g_vk->queue));
    vkFreeCommandBuffers(g_vk->device, g_vk->command_pool, 1, &tmp_cmd);

    // --- Descriptor set (ImTextureID) ---
    tex.descriptor_set = ImGui_ImplVulkan_AddTexture(
        g_vk->texture_sampler,
        tex.image_view,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

inline void vk_destroy_texture(VulkanTexture &tex) {
    vkDeviceWaitIdle(g_vk->device);
    if (tex.descriptor_set)
        ImGui_ImplVulkan_RemoveTexture(tex.descriptor_set);
    if (tex.image_view)
        vkDestroyImageView(g_vk->device, tex.image_view, nullptr);
    if (tex.image)
        vkDestroyImage(g_vk->device, tex.image, nullptr);
    if (tex.staging_mapped)
        vkUnmapMemory(g_vk->device, tex.staging_memory);
    if (tex.staging_buffer)
        vkDestroyBuffer(g_vk->device, tex.staging_buffer, nullptr);
    if (tex.staging_memory)
        vkFreeMemory(g_vk->device, tex.staging_memory, nullptr);
    if (tex.image_memory)
        vkFreeMemory(g_vk->device, tex.image_memory, nullptr);
    tex = VulkanTexture{};
}

inline void vk_allocate_textures(int num_cams, uint32_t *widths, uint32_t *heights) {
    g_vk->textures.resize(num_cams);
    for (int j = 0; j < num_cams; j++)
        vk_create_texture(g_vk->textures[j], widths[j], heights[j]);
}

inline bool vk_begin_frame() {
    int cf = g_vk->current_frame;
    VK_CHECK(vkWaitForFences(g_vk->device, 1, &g_vk->in_flight_fences[cf],
                              VK_TRUE, UINT64_MAX));
    VkResult result = vkAcquireNextImageKHR(g_vk->device, g_vk->swapchain,
                                             UINT64_MAX,
                                             g_vk->image_available[cf],
                                             VK_NULL_HANDLE,
                                             &g_vk->image_index);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
        // window minimized or resized — skip frame
        return false;
    }
    VK_CHECK(vkResetFences(g_vk->device, 1, &g_vk->in_flight_fences[cf]));

    VkCommandBuffer cmd = g_vk->command_buffers[cf];
    VK_CHECK(vkResetCommandBuffer(cmd, 0));

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &begin_info));
    return true;
}

inline void vk_upload_texture(int cam_idx, const uint8_t *rgba,
                               uint32_t w, uint32_t h) {
    VulkanTexture &tex = g_vk->textures[cam_idx];
    VkCommandBuffer cmd = g_vk->command_buffers[g_vk->current_frame];

    memcpy(tex.staging_mapped, rgba, (size_t)w * h * 4);

    // SHADER_READ_ONLY → TRANSFER_DST
    vk_transition_image_layout(cmd, tex.image,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkBufferImageCopy region = {};
    region.imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent       = {w, h, 1};
    vkCmdCopyBufferToImage(cmd, tex.staging_buffer, tex.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // TRANSFER_DST → SHADER_READ_ONLY
    vk_transition_image_layout(cmd, tex.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
}

inline void vk_begin_rendering() {
    VkCommandBuffer cmd        = g_vk->command_buffers[g_vk->current_frame];
    VkImageView     target_view = g_vk->swapchain_image_views[g_vk->image_index];
    VkImage         target_img  = g_vk->swapchain_images[g_vk->image_index];

    // Undefined → COLOR_ATTACHMENT_OPTIMAL
    vk_transition_image_layout(cmd, target_img,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

    VkClearValue clear_val = {};
    clear_val.color = {0.1f, 0.1f, 0.1f, 1.0f};

    VkRenderingAttachmentInfoKHR color_attach = {};
    color_attach.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    color_attach.imageView   = target_view;
    color_attach.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_attach.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attach.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    color_attach.clearValue  = clear_val;

    VkRenderingInfoKHR rendering_info = {};
    rendering_info.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
    rendering_info.renderArea.offset    = {0, 0};
    rendering_info.renderArea.extent    = g_vk->swapchain_extent;
    rendering_info.layerCount           = 1;
    rendering_info.colorAttachmentCount = 1;
    rendering_info.pColorAttachments    = &color_attach;

    g_vk->fn_begin_rendering(cmd, &rendering_info);
}

inline void vk_end_frame() {
    int cf  = g_vk->current_frame;
    VkCommandBuffer cmd       = g_vk->command_buffers[cf];
    VkImage         target_img = g_vk->swapchain_images[g_vk->image_index];

    g_vk->fn_end_rendering(cmd);

    // COLOR_ATTACHMENT_OPTIMAL → PRESENT_SRC_KHR
    vk_transition_image_layout(cmd, target_img,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit = {};
    submit.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount   = 1;
    submit.pWaitSemaphores      = &g_vk->image_available[cf];
    submit.pWaitDstStageMask    = &wait_stage;
    submit.commandBufferCount   = 1;
    submit.pCommandBuffers      = &cmd;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores    = &g_vk->render_finished[cf];
    VK_CHECK(vkQueueSubmit(g_vk->queue, 1, &submit, g_vk->in_flight_fences[cf]));

    VkPresentInfoKHR present_info = {};
    present_info.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores    = &g_vk->render_finished[cf];
    present_info.swapchainCount     = 1;
    present_info.pSwapchains        = &g_vk->swapchain;
    present_info.pImageIndices      = &g_vk->image_index;
    vkQueuePresentKHR(g_vk->queue, &present_info);

    g_vk->current_frame = (g_vk->current_frame + 1) % VulkanContext::FRAMES_IN_FLIGHT;
}

inline void vk_cleanup() {
    vkDeviceWaitIdle(g_vk->device);

    for (auto &tex : g_vk->textures)
        vk_destroy_texture(tex);

    ImGui_ImplVulkan_Shutdown();

    if (g_vk->texture_sampler)
        vkDestroySampler(g_vk->device, g_vk->texture_sampler, nullptr);

    for (int i = 0; i < VulkanContext::FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(g_vk->device, g_vk->image_available[i], nullptr);
        vkDestroySemaphore(g_vk->device, g_vk->render_finished[i], nullptr);
        vkDestroyFence(g_vk->device, g_vk->in_flight_fences[i], nullptr);
    }

    vkDestroyCommandPool(g_vk->device, g_vk->command_pool, nullptr);
    vkDestroyDescriptorPool(g_vk->device, g_vk->descriptor_pool, nullptr);

    for (auto &iv : g_vk->swapchain_image_views)
        vkDestroyImageView(g_vk->device, iv, nullptr);

    vkDestroySwapchainKHR(g_vk->device, g_vk->swapchain, nullptr);
    vkDestroySurfaceKHR(g_vk->instance, g_vk->surface, nullptr);
    vkDestroyDevice(g_vk->device, nullptr);
    vkDestroyInstance(g_vk->instance, nullptr);

    delete g_vk;
    g_vk = nullptr;
}

#endif // __APPLE__
#endif // VULKAN_CONTEXT_H
