///
/// Copyright 2019 Govind Pimpale
/// vulkan_methods.h
///
///  Created on: Aug 8, 2018
///      Author: gpi
///

#ifndef SRC_VULKAN_UTILS_H_
#define SRC_VULKAN_UTILS_H_

#include <stdbool.h>
#include <stdint.h>

#include <vulkan/vulkan.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "errors.h"
#include "vertex.h"

/// Creates a new VkInstance with the specified extensions and layers
/// --- PRECONDITIONS ---
/// * `ppEnabledExtensionNames` must be a pointer to at least
/// * `enabledExtensionCount` extensions `ppEnabledLayerNames` must be a pointer
/// to at least `enabledLayerCount` layers
/// * `appname` is a UTF8 null terminated string
/// * `pCallback` must be a valid pointer
/// --- POSTCONDITONS ---
/// * Returns the error status
/// * If enableGLFWRequiredExtensions, then all extensions needed by GLFW will
/// be enabled
/// * Creates a new debug callback that prints validation layer errors to stdout
/// or stderr, depending on their severity
/// * on success, `*pCallback` is set to a valid callback
/// --- PANICS ---
/// Panics if memory allocation fails
/// Panics if Vulkan is not supported by GLFW
/// --- CLEANUP ---
/// call delete_Instance on the created instance
/// call delete_DebugCallback on the created debugCallback
void new_Instance(                              //
    VkInstance *pInstance,                      //
    VkDebugUtilsMessengerEXT *pCallback,        //
    const uint32_t enabledLayerCount,           //
    const char *const *ppEnabledLayerNames,     //
    const uint32_t enabledExtensionCount,       //
    const char *const *ppEnabledExtensionNames, //
    const bool enableGLFWRequiredExtensions,    //
    const char *appname                         //
);

/// Destroys the instance created in new_Instance
/// --- PRECONDITIONS ---
/// pInstance is a valid pointer to an instance created by new_Instance
/// --- POSTCONDITONS ---
/// the instance is no longer valid
/// pInstance has been set to VK_NULL_HANDLE
void delete_Instance(VkInstance *pInstance);

/// Gets the size of the framebuffer of the window
/// --- PRECONDITIONS ---
/// * `pExtent` is a valid pointer
/// * `pWindow` is a valid pointer to a GLFWwindow
/// --- POSTCONDITONS ---
/// * returns error status
/// * on success, pExtent is set to the size of the framebuffer of pWindow
void getExtentWindow(VkExtent2D *pExtent, GLFWwindow *pWindow);

/// Destroys the debug callback
/// --- PRECONDITIONS ---
/// `*pCallback` must have been created with `new_DebugCallback`
/// `instance` must be the instance with which `pCallback` was created
/// --- POSTCONDITONS ---
/// `*pCallback` is no longer a valid callback
/// `*pCallback` is set to VK_NULL_HANDLE
void delete_DebugCallback(               //
    VkDebugUtilsMessengerEXT *pCallback, //
    const VkInstance instance            //
);

/// Gets a the first physical device with both graphics and compute capabilities
/// --- PRECONDITIONS ---
/// `pDevice` must be a valid pointer
/// `instance` must be a valid instance
/// --- POSTCONDITONS ---
/// * returns error status
/// * on success, sets `*pDevice` to a valid physical device supporting graphics
/// and compute
ErrVal getPhysicalDevice(VkPhysicalDevice *pDevice, const VkInstance instance);

/// Creates a new logical device with the given physical device which can
/// raytrace
/// --- PRECONDITIONS ---
/// * `pDevice` must be a valid pointer
/// * `physicalDevice` must be a valid physical device created from
/// * `getPhysicalDevice` `queueFamilyIndex` must be the index of the queue
/// family to use `ppEnabledExtensionNames` must be a pointer to at least
/// * `enabledExtensionCount` extensions
/// --- POSTCONDITIONS ---
/// returns error status
/// on success, `*pDevice` will be a new logical device
/// on success, `*pDevice` will be a new logical device
/// --- CLEANUP ---
/// call delete_Device
void new_RayTracingEnabledDevice(              //
    VkDevice *pDevice,                         //
    const VkPhysicalDevice physicalDevice,     //
    const uint32_t queueFamilyIndex,           //
    const uint32_t queueCount,                 //
    const uint32_t enabledExtensionCount,      //
    const char *const *ppEnabledExtensionNames //
);

/// Deletes a logical device created from new_Device
/// --- PRECONDITIONS ---
/// * `pDevice` must be a valid pointer to a logical device created from
/// new_Device
/// --- POSTCONDITIONS ---
/// * `*pDevice` is no longer a valid logical device
/// * `*pDevice` is set to VK_NULL_HANDLE
void delete_Device(VkDevice *pDevice);

/// Gets the first queue family index with the stated capabilities
/// --- PRECONDITIONS ---
/// * `pQueueFamilyIndex` must be a valid pointer or NULL
/// * `pQueueCount` must be a valid pointer or NULL
/// * `device` must be created by getPhysicalDevice
/// * `surface` must be VK_NULL_HANDLE or a valid surface from the same instance
/// as `physicalDevice`
/// --- POSTCONDITIONS ---
/// * if no family is found that has at least `minQueueCount` queues, supports
/// `bits`, and can present to `surface`, then ERR_NOTSUPPORTED is returned.
/// * otherwise, if a family is found, returns ERR_OK
/// * if not NULL, sets `*pQueueFamilyIndex` to this queue family index
/// * if not NULL, sets `*pQueueCount` to the number of queues supported by this
/// queue family index
ErrVal getQueueFamilyIndex(                //
    uint32_t *pQueueFamilyIndex,           //
    uint32_t *pQueueCount,                 //
    const VkPhysicalDevice physicalDevice, //
    const VkQueueFlags bits,               //
    const uint32_t minQueueCount,          //
    const VkSurfaceKHR surface             //
);

/// Gets the queue associated with the queue family
/// --- PRECONDITIONS ---
/// * `pQueue` is a valid pointer
/// * `device` is a logical device created by `new_Device`
/// * `queueFamilyIndex` is a valid index for a queue family in the
/// corresponding physical device
/// --- POSTCONDITIONS ---
/// * returns error status
/// * on success, `pQueue` is set to a new 	queue in the given queue family
ErrVal getQueue(                     //
    VkQueue *pQueue,                 //
    const VkDevice device,           //
    const uint32_t queueFamilyIndex, //
    const uint32_t queueIndex        //
);

/// Gets a surface format that can be rendered to
/// --- PRECONDITIONS ---
/// * `pSurfaceFormat` is a valid pointer
/// * `surface` has been allocated from the same instance as `physicalDevice`
/// --- POSTCONDITIONS ---
/// * returns error status
/// * `*pSurfaceFormat` contains a format that we can render to
ErrVal getPreferredSurfaceFormat(VkSurfaceFormatKHR *pSurfaceFormat,
                                 const VkPhysicalDevice physicalDevice,
                                 const VkSurfaceKHR surface);

/// Creates a new swapchain, possibly reusing the old one
/// --- PRECONDITIONS ---
/// * All vulkan objects come from the same instance
/// * `oldSwapchain` is a valid pointer
/// * `pSwapchainImageCount` is a valid pointer
/// * `oldSwapchain` is either VK_NULL_HANDLE or a swapchain created from
/// new_Swapchain
/// * `surfaceFormat` is from getPreferredSurfaceFormat called with
/// `physicalDevice` and `surface`
/// * `surface` has been allocated from `physicalDevice`
/// * `device` has been allocated from `physicalDevice`
/// * `extent` is the current extent of `surface`
/// * `graphicsIndex` is the queue family index for graphics operations
/// * `presentIndex` is the queue family index to submit present operations
/// --- POSTCONDITIONS ---
/// * returns error status
/// * on success, `*pSwapchain` is set to a new swapchain
/// * on success, `*pSwapchainImageCount` is set to the number of images in
/// the swapchain
/// --- CLEANUP ---
/// * call `delete_Swapchain` to free resources associated with this swapchain
ErrVal new_Swapchain(                       //
    VkSwapchainKHR *pSwapchain,             //
    uint32_t *pSwapchainImageCount,         //
    const VkSwapchainKHR oldSwapchain,      //
    const VkSurfaceFormatKHR surfaceFormat, //
    const VkPhysicalDevice physicalDevice,  //
    const VkDevice device,                  //
    const VkSurfaceKHR surface,             //
    const VkExtent2D extent                 //
);

/// Deletes a swapchain created from new_Swapchain
/// --- PRECONDITIONS ---
/// * `pSwapchain` must be a valid pointer to a swapchain created from
/// new_Swapchain
/// * `device` must be the logical device from which `*pSwapchain` was
/// allocated
/// --- POSTCONDITIONS ---
/// * Resources associated with `*pSwapchain` have been released
/// * `*pSwapchain` is no longer a valid swapchain
/// * `*pSwapchain` is set to VK_NULL_HANDLE
void delete_Swapchain(VkSwapchainKHR *pSwapchain, const VkDevice device);

/// gets swapchain images from the swapchain
/// --- PRECONDITIONS --
/// * swapchain
ErrVal getSwapchainImages(         //
    VkImage *pSwapchainImages,     //
    const uint32_t imageCount,     //
    const VkDevice device,         //
    const VkSwapchainKHR swapchain //
);

void new_Image(                             //
    VkImage *pImage,                        //
    VkDeviceMemory *pImageMemory,           //
    const VkExtent2D dimensions,            //
    const VkFormat format,                  //
    const VkImageTiling tiling,             //
    const VkImageUsageFlags usage,          //
    const VkMemoryPropertyFlags properties, //
    const VkPhysicalDevice physicalDevice,  //
    const VkDevice device                   //
);

/// Deletes a image created from new_Image
/// --- PRECONDITIONS ---
/// * `pImage` must be a valid pointer to a image created from new_Image
/// * `device` must be the logical device from which `*pImage` was allocated
/// --- POSTCONDITIONS ---
/// * Resources associated with `*pImage` have been released
/// * `*pImage` is no longer a valid image
/// * `*pImage` is set to VK_NULL_HANDLE
void delete_Image(VkImage *pImage, const VkDevice device);

void new_ImageView(           //
    VkImageView *pImageView,  //
    const VkDevice device,    //
    const VkImage image,      //
    const VkFormat format,    //
    const uint32_t aspectMask //
);

/// Deletes a imageView created from new_ImageView
/// --- PRECONDITIONS ---
/// * `pImageView` must be a valid pointer to a imageView created from
/// new_ImageView
/// * `device` must be the logical device from which `*pImageView` was
/// allocated
/// --- POSTCONDITIONS ---
/// * Resources associated with `*pImageView` have been released
/// * `*pImageView` is no longer a valid imageView
/// * `*pImageView` is set to VK_NULL_HANDLE
void delete_ImageView(VkImageView *pImageView, VkDevice device);

void new_SwapchainImageViews(        //
    VkImageView *pImageViews,        //
    const VkImage *pSwapchainImages, //
    const uint32_t imageCount,       //
    const VkDevice device,           //
    const VkFormat format            //
);

void delete_SwapchainImageViews( //
    VkImageView *pImageViews,    //
    const uint32_t imageCount,   //
    const VkDevice device        //
);

ErrVal new_ShaderModule(VkShaderModule *pShaderModule, const VkDevice device,
                        const uint32_t codeSize, const uint32_t *pCode);

/// Deletes a shaderModule created from new_ShaderModule
/// --- PRECONDITIONS ---
/// * `pShaderModule` must be a valid pointer to a shaderModule created from
/// new_ShaderModule
/// * `device` must be the logical device from which `*pShaderModule` was
/// allocated
/// --- POSTCONDITIONS ---
/// * Resources associated with `*pShaderModule` have been released
/// * `*pShaderModule` is no longer a valid shaderModule
/// * `*pShaderModule` is set to VK_NULL_HANDLE
void delete_ShaderModule(VkShaderModule *pShaderModule, const VkDevice device);

ErrVal new_VertexDisplayRenderPass(VkRenderPass *pRenderPass,
                                   const VkDevice device,
                                   const VkFormat swapchainImageFormat);

void delete_RenderPass(VkRenderPass *pRenderPass, const VkDevice device);

void new_VertexDisplayPipelineLayoutDescriptorSetLayout(      //
    VkPipelineLayout *pVertexDisplayPipelineLayout,           //
    VkDescriptorSetLayout *pVertexDisplayDescriptorSetLayout, //
    const VkDevice device                                     //
);

void delete_VertexDisplayPipelineLayoutDescriptorSetLayout( //
    VkPipelineLayout *pPipelineLayout,                      //
    VkDescriptorSetLayout *pDescriptorSetLayout,            //
    const VkDevice device                                   //
);

void new_VertexDisplayPipeline(VkPipeline *pVertexDisplayPipeline,
                               const VkDevice device,
                               const VkShaderModule vertShaderModule,
                               const VkShaderModule fragShaderModule,
                               const VkExtent2D extent,
                               const VkRenderPass renderPass,
                               const VkPipelineLayout pipelineLayout);

void delete_Pipeline(VkPipeline *pPipeline, const VkDevice device);

void new_Framebuffer(VkFramebuffer *pFramebuffer, const VkDevice device,
                     const VkRenderPass renderPass, const VkImageView imageView,
                     const VkImageView depthImageView,
                     const VkExtent2D swapchainExtent);

void delete_Framebuffer(VkFramebuffer *pFramebuffer, VkDevice device);

void new_SwapchainFramebuffers(VkFramebuffer *pFramebuffers,
                               const VkDevice device,
                               const VkRenderPass renderPass,
                               const VkExtent2D swapchainExtent,
                               const uint32_t imageCount,
                               const VkImageView depthImageView,
                               const VkImageView *pSwapchainImageViews);

void delete_SwapchainFramebuffers(VkFramebuffer *pFramebuffers,
                                  const uint32_t imageCount,
                                  const VkDevice device);

ErrVal new_CommandPool(             //
    VkCommandPool *pCommandPool,    //
    const VkDevice device,          //
    const uint32_t queueFamilyIndex //
);

void delete_CommandPool(VkCommandPool *pCommandPool, const VkDevice device);

ErrVal new_CommandBuffers(             //
    VkCommandBuffer *pCommandBuffers,  //
    const uint32_t commandBufferCount, //
    const VkCommandPool commandPool,   //
    const VkDevice device              //
);

void delete_CommandBuffers(            //
    VkCommandBuffer *pCommandBuffers,  //
    const uint32_t commandBufferCount, //
    const VkCommandPool commandPool,   //
    const VkDevice device              //
);

ErrVal recordVertexDisplayCommandBuffer(                //
    VkCommandBuffer commandBuffer,                      //
    const VkFramebuffer swapchainFramebuffer,           //
    const uint32_t vertexBufferCount,                   //
    const VkBuffer *pVertexBuffers,                     //
    const uint32_t *pVertexCounts,                      //
    const VkRenderPass renderPass,                      //
    const VkPipelineLayout vertexDisplayPipelineLayout, //
    const VkPipeline vertexDisplayPipeline,             //
    const VkExtent2D swapchainExtent,                   //
    const mat4x4 cameraTransform,                       //
    const VkDescriptorSet vertexDisplayDescriptorSet,   //
    const VkClearColorValue clearColor                  //
);

ErrVal new_Semaphore(VkSemaphore *pSemaphore, const VkDevice device);

void delete_Semaphore(VkSemaphore *pSemaphore, const VkDevice device);

ErrVal new_Semaphores(VkSemaphore *pSemaphores, const uint32_t semaphoreCount,
                      const VkDevice device);

void delete_Semaphores(VkSemaphore *pSemaphores, const uint32_t semaphoreCount,
                       const VkDevice device);

ErrVal new_Fence(VkFence *pFence, const VkDevice device, const bool signaled);

void delete_Fence(VkFence *pFence, const VkDevice device);

ErrVal waitAndResetFence(VkFence fence, const VkDevice device);

ErrVal new_Fences(VkFence *pFences, const uint32_t fenceCount,
                  const VkDevice device, const bool allSignaled);

void delete_Fences(VkFence *pFences, const uint32_t fenceCount,
                   const VkDevice device);

ErrVal getNextSwapchainImage(           //
    uint32_t *pImageIndex,              //
    const VkSwapchainKHR swapchain,     //
    const VkDevice device,              //
    VkSemaphore imageAvailableSemaphore //
);

ErrVal drawFrame(                        //
    VkCommandBuffer commandBuffer,       //
    VkSwapchainKHR swapchain,            //
    const uint32_t swapchainImageIndex,  //
    VkSemaphore imageAvailableSemaphore, //
    VkSemaphore renderFinishedSemaphore, //
    VkFence inFlightFence,               //
    const VkQueue graphicsQueue,         //
    const VkQueue presentQueue           //
);

ErrVal new_SurfaceFromGLFW(VkSurfaceKHR *pSurface, GLFWwindow *pWindow,
                           const VkInstance instance);

void delete_Surface(VkSurfaceKHR *pSurface, const VkInstance instance);

ErrVal new_Buffer_DeviceMemory(VkBuffer *pBuffer, VkDeviceMemory *pBufferMemory,
                               const VkDeviceSize size,
                               const VkPhysicalDevice physicalDevice,
                               const VkDevice device,
                               const VkBufferUsageFlags usage,
                               const VkMemoryPropertyFlags properties);

void copyBuffer(VkBuffer destinationBuffer, const VkBuffer sourceBuffer,
                const VkDeviceSize size, const VkCommandPool commandPool,
                const VkQueue queue, const VkDevice device);

void updateBuffer(VkBuffer destinationBuffer, const void *pSource,
                  const VkDeviceSize size, const VkCommandPool commandPool,
                  const VkQueue queue, const VkDevice device);

void delete_Buffer(VkBuffer *pBuffer, const VkDevice device);

void delete_DeviceMemory(VkDeviceMemory *pDeviceMemory, const VkDevice device);

void copyToDeviceMemory(VkDeviceMemory *pDeviceMemory,
                        const VkDeviceSize deviceSize, const void *source,
                        const VkDevice device);

void getDepthFormat(VkFormat *pFormat);

void new_DepthImageView(VkImageView *pImageView, const VkDevice device,
                        const VkImage depthImage);

void new_DepthImage(VkImage *pImage, VkDeviceMemory *pImageMemory,
                    const VkExtent2D swapchainExtent,
                    const VkPhysicalDevice physicalDevice,
                    const VkDevice device);

ErrVal getMemoryTypeIndex(uint32_t *memoryTypeIndex,
                          const uint32_t memoryTypeBits,
                          const VkMemoryPropertyFlags memoryPropertyFlags,
                          const VkPhysicalDevice physicalDevice);

void new_TextureImage(                     //
    VkImage *pImage,                       //
    VkDeviceMemory *pImageMemory,          //
    const uint8_t *rgbaPxArr,              //
    const VkExtent2D dimensions,           //
    const VkDevice device,                 //
    const VkPhysicalDevice physicalDevice, //
    const VkCommandPool commandPool,       //
    const VkQueue queue                    //
);

void new_TextureImageView(          //
    VkImageView *pTextureImageView, //
    const VkImage textureImage,     //
    const VkDevice device           //
);

void new_VertexDisplayDescriptorPoolAndSet(                       //
    VkDescriptorPool *pDescriptorPool,                            //
    VkDescriptorSet *pDescriptorSet,                              //
    const VkDescriptorSetLayout vertexDisplayDescriptorSetLayout, //
    const VkDevice device,                                        //
    const VkSampler textureSampler,                               //
    const VkImageView textureImageView                            //
);

void new_TextureSampler(VkSampler *pTextureSampler, const VkDevice device);
void delete_TextureSampler(VkSampler *pTextureSampler, const VkDevice device);

void delete_DescriptorPool(VkDescriptorPool *pDescriptorPool,
                           const VkDevice device);

#endif /* SRC_VULKAN_UTILS_H_ */
