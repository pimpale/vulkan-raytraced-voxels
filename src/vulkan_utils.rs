use ash::vk;

unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let severity = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "[Verbose]",
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "[Warning]",
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "[Error]",
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "[Info]",
        _ => "[Unknown]",
    };
    let types = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
        _ => "[Unknown]",
    };
    let message = CStr::from_ptr((*p_callback_data).p_message);
    println!("[Debug]{}{}{:?}", severity, types, message);

    vk::FALSE
}

/* Creates new VkInstance with sensible defaults */
fn new_Instance(
    layers: Vec<String>,
    extensions: Vec<String>,
    appname: &str
) -> vk::Instance {
  // first create a new malloced list of all extensions we need

  // append our custom enabled extensions
  for (size_t i = 0; i < enabledExtensionCount; i++) {
    ppAllExtensionNames[currentPosition] = ppEnabledExtensionNames[i];
    currentPosition++;
  }

  /* Create app info */
  VkApplicationInfo appInfo = {0};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = appname;
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "vulkan_utils.rs";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_3;

  /* Create info */
  VkInstanceCreateInfo createInfo = {0};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;
  createInfo.enabledExtensionCount = allExtensionCount;
  createInfo.ppEnabledExtensionNames = ppAllExtensionNames;
  createInfo.enabledLayerCount = enabledLayerCount;
  createInfo.ppEnabledLayerNames = ppEnabledLayerNames;
  /* Actually create instance */
  VkResult result = vkCreateInstance(&createInfo, NULL, pInstance);
  if (result != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL, "Failed to create instance, error code: %s",
                   vkstrerror(result));
    PANIC();
  }

}

/**
 * Requires the debug utils extension
 *
 * Creates a new debug callback that prints validation layer errors to stdout or
 * stderr, depending on their severity
 */
ErrVal new_DebugCallback(VkDebugUtilsMessengerEXT *pCallback,
                         const VkInstance instance) {
 vk::DebugUtilsMessengerCreateInfoEXT {
        s_type: vk::StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        p_next: ptr::null(),
        flags: vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING |
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE |
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO |
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        pfn_user_callback: Some(vulkan_debug_utils_callback),
        p_user_data: ptr::null_mut(),
    }

  /* Returns a function pointer */
  PFN_vkCreateDebugUtilsMessengerEXT func =
      (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
          instance, "vkCreateDebugUtilsMessengerEXT");
  if (!func) {
    LOG_ERROR(ERR_LEVEL_FATAL, "Failed to find extension function");
    PANIC();
  }
  VkResult result = func(instance, &createInfo, NULL, pCallback);
  if (result != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL,
                   "Failed to create debug callback, error code: %s",
                   vkstrerror(result));
    PANIC();
  }
  return (ERR_NOTSUPPORTED);

}

/**
 * Requires the debug utils extension
 * Deletes callback created in new_DebugCallback
 */
void delete_DebugCallback(VkDebugUtilsMessengerEXT *pCallback,
                          const VkInstance instance) {
  PFN_vkDestroyDebugUtilsMessengerEXT func =
      (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
          instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != NULL) {
    func(instance, *pCallback, NULL);
  }
}
/**
 * gets the best physical device, checks if it has all necessary capabilities.
 */
ErrVal getPhysicalDevice(VkPhysicalDevice *pDevice, const VkInstance instance) {
  uint32_t deviceCount = 0;
  VkResult res = vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
  if (res != VK_SUCCESS || deviceCount == 0) {
    LOG_ERROR(ERR_LEVEL_WARN, "no Vulkan capable device found");
    return (ERR_NOTSUPPORTED);
  }
  VkPhysicalDevice *arr = malloc(deviceCount * sizeof(VkPhysicalDevice));
  if (!arr) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL, "failed to get physical device: %s",
                   strerror(errno));
    PANIC();
  }
  vkEnumeratePhysicalDevices(instance, &deviceCount, arr);

  VkPhysicalDeviceProperties deviceProperties;
  VkPhysicalDevice selectedDevice = VK_NULL_HANDLE;
  for (uint32_t i = 0; i < deviceCount; i++) {
    /* TODO confirm it has required properties */
    vkGetPhysicalDeviceProperties(arr[i], &deviceProperties);
    uint32_t deviceQueueIndex;
    uint32_t deviceQueueCount;
    uint32_t ret = getQueueFamilyIndexByCapability(
        &deviceQueueIndex, &deviceQueueCount, arr[i],
        VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT);
    if (ret == VK_SUCCESS) {
      selectedDevice = arr[i];
      break;
    }
  }
  free(arr);
  if (selectedDevice == VK_NULL_HANDLE) {
    LOG_ERROR(ERR_LEVEL_WARN, "no suitable Vulkan device found");
    return (ERR_NOTSUPPORTED);
  } else {
    *pDevice = selectedDevice;
    return (ERR_OK);
  }
}

/**
 * Deletes VkDevice created in new_Device
 */
void delete_Device(VkDevice *pDevice) {
  vkDestroyDevice(*pDevice, NULL);
  *pDevice = VK_NULL_HANDLE;
}

/**
 * Sets deviceQueueIndex to queue family index corresponding to the bit passed
 * in for the device
 */
ErrVal getQueueFamilyIndexByCapability(uint32_t *pQueueFamilyIndex,
                                       uint32_t *pQueueCount,
                                       const VkPhysicalDevice device,
                                       const VkQueueFlags bit) {
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, NULL);
  if (queueFamilyCount == 0) {
    LOG_ERROR(ERR_LEVEL_WARN, "no device queues found");
    return (ERR_NOTSUPPORTED);
  }
  VkQueueFamilyProperties *pFamilyProperties =
      (VkQueueFamilyProperties *)malloc(queueFamilyCount *
                                        sizeof(VkQueueFamilyProperties));
  if (!pFamilyProperties) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL, "Failed to get device queue index: %s",
                   strerror(errno));
    PANIC();
  }
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                           pFamilyProperties);
  for (uint32_t i = 0; i < queueFamilyCount; i++) {
    if (pFamilyProperties[i].queueCount > 0 &&
        (pFamilyProperties[0].queueFlags & bit)) {
      *pQueueFamilyIndex = i;
      *pQueueCount = pFamilyProperties[i].queueCount;
      free(pFamilyProperties);
      return (ERR_OK);
    }
  }
  free(pFamilyProperties);
  LOG_ERROR(ERR_LEVEL_ERROR, "no suitable device queue found");
  return (ERR_NOTSUPPORTED);
}

ErrVal getPresentQueueFamilyIndex(uint32_t *pQueueFamilyIndex,
                                  const VkPhysicalDevice physicalDevice,
                                  const VkSurfaceKHR surface) {
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                           NULL);
  if (queueFamilyCount == 0) {
    LOG_ERROR(ERR_LEVEL_WARN, "no queues found");
    return (ERR_NOTSUPPORTED);
  }
  VkQueueFamilyProperties *arr = (VkQueueFamilyProperties *)malloc(
      queueFamilyCount * sizeof(VkQueueFamilyProperties));
  if (!arr) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL, "Failed to get present queue index: %s",
                   strerror(errno));
    PANIC();
  }
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                           arr);
  for (uint32_t i = 0; i < queueFamilyCount; i++) {
    VkBool32 surfaceSupport;
    VkResult res = vkGetPhysicalDeviceSurfaceSupportKHR(
        physicalDevice, i, surface, &surfaceSupport);
    if (res == VK_SUCCESS && surfaceSupport) {
      *pQueueFamilyIndex = i;
      free(arr);
      return (ERR_OK);
    }
  }
  free(arr);
  return (ERR_NOTSUPPORTED);
}

ErrVal new_Device(                             //
    VkDevice *pDevice,                         //
    const VkPhysicalDevice physicalDevice,     //
    const uint32_t queueFamilyIndex,           //
    const uint32_t queueCount,                 //
    const uint32_t enabledExtensionCount,      //
    const char *const *ppEnabledExtensionNames //
) {

  float *pQueuePriorities = malloc(queueCount * sizeof(float));
  for (uint32_t i = 0; i < queueCount; i++) {
    pQueuePriorities[i] = 1.0f;
  }

  VkPhysicalDeviceFeatures deviceFeatures = {0};
  VkDeviceQueueCreateInfo queueCreateInfo = {0};
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
  queueCreateInfo.queueCount = queueCount;
  queueCreateInfo.pQueuePriorities = pQueuePriorities;

  VkDeviceCreateInfo createInfo = {0};
  createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  createInfo.pQueueCreateInfos = &queueCreateInfo;
  createInfo.queueCreateInfoCount = 1;
  createInfo.pEnabledFeatures = &deviceFeatures;
  createInfo.enabledExtensionCount = enabledExtensionCount;
  createInfo.ppEnabledExtensionNames = ppEnabledExtensionNames;
  createInfo.enabledLayerCount = 0;

  VkResult res = vkCreateDevice(physicalDevice, &createInfo, NULL, pDevice);
  if (res != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_ERROR, "Failed to create device, error code: %s",
                   vkstrerror(res));
    PANIC();
  }

  free(pQueuePriorities);

  return (ERR_OK);
}

ErrVal getQueue(VkQueue *pQueue, const VkDevice device,
                const uint32_t deviceQueueIndex, const uint32_t queueIndex) {
  vkGetDeviceQueue(device, deviceQueueIndex, queueIndex, pQueue);
  return (ERR_OK);
}

ErrVal new_Swapchain(VkSwapchainKHR *pSwapchain, uint32_t *pImageCount,
                     const VkSwapchainKHR oldSwapchain,
                     const VkSurfaceFormatKHR surfaceFormat,
                     const VkPhysicalDevice physicalDevice,
                     const VkDevice device, const VkSurfaceKHR surface,
                     const VkExtent2D extent, const uint32_t graphicsIndex,
                     const uint32_t presentIndex) {
  VkSurfaceCapabilitiesKHR capabilities;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface,
                                            &capabilities);

  // it's important to note that minImageCount isn't necessarily the size of the
  // swapchain we get
  VkSwapchainCreateInfoKHR createInfo = {0};
  createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  createInfo.surface = surface;
  createInfo.minImageCount = capabilities.minImageCount + 1;
  createInfo.imageFormat = surfaceFormat.format;
  createInfo.imageColorSpace = surfaceFormat.colorSpace;
  createInfo.imageExtent = extent;
  createInfo.imageArrayLayers = 1;
  createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

  uint32_t queueFamilyIndices[] = {graphicsIndex, presentIndex};
  if (graphicsIndex != presentIndex) {
    createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    createInfo.queueFamilyIndexCount = 2;
    createInfo.pQueueFamilyIndices = queueFamilyIndices;
  } else {
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.queueFamilyIndexCount = 0;  /* Optional */
    createInfo.pQueueFamilyIndices = NULL; /* Optional */
  }

  createInfo.preTransform = capabilities.currentTransform;
  createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  /* guaranteed to be available */
  createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  createInfo.clipped = VK_TRUE;
  createInfo.oldSwapchain = oldSwapchain;
  VkResult res = vkCreateSwapchainKHR(device, &createInfo, NULL, pSwapchain);
  if (res != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_ERROR,
                   "Failed to create swap chain, error code: %s",
                   vkstrerror(res));
    PANIC();
  }

  VkResult imageCountRes =
      vkGetSwapchainImagesKHR(device, *pSwapchain, pImageCount, NULL);
  if (imageCountRes != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_ERROR,
                   "Failed to retrieve swap chain image count, error code: %s",
                   vkstrerror(res));
    PANIC();
  }

  return (ERR_OK);
}

void delete_Swapchain(VkSwapchainKHR *pSwapchain, const VkDevice device) {
  vkDestroySwapchainKHR(device, *pSwapchain, NULL);
  *pSwapchain = VK_NULL_HANDLE;
}

ErrVal getSwapchainImages(         //
    VkImage *pSwapchainImages,     //
    const uint32_t imageCount,     //
    const VkDevice device,         //
    const VkSwapchainKHR swapchain //
) {

  // we are going to try to write in imageCount images, but its possible that
  // we get less or more
  uint32_t imagesWritten = imageCount;
  VkResult res = vkGetSwapchainImagesKHR(device, swapchain, &imagesWritten,
                                         pSwapchainImages);
  if (res != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_ERROR,
                   "failed to get swap chain images, error: %s",
                   vkstrerror(res));
    return (ERR_UNKNOWN);
  }

  if (imagesWritten != imageCount) {
    LOG_ERROR_ARGS(ERR_LEVEL_ERROR, "expected %u images, but only %u written",
                   imageCount, imagesWritten);
  }

  return (ERR_OK);
}

ErrVal getPreferredSurfaceFormat(VkSurfaceFormatKHR *pSurfaceFormat,
                                 const VkPhysicalDevice physicalDevice,
                                 const VkSurfaceKHR surface) {
  uint32_t formatCount = 0;
  VkSurfaceFormatKHR *pSurfaceFormats;
  vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount,
                                       NULL);
  if (formatCount == 0) {
    return (ERR_NOTSUPPORTED);
  }

  pSurfaceFormats =
      (VkSurfaceFormatKHR *)malloc(formatCount * sizeof(VkSurfaceFormatKHR));
  if (!pSurfaceFormats) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL, "could not get preferred format: %s",
                   strerror(errno));
    PANIC();
  }
  vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount,
                                       pSurfaceFormats);

  VkSurfaceFormatKHR preferredFormat = {0};
  if (formatCount == 1 && pSurfaceFormats[0].format == VK_FORMAT_UNDEFINED) {
    /* If it has no preference, use our own */
    preferredFormat = pSurfaceFormats[0];
  } else if (formatCount != 0) {
    /* we default to the first one in the list */
    preferredFormat = pSurfaceFormats[0];
    /* However,  we check to make sure that what we want is in there
     */
    for (uint32_t i = 0; i < formatCount; i++) {
      VkSurfaceFormatKHR availableFormat = pSurfaceFormats[i];
      if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
          availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
        preferredFormat = availableFormat;
      }
    }
  } else {
    LOG_ERROR(ERR_LEVEL_ERROR, "no formats available");
    free(pSurfaceFormats);
    return (ERR_NOTSUPPORTED);
  }

  free(pSurfaceFormats);

  *pSurfaceFormat = preferredFormat;
  return (ERR_OK);
}

void new_ImageView(VkImageView *pImageView, const VkDevice device,
                   const VkImage image, const VkFormat format,
                   const uint32_t aspectMask) {
  VkImageViewCreateInfo createInfo = {0};
  createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  createInfo.image = image;
  createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  createInfo.format = format;
  createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  createInfo.subresourceRange.aspectMask = aspectMask;
  createInfo.subresourceRange.baseMipLevel = 0;
  createInfo.subresourceRange.levelCount = 1;
  createInfo.subresourceRange.baseArrayLayer = 0;
  createInfo.subresourceRange.layerCount = 1;
  VkResult ret = vkCreateImageView(device, &createInfo, NULL, pImageView);
  if (ret != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL,
                   "could not create image view, error code: %s",
                   vkstrerror(ret));
    PANIC();
  }
}

void delete_ImageView(VkImageView *pImageView, VkDevice device) {
  vkDestroyImageView(device, *pImageView, NULL);
  *pImageView = VK_NULL_HANDLE;
}

void new_SwapchainImageViews(        //
    VkImageView *pImageViews,        //
    const VkImage *pSwapchainImages, //
    const uint32_t imageCount,       //
    const VkDevice device,           //
    const VkFormat format            //
) {
  for (uint32_t i = 0; i < imageCount; i++) {
    new_ImageView(&(pImageViews[i]), device, pSwapchainImages[i], format,
                  VK_IMAGE_ASPECT_COLOR_BIT);
  }
}

void delete_SwapchainImageViews( //
    VkImageView *pImageViews,    //
    const uint32_t imageCount,   //
    const VkDevice device        //
) {
  for (uint32_t i = 0; i < imageCount; i++) {
    delete_ImageView(&pImageViews[i], device);
  }
}
static VkCommandBuffer
createBeginOneTimeCmdBuffer(const VkCommandPool commandPool,
                            const VkDevice device) {
  VkCommandBuffer commandBuffer;
  new_CommandBuffers(&commandBuffer, 1, commandPool, device);

  VkCommandBufferBeginInfo beginInfo = {0};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

  VkResult beginRet = vkBeginCommandBuffer(commandBuffer, &beginInfo);
  if (beginRet != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL,
                   "Failed to begine one time command buffer: %s",
                   vkstrerror(beginRet));
    PANIC();
  }

  return commandBuffer;
}

static void submitEndOneTimeCmdBuffer(VkCommandBuffer buffer,
                                      const VkQueue queue,
                                      const VkDevice device) {

  // End buffer
  VkResult bufferEndResult = vkEndCommandBuffer(buffer);
  if (bufferEndResult != VK_SUCCESS) {
    /* Clean up resources */
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL, "failed to end command buffer: %s",
                   vkstrerror(bufferEndResult));
    PANIC();
  }

  // wait for completion with a fence
  VkFence fence;
  if (new_Fence(&fence, device, false) != ERR_OK) {
    LOG_ERROR(ERR_LEVEL_FATAL, "Failed to make fence to wait for buffer");
    PANIC();
  }

  VkSubmitInfo submitInfo = {0};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &buffer;

  VkResult queueSubmitResult = vkQueueSubmit(queue, 1, &submitInfo, fence);
  if (queueSubmitResult != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL,
                   "failed to submit command buffer to queue: %s",
                   vkstrerror(queueSubmitResult));
    PANIC();
  }

  VkResult waitRet = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
  if (waitRet != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL, "failed to wait for fence: %s",
                   vkstrerror(waitRet));
    PANIC();
  }

  delete_Fence(&fence, device);
}

static void transitionImageLayout(   //
    VkImage image,                   //
    const VkImageLayout oldLayout,   //
    const VkImageLayout newLayout,   //
    const VkCommandPool commandPool, //
    const VkDevice device,           //
    const VkQueue queue              //
) {
  VkCommandBuffer commandBuffer =
      createBeginOneTimeCmdBuffer(commandPool, device);

  VkImageMemoryBarrier barrier = {0};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = oldLayout;
  barrier.newLayout = newLayout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;

  VkPipelineStageFlags sourceStage;
  VkPipelineStageFlags destinationStage;

  if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
      newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
             newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  } else {
    LOG_ERROR(ERR_LEVEL_FATAL, "Unsupported layout transition");
    PANIC();
  }

  vkCmdPipelineBarrier(              //
      commandBuffer,                 //
      sourceStage, destinationStage, //
      0,                             //
      0, NULL,                       //
      0, NULL,                       //
      1, &barrier                    //
  );                                 //

  submitEndOneTimeCmdBuffer(commandBuffer, queue, device);
}

static void copyBufferToImage(       //
    VkImage image,                   //
    const VkBuffer buffer,           //
    VkExtent2D dimensions,           //
    const VkCommandPool commandPool, //
    const VkDevice device,           //
    const VkQueue queue              //
) {
  VkCommandBuffer commandBuffer =
      createBeginOneTimeCmdBuffer(commandPool, device);

  VkBufferImageCopy region = {0};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;
  region.imageOffset = (VkOffset3D){0, 0, 0};
  region.imageExtent = (VkExtent3D){dimensions.width, dimensions.height, 1};

  vkCmdCopyBufferToImage(commandBuffer, buffer, image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  submitEndOneTimeCmdBuffer(commandBuffer, queue, device);
}

void new_TextureImage(                     //
    VkImage *pImage,                       //
    VkDeviceMemory *pImageMemory,          //
    const uint8_t *rgbaPxArr,              //
    const VkExtent2D dimensions,           //
    const VkDevice device,                 //
    const VkPhysicalDevice physicalDevice, //
    const VkCommandPool commandPool,       //
    const VkQueue queue                    //
) {

  // each pix has 4 channels
  VkDeviceSize bufferSize = dimensions.height * dimensions.width * 4;

  VkBuffer stagingBuffer;
  VkDeviceMemory stagingBufferMemory;

  ErrVal stagingBufferCreateResult = new_Buffer_DeviceMemory(
      &stagingBuffer, &stagingBufferMemory, bufferSize, physicalDevice, device,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  if (stagingBufferCreateResult != ERR_OK) {
    LOG_ERROR(ERR_LEVEL_FATAL,
              "failed to create texture: failed to create staging buffer");
    PANIC();
  }

  copyToDeviceMemory(&stagingBufferMemory, bufferSize, rgbaPxArr, device);

  // create new image
  new_Image(                                                        //
      pImage,                                                       //
      pImageMemory,                                                 //
      dimensions,                                                   //
      VK_FORMAT_R8G8B8A8_SRGB,                                      //
      VK_IMAGE_TILING_OPTIMAL,                                      //
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, //
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,                          //
      physicalDevice,                                               //
      device                                                        //
  );                                                                //

  // prepare image for data transfer
  transitionImageLayout(                    //
      *pImage,                              //
      VK_IMAGE_LAYOUT_UNDEFINED,            //
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, //
      commandPool,                          //
      device,                               //
      queue                                 //
  );

  copyBufferToImage(*pImage, stagingBuffer, dimensions, commandPool, device,
                    queue);

  // prepare image to only be read by shaders
  transitionImageLayout(                        //
      *pImage,                                  //
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,     //
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, //
      commandPool,                              //
      device,                                   //
      queue                                     //
  );

  /* Delete the temporary staging buffers */
  delete_Buffer(&stagingBuffer, device);
  delete_DeviceMemory(&stagingBufferMemory, device);
}

void new_TextureImageView(          //
    VkImageView *pTextureImageView, //
    const VkImage textureImage,     //
    const VkDevice device           //
) {
  new_ImageView(pTextureImageView, device, textureImage,
                VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);
}

ErrVal new_ShaderModule(VkShaderModule *pShaderModule, const VkDevice device,
                        const uint32_t codeSize, const uint32_t *pCode) {
  VkShaderModuleCreateInfo createInfo = {0};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = codeSize;
  createInfo.pCode = pCode;
  VkResult res = vkCreateShaderModule(device, &createInfo, NULL, pShaderModule);
  if (res != VK_SUCCESS) {
    LOG_ERROR(ERR_LEVEL_FATAL, "failed to create shader module");
    return (ERR_UNKNOWN);
  }
  return (ERR_OK);
}

void new_TextureSampler(VkSampler *pTextureSampler, const VkDevice device) {
  VkSamplerCreateInfo samplerInfo = {0};
  samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.magFilter = VK_FILTER_NEAREST;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  // white border to check for error in uv mapping
  samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_WHITE;
  // we clamp to border to make errors more obvious
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  // no anisotropy for now
  samplerInfo.anisotropyEnable = VK_FALSE;
  samplerInfo.maxAnisotropy = 1.0f;
  samplerInfo.unnormalizedCoordinates = VK_FALSE;
  samplerInfo.compareEnable = VK_FALSE;
  samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

  VkResult ret = vkCreateSampler(device, &samplerInfo, NULL, pTextureSampler);
  if (ret != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL, "failed to create sampler: %s",
                   vkstrerror(ret));
    PANIC();
  }
}

void delete_TextureSampler(VkSampler *pTextureSampler, const VkDevice device) {
  vkDestroySampler(device, *pTextureSampler, NULL);
  *pTextureSampler = VK_NULL_HANDLE;
}

void delete_ShaderModule(VkShaderModule *pShaderModule, const VkDevice device) {
  vkDestroyShaderModule(device, *pShaderModule, NULL);
  *pShaderModule = VK_NULL_HANDLE;
}

ErrVal new_VertexDisplayRenderPass(VkRenderPass *pRenderPass,
                                   const VkDevice device,
                                   const VkFormat swapchainImageFormat) {
  VkAttachmentDescription colorAttachment = {0};
  colorAttachment.format = swapchainImageFormat;
  colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentDescription depthAttachment = {0};
  getDepthFormat(&depthAttachment.format);
  depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  depthAttachment.finalLayout =
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkAttachmentDescription pAttachments[2];
  pAttachments[0] = colorAttachment;
  pAttachments[1] = depthAttachment;

  VkAttachmentReference colorAttachmentRef = {0};
  colorAttachmentRef.attachment = 0;
  colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentReference depthAttachmentRef = {0};
  depthAttachmentRef.attachment = 1;
  depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass = {0};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &colorAttachmentRef;
  subpass.pDepthStencilAttachment = &depthAttachmentRef;

  VkRenderPassCreateInfo renderPassInfo = {0};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = 2;
  renderPassInfo.pAttachments = pAttachments;
  renderPassInfo.subpassCount = 1;
  renderPassInfo.pSubpasses = &subpass;

  VkSubpassDependency dependency = {0};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.srcAccessMask = 0;
  dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  renderPassInfo.dependencyCount = 1;
  renderPassInfo.pDependencies = &dependency;

  VkResult res = vkCreateRenderPass(device, &renderPassInfo, NULL, pRenderPass);
  if (res != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL, "Could not create render pass, error: %s",
                   vkstrerror(res));
    PANIC();
  }
  return (ERR_OK);
}

void delete_RenderPass(VkRenderPass *pRenderPass, const VkDevice device) {
  vkDestroyRenderPass(device, *pRenderPass, NULL);
  *pRenderPass = VK_NULL_HANDLE;
}

// makes a new descriptor set layout and pipeline layout
void new_VertexDisplayPipelineLayoutDescriptorSetLayout(      //
    VkPipelineLayout *pVertexDisplayPipelineLayout,           //
    VkDescriptorSetLayout *pVertexDisplayDescriptorSetLayout, //
    const VkDevice device                                     //
) {
  // create a descriptor set at 0 for the sampler
  VkDescriptorSetLayoutBinding samplerLayoutBinding = {0};
  samplerLayoutBinding.binding = 0;
  samplerLayoutBinding.descriptorCount = 1;
  samplerLayoutBinding.descriptorType =
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  samplerLayoutBinding.pImmutableSamplers = NULL;
  samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  VkDescriptorSetLayoutCreateInfo layoutInfo = {0};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = 1;
  layoutInfo.pBindings = &samplerLayoutBinding;

  VkResult ret = vkCreateDescriptorSetLayout(device, &layoutInfo, NULL,
                                             pVertexDisplayDescriptorSetLayout);
  if (ret != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL,
                   "failed to create descriptor set layout with error: %s",
                   vkstrerror(ret));
    PANIC();
  }

  // push our mvp matrix
  VkPushConstantRange pushConstantRange = {0};
  pushConstantRange.offset = 0;
  pushConstantRange.size = sizeof(mat4x4);
  pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

  VkPipelineLayoutCreateInfo pipelineLayoutInfo = {0};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = pVertexDisplayDescriptorSetLayout;
  pipelineLayoutInfo.pushConstantRangeCount = 1;
  pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
  VkResult res = vkCreatePipelineLayout(device, &pipelineLayoutInfo, NULL,
                                        pVertexDisplayPipelineLayout);
  if (res != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL,
                   "failed to create pipeline layout with error: %s",
                   vkstrerror(res));
    PANIC();
  }
}

void delete_VertexDisplayPipelineLayoutDescriptorSetLayout( //
    VkPipelineLayout *pPipelineLayout,                      //
    VkDescriptorSetLayout *pDescriptorSetLayout,            //
    const VkDevice device                                   //
) {
  vkDestroyPipelineLayout(device, *pPipelineLayout, NULL);
  *pPipelineLayout = VK_NULL_HANDLE;
  vkDestroyDescriptorSetLayout(device, *pDescriptorSetLayout, NULL);
  *pDescriptorSetLayout = VK_NULL_HANDLE;
}

void new_VertexDisplayPipeline(VkPipeline *pGraphicsPipeline,
                               const VkDevice device,
                               const VkShaderModule vertShaderModule,
                               const VkShaderModule fragShaderModule,
                               const VkExtent2D extent,
                               const VkRenderPass renderPass,
                               const VkPipelineLayout pipelineLayout) {
  VkPipelineShaderStageCreateInfo vertShaderStageInfo = {0};
  vertShaderStageInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
  vertShaderStageInfo.module = vertShaderModule;
  vertShaderStageInfo.pName = "main";

  VkPipelineShaderStageCreateInfo fragShaderStageInfo = {0};
  fragShaderStageInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  fragShaderStageInfo.module = fragShaderModule;
  fragShaderStageInfo.pName = "main";

  VkPipelineShaderStageCreateInfo shaderStages[2] = {vertShaderStageInfo,
                                                     fragShaderStageInfo};

  VkVertexInputBindingDescription bindingDescription = {0};
  bindingDescription.binding = 0;
  bindingDescription.stride = sizeof(Vertex);
  bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  VkVertexInputAttributeDescription attributeDescriptions[3];

  attributeDescriptions[0].binding = 0;
  attributeDescriptions[0].location = 0;
  attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributeDescriptions[0].offset = offsetof(Vertex, position);

  attributeDescriptions[1].binding = 0;
  attributeDescriptions[1].location = 1;
  attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributeDescriptions[1].offset = offsetof(Vertex, normal);

  attributeDescriptions[2].binding = 0;
  attributeDescriptions[2].location = 2;
  attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
  attributeDescriptions[2].offset = offsetof(Vertex, texCoords);

  VkPipelineVertexInputStateCreateInfo vertexInputInfo = {0};
  vertexInputInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInputInfo.vertexBindingDescriptionCount = 1;
  vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
  vertexInputInfo.vertexAttributeDescriptionCount = 3;
  vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions;

  VkPipelineInputAssemblyStateCreateInfo inputAssembly = {0};
  inputAssembly.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  inputAssembly.primitiveRestartEnable = VK_FALSE;

  VkViewport viewport = {0};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = (float)extent.width;
  viewport.height = (float)extent.height;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  VkRect2D scissor = {0};
  scissor.offset.x = 0;
  scissor.offset.y = 0;
  scissor.extent = extent;

  VkPipelineDepthStencilStateCreateInfo depthStencil = {0};
  depthStencil.sType =
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthStencil.depthTestEnable = VK_TRUE;
  depthStencil.depthWriteEnable = VK_TRUE;
  depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
  depthStencil.depthBoundsTestEnable = VK_FALSE;
  depthStencil.stencilTestEnable = VK_FALSE;

  VkPipelineViewportStateCreateInfo viewportState = {0};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.pViewports = &viewport;
  viewportState.scissorCount = 1;
  viewportState.pScissors = &scissor;

  VkPipelineRasterizationStateCreateInfo rasterizer = {0};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
  rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;

  VkPipelineMultisampleStateCreateInfo multisampling = {0};
  multisampling.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineColorBlendAttachmentState colorBlendAttachment = {0};
  colorBlendAttachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  colorBlendAttachment.blendEnable = VK_FALSE;

  VkPipelineColorBlendStateCreateInfo colorBlending = {0};
  colorBlending.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.logicOp = VK_LOGIC_OP_COPY;
  colorBlending.attachmentCount = 1;
  colorBlending.pAttachments = &colorBlendAttachment;
  colorBlending.blendConstants[0] = 0.0f;
  colorBlending.blendConstants[1] = 0.0f;
  colorBlending.blendConstants[2] = 0.0f;
  colorBlending.blendConstants[3] = 0.0f;

  VkGraphicsPipelineCreateInfo pipelineInfo = {0};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.stageCount = 2;
  pipelineInfo.pStages = shaderStages;
  pipelineInfo.pVertexInputState = &vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisampling;
  pipelineInfo.pColorBlendState = &colorBlending;
  pipelineInfo.pDepthStencilState = &depthStencil;
  pipelineInfo.layout = pipelineLayout;
  pipelineInfo.renderPass = renderPass;
  pipelineInfo.subpass = 0;
  pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

  if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, NULL,
                                pGraphicsPipeline) != VK_SUCCESS) {
    LOG_ERROR(ERR_LEVEL_FATAL, "failed to create graphics pipeline!");
    PANIC();
  }
}

void delete_Pipeline(VkPipeline *pPipeline, const VkDevice device) {
  vkDestroyPipeline(device, *pPipeline, NULL);
}

void new_Framebuffer(VkFramebuffer *pFramebuffer, const VkDevice device,
                     const VkRenderPass renderPass, const VkImageView imageView,
                     const VkImageView depthImageView,
                     const VkExtent2D swapchainExtent) {
  VkFramebufferCreateInfo framebufferInfo = {0};
  framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebufferInfo.renderPass = renderPass;
  framebufferInfo.attachmentCount = 2;
  framebufferInfo.pAttachments = (VkImageView[]){imageView, depthImageView};
  framebufferInfo.width = swapchainExtent.width;
  framebufferInfo.height = swapchainExtent.height;
  framebufferInfo.layers = 1;
  VkResult res =
      vkCreateFramebuffer(device, &framebufferInfo, NULL, pFramebuffer);
  if (res != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL, "failed to create framebuffers: %s",
                   vkstrerror(res));
    PANIC();
  }
}

void delete_Framebuffer(VkFramebuffer *pFramebuffer, VkDevice device) {
  vkDestroyFramebuffer(device, *pFramebuffer, NULL);
  *pFramebuffer = VK_NULL_HANDLE;
}

void new_SwapchainFramebuffers(VkFramebuffer *pFramebuffers,
                               const VkDevice device,
                               const VkRenderPass renderPass,
                               const VkExtent2D swapchainExtent,
                               const uint32_t imageCount,
                               const VkImageView depthImageView,
                               const VkImageView *pSwapchainImageViews) {
  for (uint32_t i = 0; i < imageCount; i++) {
    new_Framebuffer(&pFramebuffers[i], device, renderPass,
                    pSwapchainImageViews[i], depthImageView, swapchainExtent);
  }
}

void delete_SwapchainFramebuffers(VkFramebuffer *pFramebuffers,
                                  const uint32_t imageCount,
                                  const VkDevice device) {
  for (uint32_t i = 0; i < imageCount; i++) {
    delete_Framebuffer(&pFramebuffers[i], device);
  }
}

ErrVal new_CommandPool(VkCommandPool *pCommandPool, const VkDevice device,
                       const uint32_t queueFamilyIndex) {
  VkCommandPoolCreateInfo poolInfo = {0};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.queueFamilyIndex = queueFamilyIndex;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  VkResult ret = vkCreateCommandPool(device, &poolInfo, NULL, pCommandPool);
  if (ret != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_ERROR, "failed to create command pool %s",
                   vkstrerror(ret));
    return (ERR_UNKNOWN);
  }
  return (ERR_OK);
}

void delete_CommandPool(VkCommandPool *pCommandPool, const VkDevice device) {
  vkDestroyCommandPool(device, *pCommandPool, NULL);
}

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
) {
  VkCommandBufferBeginInfo beginInfo = {0};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  VkResult beginRet = vkBeginCommandBuffer(commandBuffer, &beginInfo);

  if (beginRet != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL,
                   "failed to record into graphics command buffer: %s",
                   vkstrerror(beginRet));
    PANIC();
  }

  VkRenderPassBeginInfo renderPassInfo = {0};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassInfo.renderPass = renderPass;
  renderPassInfo.framebuffer = swapchainFramebuffer;
  renderPassInfo.renderArea.offset = (VkOffset2D){0, 0};
  renderPassInfo.renderArea.extent = swapchainExtent;

  VkClearValue pClearColors[2];
  pClearColors[0].color = clearColor;
  pClearColors[1].depthStencil.depth = 1.0f;
  pClearColors[1].depthStencil.stencil = 0;

  renderPassInfo.clearValueCount = 2;
  renderPassInfo.pClearValues = pClearColors;

  vkCmdBeginRenderPass(commandBuffer, &renderPassInfo,
                       VK_SUBPASS_CONTENTS_INLINE);
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    vertexDisplayPipeline);
  vkCmdPushConstants(commandBuffer, vertexDisplayPipelineLayout,
                     VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(mat4x4),
                     cameraTransform);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          vertexDisplayPipelineLayout, 0, 1,
                          &vertexDisplayDescriptorSet, 0, NULL);
  // bind all vertex buffers, assume offsets are zero
  for (uint32_t i = 0; i < vertexBufferCount; i++) {
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &pVertexBuffers[i], offsets);
    vkCmdDraw(commandBuffer, pVertexCounts[i], 1, 0, 0);
  }
  vkCmdEndRenderPass(commandBuffer);

  VkResult endCommandBufferRetVal = vkEndCommandBuffer(commandBuffer);
  if (endCommandBufferRetVal != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL,
                   "Failed to record command buffer, error code: %s",
                   vkstrerror(endCommandBufferRetVal));
    PANIC();
  }
  return (ERR_OK);
}

ErrVal new_Semaphore(VkSemaphore *pSemaphore, const VkDevice device) {
  VkSemaphoreCreateInfo semaphoreInfo = {0};
  semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  VkResult ret = vkCreateSemaphore(device, &semaphoreInfo, NULL, pSemaphore);
  if (ret != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_ERROR, "failed to create semaphore: %s",
                   vkstrerror(ret));
    return (ERR_UNKNOWN);
  }
  return (ERR_OK);
}

void delete_Semaphore(VkSemaphore *pSemaphore, const VkDevice device) {
  vkDestroySemaphore(device, *pSemaphore, NULL);
  *pSemaphore = VK_NULL_HANDLE;
}

ErrVal new_Semaphores(VkSemaphore *pSemaphores, const uint32_t semaphoreCount,
                      const VkDevice device) {
  for (uint32_t i = 0; i < semaphoreCount; i++) {
    ErrVal retVal = new_Semaphore(&pSemaphores[i], device);
    if (retVal != ERR_OK) {
      delete_Semaphores(pSemaphores, i, device);
      return (retVal);
    }
  }
  return (ERR_OK);
}

void delete_Semaphores(VkSemaphore *pSemaphores, const uint32_t semaphoreCount,
                       const VkDevice device) {
  for (uint32_t i = 0; i < semaphoreCount; i++) {
    delete_Semaphore(&pSemaphores[i], device);
  }
}

// Note we're creating the fence already signaled!
ErrVal new_Fence(VkFence *pFence, const VkDevice device, const bool signaled) {
  VkFenceCreateInfo fenceInfo = {0};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  if (signaled) {
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  }
  VkResult ret = vkCreateFence(device, &fenceInfo, NULL, pFence);
  if (ret != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL, "failed to create fence: %s",
                   vkstrerror(ret));
    PANIC();
  }
  return (ERR_OK);
}

void delete_Fence(VkFence *pFence, const VkDevice device) {
  vkDestroyFence(device, *pFence, NULL);
  *pFence = VK_NULL_HANDLE;
}

ErrVal new_Fences(             //
    VkFence *pFences,          //
    const uint32_t fenceCount, //
    const VkDevice device,     //
    const bool allSignaled     //
) {
  for (uint32_t i = 0; i < fenceCount; i++) {
    ErrVal retVal = new_Fence(&pFences[i], device, allSignaled);
    if (retVal != ERR_OK) {
      /* Clean up memory */
      delete_Fences(pFences, i, device);
      return (retVal);
    }
  }
  return (ERR_OK);
}

void delete_Fences(VkFence *pFences, const uint32_t fenceCount,
                   const VkDevice device) {
  for (uint32_t i = 0; i < fenceCount; i++) {
    delete_Fence(&pFences[i], device);
  }
}

ErrVal waitAndResetFence(VkFence fence, const VkDevice device) {
  // Wait for the current frame to finish processing
  VkResult waitRet = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
  if (waitRet != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL, "failed to wait for fence: %s",
                   vkstrerror(waitRet));
    PANIC();
  }

  // reset the fence
  VkResult resetRet = vkResetFences(device, 1, &fence);
  if (resetRet != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL, "failed to reset fence: %s",
                   vkstrerror(resetRet));
    PANIC();
  }

  return (ERR_OK);
}

ErrVal getNextSwapchainImage(           //
    uint32_t *pImageIndex,              //
    const VkSwapchainKHR swapchain,     //
    const VkDevice device,              //
    VkSemaphore imageAvailableSemaphore //
) {
  // get the next image from the swapchain
  VkResult nextImageResult = vkAcquireNextImageKHR(
      device, swapchain, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE,
      pImageIndex);
  if (nextImageResult == VK_ERROR_OUT_OF_DATE_KHR || nextImageResult  == VK_SUBOPTIMAL_KHR) {
    // If the window has been resized, the result will be an out of date error,
    // meaning that the swap chain must be resized
    return (ERR_OUTOFDATE);
  } else if (nextImageResult != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL, "failed to get next frame: %s",
                   vkstrerror(nextImageResult));
    PANIC();
  }

  return (ERR_OK);
}

// Draws a frame to the surface provided, and sets things up for the next frame
ErrVal drawFrame(                        //
    VkCommandBuffer commandBuffer,       //
    VkSwapchainKHR swapchain,            //
    const uint32_t swapchainImageIndex,  //
    VkSemaphore imageAvailableSemaphore, //
    VkSemaphore renderFinishedSemaphore, //
    VkFence inFlightFence,               //
    const VkQueue graphicsQueue,         //
    const VkQueue presentQueue           //
) {

  // Sets up for next frame
  VkPipelineStageFlags waitStages[] = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

  VkSubmitInfo submitInfo = {0};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.waitSemaphoreCount = 1;
  submitInfo.pWaitSemaphores = &imageAvailableSemaphore;
  submitInfo.pWaitDstStageMask = waitStages;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;

  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = &renderFinishedSemaphore;

  VkResult queueSubmitResult =
      vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFence);
  if (queueSubmitResult != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL, "failed to submit queue: %s",
                   vkstrerror(queueSubmitResult));
    PANIC();
  }

  // Present frame to screen
  VkPresentInfoKHR presentInfo = {0};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = &renderFinishedSemaphore;
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = &swapchain;
  presentInfo.pImageIndices = &swapchainImageIndex;
  vkQueuePresentKHR(presentQueue, &presentInfo);

  return (ERR_OK);
}

// Deletes a VkSurfaceKHR
void delete_Surface(VkSurfaceKHR *pSurface, const VkInstance instance) {
  vkDestroySurfaceKHR(instance, *pSurface, NULL);
  *pSurface = VK_NULL_HANDLE;
}

/* Gets the extent of the given GLFW window */
ErrVal getExtentWindow(VkExtent2D *pExtent, GLFWwindow *pWindow) {
  int width;
  int height;
  glfwGetFramebufferSize(pWindow, &width, &height);
  pExtent->width = (uint32_t)width;
  pExtent->height = (uint32_t)height;
  return (ERR_OK);
}

/* Creates a new window using the GLFW library. */
ErrVal new_GlfwWindow(GLFWwindow **ppGlfwWindow, const char *name,
                      VkExtent2D dimensions) {
  /* Not resizable */
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  *ppGlfwWindow = glfwCreateWindow((int)dimensions.width,
                                   (int)dimensions.height, name, NULL, NULL);
  if (*ppGlfwWindow == NULL) {
    LOG_ERROR(ERR_LEVEL_ERROR, "failed to create GLFW window");
    return (ERR_UNKNOWN);
  }
  return (ERR_OK);
}

/* Creates a new window surface using the glfw libraries. This must be deleted
 * with the delete_Surface function*/
ErrVal new_SurfaceFromGLFW(VkSurfaceKHR *pSurface, GLFWwindow *pWindow,
                           const VkInstance instance) {
  VkResult res = glfwCreateWindowSurface(instance, pWindow, NULL, pSurface);
  if (res != VK_SUCCESS) {
    LOG_ERROR(ERR_LEVEL_FATAL, "failed to create surface, quitting");
    PANIC();
  }
  return (ERR_OK);
}

/* returns any errors encountered. Finds the index of the correct type of memory
 * to allocate for a given physical device using the bits and flags that are
 * requested. */
ErrVal getMemoryTypeIndex(uint32_t *memoryTypeIndex,
                          const uint32_t memoryTypeBits,
                          const VkMemoryPropertyFlags memoryPropertyFlags,
                          const VkPhysicalDevice physicalDevice) {

  /* Retrieve memory properties */
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
  /* Check each memory type to see if it conforms to our requirements */
  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((memoryTypeBits &
         (1 << i)) && /* TODO figure out what's going on over here */
        (memProperties.memoryTypes[i].propertyFlags & memoryPropertyFlags) ==
            memoryPropertyFlags) {
      *memoryTypeIndex = i;
      return (ERR_OK);
    }
  }
  LOG_ERROR(ERR_LEVEL_ERROR, "failed to find suitable memory type");
  return (ERR_MEMORY);
}

ErrVal new_Buffer_DeviceMemory(VkBuffer *pBuffer, VkDeviceMemory *pBufferMemory,
                               const VkDeviceSize size,
                               const VkPhysicalDevice physicalDevice,
                               const VkDevice device,
                               const VkBufferUsageFlags usage,
                               const VkMemoryPropertyFlags properties) {
  VkBufferCreateInfo bufferInfo = {0};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  /* Create buffer */
  VkResult bufferCreateResult =
      vkCreateBuffer(device, &bufferInfo, NULL, pBuffer);
  if (bufferCreateResult != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_ERROR, "failed to create buffer: %s",
                   vkstrerror(bufferCreateResult));
    return (ERR_UNKNOWN);
  }
  /* Allocate memory for buffer */
  VkMemoryRequirements memoryRequirements;
  vkGetBufferMemoryRequirements(device, *pBuffer, &memoryRequirements);

  VkMemoryAllocateInfo allocateInfo = {0};
  allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocateInfo.allocationSize = memoryRequirements.size;
  /* Get the type of memory required, handle errors */
  ErrVal getMemoryTypeRetVal = getMemoryTypeIndex(
      &allocateInfo.memoryTypeIndex, memoryRequirements.memoryTypeBits,
      properties, physicalDevice);
  if (getMemoryTypeRetVal != ERR_OK) {
    LOG_ERROR(ERR_LEVEL_ERROR, "failed to get type of memory to allocate");
    return (ERR_MEMORY);
  }

  /* Actually allocate memory */
  VkResult memoryAllocateResult =
      vkAllocateMemory(device, &allocateInfo, NULL, pBufferMemory);
  if (memoryAllocateResult != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_ERROR, "failed to allocate memory for buffer: %s",
                   vkstrerror(memoryAllocateResult));
    return (ERR_ALLOCFAIL);
  }
  vkBindBufferMemory(device, *pBuffer, *pBufferMemory, 0);
  return (ERR_OK);
}

// submits a copy to the queue, you'll later need to wait for idle
void copyBuffer(VkBuffer destinationBuffer, const VkBuffer sourceBuffer,
                const VkDeviceSize size, const VkCommandPool commandPool,
                const VkQueue queue, const VkDevice device) {

  VkCommandBuffer copyCommandBuffer =
      createBeginOneTimeCmdBuffer(commandPool, device);

  VkBufferCopy copyRegion = {.size = size, .srcOffset = 0, .dstOffset = 0};
  vkCmdCopyBuffer(copyCommandBuffer, sourceBuffer, destinationBuffer, 1,
                  &copyRegion);

  submitEndOneTimeCmdBuffer(copyCommandBuffer, queue, device);
  return;
}

void delete_Buffer(VkBuffer *pBuffer, const VkDevice device) {
  vkDestroyBuffer(device, *pBuffer, NULL);
  *pBuffer = VK_NULL_HANDLE;
}

void delete_DeviceMemory(VkDeviceMemory *pDeviceMemory, const VkDevice device) {
  vkFreeMemory(device, *pDeviceMemory, NULL);
  *pDeviceMemory = VK_NULL_HANDLE;
}

// updates, you'll later need to wait for idle
void updateBuffer(VkBuffer destinationBuffer, const void *pSource,
                  const VkDeviceSize size, const VkCommandPool commandPool,
                  const VkQueue queue, const VkDevice device) {

  VkCommandBuffer copyCommandBuffer =
      createBeginOneTimeCmdBuffer(commandPool, device);

  vkCmdUpdateBuffer(copyCommandBuffer, destinationBuffer, 0, size, pSource);

  submitEndOneTimeCmdBuffer(copyCommandBuffer, queue, device);
  return;
}

// creates a command buffer that hasn't yet been begun
ErrVal new_CommandBuffers(             //
    VkCommandBuffer *pCommandBuffer,   //
    const uint32_t commandBufferCount, //
    const VkCommandPool commandPool,   //
    const VkDevice device              //
) {
  VkCommandBufferAllocateInfo allocateInfo = {0};
  allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocateInfo.commandPool = commandPool;
  allocateInfo.commandBufferCount = commandBufferCount;

  VkResult allocateResult =
      vkAllocateCommandBuffers(device, &allocateInfo, pCommandBuffer);
  if (allocateResult != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL, "failed to allocate command buffers: %s",
                   vkstrerror(allocateResult));
    PANIC();
  }

  return (ERR_OK);
}

void delete_CommandBuffers(            //
    VkCommandBuffer *pCommandBuffers,  //
    const uint32_t commandBufferCount, //
    const VkCommandPool commandPool,   //
    const VkDevice device              //
) {
  vkFreeCommandBuffers(device, commandPool, commandBufferCount,
                       pCommandBuffers);
  for (uint32_t i = 0; i < commandBufferCount; i++) {
    pCommandBuffers[i] = VK_NULL_HANDLE;
  }
}

void copyToDeviceMemory(VkDeviceMemory *pDeviceMemory,
                        const VkDeviceSize deviceSize, const void *source,
                        const VkDevice device) {
  void *data;
  VkResult mapMemoryResult =
      vkMapMemory(device, *pDeviceMemory, 0, deviceSize, 0, &data);

  /* On failure */
  if (mapMemoryResult != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL,
                   "failed to copy to device memory: failed to map memory: %s",
                   vkstrerror(mapMemoryResult));
    PANIC();
  }

  // If it was successful, go on and actually copy it, making sure to unmap once
  // done
  memcpy(data, source, (size_t)deviceSize);
  vkUnmapMemory(device, *pDeviceMemory);
}

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
) {
  VkImageCreateInfo imageInfo = {0};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.extent.width = dimensions.width;
  imageInfo.extent.height = dimensions.height;
  imageInfo.extent.depth = 1;
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.format = format;
  imageInfo.tiling = tiling;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage = usage;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkResult createImageResult = vkCreateImage(device, &imageInfo, NULL, pImage);
  if (createImageResult != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL, "failed to create image: %s",
                   vkstrerror(createImageResult));
    PANIC();
  }

  VkMemoryRequirements memRequirements;
  vkGetImageMemoryRequirements(device, *pImage, &memRequirements);

  VkMemoryAllocateInfo allocInfo = {0};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;

  ErrVal memGetResult = getMemoryTypeIndex(&allocInfo.memoryTypeIndex,
                                           memRequirements.memoryTypeBits,
                                           properties, physicalDevice);

  if (memGetResult != ERR_OK) {
    LOG_ERROR(ERR_LEVEL_FATAL,
              "failed to create image: could not find right memory type");
    PANIC();
  }

  VkResult allocateResult =
      vkAllocateMemory(device, &allocInfo, NULL, pImageMemory);
  if (allocateResult != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL,
                   "failed to create image: could not allocate: %s",
                   vkstrerror(allocateResult));
    PANIC();
  }

  VkResult bindResult = vkBindImageMemory(device, *pImage, *pImageMemory, 0);
  if (bindResult != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL,
                   "failed to create image: could not bind memory: %s",
                   vkstrerror(bindResult));
    PANIC();
  }
}

void delete_Image(VkImage *pImage, const VkDevice device) {
  vkDestroyImage(device, *pImage, NULL);
  *pImage = VK_NULL_HANDLE;
}

/* Gets image format of depth */
void getDepthFormat(VkFormat *pFormat) {
  /* TODO we might want to redo this so that there are more compatible images */
  *pFormat = VK_FORMAT_D32_SFLOAT;
}

void new_DepthImage(VkImage *pImage, VkDeviceMemory *pImageMemory,
                    const VkExtent2D swapchainExtent,
                    const VkPhysicalDevice physicalDevice,
                    const VkDevice device) {
  VkFormat depthFormat;
  getDepthFormat(&depthFormat);
  new_Image(pImage, pImageMemory, swapchainExtent, depthFormat,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, physicalDevice, device);
}

void new_DepthImageView(VkImageView *pImageView, const VkDevice device,
                        const VkImage depthImage) {
  VkFormat depthFormat;
  getDepthFormat(&depthFormat);
  new_ImageView(pImageView, device, depthImage, depthFormat,
                VK_IMAGE_ASPECT_DEPTH_BIT);
}

// creates a descriptor pool to render an image sampler at binding 0
// creates a descriptor set with the given texture sampler and texture image
// view we can do this since in this application, the descriptors are 100%
// static
void new_VertexDisplayDescriptorPoolAndSet(                       //
    VkDescriptorPool *pDescriptorPool,                            //
    VkDescriptorSet *pDescriptorSet,                              //
    const VkDescriptorSetLayout vertexDisplayDescriptorSetLayout, //
    const VkDevice device,                                        //
    const VkSampler textureSampler,                               //
    const VkImageView textureImageView                            //
) {
  VkDescriptorPoolSize descriptorPoolSize = {0};
  descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  descriptorPoolSize.descriptorCount = 1;

  VkDescriptorPoolCreateInfo poolInfo = {0};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = &descriptorPoolSize;
  poolInfo.maxSets = 1;

  /* Actually create descriptor pool */
  VkResult ret =
      vkCreateDescriptorPool(device, &poolInfo, NULL, pDescriptorPool);

  if (ret != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL, "failed to create descriptor pool: %s",
                   vkstrerror(ret));
    PANIC();
  }

  VkDescriptorSetAllocateInfo allocInfo = {0};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = *pDescriptorPool;
  // we would need an array of these corresponding to each element in the set,
  // but since we only have 1 descriptor set layout, we are good
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &vertexDisplayDescriptorSetLayout;

  VkResult sets_ret =
      vkAllocateDescriptorSets(device, &allocInfo, pDescriptorSet);
  if (sets_ret != VK_SUCCESS) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL, "failed to allocate descriptor sets: %s",
                   vkstrerror(ret));
    PANIC();
  }

  VkDescriptorImageInfo imageInfo = {0};
  imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  imageInfo.imageView = textureImageView;
  imageInfo.sampler = textureSampler;

  VkWriteDescriptorSet descriptorWrite = {0};
  descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  descriptorWrite.dstSet = *pDescriptorSet;
  descriptorWrite.dstBinding = 0;
  descriptorWrite.dstArrayElement = 0;
  descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  descriptorWrite.descriptorCount = 1;
  descriptorWrite.pImageInfo = &imageInfo;

  vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, NULL);
}

void delete_DescriptorPool(VkDescriptorPool *pDescriptorPool,
                           const VkDevice device) {
  vkDestroyDescriptorPool(device, *pDescriptorPool, NULL);
  *pDescriptorPool = VK_NULL_HANDLE;
}
