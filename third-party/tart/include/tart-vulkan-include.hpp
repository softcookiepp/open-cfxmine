#ifndef TART_VULKAN_INCLUDE
#define TART_VULKAN_INCLUDE

#ifdef ENABLE_LOCAL_LOADER
#include "vulkan/vulkan.hpp"
#else
#include <vulkan/vulkan.hpp>
#endif

#ifndef TART_USE_VMA
#define TART_USE_VMA
#endif

#endif
