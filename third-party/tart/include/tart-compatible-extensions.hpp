#ifndef TART_ERROR_HANDLING
#define TART_ERROR_HANDLING


#include <iostream>
#include <map>
#include <utility>
#include <vector>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <set>
#include <string>
#include <stdexcept>

namespace tart_extensions
{
static const std::vector<std::string> implementedExtensions(
	{
		"VK_KHR_storage_buffer_storage_class",
		"VK_KHR_16bit_storage",
		"VK_KHR_shader_float16_int8",
		
		"VK_KHR_external_fence",
#ifdef VK_USE_PLATFORM_WIN32_KHR
		"VK_KHR_external_fence_win32",
#endif
		"VK_KHR_external_fence_fd",
		// ...
		"VK_KHR_variable_pointers"
		
	}
);

} // namespace tart_extensions
#endif
