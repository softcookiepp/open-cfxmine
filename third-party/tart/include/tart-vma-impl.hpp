#ifndef TART_VMA
#define TART_VMA
#include "tart-vulkan-include.hpp"
#include "vk_mem_alloc.h"

namespace tart
{

class Allocator
{
	// the holy grail of memory management speedup, I seriously hope
	VmaAllocator mVmaAllocator;
	
	// all the stuff required for doing much of anything with VMA
	vk::Instance mInstance;
	vk::Device mDevice;
	
	uint64_t mBlockSize = 0;

	// whether or not pools are initialized
	bool mPoolInit = false;
	// one pool for device memory, one for host
	VmaPool mDevicePool;
	VmaPool mHostPool;
	void initPools();
	void initPool(VmaPool& pool, VmaMemoryUsage memoryUsage);
	
	// 
	uint32_t getMemoryTypeIndex();
	
public:
	Allocator(vk::Instance& instance, vk::PhysicalDevice& physicalDevice, vk::PhysicalDeviceProperties& deviceProps,
		vk::Device& device, bool trackMemory, bool bda);
	~Allocator();
	
	void allocateBuffer(vk::BufferUsageFlags usageFlags,
		vk::Buffer& buffer, VmaAllocation& allocation, uint64_t bufferSize,
		uint32_t queueFamilyIndex, VmaMemoryUsage usage);
		
	void deallocateBuffer(vk::Buffer& buffer, VmaAllocation& allocation);
	
	void* map(VmaAllocation& allocation);
	void unmap(VmaAllocation& allocation);
};

} // namespace tart

#endif
