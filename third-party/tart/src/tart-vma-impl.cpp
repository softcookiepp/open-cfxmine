#include <iostream>
#include "tart-vma-impl.hpp"

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

namespace tart
{

Allocator::Allocator(vk::Instance& instance, vk::PhysicalDevice& physicalDevice, vk::PhysicalDeviceProperties& deviceProps,
	vk::Device& device, bool trackMemory, bool bda):
	mDevice(device)
{
	VmaAllocatorCreateInfo info = {};
	info.vulkanApiVersion = VK_API_VERSION_1_2;
	info.physicalDevice = physicalDevice;
	info.device = device;
	info.instance = instance;
	info.flags = VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT;
	if (trackMemory)
		info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
	if (bda)
		info.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	if (VK_SUCCESS != vmaCreateAllocator(&info, &mVmaAllocator) )
		throw std::runtime_error("failed to create VmaAllocator");
	
	// now create pools
	initPools();
}

Allocator::~Allocator()
{
	if (mPoolInit)
	{
		vmaDestroyPool(mVmaAllocator, mHostPool);
		vmaDestroyPool(mVmaAllocator, mDevicePool);
	}
	vmaDestroyAllocator(mVmaAllocator);
}

void Allocator::initPool(VmaPool& pool, VmaMemoryUsage memoryUsage)
{
	uint32_t queueFamilyIndex(0);
	vk::BufferCreateInfo bufCreateInfo{
		vk::BufferCreateFlags(),
		0x1000,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
		vk::SharingMode::eExclusive,
		1, &queueFamilyIndex
	};
	auto bufCreateInfoC = static_cast<VkBufferCreateInfo>(bufCreateInfo);
	
	VmaAllocationCreateInfo sampleAllocCreateInfo = {};
	sampleAllocCreateInfo.usage = memoryUsage;
	
	uint32_t memTypeIndex;
	if (VK_SUCCESS != vmaFindMemoryTypeIndexForBufferInfo(mVmaAllocator, &bufCreateInfoC, &sampleAllocCreateInfo, &memTypeIndex) )
		throw std::runtime_error("Could not get memory type index");
	
	VmaPoolCreateInfo poolCreateInfo = {};
	poolCreateInfo.memoryTypeIndex = memTypeIndex;
	poolCreateInfo.blockSize = mBlockSize;
	poolCreateInfo.minBlockCount = 0;
	poolCreateInfo.maxBlockCount = 0;// whatever	
	poolCreateInfo.flags = VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT;
	
	if (VK_SUCCESS != vmaCreatePool(mVmaAllocator, &poolCreateInfo, &pool) )
		throw std::runtime_error("failed to create VMA pool");
	
}

void Allocator::initPools()
{
	if (mPoolInit) return;
	// init both of them
	initPool(mDevicePool, VMA_MEMORY_USAGE_GPU_ONLY);
	initPool(mHostPool, VMA_MEMORY_USAGE_CPU_TO_GPU);
	mPoolInit = true;
}

void Allocator::allocateBuffer(vk::BufferUsageFlags usageFlags,
	vk::Buffer& buffer, VmaAllocation& allocation, uint64_t bufferSize,
	uint32_t queueFamilyIndex, VmaMemoryUsage usage)
{
	// first make the info struct
	vk::BufferCreateInfo bufCreateInfo{
		vk::BufferCreateFlags(),
		bufferSize,
		usageFlags,
		vk::SharingMode::eExclusive,
		1, &queueFamilyIndex
	};
	// cast it to Vulkan C API type to interface with VMA
	auto bufCreateInfoC = static_cast<VkBufferCreateInfo>(bufCreateInfo);
	
	VkBuffer bufferRaw;
	
	VmaAllocationCreateInfo allocationInfo = {};
	
	// set the usage and flags, if applicable
	allocationInfo.usage = usage;
	//if (usage == VMA_MEMORY_USAGE_CPU_TO_GPU)
	//	allocationInfo.flags = (VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT);
	
	if (mPoolInit)
	{
		if (usage == VMA_MEMORY_USAGE_CPU_TO_GPU)
		{
			allocationInfo.pool = mHostPool;
		}
		else if(usage == VMA_MEMORY_USAGE_GPU_ONLY)
		{
			// "Allocating buffers larger than the pool’s block size without the proper flag: Large allocations require dedicated memory or the pool block size must be sufficient." - a wise GPU in someone else's computer
			// I need to take this into account and specify the proper flags when making allocations larger than the block size
			allocationInfo.pool = mDevicePool;
		}
		else
			throw std::invalid_argument("usage must be either VMA_MEMORY_USAGE_AUTO_PREFER_HOST or VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE");
	}
	
	// now we just create the buffer! that is quite a bit easier
	if (VK_SUCCESS != vmaCreateBuffer(mVmaAllocator, &bufCreateInfoC, &allocationInfo, &bufferRaw, &allocation, nullptr) )
		throw std::runtime_error("failed to create buffer");
	// almost forgot this very important part
	buffer = bufferRaw;
}

void Allocator::deallocateBuffer(vk::Buffer& buffer, VmaAllocation& allocation)
{
	// slightly different, but ever so similar
	vmaDestroyBuffer(mVmaAllocator, buffer, allocation);
}

void* Allocator::map(VmaAllocation& allocation)
{
	void* ptr = nullptr;
	VkResult result = vmaMapMemory(mVmaAllocator, allocation, reinterpret_cast<void**>(&ptr));
	if (result != VK_SUCCESS)
	{
		std::cerr << "vmaMapMemory failed with result: " << result << std::endl;
		throw std::runtime_error("failed to map memory!");
	}
	return ptr;
}

void Allocator::unmap(VmaAllocation& allocation)
{
	vmaUnmapMemory(mVmaAllocator, allocation);
}

} // namespace tart
