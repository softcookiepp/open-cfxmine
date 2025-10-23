#include <fstream>
#include <vulkan/vulkan_extension_inspection.hpp>
#include <cstdlib>
#include <regex>

#include "tart.hpp"
#include "tart-vk-helpers.hpp"
#include "tart-compilers.hpp"

namespace tart
{
#ifdef TART_USE_VMA
Buffer::Buffer(device_ptr device, uint64_t size, vk::BufferUsageFlags usageFlags, VmaMemoryUsage usage):
	mBufferSize(size),
	mBufferOffset(0),
	mDevice(device),
	mIsView(false), // just to be safe
	mBufferUsageFlags(usageFlags),
	mVmaMemoryUsage(usage)
{	
	// use the allocator
	mDevice.lock()->mAllocator->allocateBuffer(usageFlags, mBuffer, mVmaAllocation, size,
		mDevice.lock()->mComputeQueueFamilyIndex, usage);
}

Buffer::Buffer(device_ptr device, uint64_t offset, uint64_t originalSize, vk::BufferUsageFlags usageFlags, VmaMemoryUsage usage, vk::Buffer& buffer, VmaAllocation& allocation):
	mBufferOffset(offset),
	mBufferSize(originalSize),
	mIsView(true),
	mDevice(device),
	mBufferUsageFlags(usageFlags),
	mVmaMemoryUsage(usage),
	mBuffer(buffer),
	mVmaAllocation(allocation)
{
	// nothing else is needed
}

#else
Buffer::Buffer(device_ptr device, uint64_t size, vk::BufferUsageFlags usageFlags, vk::MemoryPropertyFlags& memoryFlags):
	mBufferSize(size),
	mBufferOffset(0),
	mDevice(device),
	mIsView(false), // just to be safe
	mBufferUsageFlags(usageFlags),
	mMemoryPropertyFlags(memoryFlags)
{	
	// well isn't this more concise!
	tart_helpers::allocateBuffer(mDevice.lock()->mPhysicalDevice, mDevice.lock()->mDevice,
		usageFlags, mBuffer, mBufferMemory, size, mDevice.lock()->mComputeQueueFamilyIndex, memoryFlags);
}

Buffer::Buffer(device_ptr device, uint64_t offset, uint64_t originalSize, vk::BufferUsageFlags usageFlags, vk::MemoryPropertyFlags& memoryFlags, vk::Buffer& buffer, vk::DeviceMemory& memory):
	mBufferOffset(offset),
	mBufferSize(originalSize),
	mIsView(true),
	mDevice(device),
	mBufferUsageFlags(usageFlags),
	mMemoryPropertyFlags(memoryFlags),
	mBuffer(buffer),
	mBufferMemory(memory)
{
	// nothing else is needed
}
#endif

buffer_ptr Buffer::view(uint64_t offset)
{
	// offset must be appended to existing offset if creating nested views
	offset += mBufferOffset;
	// ensure new buffer size and buffer offset don't go out of bounds
	if (offset >= mBufferSize)
	{
		std::stringstream ss;
		ss << "Offset << " << offset << " is too large for buffer of size " << mBufferSize << ".";
		throw std::invalid_argument(ss.str());
	}
	
	// construct new buffer with view constructor, append to tracked child views
#ifdef TART_USE_VMA
	buffer_ptr newView = std::make_shared<Buffer>(mDevice.lock(), offset, mBufferSize, mBufferUsageFlags,
		mVmaMemoryUsage, mBuffer, mVmaAllocation);
#else
	buffer_ptr newView = std::make_shared<Buffer>(mDevice.lock(), offset, mBufferSize, mBufferUsageFlags,
		mMemoryPropertyFlags, mBuffer, mBufferMemory);
#endif
	mChildViews.push_back(newView);
	return newView;
}

const uint64_t Buffer::getAddress()
{
	if (!mDevice.lock()->mDeviceMetadata.bda) throw std::runtime_error("Cannot get the address of device that lacks BDA support");
	vk::BufferDeviceAddressInfo info(mBuffer);
	return mDevice.lock()->mDevice.getBufferAddress(info);
}
#ifdef TART_USE_VMA

void* Buffer::map()
{
	return mDevice.lock()->mAllocator->map(mVmaAllocation);
}

void Buffer::unmap()
{
	mDevice.lock()->mAllocator->unmap(mVmaAllocation);
}
#else
void* Buffer::map(uint64_t offset, uint64_t size)
{
	// this is not recommended...
	return mDevice.lock()->mDevice.mapMemory(mBufferMemory, offset, size);
}

void* Buffer::map()
{
	return mDevice.lock()->mDevice.mapMemory(mBufferMemory, mBufferOffset, mBufferSize);
}

void Buffer::unmap()
{
	mDevice.lock()->mDevice.unmapMemory(mBufferMemory);
}
#endif

Buffer::~Buffer()
{
	destroy();
}

void Buffer::destroy()
{
	if (mDestroyed) return;
	for (buffer_ref view : mChildViews)
	{
		// just recursively call destroy
		if ( !view.expired() ) view.lock()->destroy();
	}
	if (!mIsView)
	{
#ifdef TART_USE_VMA
		mDevice.lock()->mAllocator->deallocateBuffer(mBuffer, mVmaAllocation);
#else
		// only destroy the underlying resources if the buffer is indeed the owner
		mDevice.lock()->mDevice.freeMemory(mBufferMemory);
		mDevice.lock()->mDevice.destroyBuffer(mBuffer);
#endif
	}
	mDestroyed = true;
}

void Buffer::copyOut(void* hostbuf, uint64_t size)
{
	uint64_t stagingSize = mBufferSize - mBufferOffset;
	if (size > stagingSize) throw std::invalid_argument("Too big of a size provided!");
	stagingSize = size;
	
	buffer_ptr stagingBuffer = mDevice.lock()->allocateBuffer(stagingSize, true);
	copyTo(stagingBuffer);
	
	// now we should be able to map the staging memory, and copy it
#ifdef TART_USE_VMA
	uint8_t* mappedBuf = static_cast<uint8_t*>( mDevice.lock()->mAllocator->map(stagingBuffer->mVmaAllocation) );
#else
	uint8_t* mappedBuf = static_cast<uint8_t*>( mDevice.lock()->mDevice.mapMemory(stagingBuffer->mBufferMemory, 0, stagingSize) );
#endif
	uint8_t* castHostPtr = static_cast<uint8_t*>(hostbuf);
	for(uint64_t i = 0; i < stagingSize; i += 1)
	{
		castHostPtr[i] = mappedBuf[i];
	}
#ifdef TART_USE_VMA
	mDevice.lock()->mAllocator->unmap(stagingBuffer->mVmaAllocation);
#else
	mDevice.lock()->mDevice.unmapMemory(stagingBuffer->mBufferMemory);
#endif
	// destroy
	mDevice.lock()->deallocateBuffer(stagingBuffer);
}

void Buffer::copyIn(void* hostbuf, uint64_t size)
{
	uint64_t stagingSize = mBufferSize - mBufferOffset;
	if (size > stagingSize) throw std::invalid_argument("Too big of a size provided!");
	stagingSize = size;
	
	buffer_ptr stagingBuffer = mDevice.lock()->allocateBuffer(stagingSize, true);
#ifdef TART_USE_VMA
	uint8_t* mappedBuf = static_cast<uint8_t*>( mDevice.lock()->mAllocator->map(stagingBuffer->mVmaAllocation) );
#else
	uint8_t* mappedBuf = static_cast<uint8_t*>( mDevice.lock()->mDevice.mapMemory(stagingBuffer->mBufferMemory, 0, stagingSize) );
#endif
	uint8_t* castHostPtr = static_cast<uint8_t*>(hostbuf);
	for(uint64_t i = 0; i < stagingSize; i += 1)
	{
		mappedBuf[i] = castHostPtr[i];
	}
#ifdef TART_USE_VMA
	mDevice.lock()->mAllocator->unmap(stagingBuffer->mVmaAllocation);
#else
	mDevice.lock()->mDevice.unmapMemory(stagingBuffer->mBufferMemory);
#endif
	copyFrom(stagingBuffer);
	
	// destroy
	mDevice.lock()->deallocateBuffer(stagingBuffer);
}

bool Buffer::anyUsers()
{
	/*
	 * Since the user list is no longer going to be kept track of by each buffer,
	 * the device itself must be used to query whether or not the buffer is being used.
	 * A new function will have to be written for that.
	 */
	return mDevice.lock()->isBufferInUse(*this);
}

void Buffer::copyTo(Buffer& otherBuffer, uint64_t selfOffset, uint64_t destOffset, uint64_t size)
{
	// ensure offsets are adjusted for views if necessary
	selfOffset += mBufferOffset;
	destOffset += otherBuffer.mBufferOffset;
	
	// check to ensure copy is safe to do
	if (anyUsers() || otherBuffer.anyUsers()) throw std::runtime_error("Cannot copy a buffer while it is in use!");
	if (otherBuffer.mDevice.lock() != mDevice.lock()) throw std::runtime_error("Copying buffers of different devices is not yet implemented!");
	
	// validate offsets and size
	if (selfOffset >= mBufferSize) throw std::invalid_argument("selfOffset cannot be larger than size of buffer!");
	if (destOffset >= otherBuffer.mBufferSize) throw std::invalid_argument("destOffset cannot be larger than size of buffer!");
	if (size == 0)
	{
		// choose the smaller one to be safe
		if (mBufferSize - selfOffset > otherBuffer.mBufferSize - destOffset) size = otherBuffer.mBufferSize - destOffset;
		else size = mBufferSize - selfOffset;
	}
	else
	{
		if (selfOffset + size >= mBufferSize) throw std::invalid_argument("selfOffset + size cannot be greated than buffer size!");
		if (destOffset + size >= otherBuffer.mBufferSize) throw std::invalid_argument("destOffset + size cannot be greated than buffer size!");
	}
	
	// then copy, finally
	tart_helpers::copyBuffer(mDevice.lock()->mDevice,
		mBuffer, selfOffset,
		otherBuffer.mBuffer, destOffset,
		size, mDevice.lock()->mComputeQueueFamilyIndex);
}

void Buffer::copyFrom(Buffer& otherBuffer) { otherBuffer.copyTo(*this); }

void Buffer::copyTo(buffer_ptr otherBuffer, uint64_t selfOffset, uint64_t destOffset, uint64_t size)
	{ copyTo(*otherBuffer, selfOffset, destOffset, size); }

void Buffer::copyFrom(buffer_ptr otherBuffer) { copyFrom(*otherBuffer); }

Instance::Instance(std::vector<std::string> validationLayers)
{
	// Vulkan Instance - vk::Instance
	// A Vulkan application starts with a vk::Instance, so lets create one:
	vk::ApplicationInfo AppInfo{
		"tart",				// Application Name
		1,					// Application Version
		nullptr,			// Engine Name or nullptr
		0,					// Engine Version
		VK_API_VERSION_1_2  // Vulkan API version
	};
	
	// enable validation layers if supplied
	std::vector<const char*> Layers = {};
	for (std::string& layer : validationLayers) Layers.push_back( layer.c_str() );
	vk::InstanceCreateInfo InstanceCreateInfo(vk::InstanceCreateFlags(), &AppInfo, Layers.size(), Layers.data());
	mVkInstance = vk::createInstance(InstanceCreateInfo);
	std::vector<vk::PhysicalDevice> allDevices = mVkInstance.enumeratePhysicalDevices();

	// take limited device visibility into account
	std::vector<uint32_t> deviceIndices;
	if (const char* visibleDeviceEnv = std::getenv("TART_VISIBLE_DEVICES") )
	{
		std::string visibleDeviceArg(visibleDeviceEnv);
		std::regex filter("[a-zA-Z 	]*");
		
		std::stringstream ss;
		ss << std::regex_replace(visibleDeviceArg, filter, "");
		std::string visibleDeviceArgParsed = ss.str();
		std::cout << "TART_VISIBLE_DEVICES=" << visibleDeviceArgParsed << std::endl;
		
		ss.str("");
		ss.clear();
		for (char c : visibleDeviceArgParsed)
		{
			if (c == ',')
			{
				std::string idxStr = ss.str();
				if (idxStr.size() > 0)
				{
					uint32_t idx = std::stoi(idxStr);
					if (idx < allDevices.size()) deviceIndices.push_back(idx);
				}
				ss.str("");
				ss.clear();
			}
			else
			{
				ss << c;
			}
		}
		
		std::string idxStr = ss.str();
		if (idxStr.size() > 0)
		{
			uint32_t idx = std::stoi(idxStr);
			if (idx < allDevices.size()) deviceIndices.push_back(idx);
		}
		
		for (uint32_t idx : deviceIndices)
		{
			mPhysicalDevices.push_back(allDevices[idx]);
		}
	}
	
	// just default to all
	if (deviceIndices.size() == 0)
		mPhysicalDevices = allDevices;
	
	// initialize vector of internal devices with nullptr at every index
	mDevices.resize( mPhysicalDevices.size() );
}

Instance::~Instance()
{
	for (device_ptr device : mDevices)
	{
		if (device) device->destroy();
	}
	mDevices.clear();
	mVkInstance.destroy();
	mDestroyed = true;
}

device_ptr Instance::createDevice(uint32_t index, std::vector<std::string> requiredExtensionNames)
{
	if (index >= mDevices.size() )
	{
		throw std::runtime_error("Invalid device index!");
	}
	
	device_ptr dev = mDevices[index];
	if(!dev)
	{
		dev = std::make_shared<Device>(mVkInstance, mPhysicalDevices[index], index, requiredExtensionNames);
		mDevices[index] = dev;
		// set self as well
		dev->initDevice(dev);
	}
	return dev;
}

} // namespace tart
