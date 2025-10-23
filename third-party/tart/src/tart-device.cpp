#include <fstream>
#include "tart.hpp"
#include "tart-hardware-profiling.hpp"
#include "tart-internal.hpp"
#include "tart-compilers.hpp"
#include "tart-vk-helpers.hpp"
#include <vulkan/vulkan_extension_inspection.hpp>
//#include "tart-vma-impl.hpp"


namespace tart
{
#ifdef TART_USE_VMA
buffer_ptr Device::allocateBuffer(uint64_t bufferSize, VmaMemoryUsage usage)
#else
buffer_ptr Device::allocateBuffer(uint64_t bufferSize, vk::MemoryPropertyFlags memoryFlags)
#endif
{
	std::lock_guard<std::recursive_mutex> guard( *mDeviceMutex );
	
	if (bufferSize > mDeviceMetadata.maxMemoryAllocationSize)
	{
		std::stringstream msgStream;
		msgStream << "Tried to allocate buffer with size of " << bufferSize
			<< " bytes, but max memory size for device " << mDeviceIndex
			<< " is " << mDeviceMetadata.maxMemoryAllocationSize << ".";
		throw std::runtime_error(msgStream.str());
	}
	
#ifdef TART_USE_VMA
	buffer_ptr buf = std::make_shared<Buffer>(mSelf.lock(), bufferSize, DEFAULT_BUFFER_FLAG_BITS, usage);
#else
	// pretty sure VMA will handle this for us
	if (mAllocatedBuffers.size() + 1 >= mPhysicalDeviceProperties.limits.maxMemoryAllocationCount)
	{
		std::string msg("Cannot allocate more than ");
		msg += std::to_string(mPhysicalDeviceProperties.limits.maxMemoryAllocationCount);
		msg += " buffers!";
		throw std::runtime_error(msg);
	}
	
	buffer_ptr buf = std::make_shared<Buffer>(mSelf.lock(), bufferSize, DEFAULT_BUFFER_FLAG_BITS, memoryFlags);
#endif
	mAllocatedBuffers.insert(buf);
	return buf;
}

buffer_ptr Device::allocateBuffer(uint64_t bufferSize, bool host)
{
#ifdef TART_USE_VMA
	VmaMemoryUsage usage;
	if (host) usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
	else usage = VMA_MEMORY_USAGE_GPU_ONLY; // use gpu memory
	return allocateBuffer(bufferSize, usage);
#else
	vk::MemoryPropertyFlags memoryFlags;
	if (host) memoryFlags = mDefaultHostMemoryFlags;
	else memoryFlags = mDefaultDeviceMemoryFlags;
	return allocateBuffer(bufferSize, memoryFlags);
#endif
}

command_sequence_ptr Device::dispatchPipeline(
	Pipeline& pipeline,
	std::vector<uint32_t> workGroup,
	std::vector<buffer_ptr> buffers,
	std::vector<uint8_t> pushConstants)
{	
	std::lock_guard<std::recursive_mutex> guard( *mDeviceMutex );
	command_sequence_ptr sequence = mSelf.lock()->createSequence();
	sequence->recordPipeline(pipeline, workGroup, buffers, pushConstants);
	
	// default to 0 and false for now
	return mSelf.lock()->submitSequence(sequence, 0, nullptr);
}

bool Device::addExtensionIfSupported(std::string shader, std::vector<std::string>& shaderList)
{
	std::lock_guard<std::recursive_mutex> guard( *mDeviceMutex );
	bool result = supportsExtension(shader);
	if (result) shaderList.push_back(shader);
	return result;
}

std::vector<std::string> Device::initDefaultExtensions()
{
	std::lock_guard<std::recursive_mutex> guard( *mDeviceMutex );
	std::vector<std::string> enabledDefaults;
	std::cerr << "WARNING: Device::initDefaultExtensions not completely implemented!" << std::endl;
	static const std::vector<std::string> defaults( {
			// 16-bit float capabilities
			"VK_KHR_storage_buffer_storage_class",
			"VK_KHR_16bit_storage",
			"VK_KHR_shader_float16_int8"
			
			// 
		}
	);
	
	// these compilers can be enabled as supported by default
	mCompilerSupport.glsl = true;
	mCompilerSupport.dxc = true;
	
	if ( addExtensionIfSupported("VK_KHR_variable_pointers", enabledDefaults)
		&& addExtensionIfSupported("VK_KHR_storage_buffer_storage_class", enabledDefaults)
		&& addExtensionIfSupported("VK_KHR_shader_non_semantic_info", enabledDefaults) )
		mCompilerSupport.clspv = true;
	if ( addExtensionIfSupported("VK_KHR_storage_buffer_storage_class", enabledDefaults)
		&& addExtensionIfSupported("VK_KHR_16bit_storage", enabledDefaults)
		&& addExtensionIfSupported("VK_KHR_shader_float16_int8", enabledDefaults) )
	{
		mDeviceMetadata.half_ = true;
	}
	
	return enabledDefaults;
}

bool Device::supportsExtension(std::string ext)
{
	std::lock_guard<std::recursive_mutex> guard( *mDeviceMutex );
	for (vk::ExtensionProperties& props : mDeviceExtensionProperties)
	{
		if ( strcmp(props.extensionName, ext.c_str() ) == 0 ) return true;
	}
	return false;
}

cl_program_ptr Device::createCLProgram(shader_module_ptr shaderModule)
{
	return std::make_shared<CLProgram>(mSelf, shaderModule);
}

pipeline_ptr Device::createPipeline(shader_module_ptr shaderModule, std::string entryPoint, std::vector<uint8_t> specConstants, std::vector<uint8_t> defaultPushConstants)
{
	std::lock_guard<std::recursive_mutex> guard( *mDeviceMutex );
	return std::make_shared<Pipeline>(shaderModule, entryPoint, specConstants, defaultPushConstants);
}

void Device::initDevice(device_ref self)
{
	std::lock_guard<std::recursive_mutex> guard( *mDeviceMutex );
	
	// only needs to be done once
	if (mDeviceInitialized) return;
	mDeviceInitialized = true;
	
	// some stuffs
	std::cout << "Device Name    : " << mPhysicalDeviceProperties.deviceName << std::endl;
	const uint32_t ApiVersion = mPhysicalDeviceProperties.apiVersion;
	std::cout << "Vulkan Version : " << VK_VERSION_MAJOR(ApiVersion) << "." << VK_VERSION_MINOR(ApiVersion) << "." << VK_VERSION_PATCH(ApiVersion) << std::endl;
	vk::PhysicalDeviceLimits DeviceLimits = mPhysicalDeviceProperties.limits;
	
	// TODO: get maintenance limits
	std::cout << "Max Compute Shared Memory Size: " << DeviceLimits.maxComputeSharedMemorySize / 1024 << " KB" << std::endl;
	// Here I’m just printing some information from the first physical device available in the machine.
	
	// create a temporary char* vector for the required extensions
	std::vector<const char*> requiredExtensions;
	for (std::string& ext : mUsedExtensionNames)
	{
		requiredExtensions.push_back( ext.c_str() );
	}
	
	// Vulkan Device - vk::Device
	// Creating a device requires a vk::DeviceQueueCreateInfo and a vk::DeviceCreateInfo:
	// Just to avoid a warning from the Vulkan Validation Layer
	const float QueuePriority = 1.0f;
	vk::DeviceQueueCreateInfo DeviceQueueCreateInfo(vk::DeviceQueueCreateFlags(),		// Flags
													mComputeQueueFamilyIndex,	// Queue Family Index
													1,									// Number of Queues ( need to update this if transfer queues are implemented
													&QueuePriority);
	vk::DeviceCreateInfo DeviceCreateInfo(
		vk::DeviceCreateFlags(),	// Flags
		1,							// DeviceQueueCreateInfo count
		&DeviceQueueCreateInfo,		// Device Queue Create Info struct
		0,							// layer count
		nullptr,					// layers
		requiredExtensions.size(),	// extension count
		requiredExtensions.data()	// extensions
	);
	mDevice = mPhysicalDevice.createDevice(DeviceCreateInfo);
	
	// verify that device was created
	if (mDevice == vk::Device{}) throw std::runtime_error("Device initialization failed for some reason!");

	mAllocator = std::make_unique<Allocator>(mVkInstance, mPhysicalDevice,mPhysicalDeviceProperties, mDevice,
		mDeviceMetadata.canTrackUsedMemory, mDeviceMetadata.bda);
	
	// post device initialization actions
	mPipelineCache = mDevice.createPipelineCache(vk::PipelineCacheCreateInfo());
	mComputeQueue = mDevice.getQueue(mComputeQueueFamilyIndex, 0);
	
	// init other members that require self reference
	if (self.lock().get() != this) throw std::runtime_error("Cannot set internal device reference to other internal device instance!");
	mSelf = self;
	mDescriptorSetLayoutCache = std::make_shared<DescriptorSetLayoutCache>(self);
	mDescriptorSetCache = std::make_shared<DescriptorSetCache>(self);

	mCommandPool = tart_helpers::initCommandPool(mDevice, mComputeQueueFamilyIndex);
}

void Device::destroy()
{
	std::lock_guard<std::recursive_mutex> guard( *mDeviceMutex );
	if (mDestroyed) return;
	/* 
	 * The descriptor set cache must be freed first, since it may
	 * depend on resources from the descriptor layout cache.
	 * After that, the descriptor layout cache may be freed.
	*/
	mDescriptorSetCache = nullptr;
	mDescriptorSetLayoutCache = nullptr;
	
	// safely destroy all command sequences
	mSubmittedSequences.clear();
	for (command_sequence_ref sequence : mTrackedSequences) destroySequence(sequence);
	
	// when the device dies, everything it owns has to go too.
	for (shader_module_ptr shaderModule : mAllocatedShaderModules)
	{
		shaderModule->destroy();
	}
	
	for (buffer_ptr buf : mAllocatedBuffers)
	{
		buf->destroy();
	}
	mAllocatedBuffers.clear();
	
	mDevice.destroyCommandPool(mCommandPool);

	// ensure allocator is freed before device is destroyed
	mAllocator = nullptr;

	if (mDeviceInitialized)
	{
		mDevice.destroyPipelineCache(mPipelineCache);
		mDevice.destroy();
	}
	mDestroyed = true;
}

vk::Queue& Device::getComputeQueue()
{
	std::lock_guard<std::recursive_mutex> guard( *mDeviceMutex );
	return mComputeQueue;
}

#if 0
vk::Queue& Device::getTransferQueue()
{
	std::lock_guard<std::recursive_mutex> guard( *mDeviceMutex );
	throw std::runtime_error("not implemented!");
	return mTransferQueue;
}
#endif

command_sequence_ptr Device::createSequence()
{
	std::lock_guard<std::recursive_mutex> guard( *mDeviceMutex );
	if (mDestroyed) throw std::runtime_error("cannot create sequence with destroyed device!");
	command_sequence_ptr sequence = std::make_shared<CommandSequence>(mSelf.lock(), true);
	
	// track weak refs to sequence so they can be safely destroyed later
	mTrackedSequences.push_back(sequence);
	return sequence;
}

void Device::destroySequence(command_sequence_ref sequence)
{
	std::lock_guard<std::recursive_mutex> guard( *mDeviceMutex );
	if (!sequence.expired()) sequence.lock()->destroy();
}

command_sequence_ptr Device::submitSequence(command_sequence_ptr commandSequence, uint32_t queueIndex, std::vector<command_sequence_ptr> listToWaitFor)
{
	// lock it
	std::lock_guard<std::recursive_mutex> guard( *mDeviceMutex );
	
	// force sync to prevent driver stall if needed
	mDispatchCount += 1;
	if (mDispatchCount > mDispatchLimit)
	{
		sync();
		mDispatchCount = 1;
	}
	
	// much more important to check for this first
	if (!(commandSequence->mSynced))
	{
		std::cerr << "WARNING! Cannot re-submit sequence that has been submitted without first calling tart::Device::sync()\n"
			"	This sequence submission will be canceled." << std::endl;
		return commandSequence;
	}

	for (command_sequence_ptr toWaitFor : listToWaitFor)
	{
		if (std::find(mSubmittedSequences.begin(), mSubmittedSequences.end(), toWaitFor) == mSubmittedSequences.end() )
			throw std::runtime_error("Attempted to wait for one or more command sequences that have not been submitted yet!");
	}

	// get the semaphores from the wait list
	std::vector<vk::Semaphore> waitSemaphores(listToWaitFor.size());
	std::vector<vk::PipelineStageFlags> waitSemaphoreFlags(listToWaitFor.size());
	if (listToWaitFor.size() > 0)
	{
		// now we must assemble the semaphore list!
		for (size_t i = 0; i < waitSemaphores.size(); i += 1)
		{
			// their signal is our wait
			waitSemaphores[i] = listToWaitFor[i]->mSignalSemaphore;
			waitSemaphoreFlags[i] = listToWaitFor[i]->mSemaphoreFlags;
		}
	}
	
	// i totally forgot it needs to call dispatch
	commandSequence->dispatch((uint32_t)waitSemaphores.size(), waitSemaphores, waitSemaphoreFlags);
	
	mSubmittedSequences.insert(commandSequence);
	
	// this method may be re-used later on for single-shader submissions, so this may be wise to keep track of...
	return commandSequence;
}

void Device::sync()
{
	std::vector<command_sequence_ptr> sequences;
	return sync(sequences);
}

void Device::sync(std::vector<command_sequence_ptr> sequences)
{
	std::lock_guard<std::recursive_mutex> guard( *mDeviceMutex );
	
	// get fences to wait on and wait for them.
	// if none are there, just skip the entire process
	std::vector<vk::Fence> fencesToWaitOn;
	if (sequences.size() > 0)
	{
		for (auto commandSequence : sequences)
		{
			// only get the desynced ones, we have no idea what users will try to throw at this lool
			if (!commandSequence->isSynced() )
				fencesToWaitOn.push_back(commandSequence->mFence);
			else
				// just in case
				mSubmittedSequences.erase(commandSequence);
		}
	}
	else
	{
		for (auto commandSequence : mSubmittedSequences) fencesToWaitOn.push_back(commandSequence->mFence);
	}
	if (fencesToWaitOn.size() == 0) return; // no reason to continue
	
	// get the result
	vk::Result result = mDevice.waitForFences( fencesToWaitOn, true, uint64_t(-1) );
	
	// do post-sync cleanup for each command sequence, then clear them
	if (sequences.size() == 0)
	{
		// when waiting for everything
		for (auto commandSequence : mSubmittedSequences) commandSequence->postSync();
		mSubmittedSequences.clear();
	}
	else
	{
		// when only waiting for a few
		for (auto commandSequence : sequences)
		{
			commandSequence->postSync();
			mSubmittedSequences.erase(commandSequence);
		}
	}
}

shader_module_ptr Device::compileGLSL(const std::string& src)
{
	std::lock_guard<std::recursive_mutex> guard( *mDeviceMutex );
	std::vector<uint32_t> spv = tart_compilers::compileGLSL(src);
	return loadShader(spv);
}

shader_module_ptr Device::compileCL(const std::string& src)
{
	std::lock_guard<std::recursive_mutex> guard( *mDeviceMutex );
	auto spv = tart_compilers::compileCL(src);
	return loadShader(spv);
}

shader_module_ptr Device::loadShader(std::vector<uint32_t>& shaderContents)
{
	std::lock_guard<std::recursive_mutex> guard( *mDeviceMutex );
	shader_module_ptr tartShaderModule = std::make_shared<ShaderModule>(mSelf.lock(), shaderContents);
	mAllocatedShaderModules.insert(tartShaderModule);
	return tartShaderModule;
}

shader_module_ptr Device::loadShaderFromPath(const std::string shaderPath)
{
	std::lock_guard<std::recursive_mutex> guard( *mDeviceMutex );
	std::vector<uint32_t> shaderContents;
	if (std::ifstream shaderFile{ shaderPath, std::ios::binary | std::ios::ate } )
	{
		const size_t fileSize = shaderFile.tellg();
		shaderFile.seekg(0);
		shaderContents.resize(fileSize / 4, '\0');
		shaderFile.read( (char*)shaderContents.data(), fileSize);
	}
	return loadShader(shaderContents);
}

Device::Device(vk::Instance& instance,
	vk::PhysicalDevice& physicalDevice,
	const uint32_t physicalDeviceIndex,
	std::vector<std::string> requiredExtensionNames):
	mVkInstance(instance),
	mDeviceIndex(physicalDeviceIndex)
{	
	mDeviceMutex = std::make_shared<std::recursive_mutex>();
	mPhysicalDevice = physicalDevice;
	getDeviceHardwareInfo(mPhysicalDevice, mPhysicalDeviceProperties, mMemoryProperties, requiredExtensionNames,
		mDeviceMetadata, mComputeQueueFamilyIndex,
		mDefaultDeviceMemoryFlags,
		mDefaultStagingMemoryFlags,
		mDefaultHostMemoryFlags);
	
	if (mPhysicalDevice == VK_NULL_HANDLE)
	{
		for (auto dev : mVkInstance.enumeratePhysicalDevices())
			std::cout << "device? " << (dev == VK_NULL_HANDLE) << std::endl;
		throw std::runtime_error("Failed to get physical device! Why? I really do not know.");
	}
	
	// extension properties!
	mDeviceExtensionProperties = mPhysicalDevice.enumerateDeviceExtensionProperties();

	// enable some default extensions that are useful
	for (std::string& ext : initDefaultExtensions() )
	{
		// add any of the defaults to requiredExtensionNames that aren't already there
		if (std::find(requiredExtensionNames.begin(), requiredExtensionNames.end(), ext) == requiredExtensionNames.end() )
			requiredExtensionNames.push_back(ext);
	}

	for (std::string& requiredExtensionName : requiredExtensionNames)
	{
		if (!supportsExtension(requiredExtensionName) )
		{
			std::string error_msg("extension unsupported by device ");
			error_msg += std::to_string(physicalDeviceIndex);
			error_msg += " ( ";
			error_msg += mPhysicalDeviceProperties.deviceName.data();
			error_msg += " ):";
			error_msg += requiredExtensionName;
			throw std::runtime_error(error_msg);
		}
		
		mUsedExtensionNames.push_back(requiredExtensionName);
	}
	std::cout << "max descriptor sets: " << mPhysicalDeviceProperties.limits.maxBoundDescriptorSets << std::endl;
	
	// 
	mPhysicalDeviceProperties = mPhysicalDevice.getProperties();
}

void Device::deallocateBuffer(buffer_ptr buf)
{
	std::lock_guard<std::recursive_mutex> guard( *mDeviceMutex );
	
	// for now, deallocation of buffers while they are in use is not supported.
	if (buf->anyUsers()) throw std::runtime_error("Cannot deallocate a buffer while it is in use!");
	
	// erase all entries from the descriptor set cache
	mDescriptorSetCache->destroyAnyDescriptorSetsWithBuffer(buf);
	// invoke the destroy method so that all the buffer resources themselves are freed
	buf->destroy();
	mAllocatedBuffers.erase(buf);
}

bool Device::isBufferInUse(Buffer& buf)
{
	std::lock_guard<std::recursive_mutex> guard( *mDeviceMutex );
	for (command_sequence_ptr sequence : mSubmittedSequences)
	{
		for (buffer_ptr bufInUse : sequence->mResourcesInUse)
		{
			if ( buf == *bufInUse) return true;
		}
	}
	return false;
}

std::vector<command_sequence_ptr> Device::getSequencesToWaitFor(const std::vector<buffer_ptr>& bufs)
{
	std::lock_guard<std::recursive_mutex> guard( *mDeviceMutex );
	
	std::vector<command_sequence_ptr> toWaitFor;
	
	for (command_sequence_ptr sequence : mSubmittedSequences)
	{
		// check to see if it has any buffers in common
		for (buffer_ptr bufOutOfSync : sequence->mResourcesInUse)
		{
			if ( std::find(bufs.begin(), bufs.end(), bufOutOfSync) != bufs.end() )
				toWaitFor.push_back(sequence);
		}
	}
	return toWaitFor;
}

void Device::getRequiredWaitSemaphores(const std::vector<buffer_ptr>& bufs,
	std::vector<vk::Semaphore>& semaphores, std::vector<vk::PipelineStageFlags>& semaphoreFlags)
{
	std::lock_guard<std::recursive_mutex> guard( *mDeviceMutex );
	for (command_sequence_ptr sequence : getSequencesToWaitFor(bufs) )
	{
		semaphores.push_back(sequence->mSignalSemaphore);
		semaphoreFlags.push_back(sequence->mSemaphoreFlags);
	}
}

} // namespace tart
