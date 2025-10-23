#ifndef TART_GPU
#define TART_GPU

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
#include <memory>
#include <string>
#include <mutex>
#include <sstream>

#include "tart-vulkan-include.hpp"
#include "tart-metadata.hpp"

#include "tart-vma-impl.hpp"


namespace tart
{

class Buffer;

typedef std::shared_ptr<Buffer> buffer_ptr;
typedef std::weak_ptr<Buffer> buffer_ref;

class ShaderModule;
typedef std::shared_ptr<ShaderModule> shader_module_ptr;
typedef std::weak_ptr<ShaderModule> shader_module_ref;

class Pipeline;
typedef std::shared_ptr<Pipeline> pipeline_ptr;
typedef std::weak_ptr<Pipeline> pipeline_ref;

class CommandSequence;
typedef std::shared_ptr<CommandSequence> command_sequence_ptr;
typedef std::weak_ptr<CommandSequence> command_sequence_ref;

class Device;
typedef std::shared_ptr<Device> device_ptr;
typedef std::weak_ptr<Device> device_ref;

class DescriptorSetCache;
typedef std::shared_ptr<DescriptorSetCache> descriptor_set_cache_ptr;
typedef std::weak_ptr<DescriptorSetCache> descriptor_set_cache_ref;

class DescriptorSetLayoutCache;
typedef std::shared_ptr<DescriptorSetLayoutCache> descriptor_set_layout_cache_ptr;

class CLProgram;
typedef std::shared_ptr<CLProgram> cl_program_ptr;
typedef std::shared_ptr<CLProgram> cl_program_ref;

typedef std::shared_ptr<std::recursive_mutex> mutex_ptr;

typedef vk::Flags<vk::BufferUsageFlagBits> buffer_flags_t;

const vk::BufferUsageFlags DEFAULT_BUFFER_FLAG_BITS = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;

// number of dispatched command sequences to tolerate before forcing sync
constexpr size_t DEFAULT_DISPATCH_LIMIT = 64;

// method for converting arbitrary structs/objects to bytes, will be used for push constant serialization
template <typename T>
std::vector<uint8_t> packConstants(T& inp)
{
	std::vector<uint8_t> out(sizeof(inp) );
	T* inpPtr = &inp;
	std::memcpy(out.data(), &inp, sizeof(inp));
	return out;
}

// same but with vectors of primitive types
template <typename T>
std::vector<uint8_t> packConstants(std::vector<T>& inp)
{
	std::vector<uint8_t> out(sizeof(T)*inp.size() );
	std::memcpy(out.data(), inp.data(), sizeof(T)*inp.size());
	return out;
}

/*
 * This is now the device class.
 * Previously there were 2 (one for user interface, this one for
 * acutal internal function,) but it became to bloated.
 */
class Device
{
private:

	// internal vulkan stuff
	vk::Instance mVkInstance;
	std::vector<vk::ExtensionProperties> mDeviceExtensionProperties;
	
	// device index used for error messages, etc. in case devices are limited by the TART_VISIBLE_DEVICES variable
	uint32_t mDeviceIndex;
	
	// list of allocated shader modules
	std::set<shader_module_ptr> mAllocatedShaderModules;
	
	// enables default extensions and also extracts important information
	// regarding certain properties, listed directly below it
	std::vector<std::string> initDefaultExtensions();
	bool addExtensionIfSupported(std::string shader, std::vector<std::string>& shaderList);
	
	// supported compilers.
	// may move this to the instance. or not? I am not sure yet, it depends on where the embedded compilers will reside.
	struct CompilerSupport mCompilerSupport;
	
	// data type support
	struct DeviceMetadata mDeviceMetadata;

	bool mDeviceInitialized = false;
	vk::PhysicalDevice mPhysicalDevice;
	vk::Device mDevice;
	vk::PhysicalDeviceProperties mPhysicalDeviceProperties;
	vk::PhysicalDeviceMemoryProperties mMemoryProperties;
	
	// interface to VMA
	std::unique_ptr<Allocator> mAllocator = nullptr;
	
	// internal limit and counter for number of queue submissions.
	size_t mDispatchLimit = DEFAULT_DISPATCH_LIMIT;
	size_t mDispatchCount = 0;
	
	// the one and only compute queue; we will implement multiple queues at some point later, but that will take time that I do not have at the moment.
	vk::Queue mComputeQueue;
	uint32_t mComputeQueueFamilyIndex;
	
	// get the compute queue
	vk::Queue& getComputeQueue();
#if 0 // not yet implemented; single queue only for now
	vk::Queue& getTransferQueue();
#endif
	vk::MemoryPropertyFlags mDefaultDeviceMemoryFlags;
	vk::MemoryPropertyFlags mDefaultStagingMemoryFlags;
	vk::MemoryPropertyFlags mDefaultHostMemoryFlags;
	
	// the command pool
	vk::CommandPool mCommandPool;
	
	// weak reference to self. is this a good idea?
	device_ref mSelf;
	
	vk::PipelineCache mPipelineCache;
	
	// mutex to keep everything related to this device thread-safe
	mutex_ptr mDeviceMutex;
	
	// cache for any descriptor sets created with this device
	descriptor_set_cache_ptr mDescriptorSetCache;
	
	// cache
	descriptor_set_layout_cache_ptr mDescriptorSetLayoutCache;
	
	// list of initialized extension names
	std::vector<std::string> mUsedExtensionNames;
	
	// list of allocated buffers.
	std::set<buffer_ptr> mAllocatedBuffers;

	// set of command sequences that have been submitted.
	// Upon every sync, these should be cleared.
	std::set<command_sequence_ptr> mSubmittedSequences;
	
	/*
	 * we use a collection of weak pointers to command sequences to track
	 * all command sequences created by this device.
	 * If any are still alive upon object destruction, their resources are manually freed.
	 */
	std::vector<command_sequence_ref> mTrackedSequences;
	
	// destroyer of objects!
	bool mDestroyed = false;
	void destroy();
	
	void initDevice(device_ref self);
	
	bool isBufferInUse(Buffer& buf);
	
	// command sequences will eventually use this in order to automatically retrieve wait semaphores
	std::vector<command_sequence_ptr> getSequencesToWaitFor(const std::vector<buffer_ptr>& bufs);
	
	// and this will be used to actually get the wait semaphores directly
#if 0
	std::vector<vk::Semaphore> getRequiredWaitSemaphores(const std::vector<buffer_ptr>& bufs);
#else
	void getRequiredWaitSemaphores(const std::vector<buffer_ptr>& bufs, std::vector<vk::Semaphore>& semaphores, std::vector<vk::PipelineStageFlags>& semaphoreFlags);
#endif
	
public:
	Device(vk::Instance& instance, vk::PhysicalDevice& physicalDevice,
		const uint32_t physicalDeviceIndex,
		std::vector<std::string> requiredExtensionNames);
	~Device()
	{
		std::lock_guard<std::recursive_mutex> guard( *mDeviceMutex );
		destroy();
	}
	
	// whether or not the device is dead
	const bool isDestroyed() { return mDestroyed; }
	DeviceMetadata getMetadata() { return mDeviceMetadata; }
	
	// Allocate a buffer on the device's memory, or at least in memory associated with the device
#ifdef TART_USE_VMA
	buffer_ptr allocateBuffer(uint64_t bufferSize, VmaMemoryUsage usage);
#else
	buffer_ptr allocateBuffer(uint64_t bufferSize, vk::MemoryPropertyFlags memoryFlags);
#endif
	buffer_ptr allocateBuffer(uint64_t bufferSize, bool host = false);
	
	// same, but also supply data upon allocation to be copied to it
	template <typename T>
	buffer_ptr allocateBuffer(std::vector<T>& data, bool host = false)
	{
		std::lock_guard<std::recursive_mutex> guard( *mDeviceMutex );
		uint64_t bufferSize = sizeof(T)*data.size();
		buffer_ptr out = allocateBuffer(bufferSize, host);
		// now copy!
		out->copyIn(data);
		return out;
	}
	template <typename T>
	buffer_ptr allocateBuffer(const std::vector<T>& data, bool host = false)
	{
		std::lock_guard<std::recursive_mutex> guard( *mDeviceMutex );
		uint64_t bufferSize = sizeof(T)*data.size();
		buffer_ptr out = allocateBuffer(bufferSize, host);
		// now copy!
		out->copyIn(data);
		return out;
	}
	
	// load a shader module from SPIR-V code, or from a path to a file containing SPIR-V code
	shader_module_ptr loadShader(std::vector<uint32_t>& ShaderContents);
	shader_module_ptr loadShaderFromPath(const std::string shaderPath);
	
	// Compile source in your language of choice, and return a shader module
	shader_module_ptr compileGLSL(const std::string& src); // options will be added later, for now I just need it to work at all
	shader_module_ptr compileCL(const std::string& src);
	
	// create a command sequence to record operations to
	command_sequence_ptr createSequence();
	
	// deallocate a sequence, so long as it is not synced or anything
	void destroySequence(command_sequence_ref sequence);
	
	// dispatch a sequence
	command_sequence_ptr submitSequence(command_sequence_ptr commandSequence, uint32_t queueIndex, std::vector<command_sequence_ptr> listToWaitFor);
	command_sequence_ptr submitSequence(command_sequence_ptr commandSequence, uint32_t queueIndex = 0, command_sequence_ptr waitFor = nullptr)
	{
		std::lock_guard<std::recursive_mutex> guard( *mDeviceMutex );
		std::vector<command_sequence_ptr> listToWaitFor;
		if (waitFor) listToWaitFor.push_back(waitFor);
		return submitSequence(commandSequence, queueIndex, listToWaitFor);
	}
	
	// dispatch an invocation of a pipeline without needing to manually create a sequence and record it
	command_sequence_ptr dispatchPipeline(Pipeline& pipeline, std::vector<uint32_t> workGroup, std::vector<buffer_ptr> buffers, std::vector<uint8_t> pushConstants = {});
	command_sequence_ptr dispatchPipeline(pipeline_ptr pipeline, std::vector<uint32_t> workGroup, std::vector<buffer_ptr> buffers, std::vector<uint8_t> pushConstants = {})
	{
		std::lock_guard<std::recursive_mutex> guard( *mDeviceMutex );
		return dispatchPipeline(*pipeline, workGroup, buffers, pushConstants);
	}
	
	// whether or not the device supports an extension
	bool supportsExtension(std::string ext);

	// Wait for all dispatched command sequences to complete (no arguments,) or a list of specified dispatched sequences.
	// Has no effect if no sequences are dispatched or the specified sequences are already synced
	void sync();
	void sync(std::vector<command_sequence_ptr> sequences);
	
	// create OpenCL-like program, based on clspv-compiled shader module.
	cl_program_ptr createCLProgram(shader_module_ptr shaderModule);
	
	// create a pipeline from a specified shader module, entry point, and spec constants
	pipeline_ptr createPipeline(shader_module_ptr shaderModule, std::string entryPoint, std::vector<uint8_t> specConstants = {}, std::vector<uint8_t> defaultPushConstants = {} );

	// free a buffer from memory (should throw an error if the buffer is in use, but I don't remember :c )
	void deallocateBuffer(buffer_ptr buf);
	
	// lotsa fwends
	friend class Buffer;
	friend class CommandSequence;
	friend class ShaderModule;
	friend class Pipeline;
	friend class DescriptorSetCache;
	friend class DescriptorSetContainer;
	friend class DescriptorSetLayoutCache;
	friend class Instance;
};

class Pipeline
{
private:
	device_ref mDevice;
	
	// shader stuff
	std::string mEntryPoint;
	shader_module_ref mShaderModule;
	
	// descriptor set layouts
	std::vector<vk::DescriptorSetLayout> mDescriptorSetLayouts;
	std::vector<size_t> mDescriptorSetBindingCounts;
	size_t mTotalBindingCount = 0;
	
	// pipeline
	vk::PipelineLayout mPipelineLayout;
	vk::Pipeline mComputePipeline;
	
	// tart
	std::vector<buffer_ref> mCurrentBuffers;
	const bool checkBuffersExpired();
	
	// function to validate the buffer updates
	// returns true if they pass, false if no
	const bool validateBufferUpdates(std::vector<buffer_ptr>& tartBuffers);
	
	// function to execute the shader with an external command buffer.
	// will be used by CommandSequence
	void invoke(std::vector<uint32_t> workGroup, std::vector<buffer_ptr> buffers,
		std::vector<uint8_t>& pushConstants, vk::CommandBuffer& cmdBuffer,
		descriptor_set_cache_ptr descriptorSetCache,
		std::vector<buffer_ptr>& buffersNeedingBarrier);
	
	// same but for the descriptor sets

	void rebuildDescriptorSets(std::vector<vk::DescriptorSet>& descriptorSets, descriptor_set_cache_ptr descriptorSetCache);

	// have buffers been assigned at all
	bool mBuffersUpdatedAtLeastOnce = false;
	
	// push constant parameters.
	// if numPushConstantBlocks is 0, then push constants will be disabled.
	uint32_t mNumPushConstantBlocks = 0;
	uint32_t mPushConstantBlockOffset = 0;
	uint32_t mPushConstantBlockSize = 0;
	
	// fallback in the event that none are supplied
	std::vector<uint8_t> mDefaultPushConstantBlockData;
	
	// spec constants
	std::vector<uint8_t> mSpecConstants;

public:
	Pipeline(shader_module_ptr shaderModule, std::string entryPoint, std::vector<uint8_t>& specConsts, std::vector<uint8_t>& defaultPushConstants);
	~Pipeline();
	
	void updateBuffers(std::vector<buffer_ptr> buffers);

	void execute(std::vector<uint32_t> workGroup, std::vector<buffer_ptr> buffers, std::vector<uint8_t> pushConstants);
	void execute(std::vector<uint32_t> workGroup, std::vector<buffer_ptr> buffers)
	{
		std::vector<uint8_t> pushConstants;
		return execute(workGroup, buffers, pushConstants);
	}

	command_sequence_ptr dispatch(std::vector<uint32_t> workGroup, std::vector<buffer_ptr> buffers, std::vector<uint8_t>& pushConstants);
	command_sequence_ptr dispatch(std::vector<uint32_t> workGroup, std::vector<buffer_ptr> buffers);
	
	const size_t getNumBufferArguments() { return mTotalBindingCount; }
	const std::vector<uint8_t>& getSpecializationConstants() { return mSpecConstants; }
	const std::vector<uint8_t>& getDefaultPushConstants() { return mDefaultPushConstantBlockData; }
	
	friend class CommandSequence;
};

/*
 * class for handling clspv-compiled SPIR-V modules, which typically
 * require pipeline reconstruction when a different local size is specified.
 */
class CLProgram
{
private:
	device_ref mDevice;
	shader_module_ptr mShaderModule = nullptr;
	
	// all the pipelines constructed for given local workgroup sizes
	std::map<std::pair<std::string, std::array<uint32_t, 3>>, pipeline_ptr> mPipelines;
	
	// serialize key
	std::array<uint32_t, 3> getWorkgroupKey(std::vector<uint32_t>& wg);
	
public:
	CLProgram(device_ref device, shader_module_ptr shaderModule);
	~CLProgram();
	
	pipeline_ptr getPipeline(std::string& entryPoint, std::vector<uint32_t> localSize, std::vector<uint8_t> pushConstants = {});
	command_sequence_ptr dispatch(std::string entryPoint, std::vector<uint32_t> globalSize,
		std::vector<uint32_t> localSize, std::vector<buffer_ptr> buffers, std::vector<uint8_t> pushConstants = {});
};

class CommandSequence
{
private:
	// parent device
	device_ref mDevice;
	
	// resources being used
	std::set<buffer_ptr> mQueuedResources;
	std::vector<buffer_ptr> mResourcesInUse;
	
	// command pool; each sequence will have its own, since there is no telling how many
	// sequences that users will want to execute concurrently
	vk::CommandPool mCommandPool;
	
	// command buffers; only one, but initialized as a vector for a specific reason that I cannot remember
	std::vector<vk::CommandBuffer> mCmdBuffers;
	vk::CommandBufferBeginInfo mCmdBufferBeginInfo;
	
	// whether or not the sequence is recording commands
	bool mRecording = false;
	
	std::vector<buffer_ptr> getBuffersNeedingBarrier(std::vector<buffer_ptr>& queryBuffers);

	// we are adding the fence back. this is literally the only thing that makes sense now
	vk::Fence mFence;
	
	// signal semaphore
	vk::Semaphore mSignalSemaphore;
	
	// flags that other sequences may use to wait on this one
	vk::PipelineStageFlags mSemaphoreFlags;
	
	// sync status of the command sequence.
	bool mSynced = true;
	
	// guess we still gotta do this..
	bool mDestroyed = false;
	void destroy();
	
	// operations to be performed post-sync
	void postSync();
	
	// number of recorded commands.
	// if 0, the sequence cannot be dispatched.
	size_t mRecordCount = 0;
	
	// dumb state management stuff
	void ensureRecording();
	void end();
	void clear();
	
	// lets just make these private. they really should not be invoked by anything directly.
	void dispatch(uint32_t numWaitSemaphores, std::vector<vk::Semaphore>& waitSemaphores,
		std::vector<vk::PipelineStageFlags>& semaphorePipelineStageFlags);
	void dispatch();
	
	vk::CommandPool& getCommandPool()
	{
#if 0
		return mDevice.lock()->mCommandPool;
#else
		return mCommandPool;
#endif
	}
	
public:
	CommandSequence(device_ptr internalDevice, bool oneTime = true);
	
	// Does this need to be public? It may, just in case...
	vk::CommandBuffer& getCommandBuffer() { return mCmdBuffers[0]; }
	
	void recordCopyBuffer(buffer_ptr dst, buffer_ptr src);
	
	void recordPipeline(
		Pipeline& pipeline,
		std::vector<uint32_t> workGroup,
		std::vector<buffer_ptr> buffers,
		std::vector<uint8_t>& pushConstants);
	
	void recordPipeline(
		pipeline_ref pipeline,
		std::vector<uint32_t> workGroup,
		std::vector<buffer_ptr> buffers,
		std::vector<uint8_t> pushConstants = {});

	const bool isDestroyed() { return mDestroyed; }
	
	const bool isSynced() { return mSynced; }
	
	~CommandSequence()
	{
		destroy();
	}
	friend class Device;
	
};

class ShaderModule
{
private:
	// time to change this to actual device
	device_ref mDevice;
	vk::ShaderModule mShaderModule;
	std::vector<uint32_t> mShaderContents;
	
	// destroyer of objects!
	bool mDestroyed = false;
	void destroy();
public:
	ShaderModule(device_ptr internalDevice, const std::vector<uint32_t>& shaderContents);
	~ShaderModule()
	{
		destroy();
	}
	
	std::vector<uint32_t> getSpv() { return mShaderContents; }
	
	vk::ShaderModule& getShaderModule() { return mShaderModule; }
	vk::Device& getDevice() { return mDevice.lock()->mDevice; }
	friend class Pipeline;
	
	friend class Device;
};

class Buffer
{
private:

	vk::Buffer mBuffer;
#ifdef TART_USE_VMA
	// this will likely replace mBufferMemory
	VmaAllocation mVmaAllocation;
	VmaMemoryUsage mVmaMemoryUsage;
#else
	vk::DeviceMemory mBufferMemory;
	vk::MemoryPropertyFlags mMemoryPropertyFlags;
#endif
	device_ref mDevice;
	
	// pooperties
	const uint64_t mBufferSize;
	const uint64_t mBufferOffset;
	vk::BufferUsageFlags mBufferUsageFlags;
	
	// descriptor type
	vk::DescriptorType mDescriptorType = vk::DescriptorType::eStorageBuffer;
	
	// destroyer of objects!
	bool mDestroyed = false;
	void destroy();
	
	vk::DescriptorSetLayoutBinding generateDescriptorSetLayoutBinding(uint32_t index)
	{
		// the shader stage will always be compute.
		// and the descriptor type should always be storage buffer.
		vk::DescriptorSetLayoutBinding binding(index, mDescriptorType, 1, vk::ShaderStageFlagBits::eCompute);
		return binding;
	}

	bool anyUsers();

	// flag to keep track of whether or not this object is a view into another buffer
	bool mIsView = false;
	
	// list of views that have been spawned by this object; a view itself may spawn child views
	std::vector<buffer_ref> mChildViews;
	
public:
#ifdef TART_USE_VMA
	// regular constructor used for allocating new memory
	Buffer(device_ptr device, uint64_t size, vk::BufferUsageFlags usageFlagBits, VmaMemoryUsage usage);
	
	// constructor for view creation
	Buffer(device_ptr device, uint64_t offset, uint64_t originalSize, vk::BufferUsageFlags usageFlags, VmaMemoryUsage usage, vk::Buffer& buffer, VmaAllocation& allocation);
#else
	// regular constructor used for allocating new memory
	Buffer(device_ptr device, uint64_t size, vk::BufferUsageFlags usageFlagBits, vk::MemoryPropertyFlags& memoryFlags);
	
	// constructor for view creation
	Buffer(device_ptr device, uint64_t offset, uint64_t originalSize, vk::BufferUsageFlags usageFlags, vk::MemoryPropertyFlags& memoryFlags, vk::Buffer& buffer, vk::DeviceMemory& memory);
#endif
	// not a constructor as you can probably tell
	~Buffer();
	
	bool operator==(Buffer& other)
	{
		// almost certain this should also work for views
		return (this == &other);
	}
	
	// create a view of this buffer
	buffer_ptr view(uint64_t offset);
	
	// properties
	vk::Buffer& getBuffer() { return mBuffer; }

	const uint64_t getSize() { return mBufferSize; }
	const uint64_t getOffset() { return mBufferOffset; }
	const bool isDestroyed() { return mDestroyed; }
	
	// convenient binding thingies
	// TODO: make these private, maybe?
#ifndef TART_USE_VMA
	void* map(uint64_t offset, uint64_t size);
#endif
	void* map();
	void unmap();
	
	// copy buffer memory to host memory
	void copyOut(void* hostbuf, uint64_t size);
	void copyIn(void* hostbuf, uint64_t size);
	
	template<typename T>
	void copyIn(const std::vector<T>& inp)
	{
		if (inp.size()*sizeof(T) > mBufferSize - mBufferOffset)
		{
			std::cout << "	input size: " << inp.size()*sizeof(T)
				<< "\n	buffer size: " << mBufferSize
				<< "\n	buffer offset: " << mBufferOffset << std::endl;
			throw std::runtime_error("Attempting to copy host data with mismatched size!");
		}
		copyIn( (void*)( inp.data() ), inp.size()*sizeof(T) );
	}
	
	// copy output buffer contents to new std::vector
	template<typename T>
	std::vector<T> copyOut(uint64_t size = 0)
	{
		// just default to maximum size
		if (size == 0)
			size = mBufferSize - mBufferOffset;
		if (size > mBufferSize - mBufferOffset)
			throw std::invalid_argument("size cannot be greated than buffer size");
		else if (size % sizeof(T) > 0)
			throw std::invalid_argument("specified size must be divisible by data type provided!");
		std::vector<T> out( ( (size_t)size ) / sizeof(T) );
		copyOut( (void*)out.data(), size);
		return out;
	}
	
	void copyTo(buffer_ptr otherBuffer, uint64_t selfOffset = 0, uint64_t destOffset = 0, uint64_t size = 0);
	void copyFrom(buffer_ptr otherBuffer);
	void copyTo(Buffer& otherBuffer, uint64_t selfOffset = 0, uint64_t destOffset = 0, uint64_t size = 0);
	void copyFrom(Buffer& otherBuffer);
	
	const vk::DescriptorType getDescriptorType() { return mDescriptorType; }
	
	// address for BDA
	const uint64_t getAddress();
	
	// fwends!
	friend class Pipeline;
	friend class ShaderModule;
	friend class Device;
	friend class CommandSequence;
};

class Instance
{
private:
	vk::Instance mVkInstance;
	std::vector<vk::PhysicalDevice> mPhysicalDevices;
	std::vector<device_ptr> mDevices;
	bool mDestroyed = false;
public:
	Instance(std::vector<std::string> validationLayers = {});
	~Instance();
	
	vk::Instance& getInstance() { return mVkInstance; }
	
	uint32_t getNumDevices() { return (uint32_t)mPhysicalDevices.size(); }
	
	device_ptr createDevice(uint32_t index, std::vector<std::string> requiredExtensionNames = {});
	const bool isDestroyed() { return mDestroyed; }
	void syncGlobal();
};

} //namespace tart

#endif
