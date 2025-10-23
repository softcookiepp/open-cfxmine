#include "tart-vk-helpers.hpp"
#include <cstring>
#include "tart-vulkan-include.hpp"

namespace tart_helpers
{

#ifdef TART_USE_VMA

#else
// allocates a buffer
void allocateBuffer(vk::PhysicalDevice& physicalDevice, vk::Device& device, tart::buffer_flags_t usageFlagBits,
	vk::Buffer& buffer, vk::DeviceMemory& bufferMemory, uint64_t bufferSize,
	uint32_t queueFamilyIndex, const vk::MemoryPropertyFlags& memoryFlags)
{
	// Creating the buffers - vk::Buffer
	vk::BufferCreateInfo BufferCreateInfo{
		vk::BufferCreateFlags(),		// Flags
		bufferSize,						// Size
		usageFlagBits,					// Usage
		vk::SharingMode::eExclusive,	// Sharing mode
		1,								// Number of queue family indices
		&queueFamilyIndex				// List of queue family indices
	};
	// why is this required? why not just pass it directly?
	// oh, its because the thing was using VMA. which I now need to use...
	auto vkBufferCreateInfo = static_cast<VkBufferCreateInfo>(BufferCreateInfo);
	buffer = device.createBuffer(BufferCreateInfo);

	// Allocating memory
	// To allocate memory in Vulkan we first need to find the type of memory we actually require to back the buffers we have.
	// The vk::Device provides a member function called vk::Device::getBufferMemoryRequirements that returns a vk::MemoryRequirements object with information so we can ask Vulkan how much memory to allocate for each buffer:
	vk::MemoryRequirements bufferMemoryRequirements = device.getBufferMemoryRequirements(buffer);

	// With this information on hand we can query Vulkan for the memory
	// type required to allocate memory that is visible from the host,
	// i.e., memory that can be mapped on the host side:
	vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice.getMemoryProperties();
	
	// find the index for the memory type we want to allocate
	uint32_t memoryTypeIndex = uint32_t(~0);
	vk::DeviceSize memoryHeapSize = uint32_t(~0);
	bool found = false;
	for (uint32_t CurrentMemoryTypeIndex = 0; CurrentMemoryTypeIndex < memoryProperties.memoryTypeCount; ++CurrentMemoryTypeIndex)
	{
		vk::MemoryType MemoryType = memoryProperties.memoryTypes[CurrentMemoryTypeIndex];
		if(memoryFlags & MemoryType.propertyFlags)
		{
			memoryHeapSize = memoryProperties.memoryHeaps[MemoryType.heapIndex].size;
			memoryTypeIndex = CurrentMemoryTypeIndex;
			found = true;
			break;
		}
	}
	if (!found) throw std::runtime_error("failed to find device memory type index!");
	
	// And finally we can ask the device to allocate the required memory for our buffers:
	vk::MemoryAllocateInfo bufferMemoryAllocateInfo(bufferMemoryRequirements.size, memoryTypeIndex);
	bufferMemory = device.allocateMemory(bufferMemoryAllocateInfo);
	
	// bind the memory to the buffer
	device.bindBufferMemory(buffer, bufferMemory, 0);
}
#endif

void copyBuffer(vk::Device& device,
		vk::Buffer& srcBuf, vk::DeviceSize srcOffset,
		vk::Buffer& dstBuf, vk::DeviceSize dstOffset,
		vk::DeviceSize& bufferSize,
		uint32_t queueFamilyIndex)
{
	// oh god what the hell is this and why did I write it?
	// create command pool
	vk::CommandPoolCreateInfo CommandPoolCreateInfo(vk::CommandPoolCreateFlags(), queueFamilyIndex);
	vk::CommandPool commandPool = device.createCommandPool(CommandPoolCreateInfo);
	
	// allocate command buffer(s)
	vk::CommandBufferAllocateInfo cmdBufferAllocateInfo(commandPool, vk::CommandBufferLevel::ePrimary, 1);
	std::vector<vk::CommandBuffer> cmdBuffers = device.allocateCommandBuffers(cmdBufferAllocateInfo);

	vk::CommandBuffer& cmdBuffer = cmdBuffers.front();
	
	vk::CommandBufferBeginInfo CmdBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
	cmdBuffer.begin(CmdBufferBeginInfo);
	
	// record the command
	// src offset, dst offset, size
	vk::BufferCopy copyRegion(srcOffset, dstOffset, bufferSize);
	cmdBuffer.copyBuffer(srcBuf, dstBuf, 1, &copyRegion);
	
	// end
	cmdBuffer.end();
	
	// get the queue
	vk::Queue queue = device.getQueue(queueFamilyIndex, 0);
	
	// we don't need a fence this time
	vk::SubmitInfo SubmitInfo(0, nullptr, nullptr, cmdBuffers.size(), &cmdBuffer);
	queue.submit({ SubmitInfo }, nullptr);
	queue.waitIdle();
	
	// free everything after copying
	device.freeCommandBuffers(commandPool, 1, cmdBuffers.data() );
	device.destroyCommandPool(commandPool);
}

void recordMemoryBarrier(vk::CommandBuffer& cmdBuffer, const vk::Buffer& buffer,
	vk::AccessFlags srcAccessFlags,
	vk::AccessFlags dstAccessFlags,
	vk::PipelineStageFlags srcPipelineStageFlags,
	vk::PipelineStageFlags dstPipelineStageFlags)
{
	// create the barrier
	vk::BufferMemoryBarrier bufferMemoryBarrier;
	bufferMemoryBarrier.buffer = buffer;
	bufferMemoryBarrier.size = VK_WHOLE_SIZE;
	bufferMemoryBarrier.srcAccessMask = srcAccessFlags;
	bufferMemoryBarrier.dstAccessMask = dstAccessFlags;
	bufferMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; // who cares, we just want to avoid race conditions regardless of queue
	bufferMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; // same
	
	// add barrier to command buffer
	cmdBuffer.pipelineBarrier(
		srcPipelineStageFlags,
		dstPipelineStageFlags,
		vk::DependencyFlags(), // what is this?
		nullptr,
		bufferMemoryBarrier,
		nullptr);
}

void generateDescriptorSetLayoutBindings(std::vector<vk::DescriptorSetLayoutBinding>& bindings, uint32_t numBindings)
{
	/*
	 * Currently, everything is assumed to be a storage buffer.
	 * This may change at a later time, once automatic determination of descriptor types from provided SPIR-V
	 * is implemented. But for now, anyone using this is stuck using storage buffers.
	 */
	for (uint32_t i = 0; i < numBindings; i += 1)
		bindings.push_back({i, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
}

std::vector<vk::DescriptorSetLayoutBinding> generateDescriptorSetLayoutBindings(uint32_t numBindings)
{
	std::vector<vk::DescriptorSetLayoutBinding> bindings;
	generateDescriptorSetLayoutBindings(bindings, numBindings);
	return bindings;
}

void generateDescriptorSetLayoutBindings(
	std::vector<vk::DescriptorSetLayoutBinding>& bindings,
	std::vector<vk::DescriptorType>& descriptorTypes)
{
	/*
	 * will this work?
	 */
	for (uint32_t i = 0; i < descriptorTypes.size(); i += 1)
		bindings.push_back({i, descriptorTypes[i], 1, vk::ShaderStageFlagBits::eCompute});
}

std::vector<vk::DescriptorSetLayoutBinding> generateDescriptorSetLayoutBindings(std::vector<vk::DescriptorType>& descriptorTypes)
{
	std::vector<vk::DescriptorSetLayoutBinding> bindings;
	generateDescriptorSetLayoutBindings(bindings, descriptorTypes);
	return bindings;
}



// we need a separate function for vk::DescriptorSetLayout creation.
vk::DescriptorSetLayout createDescriptorSetLayout(const vk::Device& device,
	std::vector<vk::DescriptorType>& descriptorTypes)
{
	std::vector<vk::DescriptorSetLayoutBinding> bindings = generateDescriptorSetLayoutBindings(descriptorTypes);
	
	// so each descriptor set has a vk::DescriptorSetLayout
	vk::DescriptorSetLayoutCreateInfo dsLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags(), bindings);
	return device.createDescriptorSetLayout(dsLayoutCreateInfo);
}

/*
 * so...how do I make this work?
 */
void createPipeline(const vk::Device& device,
		const std::vector<vk::DescriptorSetLayout>& descriptorSetLayouts,
		vk::PipelineLayout& pipelineLayout, vk::Pipeline& pipeline,
		const vk::ShaderModule& shaderModule, const std::string& entryPoint,
		const vk::PipelineCache& pipelineCache,
		const std::vector<vk::SpecializationMapEntry>& specConstEntries,
		const std::vector<uint8_t>& specConsts,
		uint32_t pushConstantOffset, uint32_t pushConstantSize)
{
	// lets do push constants!
	// only one push constant block is supported for now
	vk::PushConstantRange pushConstantRange(vk::ShaderStageFlagBits::eCompute, pushConstantOffset, pushConstantSize);
	
	// Now for the pipeline layout.
	if (pushConstantSize > 0)
	{
		vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), descriptorSetLayouts,
			pushConstantRange);
		pipelineLayout = device.createPipelineLayout(pipelineLayoutCreateInfo);
	}
	else
	{
		vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), descriptorSetLayouts);
		pipelineLayout = device.createPipelineLayout(pipelineLayoutCreateInfo);
	}
	
	// only applicable if spec constants are provided
	vk::SpecializationInfo specInfo(
		(uint32_t)specConstEntries.size(),
		specConstEntries.data(),
		specConsts.size(),
		specConsts.data()
	);
	
	// set all the flags and stuff
	vk::PipelineShaderStageCreateInfo pipelineShaderCreateInfo(
		vk::PipelineShaderStageCreateFlags(), // yes, this is incomplete. Device profiling capability is nowhere near ready :c
		vk::ShaderStageFlagBits::eCompute,
		shaderModule,
		entryPoint.c_str()
	);
	
	if (specConsts.size() > 0)
	{
		// set spec constants info
		pipelineShaderCreateInfo.pSpecializationInfo = &specInfo;
	}
	
	vk::ComputePipelineCreateInfo computePipelineCreateInfo(
		vk::PipelineCreateFlags(), // may allow vk::PipelineCreateFlagBits::eAllowDerivatives to be enabled later, if frequent shader module re-use ends up being commonplace
		pipelineShaderCreateInfo,
		pipelineLayout
	);
	
	// now to finally create it!
	auto pipelineCreationResult = device.createComputePipeline(pipelineCache, computePipelineCreateInfo);
	if (pipelineCreationResult.result != vk::Result::eSuccess)
		// Maybe I will add more error handling. I am not sure yet.
		throw std::runtime_error("Failed to create compute pipeline!");
	// but other than that we are all done!
	pipeline = pipelineCreationResult.value;
}

void allocateSingleDescriptorSet(const vk::Device& device,
		vk::DescriptorPool& descriptorPool,
		uint32_t descriptorCount,
		uint32_t imageCount,
		const std::vector<vk::DescriptorSetLayout>& descriptorSetLayouts,
		std::vector<vk::DescriptorSet>& descriptorSets)
{
	// For now, the only type of descriptor being allocated is a storage buffer.
	// That will likely change at some point
	if (imageCount > 0) throw std::runtime_error("Allocation of image descriptors has not yet been implemented");
	std::vector<vk::DescriptorPoolSize> poolSizes(1);
	poolSizes[0] = {vk::DescriptorType::eStorageBuffer, descriptorCount};
	
	// number of sets to allocate
	uint32_t maxSets = 1;
	
	vk::DescriptorPoolCreateInfo poolCreateInfo(
		vk::DescriptorPoolCreateFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet),
		maxSets,
		poolSizes.size(),
		poolSizes.data()
	);
	descriptorPool = device.createDescriptorPool(poolCreateInfo);
	
	// Now to allocate the descriptor sets themselves...
	vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo(descriptorPool,
		descriptorSetLayouts.size(),
		descriptorSetLayouts.data()
	);
	descriptorSets = device.allocateDescriptorSets(descriptorSetAllocateInfo);
}


/* 
 * This function assumes that everything provided is new if not const.
 */
void allocateDescriptorSets(
		const vk::Device& device,
		vk::DescriptorPool& descriptorPool,
		std::vector<uint32_t> descriptorCounts,
		const std::vector<vk::DescriptorSetLayout>& descriptorSetLayouts,
		std::vector<vk::DescriptorSet>& descriptorSets
	)
{
	// For now, the only type of descriptor being allocated is a storage buffer.
	// That will likely change at some point
	uint32_t totalNumDescriptors = 0;
	for (const uint32_t& count : descriptorCounts)
	{
		totalNumDescriptors += count;
	}
	
	std::vector<vk::DescriptorPoolSize> poolSizes(1);
	poolSizes[0] = {vk::DescriptorType::eStorageBuffer, totalNumDescriptors};
	
	// number of sets to allocate
	uint32_t maxSets = descriptorCounts.size();
	
	vk::DescriptorPoolCreateInfo poolCreateInfo(
		vk::DescriptorPoolCreateFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet),
		maxSets,
		poolSizes.size(),
		poolSizes.data()
	);
	descriptorPool = device.createDescriptorPool(poolCreateInfo);
	
	// Now to allocate the descriptor sets themselves...
	vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo(descriptorPool,
		descriptorSetLayouts.size(),
		descriptorSetLayouts.data()
	);
	descriptorSets = device.allocateDescriptorSets(descriptorSetAllocateInfo);
}

void writeToDescriptorSet(const vk::Device& device, const vk::DescriptorSet& descriptorSet,
	const std::vector<vk::DescriptorImageInfo>& descriptorImageInfos,
	const std::vector<vk::DescriptorBufferInfo>& descriptorBufferInfos,
	const std::vector<vk::DescriptorType>& descriptorTypes)
{
	// check inputs just to be safesies
	if (descriptorBufferInfos.size() != descriptorTypes.size() )
		throw std::runtime_error("descriptorBufferInfos must be the same size as descriptorTypes!");
	if (descriptorImageInfos.size() != descriptorTypes.size() )
		throw std::runtime_error("descriptorImageInfos must be the same size as descriptorTypes!");
	
	/*
	 * also, there is no way of validating whether or not the provided descriptor set's layout
	 * matches any of this stuff. that may need to be handled elsewhere
	 */
	
	// info to be written to the descriptor set
	std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
	for (uint32_t i = 0; i < descriptorBufferInfos.size(); i += 1)
	{
		const vk::DescriptorType& descriptorType = descriptorTypes[i];
		
		if (descriptorType == vk::DescriptorType::eStorageBuffer)
		{
			writeDescriptorSets.push_back( {descriptorSet, i, 0, 1, descriptorTypes[i], nullptr, &descriptorBufferInfos[i]} );
		}
		else if (descriptorType == vk::DescriptorType::eStorageImage)
		{
			writeDescriptorSets.push_back( {descriptorSet, i, 0, 1, descriptorTypes[i], &descriptorImageInfos[i], nullptr} );
			throw std::runtime_error("descriptor set buffer updating is not yet implemented for types other than vk::DescriptorType::eStorageBuffer");
		}
	}
	// finally we write them
	device.updateDescriptorSets(writeDescriptorSets, {});
}

/* hash of 4 members. Since hash_combine can result in collision errors, it is better to
 * just use a fixed-size array of each hash.
 * actually it is not even a hash, but im too lazy to change the name.
 * its more like just serializing them I guess.
 */
std::array<size_t, 4> serializeDescriptorSetLayoutBinding(const vk::DescriptorSetLayoutBinding& binding)
{
	std::array<size_t, 4> bindingHash;
	// have not implemented these yet.
	if ( binding.descriptorType == vk::DescriptorType::eCombinedImageSampler
		|| binding.descriptorType == vk::DescriptorType::eCombinedImageSampler )
	{
		throw std::runtime_error("Descriptor set layout caching is not implemented for vk::DescriptorType::eCombinedImageSampler or binding.descriptorType == vk::DescriptorType::eCombinedImageSampler");
	}
	
	bindingHash[0] = static_cast<size_t>(binding.binding);
	bindingHash[1] = static_cast<size_t>(binding.descriptorType);
	bindingHash[2] = static_cast<size_t>(binding.descriptorCount);
	bindingHash[3] = (size_t)( static_cast<uint32_t>(binding.stageFlags) );
	return bindingHash;
}

std::vector<std::array<size_t, 4>> serializeDescriptorSetLayoutBindings(
	const std::vector<vk::DescriptorSetLayoutBinding>& bindings)
{
	std::vector<std::array<size_t, 4>> hashes( bindings.size() );
	for (size_t i = 0; i < bindings.size(); i += 1)
	{
		hashes[i] = serializeDescriptorSetLayoutBinding(bindings[i]);
	}
	return hashes;
}


vk::CommandPool initCommandPool(vk::Device& device, uint32_t queueFamilyIndex)
{
	// create command pool
	vk::CommandPoolCreateFlags flags = vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	vk::CommandPoolCreateInfo commandPoolCreateInfo(flags, queueFamilyIndex);
	return device.createCommandPool(commandPoolCreateInfo);
}
#if 0
void initCommandBuffers()
{
	
}
#endif

} // namespace tart_helpers
