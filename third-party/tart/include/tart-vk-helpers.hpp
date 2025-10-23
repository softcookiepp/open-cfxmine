#ifndef TART_VK_HELPERS
#define TART_VK_HELPERS
#include <cstring>
#include "tart.hpp"
#include "tart-vulkan-include.hpp"

namespace tart_helpers
{

// memory flags that should work under the vast majority of systems.
static const vk::MemoryPropertyFlags HOST_MEMORY_FLAGS = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
static const vk::MemoryPropertyFlags DEVICE_MEMORY_FLAGS = vk::MemoryPropertyFlagBits::eDeviceLocal;

// should cover both transfer and compute operations for read and write
static const vk::AccessFlags DEFAULT_MEMORY_BARRIER_ACCESS_FLAGS = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
static const vk::PipelineStageFlags DEFAULT_MEMORY_BARRIER_PIPELINE_STAGE_FLAGS = vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer;

#if 0
// allocates a buffer with VMA
void allocateBuffer(vk::PhysicalDevice& physicalDevice, vk::Device& device, tart::buffer_flags_t usageFlagBits,
	vk::Buffer& buffer, vk::DeviceMemory& bufferMemory, uint64_t bufferSize, uint32_t computeQueueFamilyIndex,
	const vk::MemoryPropertyFlags& memoryFlags);)
#else
// allocates a buffer
void allocateBuffer(vk::PhysicalDevice& physicalDevice, vk::Device& device, tart::buffer_flags_t usageFlagBits,
	vk::Buffer& buffer, vk::DeviceMemory& bufferMemory, uint64_t bufferSize, uint32_t computeQueueFamilyIndex,
	const vk::MemoryPropertyFlags& memoryFlags);
#endif
void copyBuffer(vk::Device& device,
		vk::Buffer& srcBuf, vk::DeviceSize srcOffset,
		vk::Buffer& dstBuf, vk::DeviceSize dstOffset,
		vk::DeviceSize& bufferSize,
		uint32_t queueFamilyIndex);
		
void recordMemoryBarrier(vk::CommandBuffer& cmdBuffer, const vk::Buffer& buffer,
	vk::AccessFlags srcAccessFlags = DEFAULT_MEMORY_BARRIER_ACCESS_FLAGS,
	vk::AccessFlags dstAccessFlags = DEFAULT_MEMORY_BARRIER_ACCESS_FLAGS,
	vk::PipelineStageFlags srcPipelineStageFlags = DEFAULT_MEMORY_BARRIER_PIPELINE_STAGE_FLAGS,
	vk::PipelineStageFlags dstPipelineStageFlags = DEFAULT_MEMORY_BARRIER_PIPELINE_STAGE_FLAGS);
	
void generateDescriptorSetLayoutBindings(std::vector<vk::DescriptorSetLayoutBinding>& bindings, uint32_t numBindings);

std::vector<vk::DescriptorSetLayoutBinding> generateDescriptorSetLayoutBindings(uint32_t numBindings);

void generateDescriptorSetLayoutBindings(
	std::vector<vk::DescriptorSetLayoutBinding>& bindings,
	std::vector<vk::DescriptorType>& descriptorTypes);
	
std::vector<vk::DescriptorSetLayoutBinding> generateDescriptorSetLayoutBindings(std::vector<vk::DescriptorType>& descriptorTypes);

// we need a separate function for vk::DescriptorSetLayout creation.
vk::DescriptorSetLayout createDescriptorSetLayout(const vk::Device& device,
	std::vector<vk::DescriptorType>& descriptorTypes);

void createPipeline(const vk::Device& device,
		const std::vector<vk::DescriptorSetLayout>& descriptorSetLayouts,
		vk::PipelineLayout& pipelineLayout, vk::Pipeline& pipeline,
		const vk::ShaderModule& shaderModule, const std::string& entryPoint,
		const vk::PipelineCache& pipelineCache,
		const std::vector<vk::SpecializationMapEntry>& specConstEntries,
		const std::vector<uint8_t>& specConsts,
		uint32_t pushConstantOffset = 0, uint32_t pushConstantSize = 0);
		
void allocateSingleDescriptorSet(const vk::Device& device,
		vk::DescriptorPool& descriptorPool,
		uint32_t descriptorCount,
		uint32_t imageCount,
		const std::vector<vk::DescriptorSetLayout>& descriptorSetLayouts,
		std::vector<vk::DescriptorSet>& descriptorSets);

/* 
 * This function assumes that everything provided is new if not const.
 */
void allocateDescriptorSets(
		const vk::Device& device,
		vk::DescriptorPool& descriptorPool,
		std::vector<uint32_t> descriptorCounts,
		const std::vector<vk::DescriptorSetLayout>& descriptorSetLayouts,
		std::vector<vk::DescriptorSet>& descriptorSets
	);
	
void writeToDescriptorSet(const vk::Device& device, const vk::DescriptorSet& descriptorSet,
	const std::vector<vk::DescriptorImageInfo>& descriptorImageInfos,
	const std::vector<vk::DescriptorBufferInfo>& descriptorBufferInfos,
	const std::vector<vk::DescriptorType>& descriptorTypes);
	
/* hash of 4 members. Since hash_combine can result in collision errors, it is better to
 * just use a fixed-size array of each hash.
 * actually it is not even a hash, but im too lazy to change the name.
 * its more like just serializing them I guess.
 */
std::array<size_t, 4> serializeDescriptorSetLayoutBinding(const vk::DescriptorSetLayoutBinding& binding);

std::vector<std::array<size_t, 4>> serializeDescriptorSetLayoutBindings(
	const std::vector<vk::DescriptorSetLayoutBinding>& bindings);

// method for helping with command buffers
vk::CommandPool initCommandPool(vk::Device& device, uint32_t queueFamilyIndex);
void initCommandBuffers();

} // namespace tart_helpers



#endif
