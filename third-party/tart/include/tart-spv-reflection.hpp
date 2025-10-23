#ifndef TART_SPV_REFLECTION
#define TART_SPV_REFLECTION

#include "spirv_reflect.h"
#include "spirv_reflect.cpp"
#include <string>
#include <map>
#include <vulkan/vulkan_enums.hpp>

void decodePushConstantBlock(SpvReflectBlockVariable& block,
	uint32_t& pushConstantBlockOffset, uint32_t& pushConstantBlockSize)
{
	pushConstantBlockOffset = block.offset;
	pushConstantBlockSize = block.size;
}

uint32_t getTrueSpecConstCount(const std::vector<SpvReflectSpecializationConstant*>& reflectSpecConsts)
{
	std::set<uint32_t> dedupedIDs; 
	for (SpvReflectSpecializationConstant* reflectSpecConst : reflectSpecConsts)
	{
		if (!reflectSpecConst) throw std::runtime_error("Encountered null reflection spec constant!");
		dedupedIDs.insert(reflectSpecConst->constant_id);
	}
	return (uint32_t)dedupedIDs.size();
}

void decodeSpecConstant(SpvReflectSpecializationConstant& spec, std::vector<vk::SpecializationMapEntry>& entries)
{
	// constant_id maps directly to the index.
	// constant_id values >= number of spec constants are currently unsupported :c
	if (spec.constant_id >= entries.size() )
		throw std::runtime_error("Specialization constant IDs >= the number of specialization constants present is not supported!");
	vk::SpecializationMapEntry entry(spec.constant_id, spec.constant_id*sizeof(uint32_t), sizeof(uint32_t) );
	entries[spec.constant_id] = entry;
}

// HERE THERE BE POINTERS
// adapted from the tutorial here: https://hushengine.com/blog/shader-reflection-system/
void inferShaderInfo(const std::vector<uint32_t>& shaderBinary,
	const std::string& entryPoint,
	std::vector<std::vector<vk::DescriptorSetLayoutBinding>>& descriptorSetsBindings,
	uint32_t& pushConstantBlockCount, uint32_t& pushConstantBlockOffset, uint32_t& pushConstantBlockSize,
	std::vector<vk::SpecializationMapEntry>& specConstEntries)
{	
#if 0 // Disabled for now. People will just have to look at https://docs.vulkan.org/spec/latest/appendices/extensions.html
	tart::SpvExtensionParser dummy(shaderBinary);
#endif
	
	// create the shader module
	size_t bytecodeLength = shaderBinary.size()*sizeof(uint32_t);
	SpvReflectShaderModule reflectionModule;
	SpvReflectResult rc = spvReflectCreateShaderModule(bytecodeLength, shaderBinary.data(), &reflectionModule);
	if (rc != SpvReflectResult::SPV_REFLECT_RESULT_SUCCESS)
	{
		std::cout << "spv error code: " << rc << std::endl;
		if (rc == SpvReflectResult::SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_CODE_SIZE)
			std::cout << "	(SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_CODE_SIZE)" << std::endl;
		throw std::runtime_error("SPIR-V reflection failed!");
	}	
	// get some info about the entry point. we will need this later...
	const SpvReflectEntryPoint* reflectEntryPoint = spvReflectGetEntryPoint(&reflectionModule, entryPoint.c_str() );
	vk::Flags<vk::ShaderStageFlagBits> shaderStageFlags
		= static_cast<vk::Flags<vk::ShaderStageFlagBits>>(reflectEntryPoint->shader_stage);
		
	// get info about the push constants
	SpvReflectResult result = spvReflectEnumerateEntryPointPushConstantBlocks(
		&reflectionModule,
		entryPoint.c_str(),
		&pushConstantBlockCount,
		nullptr);
	
	if (result != SpvReflectResult::SPV_REFLECT_RESULT_SUCCESS)
		throw std::runtime_error("SPIR-V reflection failed!");
	
	// get push constant blocks
	std::vector<SpvReflectBlockVariable*> pushConstantBlocks(pushConstantBlockCount);
	result = spvReflectEnumerateEntryPointPushConstantBlocks(
		&reflectionModule,
		entryPoint.c_str(),
		&pushConstantBlockCount,
		pushConstantBlocks.data());
	
	if (result != SpvReflectResult::SPV_REFLECT_RESULT_SUCCESS)
		throw std::runtime_error("SPIR-V reflection failed this time!");
	
	// iterate over each one
	for (size_t i = 0; i < pushConstantBlockCount; i += 1)
	{
		/*
		 * This assumes there is only one push constant block per shader, since
		 * shader execution with multiple push constant blocks is not being implemented yet.
		 * also, a reference of a reference sometimes causes a crash, hence the tmp variables.
		 */
		uint32_t tmpOffset;
		uint32_t tmpSize;
		SpvReflectBlockVariable& block = *(pushConstantBlocks[i]);
		decodePushConstantBlock(block, tmpOffset, tmpSize);
		pushConstantBlockOffset = tmpOffset;
		pushConstantBlockSize = tmpSize;
	}
	
	uint32_t specConstCount = 0;
	result = spvReflectEnumerateSpecializationConstants(&reflectionModule, &specConstCount, nullptr);
	if (result != SpvReflectResult::SPV_REFLECT_RESULT_SUCCESS)
		throw std::runtime_error("Specialization constant enumeration failed during count!");
	
	if (specConstCount > 0)
	{
		std::vector<SpvReflectSpecializationConstant*> reflectSpecConsts(specConstCount);
		result = spvReflectEnumerateSpecializationConstants(&reflectionModule, &specConstCount, reflectSpecConsts.data() );
		if (result != SpvReflectResult::SPV_REFLECT_RESULT_SUCCESS)
			throw std::runtime_error("Specialization constant enumeration failed during enumeration!");
		
		// if multiple specialization constants have the same ID, reflection will still yield multiple.
		// therefore they must be deduped
		specConstCount = getTrueSpecConstCount(reflectSpecConsts);
		specConstEntries.resize(specConstCount);
		for (SpvReflectSpecializationConstant* reflectSpecConst : reflectSpecConsts)
		{
			if (!reflectSpecConst) throw std::runtime_error("Encountered null reflection spec constant!");
			decodeSpecConstant(*reflectSpecConst, specConstEntries);
		}
	}
	
	
	// Now we can load the descriptors
	// first it the count needs to be determined
	uint32_t descriptorCount;
	spvReflectEnumerateEntryPointDescriptorBindings(&reflectionModule, entryPoint.c_str(), &descriptorCount, nullptr);
	std::vector<SpvReflectDescriptorBinding*> descriptorBindings(descriptorCount);
	spvReflectEnumerateEntryPointDescriptorBindings(&reflectionModule, entryPoint.c_str(), &descriptorCount, descriptorBindings.data() );
	
	//std::map<uint32_t, std::vector<vk::DescriptorSetLayoutBinding> > descriptorSetsBindings;
	
	// first we have to get number of descriptor sets
	std::set<uint32_t> setIndices;
	for(SpvReflectDescriptorBinding* descriptor : descriptorBindings)
		setIndices.insert(descriptor->set);
	
	// now we can initialize our thingy with the correct number
	descriptorSetsBindings.resize( setIndices.size() );
	
	for (size_t i = 0; i < descriptorCount; i += 1)
	{
		SpvReflectDescriptorBinding* descriptor = descriptorBindings[i];
		
		// separate them all into sets
		uint32_t setIndex = descriptor->set;
		
		// reference to the correct descriptor set binding list
		std::vector<vk::DescriptorSetLayoutBinding>& layoutBindings = descriptorSetsBindings[setIndex];
		
		// cast to descriptor type
		vk::DescriptorType descriptorType( static_cast<vk::DescriptorType>(descriptor->descriptor_type) );
		
		// and add the binding!
		layoutBindings.push_back( {descriptor->binding, descriptorType, descriptor->count,
			shaderStageFlags} ); // no info is given about the shader stage. we will assume it is compute, since that is what this entire thing is for
	}
	
	// the question is, now what? Seems as if we just made some of the earlier code obsolete!
	spvReflectDestroyShaderModule(&reflectionModule);
}

#endif
