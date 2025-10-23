#include <fstream>

#include "tart.hpp"
#include "tart-internal.hpp"
#include "tart-spv-reflection.hpp"
#include "tart-vk-helpers.hpp"

#include "tart-compilers.hpp"

namespace tart
{
	
void parseWorkgroup(std::vector<uint32_t>& wg)
{
	if (wg.size() > 3) throw std::runtime_error("workgroup size cannot have more than 3 dimensions!");
	wg.resize(3, 1);
}

Pipeline::Pipeline(shader_module_ptr shaderModule, std::string entryPoint, std::vector<uint8_t>& specConsts, std::vector<uint8_t>& defaultPushConstants):
	mShaderModule(shaderModule),
	mEntryPoint(entryPoint),
	mDevice(shaderModule->mDevice),
	mSpecConstants(specConsts),
	mDefaultPushConstantBlockData(defaultPushConstants)
{
	// spec constant entries
	std::vector<vk::SpecializationMapEntry> specConstEntries;
	
	// build the pipeline
	// get list of descriptor set layout bindings and push constants
	std::vector<std::vector<vk::DescriptorSetLayoutBinding>> setsBindings;
	inferShaderInfo(mShaderModule.lock()->mShaderContents,
		mEntryPoint, setsBindings, mNumPushConstantBlocks, mPushConstantBlockOffset, mPushConstantBlockSize,
		specConstEntries);
	
	if (mSpecConstants.size() % sizeof(uint32_t) > 0)
		throw std::runtime_error("specification constants must all be 32-bit data types!");
	
	if (mSpecConstants.size() / sizeof(uint32_t) != specConstEntries.size() )
		throw std::runtime_error("Invalid number of spec constants supplied to shader!");
	
	// glslc complains about too many push constant blocks when I try to compile a shader that uses more than one.
	// so we will have to deal with it for now...
	if (mNumPushConstantBlocks > 1)
		throw std::runtime_error("Shader execution with more than one push constant block have not been implemented!");
	
	// throw an error if the user doesn't supply default push constants but the shader expects them
	if (mPushConstantBlockSize != mDefaultPushConstantBlockData.size() )
	{
		std::stringstream ss;
		ss << "Size of push constant block provided (" << mDefaultPushConstantBlockData.size()
			<< " bytes) does not match the size inferred from reflection ("
			<< mPushConstantBlockSize << " bytes)" << std::endl;
		throw std::invalid_argument(ss.str());
	}
	
	// request the descriptor set layouts from the cache
	for (std::vector<vk::DescriptorSetLayoutBinding>& bindingList : setsBindings)
	{
		// now the cache gets put to the test.
		vk::DescriptorSetLayout layout = mDevice.lock()->mDescriptorSetLayoutCache->getLayout(bindingList);
		mDescriptorSetLayouts.push_back(layout);
		// also add the binding length to the descriptor set binding counts
		mDescriptorSetBindingCounts.push_back( bindingList.size() );
		mTotalBindingCount += bindingList.size();
	}
	
	// create the pipeline
	tart_helpers::createPipeline(mDevice.lock()->mDevice, mDescriptorSetLayouts,
		mPipelineLayout, mComputePipeline,
		mShaderModule.lock()->mShaderModule, mEntryPoint,
		mDevice.lock()->mPipelineCache, specConstEntries, specConsts ,mPushConstantBlockOffset, mPushConstantBlockSize);
		
	//if (specConsts.size() > 0) throw std::runtime_error("spec constants not implemented yet!");
}

void Pipeline::invoke(
	std::vector<uint32_t> workGroup,
	std::vector<buffer_ptr> buffers,
	std::vector<uint8_t>& pushConstants,
	vk::CommandBuffer& cmdBuffer,
	descriptor_set_cache_ptr descriptorSetCache,
	std::vector<buffer_ptr>& buffersNeedingBarrier)
{
	// validate workgroup and resize if needed
	parseWorkgroup(workGroup);
	
	// this function assumes that the push constants have already been updated.
	// update the buffers
	updateBuffers(buffers);
	
	// now we do it here c:
	std::vector<vk::DescriptorSet> descriptorSets;
	rebuildDescriptorSets(descriptorSets, descriptorSetCache);
	
	// check to ensure that the size of descriptor sets is not too big
	if (descriptorSets.size() >
		mDevice.lock()->mPhysicalDeviceProperties.limits.maxBoundDescriptorSets)
	{
		throw std::runtime_error("trying to bind too many descriptor sets to command buffer!");
	}
	
	// use memory barrier if applicable
	for (buffer_ptr tartBuffer : buffersNeedingBarrier)
	{
		// just use the default options for now
		tart_helpers::recordMemoryBarrier(cmdBuffer, tartBuffer->mBuffer);
	}
	
	// record the commands
	cmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, mComputePipeline);
	cmdBuffer.bindDescriptorSets(
		vk::PipelineBindPoint::eCompute,	// Bind point
		mPipelineLayout,				    // Pipeline Layout
		0,								    // First descriptor set
		descriptorSets,						// list of descriptor sets
		{});								// Dynamic offsets
	
	
	// bind push constants if applicable
	if (mNumPushConstantBlocks > 0)
	{
		// make the reference to use the default
		std::vector<uint8_t>& pushConstantsToUse = mDefaultPushConstantBlockData;
		if (mPushConstantBlockSize > 0)
			pushConstantsToUse = pushConstants;
		if (pushConstantsToUse.size() != mPushConstantBlockSize)
			// TODO: have better error handling
			throw std::runtime_error("Attempting to execute a shader that requires push constants without supplying push constants!");

		cmdBuffer.pushConstants(
			mPipelineLayout,
			vk::ShaderStageFlagBits::eCompute,
			mPushConstantBlockOffset,
			mPushConstantBlockSize,
			(void*)pushConstantsToUse.data()
		);
	}
	
	// dispatch!
	cmdBuffer.dispatch(workGroup[0], workGroup[1], workGroup[2]);
}

Pipeline::~Pipeline()
{
	mDevice.lock()->mDevice.destroyPipelineLayout(mPipelineLayout);
	mDevice.lock()->mDevice.destroyPipeline(mComputePipeline);
}

void Pipeline::rebuildDescriptorSets(std::vector<vk::DescriptorSet>& descriptorSets,
	descriptor_set_cache_ptr descriptorSetCache)
{
	descriptorSets.resize(mDescriptorSetLayouts.size());
	
	size_t offset = 0;
	for (size_t i = 0; i < descriptorSets.size(); i += 1)
	{
		// make a new list of buffers based on which descriptor set is being written to
		std::vector<buffer_ptr> tartBuffers(mDescriptorSetBindingCounts[i]);
		for (size_t bindingIndex = 0; bindingIndex < mDescriptorSetBindingCounts[i]; bindingIndex += 1)
		{
			tartBuffers[bindingIndex] = mCurrentBuffers[bindingIndex + offset].lock();
		}
		descriptorSets[i] = descriptorSetCache->getDescriptorSet(
			tartBuffers, mDescriptorSetLayouts[0]);
		offset += mDescriptorSetBindingCounts[i];
	}
}

/*
Just a quick note here.
So it seems like the command buffer and fence should go to something else.
But should they? I don't even know yet...
I would rather stick to the simplest scenario possible for now, where 
*/
const bool Pipeline::checkBuffersExpired()
{
	for (buffer_ref ref : mCurrentBuffers)
		if (ref.expired() || ref.lock()->mDestroyed ) return true;
	return false;
}

void Pipeline::execute(std::vector<uint32_t> workGroup, std::vector<buffer_ptr> buffers, std::vector<uint8_t> pushConstants)
{
	command_sequence_ptr sequence = dispatch(workGroup, buffers, pushConstants);
	mDevice.lock()->sync();
}

command_sequence_ptr Pipeline::dispatch(std::vector<uint32_t> workGroup, std::vector<buffer_ptr> buffers, std::vector<uint8_t>& pushConstants)
{
	return mDevice.lock()->dispatchPipeline(*this, workGroup, buffers, pushConstants);
}

command_sequence_ptr Pipeline::dispatch(std::vector<uint32_t> workGroup, std::vector<buffer_ptr> buffers)
{
	return mDevice.lock()->dispatchPipeline(*this, workGroup, buffers);
}

const bool Pipeline::validateBufferUpdates(std::vector<buffer_ptr>& tartBuffers)
{
	/*
	 * This function performs a basic check on the input buffers
	 */
	if (tartBuffers.size() != mTotalBindingCount ) return false; // mismatched sizes
	for (uint32_t i = 0; i < tartBuffers.size(); i += 1)
	{
		if(mDevice.lock()->mDevice != tartBuffers[i]->mDevice.lock()->mDevice)
			throw std::runtime_error("Cannot invoke Pipeline::updateBuffers with buffer_ptr instances which do not belong to the same device as the Pipeline.");
		else if(tartBuffers[i]->isDestroyed() )
			throw std::runtime_error("Attempted to bind a destroyed buffer to Pipeline instance!");
	}
	return true;
}

void Pipeline::updateBuffers(std::vector<buffer_ptr> buffers)
{	
	// we are updating the buffers now, right here c:
	mBuffersUpdatedAtLeastOnce = true;
	
	// first, validate the buffers
	if ( !validateBufferUpdates(buffers) )
	{
		std::stringstream ss;
		ss << "Expected " << mTotalBindingCount << " buffers to be passed, got " << buffers.size() << " instead." << std::endl;
		throw std::invalid_argument(ss.str());
	}
	
	// TODO: add input size checking
	mCurrentBuffers.resize( buffers.size() );
	for (size_t i = 0; i < buffers.size(); i += 1)
		mCurrentBuffers[i] = buffers[i];
	
	// ensure buffers are not dead
	if ( checkBuffersExpired() )
		throw std::runtime_error("Attempting to add dead buffers to shader executor!");
}

CommandSequence::CommandSequence(device_ptr internalDevice, bool oneTime):
	mDevice(internalDevice)
{
	if (oneTime) mCmdBufferBeginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	
	// create command pool
	mCommandPool = tart_helpers::initCommandPool(mDevice.lock()->mDevice, mDevice.lock()->mComputeQueueFamilyIndex);
	
	// create fence
	mFence = mDevice.lock()->mDevice.createFence(vk::FenceCreateInfo());
	
	// create semaphore
	vk::SemaphoreCreateFlags semaphoreCreateFlags;
	vk::SemaphoreCreateInfo semaphoreCreateInfo(semaphoreCreateFlags);
	mSignalSemaphore = mDevice.lock()->mDevice.createSemaphore(semaphoreCreateInfo);
	
	// turns out having multiple command buffers is inefficient, we will have only one.
	vk::CommandBufferAllocateInfo cmdBufferAllocateInfo(getCommandPool(), vk::CommandBufferLevel::ePrimary, 1);
	mCmdBuffers = mDevice.lock()->mDevice.allocateCommandBuffers(cmdBufferAllocateInfo);
}

void CommandSequence::recordPipeline(
	pipeline_ref pipeline,
	std::vector<uint32_t> workGroup,
	std::vector<buffer_ptr> buffers,
	std::vector<uint8_t> pushConstants)
{
	Pipeline& pipelineDeref = *(pipeline.lock());
	recordPipeline(pipelineDeref, workGroup, buffers, pushConstants);
}

void CommandSequence::recordCopyBuffer(buffer_ptr dst, buffer_ptr src)
{
	// ensure recording is started
	ensureRecording();
	
	// increment record count
	mRecordCount += 1;
	
	// create memory barriers if necessary
	std::vector<buffer_ptr> barrierBuffers = {src, dst};
	barrierBuffers = getBuffersNeedingBarrier(barrierBuffers);
	for (buffer_ptr tartBuffer : barrierBuffers)
		tart_helpers::recordMemoryBarrier(mCmdBuffers[0], tartBuffer->getBuffer() );
	
	// check size
	if (src->getSize() != dst->getSize() )
		throw std::runtime_error("buffer size mismatch!");
	
	// record the command
	// src offset, dst offset, size
	// TODO: allow user to specify offsets, maybe? i don't know, I am not your mom
	uint32_t srcOffset = src->getOffset();
	uint32_t destOffset = dst->getOffset();
	uint32_t copySize = src->getSize();
	vk::BufferCopy copyRegion(srcOffset, destOffset, copySize);
	mCmdBuffers[0].copyBuffer(src->getBuffer(), dst->getBuffer(), 1, &copyRegion);

	mQueuedResources.insert(src);
	mQueuedResources.insert(dst);

	// add transfer to semaphore flags
	mSemaphoreFlags |= vk::PipelineStageFlagBits::eTransfer;
}

std::vector<buffer_ptr> CommandSequence::getBuffersNeedingBarrier(std::vector<buffer_ptr>& queryBuffers)
{
	std::vector<buffer_ptr> buffersNeedingBarrier;
	for (buffer_ptr buf : queryBuffers)
	{
		if (std::find(mQueuedResources.begin(), mQueuedResources.end(), buf) != mQueuedResources.end() )
			buffersNeedingBarrier.push_back(buf);
	}
	return buffersNeedingBarrier;
}

void CommandSequence::recordPipeline(
	Pipeline& pipeline,
	std::vector<uint32_t> workGroup,
	std::vector<buffer_ptr> buffers,
	std::vector<uint8_t>& pushConstants)
{
	/*
	 * For now, memory barriers will be used by default after every invocation.
	 * Eventually, this may change if submitted buffers are accounted for.
	 */
	ensureRecording();
	mRecordCount += 1;
	std::vector<buffer_ptr> buffersNeedingBarrier = getBuffersNeedingBarrier(buffers);
	pipeline.invoke(workGroup, buffers, pushConstants, mCmdBuffers[0],
		mDevice.lock()->mDescriptorSetCache, buffersNeedingBarrier);
	for (buffer_ptr buf : buffers) mQueuedResources.insert(buf);
	
	// add compute stage to semaphore flags
	mSemaphoreFlags |= vk::PipelineStageFlagBits::eComputeShader;
}
void CommandSequence::postSync()
{
	// reset fence
	mDevice.lock()->mDevice.resetFences({mFence});
	
	// extremely important! set synced to be true
	mSynced = true;

	// clear buffers in use
	// just ocurred to me that the whole problem with managing automatic
	// buffer memory barrier creation basically goes away if stuff is done with separate command queue submissions.
	// but that is less efficient....what to do? ugh

	// Hypothetically we should just be able to clear the buffers from the resources set
	mResourcesInUse.clear();
}

void CommandSequence::destroy()
{
	if (mDestroyed) return;
	
	// we may have to move the command buffer to the device...goodness.
	if (!mSynced )
		std::cout << "WARNING: Command sequence was destroyed during execution, check to ensure it has not gone out of scope!" << std::endl;
	mDevice.lock()->mDevice.destroyFence(mFence);
	mDevice.lock()->mDevice.destroySemaphore(mSignalSemaphore);
	mDevice.lock()->mDevice.freeCommandBuffers(getCommandPool(), 1, mCmdBuffers.data() );
	mDevice.lock()->mDevice.destroyCommandPool(getCommandPool());
	mDestroyed = true;
}

void CommandSequence::clear()
{
	// reset command pool in case of prior execution
	if (!mSynced) std::runtime_error("Cannot reset sequence during execution!");
	mDevice.lock()->mDevice.resetCommandPool(getCommandPool(), vk::CommandPoolResetFlags());
}

void CommandSequence::ensureRecording()
{
	if (!mSynced) 
		throw std::runtime_error("Cannot begin recording if sequence is executing!");
	if (!mRecording)
	{
		// initialize
		mRecording = true;
		clear();
		
		// re-create semaphore if needed
		if (mSignalSemaphore != VK_NULL_HANDLE)
		{
			mDevice.lock()->mDevice.destroySemaphore(mSignalSemaphore);
			vk::SemaphoreCreateFlags semaphoreCreateFlags;
			vk::SemaphoreCreateInfo semaphoreCreateInfo(semaphoreCreateFlags);
			mSignalSemaphore = mDevice.lock()->mDevice.createSemaphore(semaphoreCreateInfo);
		}
		getCommandBuffer().begin(mCmdBufferBeginInfo);
	}
}

void CommandSequence::dispatch(uint32_t numWaitSemaphores, std::vector<vk::Semaphore>& waitSemaphores,
	std::vector<vk::PipelineStageFlags>& semaphorePipelineStageFlags)
{
	if (!mSynced)
		throw std::runtime_error("Cannot dispatch again until sync!");
	if (mRecording)
	{
		// just auto-end
		end();
	}
	// set not synced
	mSynced = false;
	
	// get any additional semaphores that require waiting without any duplicates
	std::vector<buffer_ptr> resources(mQueuedResources.begin(), mQueuedResources.end());
	mResourcesInUse = resources;
	mQueuedResources.clear();

	std::vector<vk::Semaphore> finalWaitSemaphores;
	std::vector<vk::PipelineStageFlags> finalWaitSemaphoreFlags;
	mDevice.lock()->getRequiredWaitSemaphores(resources, finalWaitSemaphores, finalWaitSemaphoreFlags);
	for (size_t i = 0; i < waitSemaphores.size(); i += 1)
	{
		vk::Semaphore& semaphore = waitSemaphores[i];
		if (std::find(finalWaitSemaphores.begin(), finalWaitSemaphores.end(), semaphore) == finalWaitSemaphores.end() )
		{
			finalWaitSemaphores.push_back(semaphore);
			finalWaitSemaphoreFlags.push_back(semaphorePipelineStageFlags[i]);
		}
	}

	// we can probably eliminate this from the arguments, since it is determined here...
	numWaitSemaphores = finalWaitSemaphores.size();
	
	// get the queue from the device
	const vk::Queue& queue = mDevice.lock()->mDevice.getQueue(mDevice.lock()->mComputeQueueFamilyIndex, 0);
	
	// construct submit info, with all the semaphores necessary
	vk::SubmitInfo submitInfo(numWaitSemaphores, finalWaitSemaphores.data(),
		finalWaitSemaphoreFlags.data(), mCmdBuffers.size(), mCmdBuffers.data(), 1, &mSignalSemaphore);

	// submit with newly created fence
	queue.submit({submitInfo}, mFence );
}

void CommandSequence::end()
{
	if (mRecordCount == 0)
		throw std::runtime_error("Cannot end command sequence with nothing recorded!");
	getCommandBuffer().end();
	mRecording = false;
}

void CommandSequence::dispatch()
{
	std::vector<vk::Semaphore> waitSemaphores;
	std::vector<vk::PipelineStageFlags> semaphorePipelineStageFlags;
	dispatch(0, waitSemaphores, semaphorePipelineStageFlags);
}

ShaderModule::ShaderModule(device_ptr internalDevice, const std::vector<uint32_t>& shaderContents):
	mDevice(internalDevice),
	mShaderContents(shaderContents)
{
	vk::ShaderModuleCreateInfo ShaderModuleCreateInfo(
		vk::ShaderModuleCreateFlags(),								// Flags (not currently implemented)
		shaderContents.size() * 4,									// Code size (in bytes, thus the * 4s)
		shaderContents.data()										// SPIR-V binary
	);
	mShaderModule = mDevice.lock()->mDevice.createShaderModule(ShaderModuleCreateInfo);
}

void ShaderModule::destroy()
{
	if (mDestroyed) return;
	mDevice.lock()->mDevice.destroyShaderModule(mShaderModule);
	mDestroyed = true;
}

CLProgram::CLProgram(device_ref device, shader_module_ptr shaderModule):
	mDevice(device),
	mShaderModule(shaderModule)
{
	
}

CLProgram::~CLProgram()
{
	// do we need to do anything on destruction? I don't think so, but will leave this here just in case we do
}

std::array<uint32_t, 3> CLProgram::getWorkgroupKey(std::vector<uint32_t>& wg)
{
	std::array<uint32_t, 3> key;
	parseWorkgroup(wg);
	for (size_t i = 0; i < 3; i += 1) key[i] = wg[i];
	return key;
}

pipeline_ptr CLProgram::getPipeline(std::string& entryPoint, std::vector<uint32_t> localSize, std::vector<uint8_t> pushConstants)
{
	std::array<uint32_t, 3> key = getWorkgroupKey(localSize);
	if (mPipelines.find({entryPoint, key}) == mPipelines.end() )
	{
		mPipelines[{entryPoint, key}] = mDevice.lock()->createPipeline(mShaderModule, entryPoint, packConstants(localSize), pushConstants);
	}
	return mPipelines[{entryPoint, key}];
}

command_sequence_ptr CLProgram::dispatch(std::string entryPoint, std::vector<uint32_t> globalSize,
	std::vector<uint32_t> localSize, std::vector<buffer_ptr> buffers, std::vector<uint8_t> pushConstants)
{
	// TODO: figure out how push constants are handled
	pipeline_ptr pipeline = getPipeline(entryPoint, localSize, pushConstants);
	return pipeline->dispatch(globalSize, buffers, pushConstants);
}

} // namespace tart
