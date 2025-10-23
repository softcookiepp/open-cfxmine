
#include "tart.hpp"
#include "tart-internal.hpp"
#include "tart-vk-helpers.hpp"

namespace tart
{

DescriptorSetCache::DescriptorSetCache(device_ref internalDevice)
{
	mDevice = internalDevice;
}

DescriptorSetCache::~DescriptorSetCache()
{
	// TODO: deallocate all descriptor sets with a given thingy
	mCachedDescriptorSets.clear();
}

void DescriptorSetCache::destroyAnyDescriptorSetsWithBuffer(buffer_ptr queriedBuffer)
{
	for (auto iter = mCachedDescriptorSets.begin();
		iter != mCachedDescriptorSets.end();
		)
	{
		bool found = false;
		for (buffer_ptr buffer : iter->first.first)
		{
			if (buffer == queriedBuffer)
			{
				found = true;
				mCachedDescriptorSets.erase(iter);
				// hacky way to clear entries, but it just might work!
				iter = mCachedDescriptorSets.begin();
				break;
			}
		}
		if (! found ) iter++;
	}
}

vk::DescriptorSet DescriptorSetCache::getDescriptorSet(
	std::vector<buffer_ptr>& tartBuffers, vk::DescriptorSetLayout& descriptorSetLayout)
{
	std::pair<std::vector<buffer_ptr>, vk::DescriptorSetLayout>
		pair = std::make_pair(tartBuffers, descriptorSetLayout);
	if (mCachedDescriptorSets.find(pair) == mCachedDescriptorSets.end() )
	{
		// create a new descriptor set container for this pair
		mCachedDescriptorSets.emplace(
			pair,
			std::make_unique<DescriptorSetContainer>(mDevice, tartBuffers, descriptorSetLayout)
		);
	}
	return mCachedDescriptorSets[pair]->getDescriptorSet();
}


DescriptorSetContainer::DescriptorSetContainer(device_ref internalDevice,
	std::vector<buffer_ptr> tartBuffers, const vk::DescriptorSetLayout& descriptorSetLayout):
	mBuffers( tartBuffers.size() ) // preallocate
{
	// one layout per descriptor set.
	// which means that only one layout should be supplied to this constructor.
	// but downstream functions expect a vector, so we convert it to a vector here.
	std::vector<vk::DescriptorSetLayout> descriptorSetLayouts( {descriptorSetLayout} );
	
	// only works for one set at the moment
	// and from the way things are looking, it probably makes a lot more sense for it to stay that way.
	mDevice = internalDevice;
	
	for (size_t i = 0; i < tartBuffers.size(); i += 1)
		mBuffers[i] = tartBuffers[i];
	
	tart_helpers::allocateSingleDescriptorSet(mDevice.lock()->mDevice,
		mDescriptorPool, (uint32_t)tartBuffers.size(), 0,
		descriptorSetLayouts, mDescriptorSets);
	
	for (vk::DescriptorSet& set : mDescriptorSets)
	{
		if (set == VK_NULL_HANDLE)
			throw std::runtime_error("descriptor set is null for some reason!");
	}
	
	// and write!
	writeBuffersToDescriptorSet();
}

void DescriptorSetContainer::writeBuffersToDescriptorSet()
{
	// first we need a list of vk::DescriptorSetBufferInfos
	std::vector<vk::DescriptorBufferInfo> descriptorBufferInfos(mBuffers.size() );
	std::vector<vk::DescriptorImageInfo> descriptorImageInfos(mBuffers.size() );
	std::vector<vk::DescriptorType> descriptorTypes(mBuffers.size() );
	
	for (size_t i = 0; i < mBuffers.size(); i += 1)
	{
		descriptorTypes[i] = mBuffers[i].lock()->getDescriptorType();
		if (descriptorTypes[i] != vk::DescriptorType::eStorageBuffer)
			throw std::runtime_error("you can't use non-storage buffers for now, it is not implemented");
		vk::DescriptorBufferInfo info(mBuffers[i].lock()->getBuffer(),
			mBuffers[i].lock()->getOffset(), mBuffers[i].lock()->getSize());
		descriptorBufferInfos[i] = info;
	}
	
	tart_helpers::writeToDescriptorSet(mDevice.lock()->mDevice, mDescriptorSets.front(),
		descriptorImageInfos, descriptorBufferInfos, descriptorTypes);
}

DescriptorSetContainer::~DescriptorSetContainer()
{
	const vk::Device& device(mDevice.lock()->mDevice);
	// pretty sure we can just free the pool without freeing the descriptor sets?
	device.destroyDescriptorPool(mDescriptorPool);
}

DescriptorSetLayoutCache::DescriptorSetLayoutCache(device_ref internalDevice):
	mDevice(internalDevice)
{
	// is there anything we need to do upon init?
	// doesn't look like it.
}

DescriptorSetLayoutCache::~DescriptorSetLayoutCache()
{
	const vk::Device& device = mDevice.lock()->mDevice;
	for(auto& pair : mLayouts)
	{
		device.destroyDescriptorSetLayout(pair.second);
	}
}

vk::DescriptorSetLayout DescriptorSetLayoutCache::getLayout(
	std::vector<vk::DescriptorSetLayoutBinding>& queryBindings)
{
	std::vector<std::array<size_t, 4>> bindingHashes = tart_helpers::serializeDescriptorSetLayoutBindings(queryBindings);
	if ( mLayouts.find(bindingHashes) == mLayouts.end() )
	{
		const vk::Device& device = mDevice.lock()->mDevice;
		vk::DescriptorSetLayoutCreateInfo dsLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags(), queryBindings);
		mLayoutBindings[bindingHashes] = queryBindings;
		mLayouts[bindingHashes] = device.createDescriptorSetLayout(dsLayoutCreateInfo);
	}
	return mLayouts[bindingHashes];
}

} // namespace tart
