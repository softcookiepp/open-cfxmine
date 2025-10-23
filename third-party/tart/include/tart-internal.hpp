#ifndef TART_INTERNAL
#define TART_INTERNAL

#include "tart-vulkan-include.hpp"

#include "tart.hpp"

namespace tart
{

class Device;

class DescriptorSetCache;
typedef std::shared_ptr<DescriptorSetCache> descriptor_set_cache_ptr;
typedef std::weak_ptr<DescriptorSetCache> descriptor_set_cache_ref;

class DescriptorSetLayoutCache;
typedef std::shared_ptr<DescriptorSetLayoutCache> descriptor_set_layout_cache_ptr;

class DescriptorSetContainer;

// oh god this is going to be hard
class DescriptorSetCache
{
private:
	// using a weak pointer because this object will be owned by the same Device instance
	device_ref mDevice;
	
	// so the defining initializers of the descriptor set are going to be
	// the layout and the resources bound to it.
	// which means the key should be a std::pair of buffers and possibly the serialized layout bindings.
	// either that, or the layout handle itself, since each set of layout bindings has a guaranteed
	// 1:1 mapping with the layouts themselves.
	// layout handles should be more efficient anyways...but are they hashable?
	// the shared pointer variant of the buffers (buffer_ptr) will be used because it is hashable.
	// it will be freed using the destroyAnyDescriptorSetsWithBuffer method
	std::map<std::pair<std::vector<buffer_ptr>, vk::DescriptorSetLayout>, std::unique_ptr<DescriptorSetContainer>> mCachedDescriptorSets;
	
	// using shared pointers to Buffer instances as the basis for descriptor set caching
	// should not be too big of an issue, as long as the descriptor sets are freed upon any 
	// Device::deallocateBuffer calls
	// This method is meant to do that.
	void destroyAnyDescriptorSetsWithBuffer(buffer_ptr tartBuffer);
	
	/*
	 * These methods will either generate new descriptor sets
	 * or retrieve existing ones.
	 */
	vk::DescriptorSet getDescriptorSet(std::vector<buffer_ptr>& tartBuffers, vk::DescriptorSetLayout& descriptorSetLayout);
	
	void getDescriptorSets(std::vector<buffer_ptr>& tartBufferSets)
	{
		throw std::runtime_error("DescriptorSetCache::getDescriptorSets has not yet been implemented!");
	}
	
	bool mDestroyed = false;
	void destroy()
	{
		if (mDestroyed) return;
		
	}
	
public:
	DescriptorSetCache(device_ref internalDevice);
	~DescriptorSetCache();
	
	friend class Device;
	
	friend class Pipeline;
};

/*
 * Container that manages the lifetime of descriptor sets and the pools that allocate them.
 */
class DescriptorSetContainer
{
private:
	device_ref mDevice;
	vk::DescriptorPool mDescriptorPool;
	
	std::vector<vk::DescriptorSet> mDescriptorSets;
	
	// better to make them weak. once more automated handling of these
	// containers is in place, they will be deleted every time a buffer associated with them
	// is deallocated.
	std::vector<buffer_ref> mBuffers;
	
	// does just what is said!
	void writeBuffersToDescriptorSet();
	
public:
	DescriptorSetContainer(device_ref internalDevice,
		std::vector<buffer_ptr> tartBuffers,
		const vk::DescriptorSetLayout& descriptorSetLayout);
		
	~DescriptorSetContainer();
	
	vk::DescriptorSet getDescriptorSet() { return mDescriptorSets[0]; }
	
	friend class Pipeline;
};

// a class that will allow saving layouts, bindings, and their association.
// It will also manage the lifetime of anything associated with it.
class DescriptorSetLayoutCache
{
private:
	// the internal device
	device_ref mDevice;
	
	// central location for all descriptor set layouts
	std::map<std::vector<std::array<size_t, 4>>, vk::DescriptorSetLayout> mLayouts;
	std::map<std::vector<std::array<size_t, 4>>, std::vector<vk::DescriptorSetLayoutBinding>> mLayoutBindings;
	
	// 
public:
	DescriptorSetLayoutCache(device_ref internalDevice);
	~DescriptorSetLayoutCache();
	
	// create a descriptor layout based on a set of bindings, or get one if it already exists
	vk::DescriptorSetLayout getLayout(std::vector<vk::DescriptorSetLayoutBinding>& queryBindings);
	friend class Device;
};

} //namespace tart

#endif
