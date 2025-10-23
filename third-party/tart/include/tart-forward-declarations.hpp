#ifndef TART_FORWARD_DECLARATIONS
#define TART_FORWARD_DECLARATIONS

#include <utility>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <mutex>
#include "tart-vulkan-include.hpp"

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

typedef std::shared_ptr<std::mutex> mutex_ptr;

const vk::BufferUsageFlags DEFAULT_BUFFER_FLAG_BITS = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;

} //namespace tart

#endif
