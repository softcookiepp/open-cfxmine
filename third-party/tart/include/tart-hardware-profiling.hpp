#ifndef TART_HARDWARE_PROFILING
#define TART_HARDWARE_PROFILING
#include <cstring>
#include "tart-vulkan-include.hpp"
#include "tart-metadata.hpp"
#include <stdexcept>

namespace tart
{

enum DeviceArchitecture {
    OTHER,
    AMD_GCN,
    AMD_RDNA1,
    AMD_RDNA2,
    AMD_RDNA3,
    INTEL_XE2,
};

// whatever
#define VENDOR_ID_AMD 0x1002
#define VENDOR_ID_APPLE 0x106b
#define VENDOR_ID_INTEL 0x8086
#define VENDOR_ID_NVIDIA 0x10de

static DeviceArchitecture getDeviceArchitecture(const vk::PhysicalDevice& device)
{
	vk::PhysicalDeviceProperties props = device.getProperties();

	if (props.vendorID == VENDOR_ID_AMD)
	{
		const std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

		bool amd_shader_core_properties = false;
		bool integer_dot_product = false;
		bool subgroup_size_control = false;

		for (const auto& properties : ext_props) {
			if (strcmp("VK_AMD_shader_core_properties", properties.extensionName) == 0) {
				amd_shader_core_properties = true;
			} else if (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0) {
				integer_dot_product = true;
			} else if (strcmp("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
				subgroup_size_control = true;
			}
		}

		if (!amd_shader_core_properties || !integer_dot_product || !subgroup_size_control) {
			return DeviceArchitecture::OTHER;
		}

		vk::PhysicalDeviceProperties2 props2;
		vk::PhysicalDeviceShaderCorePropertiesAMD shader_core_props_amd;
		vk::PhysicalDeviceShaderIntegerDotProductPropertiesKHR integer_dot_props;
		vk::PhysicalDeviceSubgroupSizeControlPropertiesEXT subgroup_size_control_props;

		props2.pNext = &shader_core_props_amd;
		shader_core_props_amd.pNext = &integer_dot_props;
		integer_dot_props.pNext = &subgroup_size_control_props;

		device.getProperties2(&props2);

		if (subgroup_size_control_props.maxSubgroupSize == 64 && subgroup_size_control_props.minSubgroupSize == 64) {
			return DeviceArchitecture::AMD_GCN;
		}
		if (subgroup_size_control_props.maxSubgroupSize == 64 && subgroup_size_control_props.minSubgroupSize == 32) {
			// RDNA
			if (shader_core_props_amd.wavefrontsPerSimd == 20) {
				return DeviceArchitecture::AMD_RDNA1;
			}
			if (integer_dot_props.integerDotProduct4x8BitPackedMixedSignednessAccelerated) {
				return DeviceArchitecture::AMD_RDNA3;
			}
			return DeviceArchitecture::AMD_RDNA2;
		}
	} else if (props.vendorID == VENDOR_ID_INTEL) {
		const std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

		bool subgroup_size_control = false;

		for (const auto& properties : ext_props) {
			if (strcmp("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
				subgroup_size_control = true;
			}
		}

		if (!subgroup_size_control) {
			return DeviceArchitecture::OTHER;
		}

		vk::PhysicalDeviceProperties2 props2;
		vk::PhysicalDeviceSubgroupSizeControlPropertiesEXT subgroup_size_control_props;

		props2.pNext = &subgroup_size_control_props;
		device.getProperties2(&props2);

		if (subgroup_size_control_props.minSubgroupSize == 16) {
			// Xe2 architecture uses SIMD16 while previous Xe and Gen architecture uses SIMD8.
			// Minimum subgroup size matches the SIMD width so we distinguish architecture by checking this value.
			// https://www.intel.com/content/www/us/en/content-details/824434/2024-intel-tech-tour-xe2-and-lunar-lake-s-gpu.html
			// https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/intel-xe-gpu-architecture.html
			return DeviceArchitecture::INTEL_XE2;
		}
	}
	return DeviceArchitecture::OTHER;
}

/*
 * This is not actually used yet...
 */
static uint32_t findQueueFamilyIndex(std::vector<vk::QueueFamilyProperties>& queueFamilyProps)
{
    // find queue family index; more lines, but readable
    uint32_t i = 0;
	for (i = 0; i < queueFamilyProps.size(); i += 1)
	{
		vk::QueueFamilyProperties& qfp = queueFamilyProps[i];
		if (qfp.queueFlags & (vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eTransfer) )
		{
			return i;
		}
	}
	if (i >= queueFamilyProps.size() ) throw std::runtime_error("Could not find an available queue!");
	return 0;
}

// Pipeline configuration for RDNA1 GPUs.
static const std::unordered_map<std::string, uint32_t> rdna1_pipelines = {
    {"soft_max", 64}, {"im2col", 64},
    {"argmax", 64}, {"mul_mat_vec", 64},
    {"mul_mat_vec_f16", 32}, {"mul_mat_vec_f32_f16", 32}
};

// Pipeline configuration for RDNA2 GPUs.
static const std::unordered_map<std::string, uint32_t> rdna2_pipelines = {
    {"soft_max", 64}, {"im2col", 64},
};

static constexpr uint32_t RDNA_DEFAULT_SUBGROUP_SIZE = 32;

/*
 * Default subgroup sizes for different architectures
 */
static std::unordered_map<DeviceArchitecture, uint32_t> GPU_SUBGROUP_SIZES = {
    { DeviceArchitecture::AMD_RDNA1, RDNA_DEFAULT_SUBGROUP_SIZE },
    { DeviceArchitecture::AMD_RDNA2, RDNA_DEFAULT_SUBGROUP_SIZE },
    { DeviceArchitecture::OTHER, 0 },
    { DeviceArchitecture::AMD_GCN, 0 },
    { DeviceArchitecture::AMD_RDNA1, 0 },
    { DeviceArchitecture::AMD_RDNA2, 0 },
    { DeviceArchitecture::AMD_RDNA3, 0 },
    { DeviceArchitecture::INTEL_XE2, 0 }
};

static bool getCooperativeMatrixSupport(const vk::PhysicalDeviceProperties& props, const vk::PhysicalDeviceDriverProperties& driver_props, DeviceArchitecture arch) {
    switch (props.vendorID) {
    case VENDOR_ID_INTEL:
        // Only allowing Xe2 GPU at the moment since Xe2 GPU can gain significant performance boost,
        // while some older hardware (ex. Arc A770) has performance regressions
        return arch == DeviceArchitecture::INTEL_XE2;
    case VENDOR_ID_AMD:
        if (driver_props.driverID == vk::DriverId::eAmdProprietary || driver_props.driverID == vk::DriverId::eAmdOpenSource) {
            // Workaround for AMD proprietary driver reporting support on all GPUs
            return arch == DeviceArchitecture::AMD_RDNA3;
        }
        return true;
    default:
        return true;
    }
}

bool addExtensionIfSupported(std::string ext, std::vector<std::string>& extList,
	const std::vector<vk::ExtensionProperties>& extensionProperties)
{
	for (const vk::ExtensionProperties& props : extensionProperties)
	{
		if ( strcmp(props.extensionName, ext.c_str() ) == 0 )
		{
			extList.push_back(ext);
			return true;
		}
	}
	return false;
}

void getDeviceHardwareInfo(
	// unmodified
	const vk::PhysicalDevice& physicalDevice, 
	
	// modified/initialized
	vk::PhysicalDeviceProperties& physicalDeviceProperties,
	vk::PhysicalDeviceMemoryProperties& memoryProperties,
	std::vector<std::string>& extensionList,
	DeviceMetadata& deviceMetadata,
	uint32_t& computeQueueFamilyIndex,
	vk::MemoryPropertyFlags& deviceMemoryFlags,
	vk::MemoryPropertyFlags& stagingMemoryFlags,
	vk::MemoryPropertyFlags& hostMemoryFlags)
{
	physicalDeviceProperties = physicalDevice.getProperties();
	memoryProperties = physicalDevice.getMemoryProperties();

	// Adapted from GGML code, which seems to be very good
	const std::vector<vk::ExtensionProperties> ext_props = physicalDevice.enumerateDeviceExtensionProperties();

	DeviceArchitecture architecture = getDeviceArchitecture(physicalDevice);

#if 0
	// these will probably be important later
	// or not? i don't know...
	const char* GGML_VK_PREFER_HOST_MEMORY = getenv("GGML_VK_PREFER_HOST_MEMORY");
	device->prefer_host_memory = GGML_VK_PREFER_HOST_MEMORY != nullptr;

	const char* GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM = getenv("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM");
	device->disable_host_visible_vidmem = GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM != nullptr;
#endif
	deviceMetadata.preferHostMemory = false;

	bool maintenance4_support = false;
	bool sm_builtins = false;
	bool amd_shader_core_properties2 = false;
	bool pipeline_robustness = false;
	bool coopmat2_support = false;

	bool variablePointers = false;
	if (addExtensionIfSupported("VK_KHR_maintenance4", extensionList, ext_props))
	{
		maintenance4_support = true;
	}
	if (addExtensionIfSupported("VK_KHR_16bit_storage", extensionList, ext_props)
		&& addExtensionIfSupported("VK_KHR_shader_float16_int8", extensionList, ext_props) )
	{
		deviceMetadata.half_ = true;
	}
	if (addExtensionIfSupported("VK_KHR_8bit_storage", extensionList, ext_props)
		&& addExtensionIfSupported("VK_KHR_shader_float16_int8", extensionList, ext_props) )
	{
		deviceMetadata.char_ = true;
		deviceMetadata.uchar_ = true;
	}
	if (addExtensionIfSupported("VK_NV_shader_sm_builtins", extensionList, ext_props))
	{
		sm_builtins = true;
	}
	if (addExtensionIfSupported("VK_AMD_shader_core_properties2", extensionList, ext_props))
	{
		amd_shader_core_properties2 = true;
	}
	if (addExtensionIfSupported("VK_EXT_pipeline_robustness", extensionList, ext_props))
	{
		pipeline_robustness = true;
	}
	if (addExtensionIfSupported("VK_EXT_subgroup_size_control", extensionList, ext_props))
	{
		deviceMetadata.subgroupSizeControl = true;
	}
	if (addExtensionIfSupported("VK_KHR_cooperative_matrix", extensionList, ext_props))
	{
		deviceMetadata.coopmat = true;
		deviceMetadata.coopmatM = 0;
		deviceMetadata.coopmatN = 0;
		deviceMetadata.coopmatK = 0;
	}
	if (addExtensionIfSupported("VK_NV_cooperative_matrix2", extensionList, ext_props))
	{
		coopmat2_support = true;
	}
	if (addExtensionIfSupported("VK_KHR_shader_integer_dot_product", extensionList, ext_props))
	{
		deviceMetadata.integerDotProduct = true;
	}
	if (addExtensionIfSupported("VK_KHR_shader_bfloat16", extensionList, ext_props))
	{
		deviceMetadata.bf16_ = true;
	}
	if (addExtensionIfSupported("VK_KHR_variable_pointers", extensionList, ext_props))
	{
		variablePointers = true;
	}
	if (addExtensionIfSupported("VK_EXT_memory_budget", extensionList, ext_props) &&
		addExtensionIfSupported("VK_KHR_get_physical_device_properties2", extensionList, ext_props) )
	{
		deviceMetadata.canTrackUsedMemory = true;
	}
	if( addExtensionIfSupported("VK_KHR_buffer_device_address", extensionList, ext_props) )
	{
		deviceMetadata.bda = true;
	}

	vk::PhysicalDeviceProperties2 props2;
	vk::PhysicalDeviceMaintenance3Properties props3;
	vk::PhysicalDeviceMaintenance4Properties props4;
	vk::PhysicalDeviceSubgroupProperties subgroup_props;
	vk::PhysicalDeviceDriverProperties driver_props;
	vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV sm_props;
	vk::PhysicalDeviceShaderCoreProperties2AMD amd_shader_core_properties2_props;
	vk::PhysicalDeviceVulkan11Properties vk11_props;
	vk::PhysicalDeviceVulkan12Properties vk12_props;
	vk::PhysicalDeviceSubgroupSizeControlPropertiesEXT subgroup_size_control_props;
	vk::PhysicalDeviceShaderIntegerDotProductPropertiesKHR shader_integer_dot_product_props;

	props2.pNext = &props3;
	props3.pNext = &subgroup_props;
	subgroup_props.pNext = &driver_props;
	driver_props.pNext = &vk11_props;
	vk11_props.pNext = &vk12_props;

	VkBaseOutStructure * last_struct = (VkBaseOutStructure *)&vk12_props;

	if (maintenance4_support) {
		last_struct->pNext = (VkBaseOutStructure *)&props4;
		last_struct = (VkBaseOutStructure *)&props4;
	}
	if (sm_builtins) {
		last_struct->pNext = (VkBaseOutStructure *)&sm_props;
		last_struct = (VkBaseOutStructure *)&sm_props;
	}
	if (amd_shader_core_properties2) {
		last_struct->pNext = (VkBaseOutStructure *)&amd_shader_core_properties2_props;
		last_struct = (VkBaseOutStructure *)&amd_shader_core_properties2_props;
	}
	if (deviceMetadata.subgroupSizeControl) {
		last_struct->pNext = (VkBaseOutStructure *)&subgroup_size_control_props;
		last_struct = (VkBaseOutStructure *)&subgroup_size_control_props;
	}

#if defined(VK_NV_cooperative_matrix2)
	vk::PhysicalDeviceCooperativeMatrix2PropertiesNV coopmat2_props;
	if (coopmat2_support) {
		last_struct->pNext = (VkBaseOutStructure *)&coopmat2_props;
		last_struct = (VkBaseOutStructure *)&coopmat2_props;
	}
#endif

	if (deviceMetadata.integerDotProduct) {
		last_struct->pNext = (VkBaseOutStructure *)&shader_integer_dot_product_props;
		last_struct = (VkBaseOutStructure *)&shader_integer_dot_product_props;
	}

	physicalDevice.getProperties2(&props2);
	physicalDeviceProperties = props2.properties;
	deviceMetadata.vendorID = physicalDeviceProperties.vendorID;
	deviceMetadata.driverID = driver_props.driverID;

	if (maintenance4_support) {
		deviceMetadata.maxMemoryAllocationSize = std::min(props3.maxMemoryAllocationSize, props4.maxBufferSize);
	} else {
		deviceMetadata.maxMemoryAllocationSize = props3.maxMemoryAllocationSize;
	}

	deviceMetadata.subgroupSize = subgroup_props.subgroupSize;
	deviceMetadata.unifiedMemory = physicalDeviceProperties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;
	
	// set correct memory flags for each device type c:
	hostMemoryFlags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
	stagingMemoryFlags = hostMemoryFlags;
	if (deviceMetadata.unifiedMemory)
	{
		deviceMemoryFlags = hostMemoryFlags;
	}
	else
	{
		deviceMemoryFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
	}

	if (sm_builtins) {
		deviceMetadata.numShaderCores = sm_props.shaderSMCount;
	} else if (amd_shader_core_properties2) {
		deviceMetadata.numShaderCores = amd_shader_core_properties2_props.activeComputeUnitCount;

	} else {
		deviceMetadata.numShaderCores = 0;
	}
#if 0 // I have literally no idea what this is, so I am just going to ignore it for now :c
	device->float_controls_rte_fp16 = vk12_props.shaderRoundingModeRTEFloat16;

	device->subgroup_add = (vk11_props.subgroupSupportedStages & vk::ShaderStageFlagBits::eCompute) &&
						   (vk11_props.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eArithmetic);

	device->subgroup_shuffle = (vk11_props.subgroupSupportedStages & vk::ShaderStageFlagBits::eCompute) &&
							   (vk11_props.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eShuffle);
#endif

	if (!getCooperativeMatrixSupport(physicalDeviceProperties, driver_props, architecture)) {
		deviceMetadata.coopmat = false;
	}

	deviceMetadata.integerDotProduct = deviceMetadata.integerDotProduct && shader_integer_dot_product_props.integerDotProduct4x8BitPackedSignedAccelerated;

	std::vector<vk::QueueFamilyProperties> queueFamilyProps = physicalDevice.getQueueFamilyProperties();

	// Just use a single queue, my goodness.
	computeQueueFamilyIndex = findQueueFamilyIndex(queueFamilyProps);

#if 0
	vk::DeviceCreateInfo device_create_info;
#endif
	vk::PhysicalDeviceFeatures device_features = physicalDevice.getFeatures();

	VkPhysicalDeviceFeatures2 device_features2;
	device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
	device_features2.pNext = nullptr;
	device_features2.features = (VkPhysicalDeviceFeatures)device_features;

	VkPhysicalDeviceVulkan11Features vk11_features;
	vk11_features.pNext = nullptr;
	vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
	device_features2.pNext = &vk11_features;

	VkPhysicalDeviceVulkan12Features vk12_features;
	vk12_features.pNext = nullptr;
	vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
	vk11_features.pNext = &vk12_features;

	last_struct = (VkBaseOutStructure *)&vk12_features;

	VkPhysicalDevicePipelineRobustnessFeaturesEXT pl_robustness_features;
	pl_robustness_features.pNext = nullptr;
	pl_robustness_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_FEATURES_EXT;
	pl_robustness_features.pipelineRobustness = VK_FALSE;

	if (pipeline_robustness) {
		last_struct->pNext = (VkBaseOutStructure *)&pl_robustness_features;
		last_struct = (VkBaseOutStructure *)&pl_robustness_features;
		extensionList.push_back("VK_EXT_pipeline_robustness");
	}

	VkPhysicalDeviceSubgroupSizeControlFeaturesEXT subgroup_size_control_features;
	subgroup_size_control_features.pNext = nullptr;
	subgroup_size_control_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT;
	subgroup_size_control_features.computeFullSubgroups = false;
	subgroup_size_control_features.subgroupSizeControl = false;

	if (deviceMetadata.subgroupSizeControl) {
		last_struct->pNext = (VkBaseOutStructure *)&subgroup_size_control_features;
		last_struct = (VkBaseOutStructure *)&subgroup_size_control_features;
	}

#if defined(VK_KHR_cooperative_matrix)
	VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopmat_features;
	coopmat_features.pNext = nullptr;
	coopmat_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
	coopmat_features.cooperativeMatrix = VK_FALSE;

	if (deviceMetadata.coopmat) {
		last_struct->pNext = (VkBaseOutStructure *)&coopmat_features;
		last_struct = (VkBaseOutStructure *)&coopmat_features;
	}
#endif

#if defined(VK_NV_cooperative_matrix2)
	VkPhysicalDeviceCooperativeMatrix2FeaturesNV coopmat2_features {};
	coopmat2_features.pNext = nullptr;
	coopmat2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV;
	if (coopmat2_support) {
		last_struct->pNext = (VkBaseOutStructure *)&coopmat2_features;
		last_struct = (VkBaseOutStructure *)&coopmat2_features;
		extensionList.push_back("VK_NV_cooperative_matrix2");
	}
#endif

#if defined(VK_KHR_shader_bfloat16)
	VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16_features {};
	bfloat16_features.pNext = nullptr;
	bfloat16_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
	if (deviceMetadata.bf16_) {
		last_struct->pNext = (VkBaseOutStructure *)&bfloat16_features;
		last_struct = (VkBaseOutStructure *)&bfloat16_features;
	}
#endif

	VkPhysicalDeviceMaintenance4Features maint4_features {};
	maint4_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES;
	if (maintenance4_support) {
		last_struct->pNext = (VkBaseOutStructure *)&maint4_features;
		last_struct = (VkBaseOutStructure *)&maint4_features;
		extensionList.push_back("VK_KHR_maintenance4");
	}

	VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR shader_integer_dot_product_features {};
	shader_integer_dot_product_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR;
	if (deviceMetadata.integerDotProduct) {
		last_struct->pNext = (VkBaseOutStructure *)&shader_integer_dot_product_features;
		last_struct = (VkBaseOutStructure *)&shader_integer_dot_product_features;
		extensionList.push_back("VK_KHR_shader_integer_dot_product");
	}

	vkGetPhysicalDeviceFeatures2(physicalDevice, &device_features2);

	deviceMetadata.half_ = deviceMetadata.half_ && vk12_features.shaderFloat16;

#if defined(VK_KHR_shader_bfloat16)
	deviceMetadata.bf16_ = deviceMetadata.bf16_ && bfloat16_features.shaderBFloat16Type;
#else
	deviceMetadata.bf16_ = false;
#endif
	if (deviceMetadata.subgroupSizeControl) {
		deviceMetadata.minSubgroupSize = subgroup_size_control_props.minSubgroupSize;
		deviceMetadata.maxSubgroupSize = subgroup_size_control_props.maxSubgroupSize;
		extensionList.push_back("VK_EXT_subgroup_size_control");
	}

	deviceMetadata.subgroupSizeControl = deviceMetadata.subgroupSizeControl &&
			(subgroup_size_control_props.requiredSubgroupSizeStages & vk::ShaderStageFlagBits::eCompute) &&
			subgroup_size_control_features.subgroupSizeControl;

	if (deviceMetadata.subgroupSizeControl) {
		deviceMetadata.subgroupRequireFullSupport = subgroup_size_control_features.computeFullSubgroups;
	}

#ifdef TART_ENABLE_VALIDATION // there we go! c:
	extensionList.push_back("VK_KHR_shader_non_semantic_info");
#endif

	if (deviceMetadata.half_) extensionList.push_back("VK_KHR_shader_float16_int8");

#if 0
	device_create_info = {
		vk::DeviceCreateFlags(),
		device_queue_create_infos,
		{},
		extensionList
	};
#endif
	//device_create_info.setPNext(&device_features2);

}

} // namespace tart

#endif
