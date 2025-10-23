#ifndef TART_METADATA
#define TART_METADATA
#include "tart-vulkan-include.hpp"

namespace tart
{

struct CompilerSupport
{
	bool clspv;
	bool glsl;
	bool dxc;
};

struct DeviceMetadata
{
	/*
	 * The dtypes that a device supports.
	 */
	bool void_ = true;
	
	// integer dtypes
	bool char_ = false;
	bool uchar_ = false;
	bool short_ = true; //?
	bool ushort_ = true; //?
	bool int_ = true;
	bool uint_ = true;
	bool long_ =  false;
	bool ulong_ = false;
	
	// float dtypes
	bool fp8e4m3_ = false;
	bool fp8e5m2_ = false;
	bool half_ = false;
	bool float_ = true;
	bool double_ = false;
	bool bf16_ = false;
	
	// other stuff (i should probably call this struct something else)
	// architecture
	uint32_t numShaderCores = 0;
	bool coopmat = false;
	uint32_t coopmatM = 0;
	uint32_t coopmatN = 0;
	uint32_t coopmatK = 0;
	
	bool integerDotProduct = false;
	bool subgroupSizeControl = false;
	uint32_t subgroupSize = 0;
	uint32_t minSubgroupSize = 0;
	uint32_t maxSubgroupSize = 0;
	bool subgroupRequireFullSupport = false;

	bool bda = false;
	
	uint32_t vendorID;
	vk::DriverId driverID;
	
	// memory
	uint64_t maxMemoryAllocationSize = 0;
	bool preferHostMemory = false;
	bool unifiedMemory = false;
	bool canTrackUsedMemory = false;
	
	// queue
	bool singleQueue = true;
	
	// languages?
	CompilerSupport compilers;
};

} // namespace tart

#endif
