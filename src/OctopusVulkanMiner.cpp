#include "OctopusVulkanMiner.hpp"
#include "StratumClient.h"
#if 0
#include "vulkan/octopus.cuh"
#endif
#include "vulkan/structs.cuh"
#include "vulkan/precomputation.h"
#include "hex.h"
#include "light.h"
#include "octopus_params.h"
#include "octopus_structs.h"

#include <functional>
#include <iostream>
#include <stdexcept>

tart::Instance gTartInstance;

class VulkanDagManager {
	tart::device_ref mDevice;
	uint32_t mDagSize = 0;
	struct PushConstStruct {
		uint64_t dDagAddr;
		uint32_t dDagSize;
		
		uint64_t dLightAddr;
		uint32_t dLightSize;
		
		hash32_t dHeader;
		octopus_h256_t dBoundary;
		
#if 1
		// we should probably make this a buffer instead
		uint64_t dXAddr;
#else
		uint32_t dX[OCTOPUS_N];
#endif
	} mPushStruct;
public:
	VulkanDagManager(tart::device_ref device)
	{
		mDevice = device;
	}
	void reset(uint64_t blockHeight) {
		tart::device_ptr device = mDevice.lock();
		dagSize = octopus_get_datasize(blockHeight);
		dagNumItems = dagSize / OCTOPUS_MIX_BYTES;
		lightSize = octopus_get_cachesize(blockHeight);
		lightNumItems = lightSize / OCTOPUS_HASH_BYTES;
		if (memoryDagSize < dagSize) {
			if (h_dag)
			{
#if 1
				device->deallocateBuffer(h_dag);
#else
				checkCudaErrors(cudaFree(h_dag));
#endif
			}
			{
#if 1
				h_dag = device->allocateBuffer(dagSize);
#else
				cudaError_t err = cudaMalloc(&h_dag, dagSize);
				if (cudaSuccess != err) {
					if (cudaErrorMemoryAllocation == err) {
						fprintf(stderr, "cudaMalloc failed. Reason: Insufficient memory\n");
					} else {
						fprintf(stderr, "CUDA error RUNTIME: '%d' in func '%s' line %d",
						err, __FUNCTION__, __LINE__);
					}
					abort();
				}
#endif
			}
#if 1
			// sooo how does this work?
			// i really don't know
			// actually this can just be emulated with BDA!
			mPushStruct.dDagAddr = h_dag->getAddress();
#else			
			checkCudaErrors(cudaMemcpyToSymbol(d_dag, &h_dag, sizeof(void *)));
#endif
			memoryDagSize = dagSize;
		}
#if 1
		// this is just a constant, not an address
		mPushStruct.dDagSize = dagNumItems;
#else

		checkCudaErrors(cudaMemcpyToSymbol(d_dag_size, &dagNumItems, sizeof(u32)));
#endif
		if (memoryLightSize < lightSize) {
#if 1
			if(h_light) device->deallocateBuffer(h_light);
			h_light = device->allocateBuffer(lightSize);
			mPushStruct.dLightAddr = h_light->getAddress();
#else
			if (h_light) {
				checkCudaErrors(cudaFree(h_light));
			}
			checkCudaErrors(cudaMalloc(&h_light, lightSize));
			checkCudaErrors(cudaMemcpyToSymbol(d_light, &h_light, sizeof(void *)));
#endif
			memoryLightSize = lightSize;
		}
#if 1
		mPushStruct.dLightSize = lightNumItems;
#else
		checkCudaErrors(
			cudaMemcpyToSymbol(d_light_size, &lightNumItems, sizeof(u32)));
		checkCudaErrors(cudaDeviceSynchronize());
#endif
	}

	void FreeVulkan() {
#if 1
		if (h_dag) mDevice.lock()->deallocateBuffer(h_dag);
		if (h_light) mDevice.lock()->deallocateBuffer(h_light);
#else
		if (h_dag) {
			checkCudaErrors(cudaFree(h_dag));
		}
		if (h_light) {
			checkCudaErrors(cudaFree(h_light));
		}
#endif
	}
	PushConstStruct& refPushConsts() { return mPushStruct; }
	std::vector<uint8_t> getPushConsts() { return tart::packConstants(mPushStruct); }

public:
#if 1
	tart::buffer_ptr h_light = nullptr;
#else
  void *h_light = 0;
#endif
  u32 lightNumItems;
  size_t lightSize;
  u32 dagNumItems;
  size_t dagSize;

private:
#if 1
	tart::buffer_ptr h_dag = nullptr;
#else
  void *h_dag = 0;
#endif
  size_t memoryDagSize = 0;
  size_t memoryLightSize = 0;
};

OctopusVulkanMiner::ThreadContext::ThreadContext(std::weak_ptr<OctopusVulkanMiner> miner_,
                                               int device_id_, int context_id_)
    : mMiner(miner_), device_id(device_id_), context_id(context_id_)
      
{
	mDevice = gTartInstance.createDevice(device_id);
	dagManager = std::make_shared<VulkanDagManager>(mDevice);
}

OctopusVulkanMiner::OctopusVulkanMiner(const OctopusVulkanMinerSettings &settings)
    : AbstractMiner(), settings(settings) {
	
#if 1
	int device_count = (int)gTartInstance.getNumDevices();
#else
	int device_count;
	checkCudaErrors(cudaGetDeviceCount(&device_count));
#endif
	int context_id = 0;

	for (int device_id : settings.device_ids)
	{
		if (device_id < device_count)
		{
			device_ids.push_back(device_id);
			mThreadContexts.emplace_back(getThis(), device_id, context_id++);
		}
		else
		{
			std::cerr << "Vulkan device_id = " << device_id << " does not exist."
				<< std::endl;
		}
	}

  if (device_ids.empty()) {
    abort();
  }
}

OctopusVulkanMiner::~OctopusVulkanMiner() {}

std::shared_ptr<OctopusVulkanMiner> OctopusVulkanMiner::getThis()
{
	return shared_from_this();
}

void OctopusVulkanMiner::Start() {
  workerThreads = std::make_unique<boost::thread_group>();
  for (size_t i = 0; i < mThreadContexts.size(); ++i) {
    workerThreads->create_thread(
        boost::bind(&OctopusVulkanMiner::Work, this, &mThreadContexts[i]));
  }
}

void OctopusVulkanMiner::ThreadContext::InitVulkan() {
#if 1
	// device is already set upon thread initialization
	// all we need to do now is allocate a host-side buffer
	d_search_results = mDevice->allocateBuffer( sizeof(SearchResults) );
	
	// would be a good idea to also set up the shader modules here too...
	
#else
  checkCudaErrors(cudaSetDevice(device_id));
  checkCudaErrors(cudaMallocHost(&d_search_results, sizeof(SearchResults)));
#endif
}

void OctopusVulkanMiner::ThreadContext::InitPerEpoch(uint64_t blockHeight) {
	dagManager->reset(blockHeight);
	auto h_light = octopus_light_new(blockHeight);
#if 1
	dagManager->h_light->copyIn(h_light->cache, dagManager->lightSize);
#else
  checkCudaErrors(cudaMemcpy(dagManager->h_light, h_light->cache,
                             dagManager->lightSize, cudaMemcpyHostToDevice));
#endif
	octopus_light_delete(h_light);

	const uint32_t work = dagManager->dagSize / 8;
	const uint32_t run = mMiner.lock()->settings.initGridSize * INIT_BLOCK_SIZE;

	uint32_t base;
	for (base = 0; base <= work - run; base += run)
	{
#if 1
		mDevice->sync();
#else
		InitDagItems<<<mMiner->settings.initGridSize, INIT_BLOCK_SIZE>>>(base);
#endif
	}
	if (base < work)
	{
		const uint32_t lastGrid = ((work - base) + INIT_BLOCK_SIZE - 1) / INIT_BLOCK_SIZE;
#if 1
#else
		InitDagItems<<<lastGrid, INIT_BLOCK_SIZE>>>(base);
#endif
	}
#if 0
	checkCudaErrors(cudaDeviceSynchronize());
#endif
}

void OctopusVulkanMiner::ThreadContext::InitPerHeader(
    const octopus_h256_t headerHash, const octopus_h256_t boundary) {
#if 1
	dagManager->refPushConsts().dHeader = headerHash.b;
#else
  checkCudaErrors(
      cudaMemcpyToSymbol(d_header, headerHash.b, sizeof(headerHash)));
#endif
	{
		uint64_t buffer[4];
		for (int i = 0; i < 4; ++i)
		{
			const uint64_t b = reinterpret_cast<const uint64_t *>(boundary.b)[i];
			buffer[i] = ((b & 0xff00000000000000ULL) >> 56) |
			((b & 0x00ff000000000000ULL) >> 40) |
			((b & 0x0000ff0000000000ULL) >> 24) |
			((b & 0x000000ff00000000ULL) >> 8) |
			((b & 0x00000000ff000000ULL) << 8) |
			((b & 0x0000000000ff0000ULL) << 24) |
			((b & 0x000000000000ff00ULL) << 40) |
			((b & 0x00000000000000ffULL) << 56);
		}
#if 1
		// maybe I should move the push constants somewhere else?
		// oh well
		dagManager->refPushConsts().dBoundary = reinterpret_cast<octopus_h256_t>(buffer);
#else
		checkCudaErrors(cudaMemcpyToSymbol(d_boundary, buffer, sizeof(boundary)));
#endif
	}
	OctopusABCW p(headerHash);
	const u32 a = p.a;
	const u32 b = p.b;
	const u32 c = p.c;
	const u32 w = p.w;
	Precomputation<OCTOPUS_N> pre(a, b, c, w);
#if 1
	if (mMiner.lock()->dX == nullptr)
	{
		dX = mDevice->allocateBuffer(sizeof(uint32_t) * OCTOPUS_N);
	}
	dagManager->refPushConsts().dXAddr = dX->getAddress();
#else
	checkCudaErrors(cudaMemcpyToSymbol(d_x, pre.x, sizeof(uint32_t) * OCTOPUS_N));
	checkCudaErrors(cudaDeviceSynchronize());
#endif
}

void OctopusVulkanMiner::Work(ThreadContext *ctx) {
	ctx->InitVulkan();

	const uint32_t searchGridSize = settings.searchGridSize;
	const uint32_t batchSize = searchGridSize * SEARCH_BLOCK_SIZE;

	std::string jobId;
	uint64_t blockHeight = std::numeric_limits<uint64_t>::max();
	std::string headerHashString;
	octopus_h256_t headerHash;
	octopus_h256_t boundary;
	uint64_t nonce = ctx->context_id * batchSize;

  while (is_running.load(std::memory_order_acquire)) {
    if (workJobId == MINER_NO_WORK) {
      boost::this_thread::sleep_for(boost::chrono::milliseconds(5000));
      continue;
    }
    if (0 != memcmp(headerHash.b, workHeaderHash.b, sizeof(headerHash))) {
      jobId = workJobId;
      headerHashString = workHeaderHashString;
      if (octopus_get_epoch(blockHeight) !=
          octopus_get_epoch(workBlockHeight)) {
        ctx->InitPerEpoch(workBlockHeight);
        blockHeight = workBlockHeight;
      }
      ctx->InitPerHeader(workHeaderHash, workBoundary);
      memcpy(headerHash.b, workHeaderHash.b, sizeof(headerHash));
      memcpy(boundary.b, workBoundary.b, sizeof(boundary));
      nonce = ctx->context_id * batchSize;
    }

    volatile SearchResults &search_results =
        *reinterpret_cast<SearchResults *>(ctx->d_search_results);
    search_results.count = 0;
    Compute<<<settings.searchGridSize, SEARCH_BLOCK_SIZE>>>(
        nonce, reinterpret_cast<SearchResults *>(ctx->d_search_results));
    checkCudaErrors(cudaDeviceSynchronize());

    uint32_t found_count =
        std::min((uint32_t)search_results.count, MAX_SEARCH_RESULTS);
    for (uint32_t i = 0; i < found_count; i++) {
      uint64_t found_nonce = nonce + search_results.result[i].nonce_offset;
      std::vector<std::string> solutions;
      solutions.push_back(jobId);
      solutions.push_back("0x" + hex::to_hex_string(found_nonce));
      solutions.push_back(headerHashString);
      client->OnSolutionFound(solutions);
    }
    client->UpdateHashRate(batchSize);
    nonce += batchSize * device_ids.size();
  }

  checkCudaErrors(cudaDeviceSynchronize());
  ctx->dagManager->FreeVulkan();
  checkCudaErrors(cudaFreeHost(ctx->d_search_results));
}
