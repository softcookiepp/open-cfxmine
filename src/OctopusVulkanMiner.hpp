#pragma once

#include <boost/thread.hpp>

#include <cstdint>
#include <memory>

#include "tart.hpp"

#include "AbstractMiner.h"
#include "octopus_params.h"

// global instance used for the entire application
extern tart::Instance gTartInstance;

class StratumClient;

struct OctopusVulkanMinerSettings {
  std::vector<int> device_ids = {0};
  int initGridSize = 8192;
  int searchGridSize = 1024;
};

class VulkanDagManager;

class OctopusVulkanMiner : public AbstractMiner, std::enable_shared_from_this<OctopusVulkanMiner>
{
protected:
  struct ThreadContext {

	std::weak_ptr<OctopusVulkanMiner> mMiner;
	tart::device_ptr mDevice = nullptr;
	tart::buffer_ptr dX = nullptr;

    int device_id;
    int context_id;
	
	// doing shared instead of unique for now because of incomplete type
    std::shared_ptr<VulkanDagManager> dagManager = nullptr;

#if 1
	tart::buffer_ptr d_search_results = nullptr;
#else
    void *d_search_results;
#endif
    ThreadContext(std::weak_ptr<OctopusVulkanMiner> miner, int device_id, int context_id);

    void InitVulkan();
    void InitPerEpoch(uint64_t blockHeight);
    void InitPerHeader(const octopus_h256_t headerHash,
                       const octopus_h256_t bounadry);
  };

public:
  OctopusVulkanMiner(const OctopusVulkanMinerSettings &settings);

  ~OctopusVulkanMiner();

  void Start() override;

  void Join() override { workerThreads->join_all(); }

private:
  void Work(ThreadContext *ctx);

  std::unique_ptr<boost::thread_group> workerThreads;
  
  std::shared_ptr<OctopusVulkanMiner> getThis();

  const OctopusVulkanMinerSettings settings;

protected:
  std::vector<int> device_ids;
  std::vector<ThreadContext> mThreadContexts;
};
