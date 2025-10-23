#pragma once

#include <boost/thread.hpp>

#include <cstdint>
#include <memory>

#include <tart.hpp>

#include "AbstractMiner.h"
#include "octopus_params.h"

class StratumClient;

struct OctopusVulkanMinerSettings {
  std::vector<int> device_ids = {0};
  int initGridSize = 8192;
  int searchGridSize = 1024;
};

class VulkanDagManager;

class OctopusVulkanMiner : public AbstractMiner {
protected:
  struct ThreadContext {
    OctopusVulkanMiner *miner;
    int device_id;
    int context_id;

    VulkanDagManager *dagManager;

    void *d_search_results;

    ThreadContext(OctopusVulkanMiner *miner, int device_id, int context_id);

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

  const OctopusVulkanMinerSettings settings;

protected:
  std::vector<int> device_ids;
  std::vector<ThreadContext> threadContexts;
};
