#pragma once

#include <cassert>
#include <glog/logging.h>
#include <sys/select.h>
#include <sys/time.h>
#include <time.h>

#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      LOG(FATAL) << "call " #call " failed, code[" << error << "], errmsg["    \
                 << cudaGetErrorString(error) << "]";                          \
    }                                                                          \
  }

class Timer {
public:
  Timer() { start_ts = get_current_ts(); }
  void start() { start_ts = get_current_ts(); }
  void stop() { stop_ts = get_current_ts(); }
  int64_t elapsed_us() {
    assert(start_ts != 0);
    if (stop_ts == 0) {
      stop();
    }
    assert(stop_ts >= start_ts);
    return stop_ts - start_ts;
  }

private:
  int64_t get_current_ts() {
    timeval tval;
    gettimeofday(&tval, NULL);
    return tval.tv_sec * 1000 * 1000 + tval.tv_usec;
  }
  timeval val;
  int64_t start_ts = 0;
  int64_t stop_ts = 0;
};

inline void get_device_info(int devid) {
  cudaDeviceProp deviceProp{};
  CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, devid));
  LOG(INFO) << "Using device_no[" << devid << "]\n\tname[" << deviceProp.name
            << "]\n\twarpSize[" << deviceProp.warpSize
            << "]\n\tconcurrentKernels[" << deviceProp.concurrentKernels
            << "]\n\ttotalConstMem[" << deviceProp.totalConstMem
            << "]\n\ttotalGlobalMem[" << deviceProp.totalGlobalMem
            << "]\n\tmaxBlocksPerMultiProcessor["
            << deviceProp.maxBlocksPerMultiProcessor
            << "]\n\tmaxThreadsPerMultiProcessor["
            << deviceProp.maxThreadsPerMultiProcessor
            << "]\n\tmaxThreadsPerBlock[" << deviceProp.maxThreadsPerBlock
            << "]\n\tglobalL1CacheSupported["
            << deviceProp.globalL1CacheSupported
            << "]\n\tlocalL1CacheSupported[" << deviceProp.localL1CacheSupported
            << "]\n\tl2CacheSize[" << deviceProp.l2CacheSize
            << "]\n\tpersistingL2CacheMaxSize["
            << deviceProp.persistingL2CacheMaxSize
            << "]\n\taccessPolicyMaxWindowSize["
            << deviceProp.accessPolicyMaxWindowSize << "]\n\tasyncEngineCount["
            << deviceProp.asyncEngineCount << "]\n\tunifiedAddressing["
            << deviceProp.unifiedAddressing << "]\n\tmultiProcessorCount["
            << deviceProp.multiProcessorCount << "]\n\tcompute capability["
            << deviceProp.major << "." << deviceProp.minor
            << "]\n\tregsPerMultiprocessor[" << deviceProp.regsPerMultiprocessor
            << "]\n\tregsPerBlock[" << deviceProp.regsPerBlock
            << "]\n\tsharedMemPerBlock[" << deviceProp.sharedMemPerBlock
            << "]\n\tsharedMemPerMultiprocessor["
            << deviceProp.sharedMemPerMultiprocessor << "]\n\tcomputeMode["
            << deviceProp.computeMode << "].\n";
}