#pragma once

#include <glog/logging.h>
#include <sys/time.h>

#define CHECK_CUDA(call)                                                                                         \
  {                                                                                                              \
    const cudaError_t error = call;                                                                              \
    if (error != cudaSuccess) {                                                                                  \
      LOG(FATAL) << "call " #call " failed, code[" << error << "], errmsg[" << cudaGetErrorString(error) << "]"; \
    }                                                                                                            \
  }

inline double timestamp() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000 + static_cast<double>(tv.tv_usec * 1e-3);
}