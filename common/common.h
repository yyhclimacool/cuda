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
