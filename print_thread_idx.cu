#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>

__global__ void printThreadIndex(const int nx, const int ny) {
  int64_t ix = blockDim.x * blockIdx.x + threadIdx.x;
  int64_t iy = blockDim.y * blockIdx.y + threadIdx.y;
  int64_t global_idx = iy * nx + ix;
  printf("threadIdx(%d, %d), blockIdx(%d, %d), matrixIdx(%ld, %ld), "
         "global_linear_idx(%ld)\n",
         threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, global_idx);
}

int main(int argc, char **argv) {
  int device_count = 0;
  auto error = cudaGetDeviceCount(&device_count);
  if (error == cudaSuccess) {
    printf("cuda device count: %d\n", device_count);
    for (int i = 0; i < device_count; ++i) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      printf("device: %d, name: %s\n", i, prop.name);
      printf("    asyncEngineCount: %d\n", prop.asyncEngineCount);
    }
  }

  dim3 all_threads_in_one_block(16, 16, 1);
  dim3 all_threads_in_various_blocks(4, 4, 1);
  dim3 blocks_form_a_grid(4, 4, 1);
  cudaSetDevice(0);
  // printThreadIndex<<<1, all_threads_in_one_block>>>(16, 16);
  printThreadIndex<<<blocks_form_a_grid, all_threads_in_various_blocks>>>(16,
                                                                          16);
  cudaDeviceSynchronize();
  return 0;
}