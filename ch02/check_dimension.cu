/*************************************************

Compile command: nvcc check_dimension.cu -o check_dimension --std=c++14

*************************************************/

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

__global__ void checkIndex() {
  printf(
    "threadIdx(%d, %d, %d) "
    "blockIdx(%d, %d, %d) "
    "blockDim(%d, %d, %d) "
    "gridDim(%d, %d, %d)\n",
    threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z,
    gridDim.x, gridDim.y, gridDim.z);
}

int main(int argc, char **argv) {
  int  nelem = 6;
  dim3 block(3);
  dim3 grid((nelem + block.x - 1) / block.x);

  printf("grid(%d, %d, %d)\n", grid.x, grid.y, grid.z);
  printf("block(%d, %d, %d)\n", block.x, block.y, block.z);

  checkIndex<<<grid, block>>>();

  cudaDeviceSynchronize();

  return 0;
}