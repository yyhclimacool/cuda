#include <stdio.h>
#include <cuda_runtime_api.h>
#include <iostream>

void hello_from_cpu() {
  std::cout << "Hello from CPU." << std::endl;
}

__global__ void hello_from_gpu() {
  // std::cout not supported in cuda runtime
  //std::cout << "Hello from CPU." << std::endl;
  printf("Hello from GPU, threadid[%d]\n", threadIdx.x);
}

int main() {
    hello_from_cpu();
    hello_from_gpu<<<1, 5>>>();
    cudaDeviceSynchronize();
}
