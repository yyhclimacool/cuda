#include <cuda_runtime.h>
#include <iostream>

int main() {
  int dimx = 16;
  int numbytes = dimx * sizeof(int);
  int *h_res = (int *)malloc(numbytes);

  int *d_vec = NULL;
  cudaMalloc((void **)&d_vec, numbytes);
  if (h_res == nullptr || d_vec == nullptr) {
    std::cerr << "malloc for host/device failed." << std::endl;
    return -2;
  }
  cudaMemset(d_vec, 0, numbytes);
  cudaMemcpy(h_res, d_vec, numbytes, cudaMemcpyDeviceToHost);

  for (int i = 0; i < dimx; ++i) {
    std::cout << h_res[i] << ",";
  }
  free(h_res);
  cudaFree(d_vec);
  return 0;
}