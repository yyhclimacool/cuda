#include "common/common.h"
#include <cstdlib>
#include <glog/logging.h>

void square_matrix_sum_cpu(const float *A, const float *B, float *C, int N) {
  for (int i = 0; i < N * N; i++) {
    C[i] = A[i] + B[i];
  }
}

__global__ void square_matrix_sum_gpu(const float *A, const float *B, float *C,
                                      int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N * N) {
    C[i] = A[i] + B[i];
  }
}

void initialize_square_matrix(float *A, int N) {
  for (int i = 0; i < N * N; i++) {
    A[i] = static_cast<float>(i);
  }
}

int main(int argc, char **argv) {
  get_device_info(0);
  cudaSetDevice(0);

  constexpr int N = 1 << 10;
  constexpr int size = N * N * sizeof(float);
  float *hA, *hB, *hC, *dA, *dB, *dC;
  auto start = timestamp();
  hA = (float *)malloc(size);
  hB = (float *)malloc(size);
  hC = (float *)malloc(size);
  LOG(INFO) << "malloc cost_us[" << timestamp() - start << "]";

  start = timestamp();
  cudaMalloc(&dA, size);
  cudaMalloc(&dB, size);
  cudaMalloc(&dC, size);
  LOG(INFO) << "cudaMalloc cost_us[" << timestamp() - start << "]";

  start = timestamp();
  initialize_square_matrix(hA, N * N);
  initialize_square_matrix(hB, N * N);
  LOG(INFO) << "initialize host square matrix cost_us[" << timestamp() - start
            << "]";

  start = timestamp();
  cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);
  LOG(INFO) << "cudaMemcpy H2D cost_us[" << timestamp() - start << "]";

  start = timestamp();
  dim3 block(1024, 1, 1);
  dim3 grid((N * N + 1024 - 1) / 1024, 1, 1);
  square_matrix_sum_gpu<<<grid, block>>>(dA, dB, dC, N);
  cudaDeviceSynchronize();
  LOG(INFO) << "square_matrix_sum_gpu cost_us[" << timestamp() - start << "]";
  start = timestamp();
  cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);
  LOG(INFO) << "cudaMemcpy D2H cost_us[" << timestamp() - start << "]";

  start = timestamp();
  square_matrix_sum_cpu(hA, hB, hC, N);
  LOG(INFO) << "square_matrix_sum_cpu cost_us[" << timestamp() - start << "]";

  free(hA);
  free(hB);
  free(hC);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  cudaDeviceReset();

  return 0;
}
