#include <cassert>
#include <common/common.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/select.h>
#include <sys/time.h>
#include <time.h>

template <typename T>
void check_result(const T *h_C, const T *h_C_from_device, int N) {
  bool result_valid = true;
  for (int i = 0; i < N; ++i) {
    if (h_C[i] != h_C_from_device[i]) {
      printf("Error: h_C[%d] = %f, h_C_from_device[%d] = %f\n", i, h_C[i], i,
             h_C_from_device[i]);
      result_valid = false;
    }
  }
  if (result_valid) {
    printf("Result is valid\n");
  } else {
    printf("Result is invalid\n");
  }
}

// host code, serialized code
void vec_add_cpu(const float *A, const float *B, float *C, int N) {
  for (int i = 0; i < N; ++i) {
    C[i] = A[i] + B[i];
  }
}
// Device code
__global__ void vec_add_gpu(float *A, float *B, float *C, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
}
// Device code optimized
__global__ void vec_add_gpu_opt(const float *A, const float *B, float *C,
                                int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = i * 4;
  for (int j = 0; j < 4; ++j) {
    C[idx + j] = A[idx + j] + B[idx + j];
  }
}

// Host code
int main(int argc, char **argv) {
  cudaSetDevice(0);
  int N = 1024 * 1024;
  size_t size = N * sizeof(float);

  // Allocate input vectors h_A and h_B in host memory
  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);
  float *h_C = (float *)malloc(size);
  float *h_C_from_device = (float *)malloc(size);
  // Initialize input vectors
  for (int i = 0; i < N; ++i) {
    h_A[i] = (float)(rand() % 11);
    h_B[i] = (float)(rand() % 11);
    h_C[i] = 0;
    h_C_from_device[i] = 0;
  }

  Timer vec_add_cpu_timer;
  vec_add_cpu(h_A, h_B, h_C, N);
  vec_add_cpu_timer.stop();
  printf("vec add on cpu cost %ld us\n", vec_add_cpu_timer.elapsed_us());

  // Allocate vectors in device memory
  float *d_A;
  cudaMalloc(&d_A, size);
  float *d_B;
  cudaMalloc(&d_B, size);
  float *d_C;
  cudaMalloc(&d_C, size);

  // Copy vectors from host memory to device memory
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // Invoke kernel
  int threadsPerBlock = 1024;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  dim3 block_config(threadsPerBlock, 1, 1), grid_config(blocksPerGrid, 1, 1);
  Timer vec_add_gpu_timer;
  vec_add_gpu<<<grid_config, block_config>>>(d_A, d_B, d_C, N);
  cudaDeviceSynchronize();
  vec_add_gpu_timer.stop();
  printf("vec add on gpu cost %ld us\n", vec_add_gpu_timer.elapsed_us());

  // Copy result from device memory to host memory
  // h_C contains the result in host memory
  cudaMemcpy(h_C_from_device, d_C, size, cudaMemcpyDeviceToHost);

  check_result(h_C, h_C_from_device, N);

  // Invoke kernel optimized
  {
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    dim3 block_config(256, 1, 1), grid_config(blocksPerGrid, 1, 1);
    Timer vec_add_gpu_opt_timer;
    vec_add_gpu_opt<<<grid_config, block_config>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    vec_add_gpu_opt_timer.stop();
    printf("vec add optimized on gpu cost %ld us\n",
           vec_add_gpu_opt_timer.elapsed_us());

    cudaMemcpy(h_C_from_device, d_C, size, cudaMemcpyDeviceToHost);
    check_result(h_C, h_C_from_device, N);
  }
  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_from_device);
  return 0;
}