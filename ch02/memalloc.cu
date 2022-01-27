/*************************************************

Compile command: nvcc memalloc.cu -o memalloc --std=c++14

*************************************************/

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

/*
 * This example demonstrates a simple vector sum on the GPU and on the host.
 * sumArraysOnGPU splits the work of the vector sum across CUDA threads on the
 * GPU. Only a single thread block is used in this small case, for simplicity.
 * sumArraysOnHost sequentially iterates through vector elements on the host.
 * This version of sumArrays adds host timers to measure GPU and CPU
 * performance.
 */

inline double currentMs() {
  struct timeval  tp;
  struct timezone tzp;
  int             i = gettimeofday(&tp, &tzp);
  return ((double)tp.tv_sec * 1000 + (double)tp.tv_usec * 1.e-3);
}

void checkResult(float *hostArray, float *gpuArray, const size_t n) {
  const float epsilon = 1e-6f;
  bool            allMatch = true;
  for (size_t i = 0; i < n; ++i) {
    if (fabs(hostArray[i] - gpuArray[i]) > epsilon) {
      allMatch = false;
      printf("Array match false at index %zu, hostArray[%d], gpuArray[%d]\n", i, hostArray[i], gpuArray[i]);
      break;
    }
  }

  if (allMatch) {
    printf("Arrays match!\n");
  }
}

void initArray(float *ip, const size_t n) {
  // generate different seed for random number
  time_t t;
  srand((unsigned)time(&t));

  for (int i = 0; i < n; ++i) {
    ip[i] = (float)(rand() & 0xff) / 10.0f;
  }
}

void sumArraysOnHost(float *a, float *b, float *c, const size_t n) {
  for (int i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
  }
}

__global__ void sumArraysOnGPU(float *a, float *b, float *c, const size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

int main(int argc, char **argv) {
  printf("%s starting...\n", argv[0]);

  // set up device
  int            dev = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("Using Device %d: %s\n", dev, deviceProp.name);
  cudaSetDevice(dev);

  const size_t N = 1 << 24;
  printf("Array size: %zu\n", N);

  size_t nbytes = N * sizeof(float);
  float *ha, *hb, *host_c, *gpu_c;
  ha = (float *)malloc(nbytes);
  hb = (float *)malloc(nbytes);
  host_c = (float *)malloc(nbytes);
  gpu_c = (float *)malloc(nbytes);

  memset(host_c, 0, nbytes);
  memset(gpu_c, 0, nbytes);

  double start = currentMs();
  initArray(ha, N);
  initArray(hb, N);
  double elapse = currentMs() - start;
  printf("initArray a and b on host cost[%f]ms\n", elapse);

  start = currentMs();
  sumArraysOnHost(ha, hb, host_c, N);
  elapse = currentMs() - start;
  printf("sumArraysOnHost cost[%f]ms\n", elapse);

  // malloc device global memory
  float *da, *db, *dc;
  cudaMalloc((void **)&da, nbytes);
  cudaMalloc((void **)&db, nbytes);
  cudaMalloc((void **)&dc, nbytes);

  // copy host memory to device global memory
  cudaMemcpy(da, ha, nbytes, cudaMemcpyHostToDevice);
  cudaMemcpy(db, hb, nbytes, cudaMemcpyHostToDevice);

  // invoke kernel at host side
  int  len = 512;
  dim3 block(len); // TODO: What does this mean ?
  dim3 grid((N + block.x - 1) / block.x);
  start = currentMs();
  sumArraysOnGPU<<<grid, block>>>(da, db, dc, N);
  cudaDeviceSynchronize();
  elapse = currentMs() - start;
  printf("sumArraysOnGPU<<<%d, %d>>> cost[%f]ms\n", grid.x, block.x, elapse);

  // check kernel error
  cudaGetLastError();

  // copy kernel result back to host side
  cudaMemcpy(gpu_c, dc, nbytes, cudaMemcpyDeviceToHost);

  checkResult(host_c, gpu_c, N);

  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);

  free(ha);
  free(hb);
  free(host_c);
  free(gpu_c);

  return 0;
}
