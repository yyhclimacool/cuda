#include <bits/stdc++.h>
#include <cuda_runtime.h>

__global__ void vec_add(int n, float *da, float *db, float *dc) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dc[idx] = da[idx] + db[idx] + 1.0;
  }
  if (idx / 1024 == 0) {
    printf("idx[%d]", idx);
  }
}

int main() {
  int n = 1024 * 1024;
  int nbytes = n * sizeof(float);
  float *a = 0, *b = 0, *c = 0;
  float *da = 0, *db = 0, *dc = 0;

  a = (float *)malloc(nbytes);
  b = (float *)malloc(nbytes);
  c = (float *)malloc(nbytes);

  srand(time(0));

  for (int i = 0; i < n; ++i) {
    a[i] = (float)(rand()) / (float)(rand());
    b[i] = (float)(rand()) / (float)(rand());
    c[i] = 0;
  }

  cudaMalloc((void **)&da, nbytes);
  cudaMalloc((void **)&db, nbytes);

  cudaMemcpy(da, a, nbytes, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, nbytes, cudaMemcpyHostToDevice);

  vec_add<<<(n + 255) / 256, 256>>>(n, da, db, dc);

  cudaMemcpy(c, dc, nbytes, cudaMemcpyDeviceToHost);

  double result = 0;
  for (int i = 0; i < n; ++i) {
    result += c[i];
  }
  std::cout << "result[" << result << "]" << std::endl;

  free(a);
  free(b);
  free(c);

  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
}
