#include "common/common.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <glog/logging.h>
#include <stdio.h>

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

// M(row, col) = M.elements + row * M.width + col
struct Matrix {
  Matrix() = default;
  Matrix(int n) : width(n), height(n), elements(new float[width * height]) {}
  Matrix(int w, int h)
      : width(w), height(h), elements(new float[width * height]) {}
  // kernel launch 不支持拷贝构造函数
  // Matrix(const Matrix &other)
  //     : width(other.width), height(other.height), elements(nullptr) {
  //   elements = new float[width * height];
  //   std::memcpy(elements, other.elements, width * height * sizeof(float));
  // }
  // 由于内存可能分布在显存，因此这里不能直接调用 delete，暂时忍受内存泄露。
  // ~Matrix() { delete[] elements; }
  void random_init() {
    for (int row = 0; row < width; ++row) {
      for (int col = 0; col < height; ++col) {
        elements[row * width + col] = float(rand() % 19);
      }
    }
  }
  void zero_init() { memset(elements, 0x00, width * height * sizeof(float)); }
  void pretty_print() const {
    printf("---------matrix: %ld ---------\n", int64_t(this));
    for (int row = 0; row < height; ++row) {
      for (int col = 0; col < width; ++col) {
        printf("%f, ", elements[row * width + col]);
      }
      printf("\n");
    }
  }
  int width = 0;
  int height = 0;
  float *elements = nullptr;
};

#define BLOCK_SIZE 16

// assume M.width and M.height are multiplies of BLOCK_SIZE
__host__ void matrix_mul_cpu(const Matrix *A, const Matrix *B, Matrix *C);
__global__ void matrix_mul_gpu(const Matrix A, const Matrix B, Matrix C);
// __global__ void matrix_mul_gpu_sm(const Matrix *A, const Matrix *B, Matrix
// *C);

void bench_mark(int n) {
  int N = BLOCK_SIZE * n;

  Matrix A(N), B(N), C(N), CD(N);
  A.random_init();
  B.random_init();
  C.zero_init();
  CD.zero_init();
  // A.pretty_print();
  // B.pretty_print();
  Timer timer1;
  matrix_mul_cpu(&A, &B, &C);
  timer1.stop();
  printf("matrix multiply on cpu elapsed %ld us on matrix dim(%d, %d)\n",
         timer1.elapsed_us(), A.width, A.height);

  cudaSetDevice(0);

  Matrix da, db, dc;
  da.width = A.width;
  da.height = A.height;
  db.width = B.width;
  db.height = B.height;
  dc.width = A.width;
  dc.height = B.height;
  cudaMalloc(&da.elements, da.width * da.height * sizeof(float));
  cudaMalloc(&db.elements, db.width * db.height * sizeof(float));
  cudaMalloc(&dc.elements, dc.width * dc.height * sizeof(float));

  cudaMemcpy(da.elements, A.elements, da.width * da.height * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(db.elements, B.elements, db.width * db.height * sizeof(float),
             cudaMemcpyHostToDevice);
  dim3 blockdim(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 griddim(db.width / BLOCK_SIZE, da.height / BLOCK_SIZE, 1);
  Timer timer2;
  matrix_mul_gpu<<<griddim, blockdim>>>(da, db, dc);
  timer2.stop();
  cudaMemcpy(CD.elements, dc.elements, dc.width * dc.height * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("matrix multiply on gpu elapsed %ld us on matrix dim(%d, %d)\n",
         timer2.elapsed_us(), dc.width, dc.height);
  check_result(C.elements, CD.elements, C.width * C.height);
  cudaFree(da.elements);
  cudaFree(db.elements);
  cudaFree(dc.elements);
  cudaDeviceReset();
}

// Host code
int main(int argc, char **argv) {
  bench_mark(1);
  bench_mark(8);
  return 0;
}

__host__ void matrix_mul_cpu(const Matrix *A, const Matrix *B, Matrix *C) {
  for (int i = 0; i < A->height; ++i) {
    for (int j = 0; j < B->width; ++j) {
      C->elements[i * C->width + j] = 0;
      for (int k = 0; k < A->width; ++k) {
        C->elements[i * C->width + j] +=
            A->elements[i * A->width + k] * B->elements[k * B->width + j];
      }
    }
  }
}
__global__ void matrix_mul_gpu(const Matrix A, const Matrix B, Matrix C) {
  float cvalue = 0.f;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  for (int e = 0; e < A.width; ++e) {
    cvalue += A.elements[row * A.width + e] * B.elements[e * B.height + col];
  }
  C.elements[row * C.width + col] = cvalue;
}
// __global__ void matrix_mul_gpu_sm(const Matrix *A, const Matrix *B, Matrix
// *C);