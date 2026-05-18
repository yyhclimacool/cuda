#include "ach.h"

constexpr int bin_width = 10;

__global__ void histogram_kernel(cuda::std::span<float> temp,
                                 cuda::std::span<int> histogram) {
  __shared__ int block_histogram[bin_width];
  if (threadIdx.x < bin_width) {
    block_histogram[threadIdx.x] = 0;
  }
  __syncthreads();
  int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell_idx < temp.size()) {
    int bin = static_cast<int>(temp[cell_idx] / bin_width);
    cuda::atomic_ref<int, cuda::thread_scope_block> aref(block_histogram[bin]);
    aref.fetch_add(1);
  }
  __syncthreads();
  if (threadIdx.x < bin_width) {
    cuda::atomic_ref<int, cuda::thread_scope_block> ref(histogram[threadIdx.x]);
    ref.fetch_add(block_histogram[threadIdx.x]);
  }
}

void histogram(cuda::std::span<float> temp, cuda::std::span<int> histogram,
               cudaStream_t stream) {
  int block_size = 1024;
  int grid_size = ach::ceil_div(temp.size(), block_size);
  histogram_kernel<<<grid_size, block_size, 0, stream>>>(temp, histogram);
}

int main() {
  unsigned height = 1024;
  unsigned width = 4096;

  cudaStream_t compute_stream;
  cudaStreamCreate(&compute_stream);

  thrust::device_vector<int> d_histogram(10);
  thrust::host_vector<int> h_histogram(10);

  float low = 0.0f;
  float high = 99.0f;
  thrust::host_vector<float> h_prev(height * width, low);
  thrust::device_vector<float> d_prev(height * width, low);
  thrust::device_vector<float> d_next(height * width);
  thrust::fill_n(d_prev.begin(), width, high);
  thrust::fill_n(d_prev.begin() + width * (height - 1), width, high);

  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  for (int write_step = 0; write_step < 100; write_step++) {
    h_prev = d_prev;
    thrust::fill(d_histogram.begin(), d_histogram.end(), 0);
    cudaEventRecord(begin, compute_stream);
    histogram(cuda::std::span<float>{thrust::raw_pointer_cast(d_prev.data()),
                                     height * width},
              cuda::std::span<int>{thrust::raw_pointer_cast(d_histogram.data()),
                                   d_histogram.size()},
              compute_stream);
    cudaEventRecord(end, compute_stream);
    cudaEventSynchronize(end);
    float ms{};
    cudaEventElapsedTime(&ms, begin, end);
    std::printf("histogram took %f ms\n", ms);
    h_histogram = d_histogram;

    if (thrust::reduce(h_histogram.begin(), h_histogram.end()) !=
        height * width) {
      std::printf("Error: sum of bins is not equal to number of cells\n");
    }

    ach::store(write_step, 10, h_histogram);
    ach::store(write_step, height, width, h_prev);
    for (int compute_step = 0; compute_step < 120; compute_step++) {
      ach::simulate(
          ach::temperature_grid_f{thrust::raw_pointer_cast(d_prev.data()),
                                  height, width},
          thrust::raw_pointer_cast(d_next.data()), compute_stream);
      d_prev.swap(d_next);
    }
  }
  cudaStreamSynchronize(compute_stream);

  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  cudaStreamDestroy(compute_stream);
}