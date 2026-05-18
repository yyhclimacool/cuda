// nvcc -x cu -std=c++17 -arch=native --extended-lambda thrust_example.cc -o
// thrust_example
#include "ach.h"

float max_change_naive(thrust::universal_vector<float> a,
                       thrust::universal_vector<float> b) {
  thrust::universal_vector<float> diff(a.size());
  thrust::transform(
      thrust::device, a.begin(), a.end(), b.begin(), diff.begin(),
      [] __host__ __device__(float a, float b) { return std::abs(a - b); });
  return thrust::reduce(thrust::device, diff.begin(), diff.end(), 0.0f,
                        thrust::maximum<float>());
}

float max_change_optm(thrust::universal_vector<float> a,
                      thrust::universal_vector<float> b) {
  auto zip_iter = thrust::make_zip_iterator(a.begin(), b.begin());
  auto transform_iter = thrust::make_transform_iterator(
      zip_iter, [] __host__ __device__(thrust::tuple<float, float> t) {
        return std::abs(thrust::get<0>(t) - thrust::get<1>(t));
      });
  return thrust::reduce(thrust::device, transform_iter,
                        transform_iter + a.size(), 0.0f,
                        thrust::maximum<float>());
}

float mean(thrust::universal_vector<float> vec) {
  return thrust::reduce(thrust::device, vec.begin(), vec.end(), 0.0f,
                        thrust::plus<float>{}) /
         vec.size();
}

float variance(thrust::universal_vector<float> vec) {
  float mean_val = mean(vec);
  auto trans_iter = thrust::make_transform_iterator(
      vec.begin(), [=] __host__ __device__(float x) {
        return (x - mean_val) * (x - mean_val);
      });
  return thrust::reduce(thrust::device, trans_iter, trans_iter + vec.size(),
                        0.0f, thrust::plus<float>{}) /
         vec.size();
}

thrust::universal_vector<float>
row_temperatures(int height, int width,
                 const thrust::universal_vector<int> &temp_keys,
                 const thrust::universal_vector<float> &temp) {
  thrust::universal_vector<float> result_temp(height, 0);
  thrust::reduce_by_key(thrust::device, temp_keys.begin(), temp_keys.end(),
                        temp.begin(), thrust::make_discard_iterator(),
                        result_temp.begin());
  return result_temp;
}

thrust::universal_vector<float>
row_temperatures_optm(int height, int width,
                      const thrust::universal_vector<float> &temp) {
  thrust::universal_vector<float> result_temp(height, 0);
  auto row_ids_begin = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      [=] __host__ __device__(int idx) { return idx / width; });
  auto row_ids_end = row_ids_begin + temp.size();
  thrust::reduce_by_key(thrust::device, row_ids_begin, row_ids_end,
                        temp.begin(), thrust::make_discard_iterator(),
                        result_temp.begin());
  return result_temp;
}

thrust::universal_vector<float> init(int height, int width) {
  float low = 15.0, high = 95.0;
  thrust::universal_vector<float> temp(height * width, low);
  thrust::fill(thrust::device, temp.begin(), temp.begin() + width, high);
  thrust::fill(thrust::device, temp.end() - width, temp.end(), high);
  return temp;
}

thrust::universal_vector<float>
generate_reduce_key(int height, int width,
                    const thrust::universal_vector<float> &temp) {
  thrust::universal_vector<float> temp_idx(height * width, 0);
  for (int row = 0; row < height; ++row) {
    thrust::fill(thrust::device, temp_idx.begin() + row * width,
                 temp_idx.begin() + (row + 1) * width, row);
  }
  return temp_idx;
}

thrust::universal_vector<float>
mean_temperatures(int height, int width,
                  const thrust::universal_vector<float> &temp) {
  thrust::universal_vector<float> means(height, 0);
  auto start_idx = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      [=] __host__ __device__(int idx) { return idx / width; });
  auto end_idx = start_idx + temp.size();
  thrust::reduce_by_key(thrust::device, start_idx, end_idx, temp.begin(),
                        thrust::make_discard_iterator(), means.begin());
  return means;
}

void simulate_async(int height, int width,
                    const thrust::device_vector<float> &in,
                    thrust::device_vector<float> &out, cudaStream_t stream) {
  cuda::std::mdspan in_data(thrust::raw_pointer_cast(in.data()), height, width);

  auto compute_op = [=] __host__ __device__(int idx) {
    const int row = idx / width;
    const int col = idx % width;

    // loop over all points in domain (except boundary)
    if (row > 0 && col > 0 && row < height - 1 && col < width - 1) {
      // evaluate derivatives
      float d2tdx2 =
          in_data(row, col - 1) - 2 * in_data(row, col) + in_data(row, col + 1);
      float d2tdy2 =
          in_data(row - 1, col) - 2 * in_data(row, col) + in_data(row + 1, col);

      // update temperatures
      return in_data(row, col) + 0.2f * (d2tdx2 + d2tdy2);
    } else {
      return in_data(row, col);
    }
  };
  // thrust::tabulate(thrust::device, out.begin(), out.end(), compute_op);
  auto count_iter = thrust::make_counting_iterator(0);
  cub::DeviceTransform::Transform(in.begin(), out.begin(), out.size(),
                                  compute_op, stream);
}

__global__ void single_thread_kernel(ach::mdspan2df temp_in, float *temp_out) {
  for (int idx = 0; idx < temp_in.size(); ++idx) {
    temp_out[idx] = ach::compute(idx, temp_in);
  }
}
const int num_threads = 256;
__global__ void block_kernel(ach::mdspan2df temp_in, float *temp_out) {
  int threadidx = threadIdx.x;
  for (int id = threadidx; id < temp_in.size(); id += num_threads) {
    temp_out[id] = ach::compute(id, temp_in);
  }
}

__global__ void grid_kernel(ach::mdspan2df temp_in, float *temp_out) {
  int threadidx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (int id = threadidx; id < temp_in.size(); id += num_threads) {
    temp_out[id] = ach::compute(id, temp_in);
  }
}

void simulate(ach::mdspan2df temp_in, float *temp_out, cudaStream_t stream) {
  single_thread_kernel<<<1, 1, 0, stream>>>(temp_in, temp_out);
}
void simulate_block(ach::mdspan2df temp_in, float *temp_out,
                    cudaStream_t stream) {
  block_kernel<<<1, num_threads, 0, stream>>>(temp_in, temp_out);
}
void simulate_grid(ach::mdspan2df temp_in, float *temp_out,
                   cudaStream_t stream) {
  int block_size = 1024;
  int grid_size = (temp_in.size() + block_size - 1) / block_size;
  grid_kernel<<<grid_size, block_size, 0, stream>>>(temp_in, temp_out);
}

__global__ void symmetry_check_kernel(ach::mdspan2df temp, int row) {
  int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= temp.extent(1))
    return;
  if (abs(temp(row, column) - temp(temp.extent(0) - 1 - row, column)) > 0.1) {
    printf("Error: asymmetry in (%d, %d)\n", row, column);
  }
}

void symmetry_check(ach::mdspan2df temp_in, cudaStream_t stream) {
  int target_row = 0;
  int width = temp_in.extent(1);
  int block_size = std::min(width, 1024);
  int grid_size = (width + block_size - 1) / block_size;
  symmetry_check_kernel<<<grid_size, block_size, 0, stream>>>(temp_in,
                                                              target_row);
}

void lab_cuda_kernels() {
  int height = 1024;
  int width = 5000;

  cudaStream_t compute_stream;
  cudaStreamCreate(&compute_stream);

  // // Trying to silence symmetry check error
  // {
  //   thrust::device_vector<float> d_prev((height + 1) * width);
  //   thrust::device_vector<float> d_next((height + 1) * width);
  // }

  thrust::device_vector<float> d_prev(height * width);
  // thrust::fill_n(d_prev.begin(), width, 90.0f);
  // thrust::fill_n(d_prev.begin() + width * (height - 1), width, 90.0f);
  thrust::device_vector<float> d_next(height * width);

  auto step_begin = std::chrono::high_resolution_clock::now();
  for (int compute_step = 0; compute_step < 10; compute_step++) {
    ach::mdspan2df temp_in(thrust::raw_pointer_cast(d_prev.data()), height,
                           width);
    float *temp_out = thrust::raw_pointer_cast(d_next.data());
    symmetry_check(temp_in, compute_stream);
    cudaError_t error = cudaPeekAtLastError();
    if (error != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(error));
    }
    // d_prev.swap(d_next);
  }
  cudaStreamSynchronize(compute_stream);
  auto step_end = std::chrono::high_resolution_clock::now();
  auto step_seconds =
      std::chrono::duration<double>(step_end - step_begin).count();

  std::printf("compute in %g s\n", step_seconds);

  cudaStreamDestroy(compute_stream);
}

int main() {
  // thrust::universal_vector<float> a(1 << 28);
  // thrust::universal_vector<float> b(1 << 28);

  // thrust::sequence(a.begin(), a.end());
  // thrust::sequence(b.rbegin(), b.rend());

  // auto start_naive = std::chrono::high_resolution_clock::now();
  // max_change_naive(a, b);
  // auto end_naive = std::chrono::high_resolution_clock::now();
  // const double naive_duration =
  //     std::chrono::duration_cast<std::chrono::milliseconds>(end_naive -
  //                                                           start_naive)
  //         .count();
  // std::printf("max change naive duration: %f ms\n", naive_duration);

  // auto start_optm = std::chrono::high_resolution_clock::now();
  // max_change_optm(a, b);
  // auto end_optm = std::chrono::high_resolution_clock::now();
  // const double optm_duration =
  //     std::chrono::duration_cast<std::chrono::milliseconds>(end_optm -
  //                                                           start_optm)
  //         .count();
  // std::printf("max change optm duration: %f ms\n", optm_duration);

  // std::printf("\n---------------------------------\n");

  // const int ambient_temp = 20;
  // const float k = 0.5;
  // thrust::universal_vector<float> temp{22, 43, 30};
  // decltype(temp) next(temp.size(), 0);
  // for (int step = 0; step < 3; ++step) {
  //   thrust::transform(thrust::device, temp.begin(), temp.end(), next.begin(),
  //                     [=] __host__ __device__(float x) {
  //                       return x + k * (ambient_temp - x);
  //                     });
  //   std::printf("step %d: , variance: %f\n", step, variance(next));
  //   temp.swap(next);
  // }
  // std::printf("\n---------------------------------\n");

  // cuda::std::array<int, 6> sd{1, 2, 3, 4, 5, 6};
  // cuda::std::mdspan md(sd.data(), 2, 3);
  // std::printf("md(0,0) = %d\n", md(0, 0));
  // std::printf("md(1,2) = %d\n", md(1, 2));

  // std::printf("size = %zu\n", md.size());
  // std::printf("height = %zu\n", md.extent(0));
  // std::printf("width = %zu\n", md.extent(1));

  // std::printf("\n---------------------------------\n");

  // int height = 32, width = 16777216;
  // auto temp = init(height, width);
  // auto temp_idx = generate_reduce_key(height, width, temp);
  // auto start_ts = std::chrono::high_resolution_clock::now();
  // auto sums = row_temperatures(height, width, temp_idx, temp);
  // auto end_ts = std::chrono::high_resolution_clock::now();
  // const double duration_ms =
  //     std::chrono::duration<double, std::milli>(end_ts - start_ts).count();
  // const double throughput_gbps =
  //     temp.size() * sizeof(float) / duration_ms / 1024 / 1024 / 1024 * 1000;
  // std::printf("cost_ms: %f, throughput: %f GB/s\n", duration_ms,
  //             throughput_gbps);

  // auto start_ts_optm = std::chrono::high_resolution_clock::now();
  // auto sums_optm = row_temperatures_optm(height, width, temp);
  // auto end_ts_optm = std::chrono::high_resolution_clock::now();
  // const double duration_ms_optm =
  //     std::chrono::duration<double, std::milli>(end_ts_optm - start_ts_optm)
  //         .count();
  // const double throughput_gbps_optm = temp.size() * sizeof(float) /
  //                                     duration_ms_optm / 1024 / 1024 / 1024 *
  //                                     1000;
  // std::printf("cost_ms_optm: %f, throughput_gbps_optm: %f GB/s\n",
  //             duration_ms_optm, throughput_gbps_optm);
  // std::printf("\n---------------------------------\n");
  // cudaStream_t copyStream, computeStream;
  // cudaStreamCreate(&copyStream);
  // cudaStreamCreate(&computeStream);

  // int height = 4096, width = 4096;
  // thrust::device_vector<float> prev = ach::init(height, width);
  // thrust::device_vector<float> next(height * width);
  // thrust::device_vector<float> d_buffer(height * width);
  // thrust::universal_host_pinned_vector<float> h_prev(height *
  //                                                    width); // pinned memory

  // for (int write_step = 0; write_step < 3; ++write_step) {
  //   std::printf("    write_step: %d\n", write_step);
  //   auto step_start_ts = std::chrono::high_resolution_clock::now();
  //   {
  //     nvtx3::scoped_range r{"copy"};
  //     cudaMemcpy(thrust::raw_pointer_cast(d_buffer.data()),
  //                thrust::raw_pointer_cast(prev.data()),
  //                height * width * sizeof(float), cudaMemcpyDeviceToDevice);
  //     // thrust::copy(h_prev.begin(), h_prev.end(), d_buffer.begin());
  //     cudaMemcpyAsync(thrust::raw_pointer_cast(h_prev.data()),
  //                     thrust::raw_pointer_cast(d_buffer.data()),
  //                     height * width * sizeof(float), cudaMemcpyDeviceToHost,
  //                     copyStream);
  //   }

  //   {
  //     nvtx3::scoped_range r{"simulate"};
  //     for (int compute_step = 0; compute_step < 3; ++compute_step) {
  //       simulate_async(height, width, prev, next, computeStream);
  //       prev.swap(next);
  //     }
  //   }

  //   auto write_start_ts = std::chrono::high_resolution_clock::now();
  //   {
  //     nvtx3::scoped_range r{"write"};
  //     cudaStreamSynchronize(copyStream);
  //     ach::store(write_step, height, width, h_prev);
  //   }
  //   auto write_end_ts = std::chrono::high_resolution_clock::now();
  //   const double write_duration_ms =
  //       std::chrono::duration<double, std::milli>(write_end_ts -
  //       write_start_ts)
  //           .count();

  //   {
  //     nvtx3::scoped_range r{"synchronize_compute_stream"};
  //     cudaStreamSynchronize(computeStream);
  //   }

  //   auto step_end_ts = std::chrono::high_resolution_clock::now();
  //   const double step_duration_ms =
  //       std::chrono::duration<double, std::milli>(step_end_ts -
  //       step_start_ts)
  //           .count();

  //   std::printf("        write duration_ms: %f\n", write_duration_ms);
  //   std::printf("        copy + simulate + write duration_ms: %f\n",
  //               step_duration_ms);
  //   std::printf("        diff duration_ms: %f\n",
  //               step_duration_ms - write_duration_ms);
  // }
  // cudaStreamDestroy(copyStream);
  // cudaStreamDestroy(computeStream);
  // std::printf("\n---------------------------------\n");
  std::printf("\n---------------------------------\n");
  lab_cuda_kernels();
  std::printf("\n---------------------------------\n");
  return 0;
}