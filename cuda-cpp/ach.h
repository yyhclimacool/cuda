#pragma once

#include <cstdio>
#include <nv/target>

#include <cuda/std/atomic>

#include <cub/device/device_for.cuh>

#include <thrust/fill.h>
#include <thrust/tabulate.h>
#include <thrust/universal_vector.h>

#include <cstdio>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/universal_vector.h>

#include <cuda/std/mdspan>

#include <fstream>

#include "nvtx3.hpp"
#include <cstdint> // CHAR_BIT

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace ach {

using mdspan2df = cuda::std::mdspan<float, cuda::std::dextents<int, 2>>;
using temperature_grid_f = mdspan2df;

using mdspan2dd = cuda::std::mdspan<double, cuda::std::dextents<int, 2>>;
using temperature_grid_d = mdspan2dd;

int ceil_div(int problem_size, int block_size) {
  return (problem_size + block_size - 1) / block_size;
}

static __host__ __device__ bool is_executed_on_gpu() {
  NV_IF_TARGET(NV_IS_HOST, (return false;));
  return true;
}

static __host__ __device__ const char *execution_space() {
  return is_executed_on_gpu() ? "GPU" : "CPU";
}

static double max_bandwidth() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  const std::size_t mem_freq =
      static_cast<std::size_t>(prop.memoryClockRate) * 1000; // kHz -> Hz
  const int bus_width = prop.memoryBusWidth;
  const std::size_t bytes_per_second = 2 * mem_freq * bus_width / CHAR_BIT;
  return static_cast<double>(bytes_per_second) / 1024 / 1024 /
         1024; // B/s -> GB/s
}

__host__ __device__ void where_am_I(const char *expected) {
  const char *actual = execution_space();
  bool equal = true;

  while (*expected != '\0' && *actual != '\0') {
    if (*expected != *actual) {
      equal = false;
      break;
    }
    ++expected;
    ++actual;
  }

  if (equal) {
    std::printf("Correct! The function is invoked on %s\n", execution_space());
  } else {
    std::printf("Wrong guess\n");
  }
}

__host__ __device__ void I_expect(const char *expected) {
  std::printf("expect %s; runs on %s;\n", expected, execution_space());
}

static thrust::universal_vector<float> init(int height, int width) {
  const float low = 15.0;
  const float high = 90.0;
  thrust::universal_vector<float> data(height * width, low);
  thrust::fill(thrust::device, data.begin(), data.begin() + width, high);
  thrust::fill(thrust::device, data.end() - width, data.end(), high);
  return data;
}

template <class ContainerT>
static void store(int step, int height, int width, const ContainerT &data) {
  thrust::host_vector<float> host_data(data.begin(), data.end());
  std::ofstream file("./tmp/heat_" + std::to_string(step) + ".bin",
                     std::ios::binary);
  file.write(reinterpret_cast<const char *>(&height), sizeof(int));
  file.write(reinterpret_cast<const char *>(&width), sizeof(int));
  file.write(reinterpret_cast<const char *>(host_data.data()),
             height * width * sizeof(float));
}

template <class ContainerT> void store(int step, int bins, ContainerT &data) {
  std::ofstream file("./tmp/hist_" + std::to_string(step) + ".bin",
                     std::ios::binary);
  file.write(reinterpret_cast<const char *>(&bins), sizeof(int));
  file.write(
      reinterpret_cast<const char *>(thrust::raw_pointer_cast(data.data())),
      bins * sizeof(int));
}

__host__ __device__ float compute(int cell_id, temperature_grid_f temp) {
  int height = temp.extent(0);
  int width = temp.extent(1);

  int column = cell_id % width;
  int row = cell_id / width;

  if (row > 0 && column > 0 && row < height - 1 && column < width - 1) {
    float d2tdx2 =
        temp(row, column - 1) - 2 * temp(row, column) + temp(row, column + 1);
    float d2tdy2 =
        temp(row - 1, column) - 2 * temp(row, column) + temp(row + 1, column);

    return temp(row, column) + 0.2f * (d2tdx2 + d2tdy2);
  } else {
    return temp(row, column);
  }
}

template <class ContainerT>
void simulate(int height, int width, const ContainerT &in, ContainerT &out) {
  cuda::std::mdspan temp_in(thrust::raw_pointer_cast(in.data()), height, width);

  thrust::tabulate(thrust::device, out.begin(), out.end(), compute);
}

__global__ void grid_kernel(temperature_grid_f in, float *out) {
  int thread_index = blockDim.x * blockIdx.x + threadIdx.x;
  int number_of_threads = blockDim.x * gridDim.x;

  for (int id = thread_index; id < in.size(); id += number_of_threads) {
    out[id] = ach::compute(id, in);
  }
}

void simulate(ach::temperature_grid_f temp_in, float *temp_out,
              cudaStream_t stream) {
  int block_size = 1024;
  int grid_size = ceil_div(temp_in.size(), block_size);

  grid_kernel<<<grid_size, block_size, 0, stream>>>(temp_in, temp_out);
}
} // namespace ach
namespace heat {
inline thrust::universal_vector<float> generate_random_data(int height,
                                                            int width) {
  const float low = 15.0;
  const float high = 90.0;
  thrust::universal_vector<float> data(height * width, low);
  thrust::fill(thrust::device, data.begin(), data.begin() + width, high);
  thrust::fill(thrust::device, data.end() - width, data.end(), high);
  return data;
}

template <class ContainerT>
void simulate(int height, int width, const ContainerT &in, ContainerT &out) {
  cuda::std::mdspan temp_in(thrust::raw_pointer_cast(in.data()), height, width);

  thrust::tabulate(
      thrust::device, out.begin(), out.end(), [=] __host__ __device__(int id) {
        const int column = id % width;
        const int row = id / width;

        // loop over all points in domain (except boundary)
        if (row > 0 && column > 0 && row < height - 1 && column < width - 1) {
          // evaluate derivatives
          float d2tdx2 = temp_in(row, column - 1) - 2 * temp_in(row, column) +
                         temp_in(row, column + 1);
          float d2tdy2 = temp_in(row - 1, column) - 2 * temp_in(row, column) +
                         temp_in(row + 1, column);

          // update temperatures
          return temp_in(row, column) + 0.2f * (d2tdx2 + d2tdy2);
        } else {
          return temp_in(row, column);
        }
      });
}
} // namespace heat

#if CUB_VERSION >= 200800
#include <cub/device/device_transform.cuh>
#else
namespace thrust {

namespace system {
namespace cuda {
using universal_host_pinned_memory_resource = detail::pinned_memory_resource;
}
} // namespace system

template <typename T>
using universal_host_pinned_allocator =
    thrust::mr::stateless_resource_allocator<
        T, thrust::system::cuda::universal_host_pinned_memory_resource>;

// Should be part of CTK 12.8
template <typename T>
using universal_host_pinned_vector =
    thrust::detail::vector_base<T, universal_host_pinned_allocator<T>>;

} // namespace thrust

// TODO Will be part of CTK 12.8
namespace cub {
struct DeviceTransform {
  template <class InputIt, class OutputIt, class UnaryOperation>
  static cudaError_t Transform(InputIt in, OutputIt out, int num_items,
                               UnaryOperation op, cudaStream_t stream = 0) {
    return cub::DeviceFor::Bulk(num_items,
                                [=] __device__(int i) { out[i] = op(in[i]); });
  }
};
} // namespace cub
#endif