#include "ach.h"

void simulate(int height, int width, const thrust::universal_vector<float> &in,
              thrust::universal_vector<float> &out) {
  const float *in_ptr = thrust::raw_pointer_cast(in.data());

  auto cell_indices = thrust::make_counting_iterator(0);
  thrust::transform(
      thrust::device, cell_indices, cell_indices + in.size(), out.begin(),
      [in_ptr, height, width] __host__ __device__(int id) {
        int column = id % width;
        int row = id / width;

        if (row > 0 && column > 0 && row < height - 1 && column < width - 1) {
          float d2tdx2 = in_ptr[(row)*width + column - 1] -
                         2 * in_ptr[row * width + column] +
                         in_ptr[(row)*width + column + 1];
          float d2tdy2 = in_ptr[(row - 1) * width + column] -
                         2 * in_ptr[row * width + column] +
                         in_ptr[(row + 1) * width + column];

          return in_ptr[row * width + column] + 0.2f * (d2tdx2 + d2tdy2);
        } else {
          return in_ptr[row * width + column];
        }
      });
}

int main() {
  int height = 64, width = 128;
  auto in = heat::generate_random_data(height, width);
  auto out = in;
  for (int step = 0; step < 100; step++) {
    simulate(height, width, in, out);
    thrust::swap(in, out);

    char filename[64];
    std::snprintf(filename, sizeof(filename), "/tmp/heat_%d.bin", step);
    FILE *f = std::fopen(filename, "wb");
    std::fwrite(&height, sizeof(int), 1, f);
    std::fwrite(&width, sizeof(int), 1, f);
    std::fwrite(thrust::raw_pointer_cast(in.data()), sizeof(float), height * width, f);
    std::fclose(f);
  }
}