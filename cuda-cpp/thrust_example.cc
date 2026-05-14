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

int main() {
  thrust::universal_vector<float> a(1 << 28);
  thrust::universal_vector<float> b(1 << 28);

  thrust::sequence(a.begin(), a.end());
  thrust::sequence(b.rbegin(), b.rend());

  auto start_naive = std::chrono::high_resolution_clock::now();
  max_change_naive(a, b);
  auto end_naive = std::chrono::high_resolution_clock::now();
  const double naive_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_naive -
                                                            start_naive)
          .count();
  std::printf("max change naive duration: %f ms\n", naive_duration);

  auto start_optm = std::chrono::high_resolution_clock::now();
  max_change_optm(a, b);
  auto end_optm = std::chrono::high_resolution_clock::now();
  const double optm_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_optm -
                                                            start_optm)
          .count();
  std::printf("max change optm duration: %f ms\n", optm_duration);

  std::printf("\n---------------------------------\n");

  const int ambient_temp = 20;
  const float k = 0.5;
  thrust::universal_vector<float> temp{22, 43, 30};
  decltype(temp) next(temp.size(), 0);
  for (int step = 0; step < 3; ++step) {
    thrust::transform(thrust::device, temp.begin(), temp.end(), next.begin(),
                      [=] __host__ __device__(float x) {
                        return x + k * (ambient_temp - x);
                      });
    std::printf("step %d: , variance: %f\n", step, variance(next));
    temp.swap(next);
  }
  std::printf("\n---------------------------------\n");
}