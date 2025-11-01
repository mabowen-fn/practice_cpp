#include <cstddef>
#include <immintrin.h>

namespace centroid {
namespace kernels{

void centroid_avx2(const double* const* rows, std::size_t N, std::size_t D, double* result_out) {
  for (std::size_t j = 0; j < D; ++j) result_out[j] = 0.0;


  const std::size_t stride = 4;
  std::size_t j = 0;

  for (; j + stride <= D; j += stride) {
    __m256d acc = _mm256_setzero_pd();

    for (std::size_t i = 0; i < N; ++i) {
      __m256d v = _mm256_loadu_pd(&rows[i][j]);
      acc = _mm256_add_pd(acc, v);
    }
    _mm256_storeu_pd(&result_out[j], acc);
  }
  
  for (; j < D; ++j) {
    double s = 0.0;
    for (std::size_t i = 0; i < N; ++i) s += rows[i][j];
    result_out[j] = s;
  }

  const double invN = 1.0 / static_cast<double>(N);
  const __m256d invN4 = _mm256_set1_pd(invN);

  j = 0;
  for (; j + stride <= D; j += stride) {
    __m256d x = _mm256_loadu_pd(&result_out[j]);
    x = _mm256_mul_pd(x, invN4);
    _mm256_storeu_pd(&result_out[j], x);
  }
  for (; j < D; ++j) result_out[j] *= invN;
}

}
}
