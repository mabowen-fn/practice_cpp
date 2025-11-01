#include <cstddef>

namespace centroid {
namespace kernels {

void centroid_scalar(const double* const* rows, std::size_t N, std::size_t D, double* result_out) {
  for (std::size_t j = 0; j < D; ++j) result_out[j] = 0.0;

  for (std::size_t i = 0; i < N; ++i) {
    const double* row = rows[i];
    for (std::size_t j = 0; j < D; ++j)
      result_out[j] += row[j];
  }

  const double invN = 1.0 / static_cast<double>(N);
  for (std::size_t j = 0; j < D; ++j)
    result_out[j] *= invN;
}

}
}
