#pragma omp parallel for schedule(static)
for (std::size_t i = 0; i < N; ++i) {
  const double* row = rows[i];
  for (std::size_t j = 0; j < D; ++j) {
    #pragma omp atomic
    result_out[j] += row[j];
  }
}
