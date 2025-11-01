#include <cstddef>
#include <immintrin.h>
#include <algorithm> // std::min

namespace centroid {
namespace kernels {

// Tunable: rows per cache block (64–256 are good starting points)
static constexpr std::size_t BLOCK_ROWS = 128;

// AVX2 + blocking + (optional) OpenMP parallelization
// Compile unit is built with -mavx2 -mfma by CMake.
void centroid_avx2(const double* const* rows, std::size_t N, std::size_t D, double* result_out) {
    const std::size_t stride = 4;
    const double invN = 1.0 / static_cast<double>(N);
    const __m256d invN4 = _mm256_set1_pd(invN);

    // SIMD region: process 4 columns at a time.
    // Parallelize over column chunks so each thread writes to distinct
    // parts of result_out (no atomics / no false sharing).
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (std::size_t j = 0; j + stride <= D; j += stride) {
        __m256d sum4 = _mm256_setzero_pd();

        // Block rows for better cache locality
        for (std::size_t i0 = 0; i0 < N; i0 += BLOCK_ROWS) {
            const std::size_t i1 = std::min(N, i0 + BLOCK_ROWS);

            __m256d acc4 = _mm256_setzero_pd();
            for (std::size_t i = i0; i < i1; ++i) {
                // rows[i] is row-major; &rows[i][j] is contiguous for 4 doubles
                __m256d v = _mm256_loadu_pd(&rows[i][j]);
                acc4 = _mm256_add_pd(acc4, v);
            }
            sum4 = _mm256_add_pd(sum4, acc4);
        }

        // Scale and store
        sum4 = _mm256_mul_pd(sum4, invN4);
        _mm256_storeu_pd(&result_out[j], sum4);
    }

    // Tail columns (D not divisible by 4) — scalar but cache-friendly
    const std::size_t j_tail = (D / stride) * stride;
    for (std::size_t j = j_tail; j < D; ++j) {
        double s = 0.0;
        // Blocking helps even in scalar tail
        for (std::size_t i0 = 0; i0 < N; i0 += BLOCK_ROWS) {
            const std::size_t i1 = std::min(N, i0 + BLOCK_ROWS);
            for (std::size_t i = i0; i < i1; ++i) {
                s += rows[i][j];
            }
        }
        result_out[j] = s * invN;
    }
}

}
}

