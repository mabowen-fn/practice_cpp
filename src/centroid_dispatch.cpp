#include "centroid/cpu_features.hpp"
#include "centroid/centroid.hpp"
#include <vector>
#include <mutex>
#include <cassert>

namespace centroid {
namespace kernels {
  void centroid_scalar(const double* const* rows, std::size_t N, std::size_t D, double* result_out);
  void centroid_avx2  (const double* const* rows, std::size_t N, std::size_t D, double* result_out);
}

using KernelFn = void(*)(const double* const*, std::size_t, std::size_t, double*);
static KernelFn g_kernel = nullptr;
static const char* g_kernel_name = "uninitialized";
static std::once_flag g_once;

static void init_kernel_once() {
  if (centroid::cpu::has_avx2()) {
    g_kernel = &centroid::kernels::centroid_avx2;
    g_kernel_name = "avx2";
  } else {
    g_kernel = &centroid::kernels::centroid_scalar;
    g_kernel_name = "scalar";
  }
}

const char* selected_kernel_name() {
  std::call_once(g_once, init_kernel_once);
  return g_kernel_name;
}

void compute(const double* const* rows, std::size_t N, std::size_t D, double* result_out) {
  assert(rows && result_out);
  assert(N > 0 && D > 0);
  std::call_once(g_once, init_kernel_once);
  g_kernel(rows, N, D, result_out);
}

std::vector<double> compute(const std::vector<std::vector<double>>& vectors) {
  const std::size_t N = vectors.size();
  assert(N > 0);
  const std::size_t D = vectors[0].size();
  assert(D > 0);
  
  #ifndef NDEBUG
  for (const auto& r : vectors) assert(r.size() = D);
  #endif
  
  std::vector<const double*> row_ptrs; row_ptrs.reserve(N);
  for (const auto& r : vectors) row_ptrs.push_back(r.data());

  std::vector<double> out(D, 0.0);
  compute(row_ptrs.data(), N, D, out.data());
  return out;
}
}
