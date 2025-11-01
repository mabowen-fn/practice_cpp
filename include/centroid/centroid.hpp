#pragma once
#include <vector>
#include <cstddef>


namespace centroid {
std::vector<double>
compute(const std::vector<std::vector<double>>& vectors);

void
compute(const double* const* rows, std::size_t N, std::size_t D, double* result_out);

// exposes which kernel is currently selected
const char* selected_kernel_name();

}
