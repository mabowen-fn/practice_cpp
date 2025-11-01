#include "centroid/centroid.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

bool nearly_equal(double a, double b, double eps = 1e-12) {
  return fabs(a - b) <= eps;
}

int main() {
  std::cout << "Running Centroid Tests ... \n";

  std::vector<std::vector<double>> small = {
    {1,2,3,4,5,6,7,8},
    {2,3,4,5,6,7,8,9},
    {3,4,5,6,7,8,9,10} 
  };

  auto c1 = centroid::compute(small);

  std::vector<const double*> ptrs;
  for (auto& v : small) ptrs.push_back(v.data());

  std::vector<double> c2(small[0].size());

  centroid::compute(ptrs.data(), small.size(), small[0].size(), c2.data());

  for (size_t i = 0; i < c1.size(); i++)
    assert(nearly_equal(c1[i], c2[i]));

  std::cout << "✅ Small dataset passed\n";

  const size_t N = 1000, D = 128;
  std::vector<std::vector<double>> large(N, std::vector<double>(D));
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < D; ++j)
      large[i][j] = static_cast<double>((i + j) % 50);

  auto c3 = centroid::compute(large);
  assert(c3.size() == D);
  

  std::cout << "✅ Large dataset passed\n";
  std::cout << "All tests passed successfully";
  return 0;
}
