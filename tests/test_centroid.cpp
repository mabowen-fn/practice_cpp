#include "centroid/centroid.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

int main() {
  std::vector<std::vector<double>> vectors = {
    {1,2,3,4,5,6,7,8},
    {2,3,4,5,6,7,8,9},
    {3,4,5,6,7,8,9,10} 
  };

  auto scalar = centroid::compute(vectors);

  std::vector<const double*> ptrs;
  for (auto& v : vectors) ptrs.push_back(v.data());
  std::vector<double> manual(vectors[0].size(), 0.0);
  centroid::compute(ptrs.data(), vectors.size(), vectors[0].size(), manual.data());

  for (size_t i = 0; i < scalar.size(); i++)
    assert(std::fabs(scalar[i] - manual[i]) < 1e - 9);

  std::cout << "✅ Test passed Successfully\n";
  return 0;
}
