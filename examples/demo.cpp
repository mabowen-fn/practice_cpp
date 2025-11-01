#include "centroid/centroid.hpp"
#include <iostream>
#include <iomanip>
#include <vector>

int main() {
  std::vector<std::vector<double>> vectors = {
    {1,2,3,4,5,6,7,8,9,10},
    {2,3,4,5,6,7,8,9,10,11},
    {3,4,5,6,7,8,9,10,11,12}
  };

  auto c = centroid::compute(vectors);


  std::cout << "Kernel: " << centroid::selected_kernel_name() << std::endl;
  std::cout << "Centroid: [ ";
  for (double x : c) std::cout << std::fixed << std::setprecision(2) << x << " ";
  std::cout << "]\n";
  return 0;
}
