#include "centroid/centroid.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <random>
#include <iomanip>

using namespace std;
using namespace std::chrono;

vector<vector<double>> make_random_vectors(size_t N, size_t D) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  vector<vector<double>> data(N, vector<double>(D));
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < D; ++j)
      data[i][j] = dist(rng);
  return data;
}

template <typename Func>
double benchmark(Func&& f, int iterations = 5) {
  double total_ms = 0.0;
  for (int i = 0; i < iterations; ++i) {
    auto start = high_resolution_clock::now();
    f();
    auto end = high_resolution_clock::now();
    total_ms += duration<double, std::milli>(end - start).count();
  }
  return total_ms / iterations;
}

int main() {
  cout << "=== Centroid Benchmark ===\n";
  cout << "Kernel selected by runtime: " << centroid::selected_kernel_name() << "\n\n";
   vector<pair<size_t, size_t>> configs = {
     {100, 64}, {1000, 128}, {5000, 256}, {10000, 512}
   };

   cout << setw(8) << "N" << setw(8) << "D" << setw(20) << "Time (ms)" << endl;
   cout << string(40, '-') << endl;

   for (auto [N, D] : configs) {
      auto data = make_random_vectors(N, D);

      double ms = benchmark([&]() {
        volatile auto c = centroid::compute(data);
      });

      cout << setw(8) << N << setw(8) << D << setw(20) << fixed << setprecision(3) << ms << endl;
   }
   cout << "\nBenchmark complete.\n"; 
   return 0;
}
