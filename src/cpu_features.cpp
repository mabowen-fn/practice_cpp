#include "centroid/cpu_features.hpp"

#if defined(__GNUC__) || defined(__clang__)
  #include <cstring>
#endif

namespace centroid {
namespace cpu {

bool has_avx2() {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_cpu_supports("avx2");
#else
  return false;
#endif
}

}
}
