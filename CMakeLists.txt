cmake_minimum_required(VERSION 3.10)
project(CentroidSIMD LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Library target
add_library(centroid STATIC
  src/centroid_dispatch.cpp
  src/centroid_scalar.cpp
  src/centroid_avx2.cpp
  src/cpu_features.cpp
)

target_include_directories(centroid PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Compile the AVX2 file with AVX2 flag only. (portable & safe)
set_source_files_properties(src/centroid_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
add_compile_options(-O3)

# Example program
add_executable(demo examples/demo.cpp)
target_link_libraries(demo PRIVATE centroid)

# Test Programs
add_executable(test_centroid tests/test_centroid.cpp)
target_link_libraries(test_centroid PRIVATE centroid)
