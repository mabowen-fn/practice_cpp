## Somehow the scalar is faster than the avx2
Obviously we can benefit from having
  * contiguous column-major blocks
  * blocking. accomulate 64 rows at a time maybe.
  * loop tiling
  * parallelizing over rows
  * maybe the compiler does a better job than me auto vectoring

=== Centroid Benchmark ===
Kernel selected by runtime: avx2

       N       D           Time (ms)
----------------------------------------
     100      64               0.003
    1000     128               0.071
    5000     256               3.448
   10000     512              12.653

Benchmark complete.
./bench_centroid  0.17s user 0.03s system 11% cpu 1.733 total

=== Centroid Benchmark ===
Kernel selected by runtime: scalar

       N       D           Time (ms)
----------------------------------------
     100      64               0.006
    1000     128               0.240
    5000     256               0.693
   10000     512               3.124

Benchmark complete.
./bench_centroid  0.11s user 0.03s system 70% cpu 0.191 total


somehow the 
