#include "euler.hpp"
#include <chrono>
#include <cxxabi.h>

#ifndef __ELT_UTILS_HPP__
#define __ELT_UTILS_HPP__

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<float, std::milli> Duration;

int test_elt_conv(int tile_size, int execution_mode, int pat_i, int pat_o,
                  int input_format, int weights_format, int output_format,
                  int blk_i, int blk_o, int blk_t, int mb, int streaming_input,
                  int streaming_weights, int streaming_output,
                  bool input_as_blocked, bool weights_as_blocked,
                  bool output_as_blocked, bool with_bias, bool with_relu);

#define time_start(name) Time::time_point __s##name = Time::now();
#define time_end(name, iterations, num_ops, deduction)                         \
  Time::time_point __e##name = Time::now();                                    \
  double ms = (Duration(__e##name - __s##name).count() - deduction) /          \
      iterations;                                                              \
  double tflops = num_ops / ms / 1e9;                                          \
  printf("%s: iterations=%d, ops=%ld, time=%.4f ms, tflops=%g\n", #name,       \
      iterations, (unsigned long)num_ops, ms, tflops);

// Test timing
#define TT(name, iters, perf, expr)                                            \
  do {                                                                         \
    time_start(name);                                                          \
    for (int i = 0; i < iters; i++) {                                          \
      (expr);                                                                  \
    }                                                                          \
    if (perf) {                                                                \
      time_end(name, iters, 0, 0);                                             \
    }                                                                          \
  } while (0)

#define MEMALIGN64(ptr, size) posix_memalign((void **)(ptr), 64, size)

#endif // __ELT_UTILS_HPP__
