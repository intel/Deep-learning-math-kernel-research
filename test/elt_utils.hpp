#include <cxxabi.h>
#include <chrono>
#include "euler.hpp"

#ifndef __ELT_UTILS_HPP__
#define __ELT_UTILS_HPP__

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<float, std::milli> Duration;

#define time_start(name) Time::time_point __s##name = Time::now();
#define time_end(name, iterations, num_ops)                                  \
  Time::time_point __e##name = Time::now();                                  \
  double ms = Duration(__e##name - __s##name).count() / iterations;          \
  double tflops = num_ops / ms / 1e9;                                        \
  printf("%s: %.2f ms, tflops=%g\n", #name, ms, tflops);

// Test timing
#define TT(name, iters, perf, expr)                                          \
  do {                                                                       \
    time_start(name);                                                        \
    for (int i = 0; i < iters; i++) {                                        \
      (expr);                                                                \
    }                                                                        \
    if (perf) {                                                              \
      time_end(name, iters, 0);                                              \
    }                                                                        \
  } while (0)

#define for_each(i, lim) for (int(i) = 0; (i) < (lim); ++(i))

#define MD(type, array, dims, ptr)                                           \
  type(&array) dims = *reinterpret_cast<type(*) dims>(ptr)

#define MEMALIGN64(ptr, size) posix_memalign((void **)ptr, 64, size);

#endif  // __ELT_UTILS_HPP__
