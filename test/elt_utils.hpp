#include <cxxabi.h>
#include <chrono>

#ifndef __ELT_UTILS_HPP__
#define __ELT_UTILS_HPP__

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<float, std::milli> Duration;

#define time_start(name) Time::time_point __s##name = Time::now();
#define time_end(name, iterations)          \
  Time::time_point __e##name = Time::now(); \
  printf("%s: %.2f ms\n", #name,            \
         Duration(__e##name - __s##name).count() / iterations);

// Test timing
#define TT(name, iters, perf, expr)   \
  do {                                \
    time_start(name);                 \
    for (int i = 0; i < iters; i++) { \
      (expr);                         \
    }                                 \
    if (perf) {                       \
      time_end(name, iters);          \
    }                                 \
  } while (0)

namespace euler {}

#endif  // __ELT_UTILS_HPP__
