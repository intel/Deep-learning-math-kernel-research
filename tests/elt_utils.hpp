#pragma once
#include <chrono>
#include <cxxabi.h>
#include <iostream>
#include "euler.hpp"


typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<float, std::milli> Duration;

int test_elt_conv(int tile_size, int execution_mode, int pat_i, int pat_o,
                  int input_format, int weights_format, int output_format,
                  int blk_i, int blk_o, int blk_t, int mb,
                  int streaming_input, int streaming_output,
                  bool input_as_blocked, bool weights_as_blocked,
                  bool output_as_blocked, bool with_bias, bool with_relu);

// Test timing
#define TT(name, iters, perf, expr)                                            \
  do {                                                                         \
    test::timer timer;                                                         \
    timer.start();                                                             \
    for (int i = 0; i < iters; i++) {                                          \
      (expr);                                                                  \
    }                                                                          \
    if (perf) {                                                                \
      timer.stop();                                                            \
      timer.report_tflops(#name, iters, 0);                                    \
    }                                                                          \
  } while (0)

#define MEMALIGN64(ptr, size) posix_memalign((void **)(ptr), 64, size)

namespace euler {
namespace test {

  class timer {
public:
    timer(): duration_(0.0) { }
    void start() { start_ = Time::now(); }
    void stop()
    {
      Time::time_point end = Time::now();
      double d = Duration(end - start_).count();
      duration_ += d;
    }
    double duration() { return duration_; }
    void report_tflops(const char *name, size_t num_iters, size_t num_ops)
    {
      double ms = duration_ / num_iters;
      double tflops = num_ops / ms / 1e9;
      std::cout << name << ": num_iters=" << num_iters
                << ", num_ops=" << num_ops << ", ms=" << ms
                << ", tflops=" << tflops << "\n";
    }

private:
    Time::time_point start_;
    double duration_;
  };

  inline void error(const char *msg)
  {
    printf("Euler:test::Error: %s\n", msg);
    abort();
  }
}
}
