#pragma once
#include <chrono>
#include <cxxabi.h>
#include <iostream>
#include "euler.hpp"


namespace euler {
namespace test {

  class timer {
    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::duration<float, std::milli> Duration;

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
    void report_tflops(std::string& name, size_t num_iters, size_t num_ops)
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
