#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cxxabi.h>
#include <chrono>
#include <algorithm>
#include "el_mdarray.hpp"

#define _T(x) x
typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<float, std::milli> Duration;

#define __tstart(n) _T(Time::time_point __s##n = Time::now());
#define __tend(n)                                                              \
  _T(Time::time_point __e##n = Time::now());                                   \
  _T(printf("time: %s, th=%d, %.2f ms\n", #n, omp_get_thread_num(),            \
      Duration(__e##n - __s##n).count()));

#define iter_each(indx, lim) for (int indx = 0; indx < (lim); ++indx)
#define revs_each(indx, lim) for (int indx = lim -1; indx >=0; -- indx)

#define MEMALIGN64(ptr, size) posix_memalign((void **)(ptr), 64, size)

// Note: 'align' must be power of 2
#define ALIGNUP(value, align) (((value) + (align) - 1) & ~((align) - 1))

#define SET_EPI32(s)                                                           \
  const __i<V> vindex = _mm<V>::set_epi32(15 * (s), 14 * (s), 13 * (s),        \
      12 * (s), 11 * (s), 10 * (s), 9 * (s), 8 * (s), 7 * (s), 6 * (s),        \
      5 * (s), 4 * (s), 3 * (s), 2 * (s), (s), 0);

namespace euler {

// convolution attributes indexes
enum {
  bias_idx = 0x1,         // with bias
  relu_idx = 0x2,         // fuse with relu
  ip_sum_idx = 0x4,       // fuse with in-place sum
  op_sum_idx = 0x8,       // fuse with out-of-place sum
  r_output_idx = 0x10,    // clear output
  s_output_idx = 0x20,    // streaming output
};

inline int set_attr(int attr, int index) {
  return attr | index;
}

inline bool get_attr(int attr, int index) {
  return (attr & index) != 0;
}

inline void el_error(const char *msg) {
  printf("Euler:Error: %s\n", msg);
  abort();
}

inline void el_warn(const char *msg) {
  printf("Euler:Warning: %s\n", msg);
}

}
