#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cxxabi.h>
#include <chrono>
#include <algorithm>
#include "el_mdarray.hpp"
#include "el_def.hpp"
#include "euler.hpp"

#define _T(x) x
typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<float, std::milli> Duration;

#define __tstart(n) _T(Time::time_point __s##n = Time::now());
#define __tend(n)                                                              \
  _T(Time::time_point __e##n = Time::now());                                   \
  _T(printf("time: %s, th=%d, %.2f ms\n", #n, estl::current_thread_index(),    \
      Duration(__e##n - __s##n).count()));

#define iter_each(indx, lim) for (int indx = 0; indx < (lim); ++indx)
#define revs_each(indx, lim) for (int indx = lim -1; indx >=0; -- indx)
#define unroll_auto(indx, lim)                                                 \
  _Pragma(STRINGIFY(unroll)) for (int indx = 0; indx < (lim); ++indx)
#define unroll_for(indx, lim)                                                  \
  _Pragma(STRINGIFY(unroll(lim))) for (int indx = 0; indx < (lim); ++indx)
#define unroll_from_to(indx, from, to)                                         \
  _Pragma(STRINGIFY(unroll((to) - (from)))) for (int indx = (from);            \
                                                 indx < (to); ++indx)

#define MEMALIGN64(ptr, size) posix_memalign((void **)(ptr), 64, size)

// Note: 'align' must be power of 2
#define ALIGNUP(value, align) (((value) + (align) - 1) & ~((align) - 1))

#define SET_EPI32(s)                                                           \
  const __i<V> vindex = _mm<V>::set_epi32(15 * (s), 14 * (s), 13 * (s),        \
      12 * (s), 11 * (s), 10 * (s), 9 * (s), 8 * (s), 7 * (s), 6 * (s),        \
      5 * (s), 4 * (s), 3 * (s), 2 * (s), (s), 0);

namespace euler {

static inline size_t alignup(size_t v, size_t a) {
  return (v + a - 1) & ~(a - 1);
}

// convolution attributes indexes
enum {
  bias_idx = 0x1,         // with bias
  relu_idx = 0x2,         // fuse with relu
  ip_sum_idx = 0x4,       // fuse with in-place sum
  op_sum_idx = 0x8,       // fuse with out-of-place sum
  r_output_idx = 0x10,    // clear output
  s_output_idx = 0x20,    // streaming output
  c_output_idx = 0x40,    // convert and restore output for int8 gemm
  has_Ir_idx = 0x100,     // has Ir
  has_Or_idx = 0x200,     // has_Or
  fma_opt_idx = 0x400,    // fma optimization
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

static inline const char *format_to_string(int fmt) {
  switch (fmt) {
    case nhwc: return "nhwc";
    case nChw16c: return "nChw16c";
    case nchw: return "nchw";
    case nChw8c: return "nChw8c";
    case oihw: return "oihw";
    case hwio: return "hwio";
    case OIhw16i16o: return "OIhw16i16o";
    case OIhw8i8o: return "OIhw8i8o";
    case goihw: return "goihw";
    case ghwio: return "ghwio";
    case gOIhw16i16o: return "gOIhw16i16o";
    case gOIhw8i8o: return "gOIhw8i8o";
    default: return "unknown"; break;
  }
  return "unknown";
}

static inline const char *algorithm_to_string(int alg) {
  switch (alg) {
    case CONV_DIRECT_1X1: return "direct_conv_1x1";
    case CONV_DIRECT: return "direct_conv";
    case CONV_DIRECT_VMG: return "direct_conv_vmg";
    case CONV_WINOGRAD: return "winograd_conv";
    case DECONV_DIRECT: return "deconv";
    default: return "unknown"; break;
  }
  return "unknown";
}

static inline const char *datatype_to_string(int dt) {
  switch (dt) {
    case f32: return "f32";
    case f16: return "f16";
    case u8: return "u8";
    case s8: return "s8";
    case s32: return "s32";
    default: return "unknown";
  }
  return "unknown";
}

static inline const char *mt_runtime_to_string(int t) {
  switch (t) {
    case MT_RUNTIME_TBB: return "TBB";
    case MT_RUNTIME_OMP: return "OMP";
    default: return "unknown";
  }
  return "unknown";
}
} // euler
