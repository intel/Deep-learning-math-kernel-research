#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cxxabi.h>
#include <chrono>
#include <algorithm>
#include <stdarg.h>
#include "el_mdarray.hpp"
#include "el_def.hpp"
#include "el_log.hpp"
#include "euler.hpp"

// Compiler
#define __GCC_COMPILER (__GNUC__ && !__INTEL_COMPILER && !__clang__)
#define __ICC_COMPILER __INTEL_COMPILER
#define __CLANG_COMPILER __clang__

// Loop unrolling
#if __ICC_COMPILER
#define ENABLE_AVX512F() _allow_cpu_features(_FEATURE_AVX512F)
#define pragma_opt_core_avx512                                                 \
  _Pragma("optimization_parameter target_arch=CORE-AVX512")
#define pragma_unroll _Pragma("unroll")
#define pragma_inline _Pragma("forceinline recursive")
#else
#define ENABLE_AVX512F() // -mavx512f
#define pragma_opt_core_avx512
#define pragma_unroll
#define pragma_inline
#endif

// Loop
#define iter_each(indx, lim) for (int indx = 0; indx < (lim); ++indx)
#define revs_each(indx, lim) for (int indx = lim -1; indx >=0; -- indx)
#define unroll_auto(indx, lim)                                                 \
  _Pragma(STRINGIFY(unroll)) for (int indx = 0; indx < (lim); ++indx)
#define unroll_for(indx, lim)                                                  \
  _Pragma(STRINGIFY(unroll(lim))) for (int indx = 0; indx < (lim); ++indx)
#define unroll_from_to(indx, from, to)                                         \
  _Pragma(STRINGIFY(unroll((to) - (from)))) for (int indx = (from);            \
                                                 indx < (to); ++indx)
// Timing
#define _(x) x
#define __tstart(n) _(std::chrono::high_resolution_clock::time_point __s##n =  \
                      std::chrono::high_resolution_clock::now());
#define __tend(n)                                                              \
  _(std::chrono::high_resolution_clock::time_point __e##n =                    \
    std::chrono::high_resolution_clock::now());                                \
  _(el_log(__DEBUG, "time: %s, th=%d, %.2f ms", #n,                            \
           estl::current_thread_index(),                                       \
           std::chrono::duration<float, std::milli>(__e##n - __s##n).count()));

// misc.
#define STRINGIFY(x) #x
#define XSTRINGIFY(x) STRINGIFY(x)
// Note: 'align' must be power of 2
#define ALIGNUP(value, align) (((value) + (align) - 1) & ~((align) - 1))

// For gather/scatter index initialization
#define SET_VINDEX_16(stride)                                                  \
  15 * (stride), 14 * (stride), 13 * (stride), 12 * (stride),                  \
  11 * (stride), 10 * (stride),  9 * (stride),  8 * (stride),                  \
   7 * (stride),  6 * (stride),  5 * (stride),  4 * (stride),                  \
   3 * (stride),  2 * (stride),  1 * (stride),  0

namespace euler {

static inline size_t alignup(size_t v, size_t a) {
  return (v + a - 1) & ~(a - 1);
}

template <typename T>
static inline int memalign64(T **ptr, size_t size) {
  return posix_memalign((void **)ptr, 64, size);
}

static inline uint32_t set_bit(const uint32_t v, const uint32_t mask) {
  return v | mask;
}

static inline bool test_bit(const uint32_t v, const uint32_t mask) {
  return v & mask;
}

static inline uint32_t clear_bit(const uint32_t v, const uint32_t mask) {
  return v & ~mask;
}

static inline const char *log_severity_to_string(int severity) {
  switch (severity) {
  case __TRACE: return "trace";
  case __DEBUG: return "debug";
  case __INFO: return "info";
  case __WARN: return "warn";
  case __ERROR: return "error";
  case __FATAL: return "fatal";
  case __PERF_TRACE: return "perf";
  default: return "log-severity-unknown";
  }
  return "unknown";
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
    default: return "format-unknown";
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
    default: return "algorithm-unknown";
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
    default: return "datatype-unknown";
  }
  return "unknown";
}

static inline const char *mt_runtime_to_string(int t) {
  switch (t) {
    case MT_RUNTIME_TBB: return "TBB";
    case MT_RUNTIME_OMP: return "OMP";
    default: return "mt-runtime-unknown";
  }
  return "unknown";
}
} // euler
