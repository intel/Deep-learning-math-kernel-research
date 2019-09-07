#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <tuple>

#define __GCC_COMPILER (__GNUC__ && !__INTEL_COMPILER && !__clang__)
#define __ICC_COMPILER __INTEL_COMPILER
#define __CLANG_COMPILER __clang__

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

#define STRINGIFY(x) #x
#define XSTRINGIFY(x) STRINGIFY(x)

#define MT_RUNTIME_OMP (1)
#define MT_RUNTIME_TBB (2)

namespace euler {

#define PAGE_SIZE 4096

enum {
  ISA_GENERIC = 0,
  ISA_AVX2 = 256,
  ISA_COSIM_AVX2,
  ISA_SKX_AVX512 = 512,
  ISA_COSIM_AVX512,
};

const unsigned FUS_MSK = 0xF0;
const unsigned FUS_I   = 0x10;
const unsigned FUS_O   = 0x20;

template <typename... Types> struct ConvImplTypes {
  static_assert(sizeof...(Types) == 5,
      "Inner types input/weights/output/bias scale data type");
  using InputType = typename std::tuple_element<0, std::tuple<Types...>>::type;
  using WeightsType = typename std::tuple_element<1, std::tuple<Types...>>::type;
  using OutputType = typename std::tuple_element<2, std::tuple<Types...>>::type;
  using BiasType = typename std::tuple_element<3, std::tuple<Types...>>::type;
  using ScaleType = typename std::tuple_element<4, std::tuple<Types...>>::type;
};

namespace conv_impl {
  using FP32 = ConvImplTypes<float, float, float, float, float>;
  using FP32_F16b = ConvImplTypes<float, float, float, short, float>;
  using FP32_F16w = ConvImplTypes<float, short, float, float, float>;
  using FP32_F16o = ConvImplTypes<float, float, short, float, float>;
  using FP32_F16iwo = ConvImplTypes<short, short, short, float, float>;
  using FP32_F16wob = ConvImplTypes<float, short, short, short, float>;
  using INT8_F32 = ConvImplTypes<uint8_t, int8_t, float, float, float>;
  using INT8_F16b = ConvImplTypes<uint8_t, int8_t, float, short, float>;
  using INT8_F16o = ConvImplTypes<uint8_t, int8_t, short, float, float>;
  using INT8_F16ob = ConvImplTypes<uint8_t, int8_t, short, short, float>;
  using INT8_INT8o = ConvImplTypes<uint8_t, int8_t, int8_t, float, float>;
  using INT8_UINT8o = ConvImplTypes<uint8_t, int8_t, uint8_t, float, float>;
};

} // namespace euler
