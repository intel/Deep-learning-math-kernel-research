#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <tuple>
#include <omp.h>

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

namespace euler {

enum {
  ISA_GENERIC = 0,
  ISA_AVX2 = 256,
  ISA_COSIM_AVX2,
  ISA_SKX_AVX512 = 512,
  ISA_COSIM_AVX512,
};

#define PAGE_SIZE 4096

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
  // TarrayTypes: FP32/FP32_F16/FP32_F16O
  // GarrayTypes: FP32/FP32_F16/FP32_F16O/INT8_F32/INT8_F16
  using FP32 = ConvImplTypes<float, float, float, float, float>;
  using FP32_F16 = ConvImplTypes<float, short, short, float, float>;
  using FP32_F16O = ConvImplTypes<float, float, short, float, float>;
  using INT8_F16 = ConvImplTypes<uint8_t, int8_t, short, float, float>;
  using INT8_F32 = ConvImplTypes<uint8_t, int8_t, float, float, float>;
};

} // namespace euler
