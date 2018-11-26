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

template <typename... Types> struct WinoTypes {
  static_assert(sizeof...(Types) == 3,
      "Winograd impl. types: transformed-input-type, "
      "transformed-weights-type, output-type-for-transform, "
      "transform-opnd-type.");
  using TinputType = typename std::tuple_element<0, std::tuple<Types...>>::type;
  using TweightsType = typename std::tuple_element<1, std::tuple<Types...>>::type;
  using ToutputType = typename std::tuple_element<2, std::tuple<Types...>>::type;
  // Using in cse of TinputType = TweightsType = ToutputType
  using TarrayType = typename std::tuple_element<0, std::tuple<Types...>>::type;
  using TscaleType = typename std::tuple_element<2, std::tuple<Types...>>::type;
};

// Types of t-buffer (tinput/tweights/toutput) unquantized (if any).
namespace wino {
  using FP32 = WinoTypes<float, float, float>;
  using FP32_F16 = WinoTypes<float, float, short>;
  using FP16 = WinoTypes<short, short, short>;
}

template <typename... Types> struct IntITFTypes {
  static_assert(sizeof...(Types) == 4,
      "gemm interface. gemm-input/weights/output/bias itf data type");
  using ITFinputType = typename std::tuple_element<0, std::tuple<Types...>>::type;
  using ITFweightsType = typename std::tuple_element<1, std::tuple<Types...>>::type;
  using ITFoutputType = typename std::tuple_element<2, std::tuple<Types...>>::type;
  using ITFbiasType = typename std::tuple_element<3, std::tuple<Types...>>::type;
  using ITFscaleType = typename std::tuple_element<2, std::tuple<Types...>>::type;
};

namespace itf_gemm {
  using FP16 = IntITFTypes<short, short, short, short>;
  using FP32 = IntITFTypes<float, float, float, float>;
  using INT8_F16 = IntITFTypes<uint8_t, int8_t, short, float>;
  using INT8_F32 = IntITFTypes<uint8_t, int8_t, float, float>;
};

} // namespace euler
