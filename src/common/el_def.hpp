#pragma once

#include <cstdint>
#include <tuple>

#define MT_RUNTIME_OMP (1)
#define MT_RUNTIME_TBB (2)

namespace euler {

enum {
  ISA_GENERIC = 0,
  ISA_AVX2 = 256,
  ISA_AVX512 = 512
};

enum {
  __ALL = 0,
  __TRACE = 1,
  __DEBUG = 2,
  __INFO = 3,
  __WARN = 4,
  __ERROR = 5,
  __FATAL = 6,
  __PERF_TRACE = 999, // ensure EULER_VERBOSE always work once set
};

constexpr size_t PAGE_SIZE = 4096;

// int8 quantization
constexpr float EL_INT8_MAX = 127.0f;
constexpr float EL_UINT8_MAX = 255.0f;

// wino fusion
constexpr uint32_t FUS_MASK = 0xF0;
constexpr uint32_t FUS_I = 0x10;
constexpr uint32_t FUS_O = 0x20;

// kernel attr
constexpr uint32_t AT_BIAS_MASK             { 1 << 0 };
constexpr uint32_t AT_RELU_MASK             { 1 << 1 };
constexpr uint32_t AT_INP_SUM_MASK          { 1 << 2 }; // inplace sum
constexpr uint32_t AT_CLEAR_OUTPUT_MASK     { 1 << 3 };
constexpr uint32_t AT_STREAMING_OUTPUT_MASK { 1 << 4 };
constexpr uint32_t AT_RESTORE_OUTPUT_MASK   { 1 << 5 };
constexpr uint32_t AT_Ir_MASK               { 1 << 6 };
constexpr uint32_t AT_Or_MASK               { 1 << 7 };
constexpr uint32_t AT_FMAOPT_MASK           { 1 << 8 }; // FMA optimization

template <typename... Types> struct ConvImplTypes {
  static_assert(sizeof...(Types) == 4,
      "Inner types input/weights/output/bias scale data type");
  using InputType = typename std::tuple_element<0, std::tuple<Types...>>::type;
  using WeightsType = typename std::tuple_element<1, std::tuple<Types...>>::type;
  using OutputType = typename std::tuple_element<2, std::tuple<Types...>>::type;
  using BiasType = typename std::tuple_element<3, std::tuple<Types...>>::type;
};

namespace conv_impl {
  using FP32 = ConvImplTypes<float, float, float, float>;
  using FP32_F16b = ConvImplTypes<float, float, float, short>;
  using FP32_F16w = ConvImplTypes<float, short, float, float>;
  using FP32_F16o = ConvImplTypes<float, float, short, float>;
  using FP32_F16iwo = ConvImplTypes<short, short, short, float>;
  using FP32_F16wob = ConvImplTypes<float, short, short, short>;
  using INT8_F32 = ConvImplTypes<uint8_t, int8_t, float, float>;
  using INT8_F16b = ConvImplTypes<uint8_t, int8_t, float, short>;
  using INT8_F16o = ConvImplTypes<uint8_t, int8_t, short, float>;
  using INT8_F16ob = ConvImplTypes<uint8_t, int8_t, short, short>;
  using INT8_INT8o = ConvImplTypes<uint8_t, int8_t, int8_t, float>;
  using INT8_UINT8o = ConvImplTypes<uint8_t, int8_t, uint8_t, float>;
};

} // namespace euler
