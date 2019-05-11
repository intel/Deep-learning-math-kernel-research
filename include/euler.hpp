#ifndef __EULER_HPP__
#define __EULER_HPP__

#include <stddef.h>
#include <float.h>
#include <tuple>

#define EULER_API __attribute__ ((visibility ("default")))

namespace euler {

template <typename... Types> struct ConvTypes {
  static_assert(sizeof...(Types) == 4,
      "Contolution types: input-type, weights-type, output-type, bias-type.");
  using InputType = typename std::tuple_element<0, std::tuple<Types...>>::type;
  using WeightsType = typename std::tuple_element<1, std::tuple<Types...>>::type;
  using OutputType = typename std::tuple_element<2, std::tuple<Types...>>::type;
  using BiasType = typename std::tuple_element<3, std::tuple<Types...>>::type;
};

namespace conv {
  using FP32 = ConvTypes<float, float, float, float>;
  using FP16 = ConvTypes<short, short, short, short>;
  using FP16O = ConvTypes<float, float, short, float>;
  using U8S8F32F32 = ConvTypes<uint8_t, int8_t, float, float>;
  using U8F32F32F32 = ConvTypes<uint8_t, float, float, float>;
  using U8F32U8F32 = ConvTypes<uint8_t, float, uint8_t, float>;
  using U8F32S8F32 = ConvTypes<uint8_t, float, int8_t, float>;
};

// Convolution algorithm
enum {
  CONV_AUTO = 0,
  CONV_DIRECT_1X1 = 1,
  CONV_DIRECT = 2,
  CONV_DIRECT_VMG = 3,
  CONV_WINOGRAD = 4,
  DECONV_DIRECT = 5
};

// Desc setup error
enum {
  ELD_OK = 0,
  ELD_UNIMPLEMENTED = 1,
  ELD_GENERAL_ERROR = 2
};

// Execution error
enum {
  ELX_OK = 0,
  ELX_UNIMPLEMENTED = 1,
  ELX_GENERAL_ERROR = 2
};

// Streaming hint
enum {
  STORE_DEFAULT = 0,
  STORE_NORMAL = 1,
  STORE_STREAMING = 2
};

enum : uint8_t {
  data_type_undef = 0,
  f32,
  f16,
  u8,
  s8,
  s32
};

// Data formats
enum {
  format_undef = 0,
  nchw,
  nhwc,
  nChw16c,
  nChw8c,
  oihw,
  hwio,
  OIhw16i16o,
  OIhw8i8o,
  goihw,
  ghwio,
  gOIhw16i16o,
  gOIhw8i8o
};

// Propagation kind
enum prop_kinds {
  forward_training,
  forward_inference,
  backward_data,
  backward_weights
};

typedef enum {
  FINE = 0,
  COARSE,
  CALIBRATED
} sampling_kind_t;

#define EL_NO_CALI (FLT_MAX)

struct elx_conv_t;

// Convolution desc
struct EULER_API eld_conv_t {
  // Conv parameters
  struct {
    int n, g, ic, oc, ih, iw, oh, ow, kh, kw;
  } dims;
  struct { int l, r, t, b; } pads;
  struct { int h, w; } strides;
  struct { int h, w; } dilations;

  // Data Type
  struct {
    union {
      struct { uint8_t input, weights, output, bias; };
      uint32_t flat;
    };
  } data_type;

  // Data layout supported:
  // - plain: nchw, oihw, nchw
  // - blocked: nChw16c, OIhw16i16o, nChw16c
  struct { int input, weights, output; } formats;

  // propagation kind
  int prop_kind;

  // Algorithm
  int algorithm; // CONV_DIRECT | CONV_WINOGRAD
  int tile_size; // for Winograd only

  bool with_relu;
  bool with_bias;
  bool with_ip_sum;
  bool with_op_sum;
  bool f16c_opt;
  bool is_inference;
  bool use_scratch_pad;
  bool disable_autoparam;

  // Performance:
  // Number of thread teams, number of threads per team
  int nthreads;
  // Execution mode
  int execution_mode;
  // Flatting/Blocking/Partition
  struct { int o, t; } flatting;
  struct { int i, o; } blocking;
  struct { int i, o; } partition;
  // Streaming hint: STORE_DEFAULT | STORE_NORMAL | STORE_STREAMING
  struct { int input, output; } streaming_hint;
  // Use blocked format internally for plain format
  struct { bool input, weights, output; } format_as_blocked;

  // quantization calibration coefficients
  // A_fp32 = scale * (A_quant - z)
  struct { float scale, z; } input_quant, wino_tinput_quant, output_quant, sum_quant;
  sampling_kind_t sampling_kind;

  void *scratch_pad;

  // Defaults
  eld_conv_t();
  ~eld_conv_t();
  eld_conv_t(const eld_conv_t&) = delete;
  eld_conv_t& operator=(const eld_conv_t&) = delete;
  int setup();

  // Auto computed by setup()
  struct { size_t input, weights, output, bias; } byte_sizes;
  struct { size_t input, weights, output, bias; } sizes;

  // Internal data used by elx
  elx_conv_t *xc;
};

// Convolution execution
int EULER_API elx_conv(eld_conv_t &desc, void *output, void *input, void *weights, void *bias);

}

#endif // __EULER_HPP__
