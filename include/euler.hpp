#ifndef __EULER_HPP__
#define __EULER_HPP__

#include <stddef.h>
#include <tuple>

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
};

// Convolution algorithm
enum {
    CONV_AUTO = 0,
    CONV_DIRECT_1X1 = 1,
    CONV_DIRECT = 2,
    CONV_WINOGRAD = 3
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

// Data formats
enum formats{
    nchw,
    nhwc,
    nChw16c,
    oihw,
    hwio,
    OIhw16i16o
};

// Propagation kind
enum prop_kinds {
    forward_training,
    forward_inference,
    backward_data,
    backward_weights
};

template<typename UserTypes> struct elx_conv_t;

// Convolution desc
template<typename UserTypes> struct eld_conv_t {

    using InputType = typename UserTypes::InputType;
    using WeightsType = typename UserTypes::WeightsType;
    using OutputType = typename UserTypes::OutputType;
    using BiasType = typename UserTypes::BiasType;

    // Conv parameters
    struct {
        struct { int n, c, h, w; } input;
        struct { int o, i, h, w; } weights;
        struct { int n, c, h, w; } output;
        struct { int c;          } bias;
    } dims;
    struct { int l, r, t, b; } pads;
    struct { int h, w; } strides;
    struct { int h, w; } dilations;

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
    struct { int weights, input, output; } streaming_hint;
    // Use blocked format internally for plain format
    struct { bool input, weights, output; } format_as_blocked;

    // Defaults
    eld_conv_t();
    ~eld_conv_t();
    int setup();
    void preprocess(WeightsType *weights);
    void clflush();

    // Auto computed by setup()
    struct { size_t input, weights, output, bias; } byte_sizes;
    struct { size_t input, weights, output, bias; } sizes;

    // Internal data used by elx
    elx_conv_t<UserTypes> *xc;
};

template struct eld_conv_t<conv::FP32>;
template struct eld_conv_t<conv::FP16>;

// Convolution execution
template <typename UserTypes>
int elx_conv(eld_conv_t<UserTypes> &desc,
    typename UserTypes::OutputType *output,
    typename UserTypes::InputType *input,
    typename UserTypes::WeightsType *weights,
    typename UserTypes::BiasType *bias);
}

#endif // __EULER_HPP__
