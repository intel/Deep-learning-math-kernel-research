#ifndef __EULER_HPP__
#define __EULER_HPP__

#include <stddef.h>

namespace euler {

// Convolution algorithm
enum {
    CONV_DIRECT = 0,
    CONV_WINOGRAD = 1
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
enum {
    nchw,
    nhwc,
    nChw16c,
    oihw,
    hwio,
    OIhw16i16o
};

// Propagation kind
enum {
    forward_training,
    forward_inference,
    backward_data,
    backward_weights
};

template<typename T> struct elx_conv_t;

// Convolution desc
template<typename T>
struct eld_conv_t {
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
    bool is_inference;

    // Performance:
    // Number of thread teams, number of threads per team
    struct { int nteams, nthreads;} threading;
    // Execution mode
    int execution_mode;
    // Blocking: 2nd level blocking unit
    struct { int i, o, t; } blocking;
    struct { int i, o; } partition;
    // Streaming hint: STORE_DEFAULT | STORE_NORMAL | STORE_STREAMING
    struct { int weights, input, output; } streaming_hint;

    // Defaults
    eld_conv_t();
    ~eld_conv_t();
    int setup();

    // Auto computed by setup()
    struct { size_t input, weights, output, bias; } byte_sizes;
    struct { size_t input, weights, output, bias; } sizes;

    // Internal data used by elx
    elx_conv_t<T> *xc;
};

template struct eld_conv_t<float>;

// Convolution execution
template<typename T>
int elx_conv(eld_conv_t<T> &desc, T *output, T *input, T *weights, T *bias);

}

#endif // __EULER_HPP__
