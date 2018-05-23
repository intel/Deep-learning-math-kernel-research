#ifndef __EULER_HPP__
#define __EULER_HPP__

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

enum {
    nchw,
    nhwc,
    nChw16c,
    oihw,
    hwio,
    OIhw16i16o
};

template<typename T> struct elx_conv_t;

// Convolution desc
template<typename T>
struct eld_conv_t {
    // Conv parameters
    struct {
        struct { int n, h, w, c; } input;
        struct { int h, w, i, o; } weights;
        struct { int n, h, w, c; } output;
        struct { int c;          } bias;
    } dims;
    struct { int l, r, t, b; } pads;
    struct { int h, w; } strides;
    struct { int h, w; } dilations;

    // Data layout
    struct { int input, weights, output; } formats;

    // Algorithm
    int algorithm; // CONV_DIRECT | CONV_WINOGRAD
    int tile_size; // for Winograd only

    bool with_relu;
    bool with_bias;

    // Defaults
    eld_conv_t();
    ~eld_conv_t();
    int setup();

    // Auto computed by setup()
    struct { int input, weights, output, bias; } byte_sizes;
    struct { int input, weights, output, bias; } sizes;

    // Internal data used by elx
    elx_conv_t<T> *xc;
};

template struct eld_conv_t<float>;

// Convolution execution
template<typename T>
int elx_conv(eld_conv_t<T> &desc, T *output, T *input, T *weights, T *bias);

}

#endif // __EULER_HPP__
