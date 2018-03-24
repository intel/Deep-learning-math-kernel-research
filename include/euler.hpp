#ifndef __EULER_HPP__

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
    nhwc,
    nChw16c,
    hwio,
    OIhw16i16o
};

struct elx_conv_t {
    int n, ih, iw, ic, oc, kh, kw, oh, ow, IC, OC;
    int v, t, ot; // {vector, tile, out-tile}-size
    int lpad, rpad, tpad, bpad;
    float *tr_weights;
};

// Convolution desc
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
    eld_conv_t ();
    template<typename T> int setup();

    // Auto computed
    struct { int input, weights, output, bias; } sizes;

    // Internal
    elx_conv_t x; // TODO: hide the structure
};

// Convolution desc setup
template<typename T>
int eld_conv_setup(eld_conv_t &desc);

// Convolution execution
template<typename T>
int elx_conv(eld_conv_t &desc, T *input, T *weights, T *output, T *bias);

}

#endif // __EULER_HPP__
