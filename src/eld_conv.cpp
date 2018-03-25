#include <stdio.h>
#include <assert.h>
#include "euler.hpp"
#include "el_utils.hpp"
#include "elk_conv.hpp"

namespace euler {


eld_conv_t::eld_conv_t() {
    pads      = { 1, 1, 1, 1 };
    strides   = { 1, 1 };
    dilations = { 1, 1 };
    sizes     = { 0, 0, 0, 0 };
    algorithm = CONV_DIRECT;
    tile_size = 0;
    with_relu = false;
    with_bias = false;
}

template<typename T> int
eld_conv_t::setup()
{
    // Dimensions
    if (dims.input.c != dims.weights.i ||
        dims.input.n  != dims.output.n ||
        dims.output.c != dims.weights.o) {
        eld_error("Dimension error");
        return ELD_GENERAL_ERROR;
    }

    sizes.input   = accumulate(dims.input.n, dims.input.c,
                               dims.input.h, dims.input.w);
    sizes.weights = accumulate(dims.weights.o, dims.weights.i,
                               dims.weights.h, dims.weights.w);
    sizes.output  = accumulate(dims.output.n, dims.output.c,
                               dims.output.h, dims.output.w);
    sizes.bias    = dims.bias.c;

    byte_sizes.input   = sizeof(T) * sizes.input;
    byte_sizes.weights = sizeof(T) * sizes.weights;
    byte_sizes.output  = sizeof(T) * sizes.output;
    byte_sizes.bias    = sizeof(T) * sizes.bias;

    x.n  = dims.input.n;
    x.ic = dims.input.c;
    x.oc = dims.output.c;
    x.ih = dims.input.h;
    x.iw = dims.input.w;
    x.oh = dims.output.h;
    x.ow = dims.output.w;
    x.kh = dims.weights.h;
    x.kw = dims.weights.w;
    // TODO:Check CPUID
    x.v = 16; // avx512

    x.OC = x.oc / x.v;
    x.IC = x.ic / x.v;
    // TODO:Check output dimensions

    // Formats
    // TODO:Check formats
    if (formats.input != nChw16c ||
        formats.output != nChw16c ||
        formats.weights != OIhw16i16o) {
        eld_error("Unimplemented");
        return ELD_UNIMPLEMENTED;
    }

    // Direct
    if (algorithm == CONV_DIRECT) {
        eld_error("Unimplemented");
        return ELD_UNIMPLEMENTED;
    }

    // Winograd
    if (algorithm == CONV_WINOGRAD) {
        if (dilations.h > 1 ||
            dilations.w > 1 ||
            tile_size != 5 ||
            strides.h != 1 ||
            strides.w != 1 ||
            dims.weights.h != 3 ||
            dims.weights.w != 3) {
            eld_error("Unimplemented");
            return ELD_UNIMPLEMENTED;
        }
        x.t = tile_size;
        x.ot = tile_size - 2;

        int size = sizeof(float) * tile_size * tile_size * x.ic * x.oc;
        x.tr_weights = (float *)malloc(size);
    }

    // Winograd
    return ELD_OK;

}

template int eld_conv_t::setup<float>();

}
