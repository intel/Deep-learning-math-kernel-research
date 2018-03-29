#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "euler.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"

namespace euler {

template<typename F>
eld_conv_t<F>::eld_conv_t() {
    pads      = { 1, 1, 1, 1 };
    strides   = { 1, 1 };
    dilations = { 1, 1 };
    sizes     = { 0, 0, 0, 0 };
    algorithm = CONV_DIRECT;
    tile_size = 0;
    with_relu = false;
    with_bias = false;
}

template<typename F>
eld_conv_t<F>::~eld_conv_t() {
    if (xc->tweights != nullptr) {
        delete xc->tweights;
    }
    if (xc != nullptr) {
        free(xc);
    }
}

template<typename F> int
eld_conv_t<F>::setup()
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

    byte_sizes.input   = sizeof(F) * sizes.input;
    byte_sizes.weights = sizeof(F) * sizes.weights;
    byte_sizes.output  = sizeof(F) * sizes.output;
    byte_sizes.bias    = sizeof(F) * sizes.bias;

    //xc = (elx_conv_t *)malloc(sizeof(elx_conv_t));
    xc = new elx_conv_t<F>();

    xc->n  = dims.input.n;
    xc->ic = dims.input.c;
    xc->oc = dims.output.c;
    xc->ih = dims.input.h;
    xc->iw = dims.input.w;
    xc->oh = dims.output.h;
    xc->ow = dims.output.w;
    xc->kh = dims.weights.h;
    xc->kw = dims.weights.w;
    // TODO:Check CPUID
    xc->V = 16; // avx512

    xc->ic2 = xc->ic / xc->V;
    xc->oc2 = xc->oc / xc->V;
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
        xc->T = tile_size;
        xc->OT = tile_size - 2;
        xc->ih2 = xc->ih / xc->OT; // TODO, padding, tail
        xc->iw2 = xc->iw / xc->OT; // TODO
        xc->oh2 = xc->oh / xc->OT;
        xc->ow2 = xc->ow / xc->OT;

        int size = sizeof(F) * tile_size * tile_size * xc->ic * xc->oc;
        xc->tweights = (float *)malloc(size);

        //trans_weights = template<> elk_trans_weights<float, xc->T, xc->kw>
        //    (elx_conv_t&, float [5][5][16][16], float [3][3][16][16]);
    }

    // Winograd
    return ELD_OK;

}

template eld_conv_t<float>::eld_conv_t();
template eld_conv_t<float>::~eld_conv_t();
template int eld_conv_t<float>::setup();

}
