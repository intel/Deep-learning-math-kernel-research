#include <stdio.h>
#include <assert.h>
#include "euler.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"

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

eld_conv_t::~eld_conv_t() {
    if (xc->tr_weights != nullptr) {
        free(xc->tr_weights);
    }
    if (xc != nullptr) {
        free(xc);
    }
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

    xc = (elx_conv_t *)malloc(sizeof(elx_conv_t));

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

    xc->OC = xc->oc / xc->V;
    xc->IC = xc->ic / xc->V;
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
        xc->To = tile_size - 2;
        xc->IH = xc->ih / xc->To; // TODO, padding, tail
        xc->IW = xc->iw / xc->To; // TODO
        xc->OH = xc->oh / xc->To;
        xc->OW = xc->ow / xc->To;

        int size = sizeof(float) * tile_size * tile_size * xc->ic * xc->oc;
        xc->tr_weights = (float *)malloc(size);
    }

    // Winograd
    return ELD_OK;

}

template int eld_conv_t::setup<float>();

}
