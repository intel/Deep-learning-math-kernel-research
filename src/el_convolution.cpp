#include <stdio.h>
#include <assert.h>
#include "euler.hpp"
#include "el_utils.hpp"
#include "elk_conv.hpp"

namespace euler {

template<typename T>
int eld_conv_setup(eld_conv_t &desc)
{
    elx_conv_t &x = desc.x;

    // Dimensions
    if (desc.dims.input.c != desc.dims.weights.i ||
        desc.dims.input.n  != desc.dims.output.n ||
        desc.dims.output.c != desc.dims.weights.o) {
        eld_error("Dimension error");
        return ELD_GENERAL_ERROR;
    }
    x.n  = desc.dims.input.n;
    x.ic = desc.dims.input.c;
    x.oc = desc.dims.output.c;
    x.ih = desc.dims.input.h;
    x.iw = desc.dims.input.w;
    x.oh = desc.dims.output.h;
    x.ow = desc.dims.output.w;
    x.kh = desc.dims.weights.h;
    x.kw = desc.dims.weights.w;
    // TODO:Check output dimensions

    // Formats
    // TODO:Check formats
    if (desc.formats.input != nChw16c ||
        desc.formats.output != nChw16c ||
        desc.formats.weights != OIhw16i16o) {
        eld_error("Unimplemented");
        return ELD_UNIMPLEMENTED;
    }
    // TODO:Check CPUID
    x.v = 16; // avx512

    printf("desc.x.oh=%d\n", desc.x.oh);

    // Direct
    if (desc.algorithm == CONV_DIRECT) {
        eld_error("Unimplemented");
        return ELD_UNIMPLEMENTED;
    }

    // Winograd
    if (desc.algorithm == CONV_WINOGRAD) {
        if (desc.dilations.h > 1 ||
            desc.dilations.w > 1 ||
            desc.tile_size != 5 ||
            desc.strides.h != 1 ||
            desc.strides.w != 1 ||
            desc.dims.weights.h != 3 ||
            desc.dims.weights.w != 3) {
            eld_error("Unimplemented");
            return ELD_UNIMPLEMENTED;
        }
        x.t = desc.tile_size;
        x.ot = desc.tile_size - 2;

        x.twei = (float *)malloc(3 * 3 * 16 * 16 * 4);
    }

    // Winograd
    return ELD_OK;

}


template<typename T>
int elx_conv_winograd(eld_conv_t &desc, T *input, T *weights, T *output, T *bias)
{
    // Notation
    // ========
    //
    // Block:
    // v := vector-size (avx512:16)
    // t := tile-size
    //
    // Data:
    // i := input
    // o := output
    // k := kernel
    //
    // Dims:
    // n := batch-size
    // h := height
    // w := width
    // c := channel
    //
    // Combinations:
    // ih, iw, oh, ow, kh, kw, ic, oc
    // to, ti=t, vi=vo=v
    // I := ic/v
    // O := oc/v
    // H := h/t
    // W := w/t

    // Tasks breakdown: n.H.W.O
    // n * oh/to * ow/to * oc/v
    // i.e. each-task to compute a to*to*v (v of tile)

    // Algorithm:
    // 1. Trans-weights
    //    kh.kw.ic.oc => O.I.t.t.vi.vo
    // 2. Trans-input (per-task)
    //    t.t.vi => t.t.vi
    // 3. FMA (per-task)
    //    Loop over I. for each vi:
    //    t.t.vi * vi.t.t.vo => t.t.vo
    // 4. Trans-output (per-task)
    //    t.t.vo => oh.ow.vo

    // Trans-weights
    // Tasks breakdown: O.i
    // hwio -> Oitt16o
    // desc.info.twei
#if 0
    for (int o = 0; o < p.OC; ++o) {
        for (int i = 0; i < p.ic; ++i) {
        }
    }
#endif
    // weights -> twei


    elx_conv_t &x = desc.x;
    float wei[3][3][16][16];
    for (int i = 0; i < 3*3*16*16; i++) {
        ((float *)wei)[i] = i;
    }
    float (&twei)[5][5][16][16] =
        *reinterpret_cast<float (*)[5][5][16][16]>(x.twei);
    for (int i = 0; i < 100000; ++i) {
        //elk_trans_weights_ref(twei, wei);
        elk_trans_weights(twei, wei);
    }
    //for (int i = 0; i < 5*5*16*16; i++) {
    //    printf("%f\n", x.twei[i]);
    //}
}

template<typename T>
int elx_conv(eld_conv_t &desc, T *input, T *weights, T *output, T *bias)
{
    // Sanity check
    if (any_null(input, weights, output)
        || (desc.with_bias && bias == nullptr)) {
        elx_error("Parameter error");
        return ELX_GENERAL_ERROR;
    }

    if (desc.algorithm == CONV_DIRECT) {
        elx_error("Unimplemented");
        return ELX_UNIMPLEMENTED;

    } else {
        assert(desc.algorithm == CONV_WINOGRAD);
        __tstart(twei);
        elx_conv_winograd<T>(desc, input, weights, output, bias);
        __tend(twei);

    }
    return ELX_OK;
}

template int elx_conv<float>(eld_conv_t &desc,
    float *input, float *weights, float *output, float *bias);

}

#if 1
int main() {
    // EXAMPLE_CODE 
    //
    using namespace euler;
    // 1. allocate memory
    float *input, *weights, *output, *bias;
    input = (float *)1;
    weights = (float *)1;
    output = (float *)1;
    // allocate memory buffers

    // 2, create convolution desc
    // In C++14
#if __cplusplus > 201402L
    eld_conv_t desc = {
        .dims = {
            .input   = { 64, 224, 224, 3 },
            .weights = { 3, 3, 3, 128 },
            .output  = { 64, 224, 224, 128 },
        },
        .formats = { nhwc, hwio, nhwc },
        .pads      = { 1, 1, 1, 1 },
        .strides   = { 1, 1 },
        .algorithm = CONV_DIRECT,
        .with_relu = false,
        .with_bias = true
    };
#else
    // Or in C++11
    eld_conv_t desc;
    desc.dims = {
        .input   = { 64, 224, 224, 3 },
        .weights = { 3, 3, 3, 128 },
        .output  = { 64, 224, 224, 128 }
    };
    desc.formats = {
        .input   = nChw16c,
        .weights = OIhw16i16o,
        .output  = nChw16c
    },
    desc.pads      = { 1, 1, 1, 1 };
    desc.with_bias = true;
    desc.algorithm = CONV_WINOGRAD;
    desc.tile_size = 5;
    desc.with_relu = false;
#endif
    eld_conv_setup<float>(desc);

    // 3. execute convolution
    if (ELX_OK != elx_conv<float>(desc, input, weights, output, bias)) {
        // error
        printf("Error\n");
    }
}
#endif
