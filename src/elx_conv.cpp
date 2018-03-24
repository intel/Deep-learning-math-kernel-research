#include <stdio.h>
#include <assert.h>
#include "euler.hpp"
#include "el_utils.hpp"
#include "elk_conv.hpp"

namespace euler {

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

template int elx_conv<float>(eld_conv_t &desc, float *input, float *weights,
                             float *output, float *bias);

}
