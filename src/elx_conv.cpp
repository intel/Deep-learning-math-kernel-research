#include <stdio.h>
#include <assert.h>
#include "euler.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv.hpp"

namespace euler {


int elk_trans_weights(elx_conv_t &xc, float *tr_weights, float *weights);

template<typename F, const int T>
int elx_trans_weights(elx_conv_t &xc, F *tr_weights, F *weights)
{
    MD(F, from, [xc.OC][xc.IC][T-2][T-2][16][16], weights);
    MD(F, to,   [xc.OC][xc.IC][T][T][16][16], tr_weights);

#pragma omp parallel for collapse(2)
    for (int o = 0; o < xc.OC; ++o) {
        for (int i = 0; i < xc.IC; ++i) {
            elk_trans_weights(to[o][i], from[o][i]);
        }
    }
    return 0;
}

template<typename F, const int T>
int elx_conv_winograd(elx_conv_t &xc, F *input, F *weights, F *output, F *bias)
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

    // weights -> tr_weights
    elx_trans_weights<F, T>(xc, xc.tr_weights, weights);
    int V = xc.V;
    MD(F, ainput, [xc.n][xc.IC][xc.ih][xc.iw][16], input);

    for (int n = 0; n < xc.n; ++n)     {
    for (int oh = 0; oh < xc.OH; ++oh) {
    for (int ow = 0; ow < xc.OW; ++ow) {
    for (int oc = 0; oc < xc.OC; ++oc) {
        F tr_output[T][T][V];
        for (int ic = 0; ic < xc.IC; ++ic) {
            // input -> tr_input
            F tr_input[T][T][V];
            elk_trans_input<F, T>(xc, tr_input, (F *)ainput[n][ic], oh, ow);
            // gemm
            //elk_gemm(xc, tr_output, tr_input, tr_weights[oc][:]);
        }
        // tr_output -> output
        //elk_trans_output(xc, output, tr_output);
    }}}}
#if 0
    for (int i = 0; i < x.ic * x.oc * 25; ++i) {
        printf("%f\n", x.tr_weights[i]);
    }
#endif
    return 0;
}

template<typename F>
int elx_conv(eld_conv_t &desc, F *input, F *weights, F *output, F *bias)
{
    elx_conv_t &xc = *desc.xc;

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
        switch(xc.T) {
        case 5:
            elx_conv_winograd<F, 5>(xc, input, weights, output, bias);
            break;
        }
        __tend(twei);

    }
    return ELX_OK;
}

template int elx_conv<float>
(eld_conv_t &desc, float *input, float *weights, float *output, float *bias);
template int elx_conv_winograd<float, 5>
(elx_conv_t &xc, float *input, float *weights, float *output, float *bias);

}
