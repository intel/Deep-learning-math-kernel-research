#include <stdio.h>
#include <assert.h>
#include "euler.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv.hpp"

namespace euler {


int elk_trans_weights(elx_conv_t<float> &xc, float *tweights, float *weights);

template<typename F, const int T, const int K>
int elx_trans_weights(elx_conv_t<F> &xc, F *tweights, F *weights)
{
#pragma omp parallel for collapse(2)
    for (int _oc2 = 0; _oc2 < xc.oc2; ++_oc2) {
        for (int _ic2 = 0; _ic2 < xc.ic2; ++_ic2) {
            int d = 16 * 16 * (_ic2 + _oc2 * xc.ic2);
            MD(F, aweights,  [K][K][16][16], weights + K * K * d);
            MD(F, atweights, [T][T][16][16], tweights + T * T * d);

            elk_trans_weights<F, T, K>(atweights, aweights);
        }
    }
    return 0;
}

template<typename F, const int T, const int K>
int elx_conv_winograd(elx_conv_t<F> &xc, F *input, F *weights, F *output, F *bias)
{
    // Notation
    // ========
    //
    // Block:
    // v := vector-size (avx512:16)
    // T := tile-size
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
    // to, ti=T, vi=vo=v
    // I := ic/v
    // O := oc/v
    // H := h/T
    // W := w/T

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

    // weights -> tweights
    elx_trans_weights<F, T, K>(xc, xc.tweights, weights);
    MD(F, ainput, [xc.n][xc.ic2][xc.ih][xc.iw][16], input);
    int V = xc.V;

#pragma omp parallel for collapse(4)
    for (int _n = 0; _n < xc.n; ++_n)     {
    for (int _oh2 = 0; _oh2 < xc.oh2; ++_oh2) {
    for (int _ow2 = 0; _ow2 < xc.ow2; ++_ow2) {
    for (int _oc2 = 0; _oc2 < xc.oc2; ++_oc2) {
        F atoutput[T][T][V];
        for (int _ic2 = 0; _ic2 < xc.ic2; ++_ic2) {
            // input -> tinput
            F atinput[T][T][V];
            elk_trans_input<F, T, K>(xc, atinput, (F *)ainput[_n][_ic2],
                                     _oh2, _ow2);
            // gemm
            //elk_gemm(xc, atoutput, atinput, tweights[oc][:]);
        }
        // toutput -> output
        //elk_trans_output(xc, output, atoutput);
    }}}}
#if 0
    for (int i = 0; i < x.ic * x.oc * 25; ++i) {
        printf("%f\n", x.tweights[i]);
    }
#endif
    return 0;
}

template<typename F>
int elx_conv(eld_conv_t<F> &desc, F *input, F *weights, F *output, F *bias)
{
    elx_conv_t<F> &xc = *desc.xc;

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
            elx_conv_winograd<F, 5, 3>(xc, input, weights, output, bias);
            break;
        }
        __tend(twei);

    }
    return ELX_OK;
}

template int elx_conv<float>
(eld_conv_t<float> &desc, float *input, float *weights, float *output, float *bias);
template int elx_conv_winograd<float, 5, 3>
(elx_conv_t<float> &xc, float *input, float *weights, float *output, float *bias);

}
