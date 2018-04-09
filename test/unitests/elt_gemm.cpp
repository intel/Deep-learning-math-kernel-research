#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h> 
#include "euler.hpp"
#include "../elt_utils.hpp"
#include "../../src/elx_conv.hpp"
#include "../../src/elk_conv.hpp"

using namespace euler;

template<typename T>
int ref_gemm_ker(T *mxp, T *mxn, T *nxp, int m, int n, int p);
template<typename F>
int ref_gemm(elx_conv_t<F> &xc, F *output, F *input, F *weights);

int iterations = 1000;

template<typename Type>
int test_gemm(bool perf, bool show_diff) {
    eld_conv_t<Type> desc;
    elx_conv_impl_t<Type, 5, 3, 16, ISA_SKX_AVX512> xc(desc);
    xc.oc2 = 4;
    xc.ic2 = 2;
    xc.T = 25;
    xc.V = 16;

    Type *tinput, *tweights, *toutput;
    int tinput_sz, tweights_sz, toutput_sz;

    tinput_sz   = xc.ic2 * xc.T * xc.V * sizeof(Type);
    tweights_sz = xc.oc2 * xc.ic2 * xc.V * xc.V * sizeof(Type);
    toutput_sz  = xc.oc2 * xc.T * xc.V * sizeof(Type);

    tinput   = (Type *)malloc(tinput_sz);
    tweights = (Type *)malloc(tweights_sz);
    toutput  = (Type *)malloc(toutput_sz);

    for (int i = 0; i < tinput_sz / sizeof(Type); i++) {
        tinput[i] = i % 18;
    }
    for (int i = 0; i < tweights_sz / sizeof(Type); i++) {
        tweights[i] = i % 32;
    }

    if (!perf) iterations = 1;

    time_start(conv);
    for (int n = 0; n < iterations; ++n) {
        elk_gemm<Type, 25, 16, ISA_SKX_AVX512>(xc, toutput, tinput, tweights);
    }
    time_end(conv, iterations);

    int error = 0;
    if (!perf) {
        Type *ref_toutput = (Type *)malloc(toutput_sz);
        ref_gemm(xc, ref_toutput, tinput, tweights);
        for (int i = 0; i < toutput_sz / sizeof(Type); i++) {
            if (ref_toutput[i] != toutput[i]) {
                error++;
                if (show_diff) {
                    printf("Not equal!: [%d]: %f != %f (ref)\n",
                           i, toutput[i], ref_toutput[i]);
                }
            }
        }
    }
    return error;
}

template<typename T>
int ref_gemm_ker(T *mxp, T *mxn, T *nxp, int m, int n, int p) {
    MD(T, amxn, [m][n], mxn);
    MD(T, anxp, [n][p], nxp);
    MD(T, amxp, [m][p], mxp);

    for (int _m = 0; _m < m; ++_m) {
        for (int _p = 0; _p < p; ++_p) {
            for (int _n = 0; _n < n; ++_n) {
                amxp[_m][_p] += amxn[_m][_n] * anxp[_n][_p];               
            }
        }
    }

    return 0;
}

template<typename F>
int ref_gemm(elx_conv_t<F> &xc, F *output, F *input, F *weights) {
    MD(F, aoutput,  [xc.oc2][xc.T][xc.V], output);
    MD(F, ainput,   [xc.ic2][xc.T][xc.V], input);
    MD(F, aweights, [xc.oc2][xc.ic2][xc.V][xc.V], weights);

    memset(output, 0, xc.oc2 * xc.T * xc.V * sizeof(F));
    for (int _oc2 = 0; _oc2 < xc.oc2; ++_oc2) {
        for (int _ic2 = 0; _ic2 < xc.ic2; ++_ic2) {
            ref_gemm_ker<F>((F *)aoutput[_oc2],
                            (F *)ainput[_ic2],
                            (F *)aweights[_oc2][_ic2],
                            xc.T, xc.V, xc.V);
        }
    }
    return 0;
}

template int test_gemm<float>(bool perf, bool show_diff);

