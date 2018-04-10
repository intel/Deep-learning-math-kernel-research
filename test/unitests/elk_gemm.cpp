#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h> 
#include "euler.hpp"
#include "lest.hpp"
#include "../elt_utils.hpp"
#include "../../src/elk_conv.hpp"
#include "../../src/elx_conv.hpp"
#include "../../src/elx_conv_wino_gemm.hpp"

using namespace euler;

template<typename T>
int ref_elk_gemm_ker(T *mxp, T *mxn, T *nxp, int m, int n, int p);
template<typename F>
int ref_elk_gemm(elx_conv_t<F> &xc, F *output, F *input, F *weights);

int iterations = 10;

template<typename Type>
int test_elk_gemm(bool perf, bool show_diff) {
    int error = 0;

    eld_conv_t<Type> desc;
    elx_conv_wino_gemm_t<Type, 5, 3, 25, 16, ISA_SKX_AVX512> xc(desc);
    xc.O2 = 18;
    xc.I2 = 32;
    xc.T = 25;
    xc.V = 16;

    Type *tinput, *tweights, *toutput;
    int tinput_sz, tweights_sz, toutput_sz;

    tinput_sz   = xc.I2 * xc.T * xc.V * sizeof(Type);
    tweights_sz = xc.O2 * xc.I2 * xc.V * xc.V * sizeof(Type);
    toutput_sz  = xc.O2 * xc.T * xc.V * sizeof(Type);

    tinput   = (Type *)malloc(tinput_sz);
    tweights = (Type *)malloc(tweights_sz);
    toutput  = (Type *)malloc(toutput_sz);

    for (int i = 0; i < tinput_sz / sizeof(Type); i++) {
        tinput[i] = i % 18;
    }
    for (int i = 0; i < tweights_sz / sizeof(Type); i++) {
        tweights[i] = i % 32;
    }

    memset(toutput, 0, xc.O2 * xc.T * xc.V * sizeof(Type));
    TT(elk_gemm, iterations, perf,
       (elk_gemm<Type, 25, 16, ISA_SKX_AVX512>(xc, toutput, tinput, tweights)));

    Type *ref_toutput = (Type *)malloc(toutput_sz);
    memset(ref_toutput, 0, xc.O2 * xc.T * xc.V * sizeof(Type));
    TT(ref_elk_gemm, iterations, perf,
       (ref_elk_gemm(xc, ref_toutput, tinput, tweights)));

    for (int i = 0; i < toutput_sz / sizeof(Type); i++) {
        if (ref_toutput[i] != lest::approx(toutput[i])) {
            error++;
            if (show_diff) {
                printf("Not equal!: [%d]: %f != %f (ref)\n",
                       i, toutput[i], ref_toutput[i]);
            }
        }
    }
    return error;
}

template<typename T>
int ref_elk_gemm_ker(T *mxp, T *mxn, T *nxp, int m, int n, int p) {
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
int ref_elk_gemm(elx_conv_t<F> &xc, F *output, F *input, F *weights) {
    MD(F, aoutput,  [xc.O2][xc.T][xc.V], output);
    MD(F, ainput,   [xc.I2][xc.T][xc.V], input);
    MD(F, aweights, [xc.O2][xc.I2][xc.V][xc.V], weights);

#pragma omp parallel for collapse(1)
    for (int _O2 = 0; _O2 < xc.O2; ++_O2) {
        for (int _I2 = 0; _I2 < xc.I2; ++_I2) {
            ref_elk_gemm_ker<F>((F *)aoutput[_O2],
                            (F *)ainput[_I2],
                            (F *)aweights[_O2][_I2],
                            xc.T, xc.V, xc.V);
        }
    }
    return 0;
}

template int test_elk_gemm<float>(bool perf, bool show_diff);

