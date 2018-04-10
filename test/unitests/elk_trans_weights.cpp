#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h> 
#include "euler.hpp"
#include "lest.hpp"
#include "elt_unitests.hpp"
#include "../elt_utils.hpp"
#include "../../src/elk_conv.hpp"
#include "../../src/elx_conv.hpp"
#include "../../src/elx_conv_wino_gemm.hpp"

using namespace euler;

template<typename Type, const int A, const int K, const int V, const int I>
int test_elk_trans_weights(bool perf, bool show_diff) {
    int error = 0;

    Type aweights[K][K][V][V];
    Type atweights[A][A][V][V];
    Type ref_atweights[A][A][V][V];

    for (int _hK = 0; _hK < K; ++_hK) {
        for (int _wK = 0; _wK < K; ++_wK) {
            for (int _iV = 0; _iV < V; ++_iV) {
                for (int _oV = 0; _oV < V; ++_oV) {
                    aweights[_hK][_wK][_iV][_oV] = _hK * _wK * _iV * _oV % 32;
                }
            }
        }
    }

    TT(elk_trans_weights, iterations, perf,
       (elk_trans_weights<Type, A, K, V, I>(atweights, aweights)));

    TT(ref_elk_trans_weights, iterations, perf,
       (elk_trans_weights<Type, A, K, V, ISA_GENERIC>(ref_atweights, aweights)));

    for (int _hK = 0; _hK < K; ++_hK) {
        for (int _wK = 0; _wK < K; ++_wK) {
            for (int _iV = 0; _iV < V; ++_iV) {
                for (int _oV = 0; _oV < V; ++_oV) {
                    if (ref_atweights[_hK][_wK][_iV][_oV] !=
                        lest::approx(atweights[_hK][_wK][_iV][_oV])) {
                        error++;
                        if (show_diff) {
                            printf("Not equal!: [%d][%d][%d][%d]: %f != %f (ref)\n",
                                   _hK, _wK, _iV, _oV,
                                   atweights[_hK][_wK][_iV][_oV],
                                   ref_atweights[_hK][_wK][_iV][_oV]);
                        }
                    }
                }
            }
        }
    }
    return error;
}

template int test_elk_trans_weights<float, 5, 3, 16, ISA_SKX_AVX512>
(bool perf, bool show_diff);

