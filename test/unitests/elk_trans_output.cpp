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

template <typename Type, const int A, const int K, const int V, const int I>
int test_elk_trans_output(bool perf, bool show_diff) {
  int error = 0;

  eld_conv_t<Type> desc;
  elx_conv_wino_gemm_t<Type, A, K, 25, V, I> xc(desc);
  xc.O2 = 18;
  xc.I2 = 32;
  xc.T = 25;
  xc.V = V;
  xc.oh = 28;
  xc.ow = 28;

  Type atoutput[A][A][V];
  Type aoutput[xc.oh][xc.ow][V];
  Type ref_aoutput[xc.oh][xc.ow][V];

  for (int _hA = 0; _hA < A; ++_hA) {
    for (int _wA = 0; _wA < A; ++_wA) {
      for (int _V = 0; _V < V; ++_V) {
        atoutput[_hA][_wA][_V] = _hA * _wA * _V % 32;
      }
    }
  }

  memset((void *)aoutput, 0, sizeof(aoutput));
  memset((void *)ref_aoutput, 0, sizeof(ref_aoutput));

  TT(elk_trans_output, iterations, perf,
     (elk_trans_output<Type, A, K, V, I>(xc, (float *)&aoutput, atoutput)));

  TT(elk_trans_input, iterations, perf,
     (elk_trans_output<Type, A, K, V, ISA_GENERIC>(xc, (float *)ref_aoutput,
                                                   atoutput)));

  for (int _oh = 0; _oh < xc.oh; ++_oh) {
    for (int _ow = 0; _ow < xc.ow; ++_ow) {
      for (int _oV = 0; _oV < V; ++_oV) {
        if (ref_aoutput[_oh][_ow][_oV] !=
            lest::approx(aoutput[_oh][_ow][_oV])) {
          error++;
          if (show_diff) {
            printf("Not equal!: [%d][%d][%d]: %f != %f (ref)\n", _oh, _ow, _oV,
                   aoutput[_oh][_ow][_oV], ref_aoutput[_oh][_ow][_oV]);
          }
        }
      }
    }
  }

  return error;
}

template int test_elk_trans_output<float, 5, 3, 16, ISA_SKX_AVX512>(
    bool perf, bool show_diff);
