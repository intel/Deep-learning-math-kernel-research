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
int test_elk_trans_input(bool perf, bool show_diff) {
  int error = 0;

  eld_conv_t<Type> desc;

  desc.dims = {{64, 224, 224, 64}, {3, 3, 64, 64}, {64, 224, 224, 64}};
  desc.formats = {nChw16c, OIhw16i16o, nChw16c};
  desc.pads = {1, 1, 1, 1};
  desc.with_bias = false;
  desc.algorithm = CONV_WINOGRAD;
  desc.tile_size = 5;
  desc.with_relu = false;

  elx_conv_wino_gemm_t<Type, A, K, V, I> xc(desc);

  Type atinput[A][A][V];
  Type ainput[xc.ih][xc.iw][V];
  Type ref_atinput[A][A][V];

  for (int _ih = 0; _ih < xc.ih; ++_ih) {
    for (int _iw = 0; _iw < xc.iw; ++_iw) {
      for (int _V = 0; _V < V; ++_V) {
        ainput[_ih][_iw][_V] = _ih * _iw * _V % 32;
      }
    }
  }

  memset(atinput, 0, sizeof(atinput));
  TT(elk_trans_input, iterations, perf,
     (elk_trans_input<Type, A, K, V, I>(xc, atinput, (Type *)&ainput)));

  memset(ref_atinput, 0, sizeof(ref_atinput));
  TT(elk_trans_input, iterations, perf,
     (elk_trans_input<Type, A, K, V, ISA_GENERIC>(xc, ref_atinput,
                                                  (Type *)&ainput)));

  for (int _hA = 0; _hA < A; ++_hA) {
    for (int _wA = 0; _wA < A; ++_wA) {
      for (int _iV = 0; _iV < V; ++_iV) {
        if (ref_atinput[_hA][_wA][_iV] !=
            lest::approx(atinput[_hA][_wA][_iV])) {
          error++;
          if (show_diff && error <= 32) {
            printf("Not equal!: [%d][%d][%d]: %f != %f (ref)\n", _hA, _wA, _iV,
                   atinput[_hA][_wA][_iV], ref_atinput[_hA][_wA][_iV]);
          }
        }
      }
    }
  }

  return error;
}

template int test_elk_trans_input<float, 5, 3, 16, ISA_SKX_AVX512>(
    bool perf, bool show_diff);
