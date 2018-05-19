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
  desc.dims = {{64, 224, 224, 64}, {3, 3, 64, 64}, {64, 224, 224, 64}, {64}};
  desc.formats = {nChw16c, OIhw16i16o, nChw16c};
  desc.pads = {1, 1, 1, 1};
  desc.with_bias = true;
  desc.algorithm = CONV_WINOGRAD;
  desc.tile_size = 5;
  desc.with_relu = false;
  elx_conv_wino_gemm_t<Type, A, K, V, I> xc(desc);

  alignas(64) Type atoutput[A][A][V];
  alignas(64) Type abias[V];
  alignas(64) Type aoutput[xc.oh][xc.ow][V];
  alignas(64) Type ref_aoutput[xc.oh][xc.ow][V];

  for (int _hA = 0; _hA < A; ++_hA) {
    for (int _wA = 0; _wA < A; ++_wA) {
      for (int _V = 0; _V < V; ++_V) {
        atoutput[_hA][_wA][_V] = _hA * _wA * _V % 32;
      }
    }
  }
  for (int _V = 0; _V < V; _V++) {
    abias[_V] = _V * 1.0f;
  }

  memset((void *)aoutput, 0, sizeof(aoutput));
  memset((void *)ref_aoutput, 0, sizeof(ref_aoutput));

  TT(elk_trans_output, iterations, perf,
      (convolution_winograd_kernel<Type, 0, A, K, V, I, true>::trans_output(
          xc, (float*)&aoutput, atoutput, abias)));

  TT(elk_trans_input, iterations, perf,
      (convolution_winograd_kernel<Type, 0, A, K, V, ISA_GENERIC,
          true>::trans_output(xc, (float*)ref_aoutput, atoutput, abias)));

  for (int _oh = 0; _oh < xc.oh; ++_oh) {
    for (int _ow = 0; _ow < xc.ow; ++_ow) {
      for (int _oV = 0; _oV < V; ++_oV) {
        if (ref_aoutput[_oh][_ow][_oV] !=
            lest::approx(aoutput[_oh][_ow][_oV])) {
          error++;
          if (show_diff && error <= 10) {
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
