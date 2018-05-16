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

template <typename T>
int ref_elk_gemm_ker(T *mxp, T *mxn, T *nxp, int m, int n, int p,
                     bool zero_out);
template <typename F>
int ref_elk_gemm(elx_conv_t<F> &xc, F *output, F *input, F *weights,
                 bool zero_out);

template <typename Type, const int T, const int V, const int I>
int test_elk_gemm(bool perf, bool show_diff) {
  int error = 0;

  eld_conv_t<Type> desc;
  desc.dims = {{64, 224, 224, 64}, {3, 3, 64, 64}, {64, 224, 224, 64}};
  desc.formats = {nChw16c, OIhw16i16o, nChw16c};
  desc.pads = {1, 1, 1, 1};
  desc.with_bias = false;
  desc.algorithm = CONV_WINOGRAD;
  desc.tile_size = 5;
  desc.with_relu = false;
  elx_conv_wino_gemm_t<Type, 5, 3, V, I> xc(desc);

  Type *tinput, *tweights, *toutput;
  int tinput_sz, tweights_sz, toutput_sz;

  tinput_sz = xc.I2 * xc.T * xc.V * sizeof(Type);
  tweights_sz = xc.O2 * xc.I2 * xc.V * xc.V * sizeof(Type);
  toutput_sz = xc.O2 * xc.T * xc.V * sizeof(Type);

  tinput = (Type *)malloc(tinput_sz);
  tweights = (Type *)malloc(tweights_sz);
  toutput = (Type *)malloc(toutput_sz);

  for (size_t i = 0; i < tinput_sz / sizeof(Type); i++) {
    tinput[i] = i % 18;
  }
  for (size_t i = 0; i < tweights_sz / sizeof(Type); i++) {
    tweights[i] = i % 32;
  }

  memset(toutput, 0, xc.O2 * xc.T * xc.V * sizeof(Type));
  TT(elk_gemm, iterations, perf,
      (convolution_winograd_kernel<Type, T, 0, 0, V, I, false>::gemm(
          xc, toutput, tinput, tweights, true)));

  Type* ref_toutput = (Type*)malloc(toutput_sz);
  memset(ref_toutput, 0, xc.O2 * xc.T * xc.V * sizeof(Type));
  TT(ref_elk_gemm, iterations, perf,
      (convolution_winograd_kernel<Type, T, 0, 0, V, ISA_GENERIC, false>::gemm(
          xc, ref_toutput, tinput, tweights, true)));

  for (size_t i = 0; i < toutput_sz / sizeof(Type); i++) {
    if (ref_toutput[i] != lest::approx(toutput[i])) {
      error++;
      if (show_diff) {
        printf("Not equal!: [%ld]: %f != %f (ref)\n", i, toutput[i],
               ref_toutput[i]);
      }
    }
  }
  return error;
}

template int test_elk_gemm<float, 25, 16, ISA_SKX_AVX512>(bool perf,
                                                          bool show_diff);
