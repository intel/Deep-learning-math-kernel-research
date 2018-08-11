#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <limits.h>
#include "gtest/gtest.h"
#include "euler.hpp"
#include "el_def.hpp"
#include "elt_unitests.hpp"
#include "tests/elt_utils.hpp"
#include "src/elk_conv_wino.hpp"
#include "src/elx_conv.hpp"
#include "src/elx_conv_wino.hpp"

int iterations = 10;
using namespace euler;

template <typename F>
int ref_elk_gemm(elx_conv_t<F> &xc, F *output, F *input, F *weights,
                 bool zero_out);

template <typename Type, const int T, const int A, const int V, const int I>
void test_elk_gemm(bool perf, bool show_diff, int execution_mode) {
  int error = 0;

  eld_conv_t<Type> desc;
  desc.dims = {{64, 64, 224, 224}, {64, 64, 3, 3}, {64, 64, 224, 224}, {64}};
  desc.formats = {nChw16c, OIhw16i16o, nChw16c};
  desc.pads = {1, 1, 1, 1};
  desc.with_bias = false;
  desc.algorithm = CONV_WINOGRAD;
  desc.with_relu = false;
  desc.execution_mode = execution_mode;
  desc.with_relu = false;
  desc.prop_kind = forward_inference;
  desc.tile_size = A;
  elx_conv_wino_t<Type, A, 3, V, I> xc(desc);
  xc.T = T;

  Type *tinput, *tweights, *toutput;
  int tinput_sz, tweights_sz, toutput_sz;

  tinput_sz = xc.I2 * xc.T * xc.V * sizeof(Type);
  tweights_sz = xc.O2 * xc.I2 * xc.V * xc.V * sizeof(Type);
  toutput_sz = xc.O2 * xc.T * xc.V * sizeof(Type);

  MEMALIGN64(&tinput, tinput_sz);
  MEMALIGN64(&tweights, tweights_sz);
  MEMALIGN64(&toutput, toutput_sz);

  /*
    tinput = (Type *)malloc(tinput_sz);
    tweights = (Type *)malloc(tweights_sz);
    toutput = (Type *)malloc(toutput_sz);
  */

  for (size_t i = 0; i < tinput_sz / sizeof(Type); i++) {
    tinput[i] = i % 18;
  }
  for (size_t i = 0; i < tweights_sz / sizeof(Type); i++) {
    tweights[i] = i % 32;
  }

  memset(toutput, 0, xc.O2 * xc.T * xc.V * sizeof(Type));
  TT(elk_gemm, iterations, perf,
     (gemm_kernel<Type, I, V, T>::gemm(
         xc, toutput, tinput, tweights, true)));

  Type *ref_toutput = (Type *)malloc(toutput_sz);
  memset(ref_toutput, 0, xc.O2 * xc.T * xc.V * sizeof(Type));
  TT(ref_elk_gemm, iterations, perf,
     (gemm_kernel<Type, ISA_GENERIC, V, T>::gemm(
         xc, ref_toutput, tinput, tweights, true)));

  for (size_t i = 0; i < toutput_sz / sizeof(Type); i++) {
    EXPECT_NEAR(ref_toutput[i], toutput[i], 1e-6);
    if (show_diff) {
      printf("actual toutput: [%ld]: %f vs reference: %f (ref)\n", i,
             toutput[i], ref_toutput[i]);
    }
  }
  free(tinput);
  free(tweights);
  free(toutput);
}

class elkGemmTest : public ::testing::TestWithParam<int> {};
INSTANTIATE_TEST_CASE_P(elk_gemm_test, elkGemmTest,
                        testing::Values(0xa040, 0xa061, 0xa448, 0xa241, 0xa000,
                                        0xa201, 0xa0e0, 0xa0e1));

bool test_perf = false;
bool show_diff = false;
TEST_P(elkGemmTest, T16_A5) {
  int execution_mode = GetParam();
  test_elk_gemm<float, 16, 5, 16, ISA_SKX_AVX512>(test_perf, show_diff,
                                                  execution_mode);
}

TEST_P(elkGemmTest, T16_A7) {
  int execution_mode = GetParam();
  test_elk_gemm<float, 16, 7, 16, ISA_SKX_AVX512>(test_perf, show_diff,
                                                  execution_mode);
}

TEST_P(elkGemmTest, T25_A5) {
  int execution_mode = GetParam();
  test_elk_gemm<float, 25, 5, 16, ISA_SKX_AVX512>(test_perf, show_diff,
                                                  execution_mode);
}

TEST_P(elkGemmTest, T25_A7) {
  int execution_mode = GetParam();
  test_elk_gemm<float, 25, 7, 16, ISA_SKX_AVX512>(test_perf, show_diff,
                                                  execution_mode);
}
