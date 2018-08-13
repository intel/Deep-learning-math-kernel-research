#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <limits.h>
#include "gtest/gtest.h"
#include "euler.hpp"
#include "elt_unitests.hpp"
#include "tests/elt_utils.hpp"
#include "src/elk_conv_wino.hpp"
#include "src/elx_conv.hpp"
#include "src/elx_conv_wino.hpp"

int iterations = 10;
using namespace euler;

template <typename Type, const int A, const int K, const int V, const int I>
void test_elk_trans_output(bool perf, bool show_diff, int execution_mode) {
  int error = 0;

  eld_conv_t<Type> desc;
  desc.dims = {{64, 64, 224, 224}, {64, 64, 3, 3}, {64, 64, 224, 224}, {64}};
  desc.formats = {nChw16c, OIhw16i16o, nChw16c};
  desc.pads = {1, 1, 1, 1};
  desc.with_bias = true;
  desc.algorithm = CONV_WINOGRAD;
  desc.tile_size = A;
  desc.with_relu = false;
  desc.execution_mode = execution_mode;
  elx_conv_wino_t<Type, A, K, V, I> xc(desc);

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
     (convolution_winograd_kernel<S_OUTPUT(
          Type, A, K, V, I, BORDER(false), BIAS(true), RELU(false),
          SUM(false))>::trans_output(xc, (float *)&aoutput, atoutput, abias,
                                     A - K, A - K)));

  TT(elk_trans_input, iterations, perf,
     (convolution_winograd_kernel<S_OUTPUT(
          Type, A, K, V, ISA_GENERIC, BORDER(false), BIAS(true), RELU(false),
          SUM(false))>::trans_output(xc, (float *)ref_aoutput, atoutput, abias,
                                     A - K, A - K)));

  for (int _oh = 0; _oh < xc.oh; ++_oh) {
    for (int _ow = 0; _ow < xc.ow; ++_ow) {
      for (int _oV = 0; _oV < V; ++_oV) {
        EXPECT_NEAR(ref_aoutput[_oh][_ow][_oV], aoutput[_oh][_ow][_oV], 1e-6);
      }
    }
  }
}

class elkTransOutputTest : public ::testing::TestWithParam<int> {};
INSTANTIATE_TEST_CASE_P(elk_trans_output_test, elkTransOutputTest,
                        testing::Values(0xa040, 0xa061, 0xa448, 0xa241, 0xa000,
                                        0xa201, 0xa0e0, 0xa0e1));

bool test_perf = false;
bool show_diff = false;
TEST_P(elkTransOutputTest, A5) {
  int execution_mode = GetParam();
  test_elk_trans_output<float, 5, 3, 16, ISA_SKX_AVX512>(test_perf, show_diff,
                                                         execution_mode);
}

TEST_P(elkTransOutputTest, A6) {
  int execution_mode = GetParam();
  test_elk_trans_output<float, 6, 3, 16, ISA_SKX_AVX512>(test_perf, show_diff,
                                                         execution_mode);
}

TEST_P(elkTransOutputTest, A7) {
  int execution_mode = GetParam();
  test_elk_trans_output<float, 7, 3, 16, ISA_SKX_AVX512>(test_perf, show_diff,
                                                         execution_mode);
}
