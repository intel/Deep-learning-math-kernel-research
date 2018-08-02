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
void test_elk_trans_input(bool perf, bool show_diff, int execution_mode) {
  int error = 0;

  eld_conv_t<Type> desc;

  desc.dims = {{64, 64, 224, 224}, {64, 64, 3, 3}, {64, 64, 224, 224}, {64}};
  desc.formats = {nChw16c, OIhw16i16o, nChw16c};
  desc.pads = {1, 1, 1, 1};
  desc.with_bias = false;
  desc.algorithm = CONV_WINOGRAD;
  desc.tile_size = A;
  desc.with_relu = false;
  desc.execution_mode = execution_mode;
  desc.prop_kind = forward_inference;

  elx_conv_wino_t<Type, A, 3, V, I> xc(desc);

  alignas(64) Type atinput[A][A][V];
  alignas(64) Type ainput[xc.ih][xc.iw][V];
  alignas(64) Type ref_atinput[A][A][V];

  for (int _ih = 0; _ih < xc.ih; ++_ih) {
    for (int _iw = 0; _iw < xc.iw; ++_iw) {
      for (int _V = 0; _V < V; ++_V) {
        ainput[_ih][_iw][_V] = _ih * _iw * _V % 32;
      }
    }
  }

  memset(atinput, 0, sizeof(atinput));
  TT(elk_trans_input, iterations, perf,
     (convolution_winograd_kernel<S_INPUT(
          Type, A, 3, V, I, BORDER(false))>::trans_input(xc, atinput,
                                                         (Type *)&ainput, 0, 4,
                                                         0, 4)));

  memset(ref_atinput, 0, sizeof(ref_atinput));
  TT(elk_trans_input, iterations, perf,
     (convolution_winograd_kernel<S_INPUT(
          Type, A, 3, V, ISA_GENERIC,
          BORDER(false))>::trans_input(xc, ref_atinput, (Type *)&ainput, 0, 4,
                                       0, 4)));

  for (int _hA = 0; _hA < A; ++_hA) {
    for (int _wA = 0; _wA < A; ++_wA) {
      for (int _iV = 0; _iV < V; ++_iV) {
        EXPECT_NEAR(ref_atinput[_hA][_wA][_iV], atinput[_hA][_wA][_iV], 1e-6);
        if (show_diff) {
          printf("actual atinput: [%d][%d][%d]: %f vs reference: %f (ref)\n",
                 _hA, _wA, _iV, atinput[_hA][_wA][_iV],
                 ref_atinput[_hA][_wA][_iV]);
        }
      }
    }
  }
}

class elkTransInputTest : public ::testing::TestWithParam<int> {};
INSTANTIATE_TEST_CASE_P(elk_trans_input_test, elkTransInputTest,
                        testing::Values(0xa040, 0xa061, 0xa448, 0xa241, 0xa000,
                                        0xa201, 0xa0e0, 0xa0e1));

bool test_perf = false;
bool show_diff = false;
TEST_P(elkTransInputTest, A5) {
  int execution_mode = GetParam();
  test_elk_trans_input<float, 5, 3, 16, ISA_SKX_AVX512>(test_perf, show_diff,
                                                        execution_mode);
}

TEST_P(elkTransInputTest, A7) {
  int execution_mode = GetParam();
  test_elk_trans_input<float, 7, 3, 16, ISA_SKX_AVX512>(test_perf, show_diff,
                                                        execution_mode);
}
