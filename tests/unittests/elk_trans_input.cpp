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
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::Bool;
using ::testing::Combine;

template <typename Type, const int A, const int K, const int V, const int I>
void test_elk_trans_input(bool perf, bool show_diff, int execution_mode,
                          int input_format, int weights_format,
                          int output_format, bool with_bias, bool with_relu,
                          int mb) {
  int error = 0;

  int ic = 64, ih = 224, iw = 224, oc = 64, oh = 224, ow = 224, kh = 3, kw = 3;
  int ph = 1, pw = 1;

  eld_conv_t<Type> desc;
  desc.dims = {{mb, ic, ih, iw},
               {oc, ic, kh, kw},
               {mb, oc, oh, ow},
               {oc}};
  desc.formats = {input_format,
                  weights_format,
                  output_format};

  desc.pads = {ph, ph, pw, pw};
  desc.with_bias = with_bias;
  desc.algorithm = CONV_WINOGRAD;
  desc.tile_size = A;
  desc.with_relu = with_relu;
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

class elkTransInputTest
    : public ::testing::TestWithParam<
          ::testing::tuple<int, int, int, int, int, bool, bool, int>> {};

INSTANTIATE_TEST_CASE_P(elk_trans_input_test_common_params, elkTransInputTest,
                        Combine(Values(5, 6, 7), // tile-size
                                Values(0xa040, 0xa061, 0xa448, 0xa241, 0xa000,
                                       0xa201, 0xa0e0,
                                       0xa0e1),           // execution-mode
                                Values(nChw16c, nchw),    // input_format
                                Values(OIhw16i16o, oihw), // weights_format
                                Values(nChw16c, nchw),    // output_format
                                Bool(),                   // with_bias
                                Bool(),                   // with_relu
                                Values(1, 64)             // batchsize
                                ));
bool test_perf = false;
bool show_diff = false;

TEST_P(elkTransInputTest, combineTest) {
  int test_tile_size = ::testing::get<0>(GetParam());
  int test_execution_mode = ::testing::get<1>(GetParam());
  int test_input_format = ::testing::get<2>(GetParam());
  int test_weights_format = ::testing::get<3>(GetParam());
  int test_output_format = ::testing::get<4>(GetParam());

  bool test_with_bias = ::testing::get<5>(GetParam());
  bool test_with_relu = ::testing::get<6>(GetParam());
  int test_mb = ::testing::get<7>(GetParam());
  switch (test_tile_size) {
  case 5:
    test_elk_trans_input<float, 5, 3, 16, ISA_SKX_AVX512>(
        test_perf, show_diff, test_execution_mode, test_input_format,
        test_weights_format, test_output_format, test_with_bias, test_with_relu,
        test_mb);
    break;
  case 6:
    test_elk_trans_input<float, 6, 3, 16, ISA_SKX_AVX512>(
        test_perf, show_diff, test_execution_mode, test_input_format,
        test_weights_format, test_output_format, test_with_bias, test_with_relu,
        test_mb);
    break;
  case 7:
    test_elk_trans_input<float, 7, 3, 16, ISA_SKX_AVX512>(
        test_perf, show_diff, test_execution_mode, test_input_format,
        test_weights_format, test_output_format, test_with_bias, test_with_relu,
        test_mb);
    break;
  default:
    el_error("Unimplemented tile size");
    break;
  }
}
