#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <limits.h>
#include "gtest/gtest.h"
#include "euler.hpp"
#include "el_def.hpp"
#include "elt_unitests.hpp"
#include "elt_utils.hpp"
#include "elk_conv_wino.hpp"
#include "elx_conv.hpp"
#include "elx_conv_wino.hpp"

int iterations = 10;
using namespace euler;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::Bool;
using ::testing::Combine;

template <typename F>
int ref_elk_gemm(elx_conv_t &xc, F *output, F *input, F *weights,
                 bool zero_out);

template <typename Type, const int T, const int A, const int V, const int I>
void test_elk_gemm(bool perf, bool show_diff, int execution_mode,
                   int input_format, int weights_format, int output_format,
                   bool with_bias, bool with_relu, int mb) {
  int ic = 64, ih = 224, iw = 224, oc = 64, oh = 224, ow = 224, kh = 3, kw = 3;
  int ph = 1, pw = 1;

  eld_conv_t desc;
  desc.dims = {{mb, ic, ih, iw}, {oc, ic, kh, kw}, {mb, oc, oh, ow}, {oc}};
  desc.formats = {input_format, weights_format, output_format};

  desc.pads = {ph, ph, pw, pw};
  desc.with_bias = with_bias;
  desc.algorithm = CONV_WINOGRAD;
  desc.with_relu = with_relu;
  desc.execution_mode = execution_mode;
  desc.prop_kind = forward_inference;
  desc.tile_size = A;
  desc.data_type = {
      euler::euler_f32, euler::euler_f32, euler::euler_f32, euler::euler_f32 };
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
    EXPECT_NEAR(ref_toutput[i], toutput[i], 0);
    if (show_diff) {
      printf("actual toutput: [%ld]: %f vs reference: %f (ref)\n", i,
             toutput[i], ref_toutput[i]);
    }
  }
  free(tinput);
  free(tweights);
  free(toutput);
}

class elkGemmTest
    : public ::testing::TestWithParam<
          ::testing::tuple<int, int, int, int, int, bool, bool, int>> {};

INSTANTIATE_TEST_CASE_P(elk_gemm_test_common_params, elkGemmTest,
                        Combine(Values(4, 5, 7), // tile-size
                                Values(0xa061, 0xa000, 0xa010,
                                       0xa071, 0xa079,
                                       0xa0e0, 0xa0e1),   // execution-mode
                                Values(nChw16c, nchw),    // input_format
                                Values(OIhw16i16o, oihw), // weights_format
                                Values(nChw16c, nchw),    // output_format
                                Bool(),                   // with_bias
                                Bool(),                   // with_relu
                                Values(1, 64)             // batchsize
                                ));

bool test_perf = false;
bool show_diff = false;
TEST_P(elkGemmTest, T16) {
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
    test_elk_gemm<float, 16, 5, 16, ISA_SKX_AVX512>(
        test_perf, show_diff, test_execution_mode, test_input_format,
        test_weights_format, test_output_format, test_with_bias, test_with_relu,
        test_mb);
    break;
//  case 6:
//    test_elk_gemm<float, 16, 6, 16, ISA_SKX_AVX512>(
//        test_perf, show_diff, test_execution_mode, test_input_format,
//        test_weights_format, test_output_format, test_with_bias, test_with_relu,
//        test_mb);
//    break;
  case 7:
    test_elk_gemm<float, 16, 7, 16, ISA_SKX_AVX512>(
        test_perf, show_diff, test_execution_mode, test_input_format,
        test_weights_format, test_output_format, test_with_bias, test_with_relu,
        test_mb);
    break;
  default:
    el_error("Unimplemented tile size");
    break;
  }
}

TEST_P(elkGemmTest, T25) {
  int test_tile_size = ::testing::get<0>(GetParam());
  int test_execution_mode = ::testing::get<1>(GetParam());
  int test_input_format = ::testing::get<2>(GetParam());
  int test_weights_format = ::testing::get<3>(GetParam());
  int test_output_format = ::testing::get<4>(GetParam());

  bool test_with_bias = ::testing::get<5>(GetParam());
  bool test_with_relu = ::testing::get<6>(GetParam());
  int test_mb = ::testing::get<7>(GetParam());

  switch (test_tile_size) {
  case 4:
    test_elk_gemm<float, 25, 4, 16, ISA_SKX_AVX512>(
        test_perf, show_diff, test_execution_mode, test_input_format,
        test_weights_format, test_output_format, test_with_bias, test_with_relu,
        test_mb);
    break;

  case 5:
    test_elk_gemm<float, 25, 5, 16, ISA_SKX_AVX512>(
        test_perf, show_diff, test_execution_mode, test_input_format,
        test_weights_format, test_output_format, test_with_bias, test_with_relu,
        test_mb);
    break;
  case 6:
//     test_elk_gemm<float, 25, 6, 16, ISA_SKX_AVX512>(
//         test_perf, show_diff, test_execution_mode, test_input_format,
//         test_weights_format, test_output_format, test_with_bias, test_with_relu,
//         test_mb);
    break;
  case 7:
    test_elk_gemm<float, 25, 7, 16, ISA_SKX_AVX512>(
        test_perf, show_diff, test_execution_mode, test_input_format,
        test_weights_format, test_output_format, test_with_bias, test_with_relu,
        test_mb);
    break;
  default:
    el_error("Unimplemented tile size");
    break;
  }
}
