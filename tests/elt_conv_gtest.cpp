#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <boost/program_options.hpp>
#include <limits.h>
#include "gtest/gtest.h"
#include "elt_utils.hpp"
#include "elt_conv_utils.hpp"
#include "euler.hpp"
#include <iostream>
#include <unordered_map>

using namespace euler;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::Combine;

int test_elt_conv(int format, int tile_size, int execution_mode, int nteams,
                  int blk_i, int blk_o, int blk_t) {
  // Covolution options
  int mb = 64, ic = 64, ih = 224, iw = 224, oc = 64, oh = 224, ow = 224, kh = 3,
      kw = 3;
  int ph = 1, pw = 1, sh = 1, sw = 1, dh = 1, dw = 1;
  bool with_bias = true, with_relu = false;
  int prop_kind = forward_inference, alg = CONV_WINOGRAD;
  int pat_i = 1, pat_o = 1;
  bool validate_results = true;
  int nthreads = nteams * 28;
  int streaming_weights = 0, streaming_input = 0, streaming_output = 0;
  bool input_as_blocked = false, weights_as_blocked = false,
       output_as_blocked = false;

  int input_format, weights_format, output_format;
  if (format == 0) {
    input_format = nChw16c;
    weights_format = OIhw16i16o;
    output_format = nChw16c;
  } else {
    input_format = nchw;
    weights_format = oihw;
    output_format = nchw;
  }

  // 1, create convolution desc
  eld_conv_t<float> desc;
  desc.dims = {.input = {mb, ic, ih, iw},
               .weights = {oc, ic, kh, kw},
               .output = {mb, oc, oh, ow},
               .bias = {oc}};
  desc.formats = {.input = input_format,
                  .weights = weights_format,
                  .output = output_format};
  desc.pads = {ph, ph, pw, pw};
  desc.with_bias = with_bias;
  desc.with_relu = with_relu;
  desc.algorithm = alg;
  desc.tile_size = tile_size;
  desc.prop_kind = prop_kind;
  desc.threading = {nteams, nthreads};
  desc.execution_mode = execution_mode;
  desc.blocking = {blk_i, blk_o, blk_t};
  desc.partition = {pat_i, pat_o};
  desc.streaming_hint = {streaming_weights, streaming_input, streaming_output};
  desc.format_as_blocked = {input_as_blocked, weights_as_blocked,
                            output_as_blocked};

  if (desc.setup() != ELD_OK) {
    printf("Fail: Convolution setup error!\n");
    return -1;
  }

  // 2. prepare data
  float *input, *weights, *output, *bias;
  test::prepare_conv_data<float>(desc, &input, &weights, &output, &bias);

  // 3. execute convolution
  int iterations = validate_results ? 1 : 6400 / mb;
  size_t num_ops = test::cal_ops(desc);
  time_start(conv);
  for (int n = 0; n < iterations; ++n) {
    if (ELX_OK != elx_conv<float>(desc, output, input, weights, bias)) {
      printf("Fail: Convolution execution error!\n");
      test::teardown_conv_data(input, weights, output, bias);
      return -1;
    }
  }
  time_end(conv, iterations, num_ops);

  // 4. cosim, setdown
  if (validate_results) {
    printf("Validation: ");
    float *ref_output = (float *)malloc(desc.byte_sizes.output);
    if (test::ref_convolution2d<float>(desc, ref_output, input, weights, bias))
      printf("Fail: Convolution ref execution error!\n");
    else if (test::compare_conv_results(desc, output, ref_output))
      printf("Fail: Convolution results not correct!\n");
    else
      printf("Convolution Pass!\n");

    free(ref_output);
  }
  test::teardown_conv_data(input, weights, output, bias);

  return 0;
}

class eltConvTest : public ::testing::TestWithParam<
                        ::std::tr1::tuple<int, int, int, int, int, int, int>> {
};

INSTANTIATE_TEST_CASE_P(elt_conv_test, eltConvTest,
                        Combine(Values(0, 1), Values(5, 7),
                                Values(0xa040, 0xa061, 0xa448, 0xa241, 0xa000,
                                       0xa201, 0xa0e0, 0xa0e1),
                                Values(1, 2), Values(0, 2, 4, 8),
                                Values(0, 2, 4, 8), Values(0, 2, 4, 8)));
TEST_P(eltConvTest, combineTest) {
  int test_format = ::testing::get<0>(GetParam());
  int test_tile_size = ::testing::get<1>(GetParam());
  int test_execution_mode = ::testing::get<2>(GetParam());

  int test_nteams = ::testing::get<3>(GetParam());
  int test_blk_i = ::testing::get<4>(GetParam());
  int test_blk_o = ::testing::get<5>(GetParam());
  int test_blk_t = ::testing::get<6>(GetParam());

  // printf("actual toutput: [%x]: (ref)\n", test_execution_mode);
  int ret = test_elt_conv(test_format, test_tile_size, test_execution_mode,
                          test_nteams, test_blk_i, test_blk_o, test_blk_t);
  EXPECT_EQ(0, ret);
}
