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
//using ::testing::Range;
using ::testing::Combine;

int test_elt_conv(int tile_size, int execution_mode, int pat_i, int pat_o,
                  int input_format, int weights_format, int output_format,
                  int blk_i, int blk_o, int blk_t, int mb,
                  int streaming_input, int streaming_output,
                  bool input_as_blocked, bool weights_as_blocked,
                  bool output_as_blocked, bool with_bias, bool with_relu) {
  // Covolution options
  int ic = 64, ih = 224, iw = 224, oc = 64, oh = 224, ow = 224, kh = 3, kw = 3;
  int ph = 1, pw = 1;
  int prop_kind = forward_inference, alg = CONV_WINOGRAD;
  bool validate_results = true;
  int nthreads = 0;

  int divisor_i = 16 * blk_i * pat_i;
  if (!(ic / divisor_i != 0 && ic % divisor_i == 0)) {
    printf("Error: blocking or partion options are invalid\n");
    printf("ic = %d, blk_i = %d, pat_i = %d\n", ic, blk_i, pat_i);
    return 0;
  }

  int divisor_o = 16 * blk_o * pat_o;
  if (!(oc / divisor_o != 0 && oc % divisor_o == 0)) {
    printf("Error: blocking or partion options are invalid\n");
    printf("oc = %d, blk_o = %d, pat_o = %d\n", oc, blk_o, pat_o);
    return 0;
  }

  printf("test options are: \n");
  printf("tile_size:%d, execution_mode:%x, pat_i:%d, pat_o:%d \n", tile_size,
         execution_mode, pat_i, pat_o);
  printf("input_format:%d, weights_format:%d, output_format:%d \n",
         input_format, weights_format, output_format);
  printf("blk_i:%d, blk_o:%d, blk_t:%d, mb:%d \n", blk_i, blk_o, blk_t, mb);
  printf("streaming_input:%d, streaming_output:%d \n",
         streaming_input, streaming_output);
  printf("input_as_blocked:%d, weights_as_blocked:%d, output_as_blocked:%d \n",
         input_as_blocked, weights_as_blocked, output_as_blocked);
  printf("with_bias:%d, with_relu:%d \n", with_bias, with_relu);

  // 1, create convolution desc
  eld_conv_t desc;
  desc.data_type = {
      euler::euler_f32, euler::euler_f32, euler::euler_f32, euler::euler_f32 };
  desc.dims = {{mb, ic, ih, iw},
               {oc, ic, kh, kw},
               {mb, oc, oh, ow},
               {oc}};
  desc.formats = {input_format,
                  weights_format,
                  output_format};
  desc.pads = {ph, ph, pw, pw};
  desc.with_bias = with_bias;
  desc.with_relu = with_relu;
  desc.algorithm = alg;
  desc.tile_size = tile_size;
  desc.prop_kind = prop_kind;
  desc.nthreads = nthreads;
  desc.execution_mode = execution_mode;
  desc.flatting = {1, blk_t};
  desc.blocking = {blk_i, blk_o};
  desc.partition = {pat_i, pat_o};
  desc.streaming_hint = {streaming_input, streaming_output};
  desc.format_as_blocked = {input_as_blocked, weights_as_blocked,
                            output_as_blocked};

  if (desc.setup() != ELD_OK) {
    printf("Fail: Convolution setup error!\n");
    return -1;
  }

  // 2. prepare data
  float *input, *weights, *output, *bias;
  float *input_dummy, *weights_dummy, *output_dummy, *bias_dummy;

  MEMALIGN64(&input, desc.byte_sizes.input);
  MEMALIGN64(&output, desc.byte_sizes.output);
  MEMALIGN64(&weights, desc.byte_sizes.weights);
  MEMALIGN64(&bias, desc.byte_sizes.bias);

  test::prepare_conv_data(desc, input, weights, output, bias,
      &input_dummy, &weights_dummy, &output_dummy, &bias_dummy,
      nullptr, nullptr, nullptr);

  // 3. execute convolution
  int iterations = validate_results ? 1 : 6400 / mb;
  size_t num_ops = test::cal_ops(desc);
  test::timer timer;
  timer.start();
  for (int n = 0; n < iterations; ++n) {
    if (ELX_OK != elx_conv(desc, output, input, weights, bias)) {
      printf("Fail: Convolution execution error!\n");
      goto ret;
    }
  }
  timer.stop();
  timer.report_tflops("conv", iterations, num_ops);

  // 4. cosim, setdown
  bool validation_pass = true;
  if (validate_results) {
    printf("Validation: ");
    float *ref_output = (float *)malloc(desc.byte_sizes.output);
    if (test::ref_convolution2d<float>(desc, ref_output, input, weights,
                                       bias)) {
      printf("Fail: Convolution ref execution error!\n");
      validation_pass = false;
    } else if (test::compare_conv_results(
        desc, output, ref_output, euler::test::FP32, false, false)) {
      printf("Fail: Convolution results not correct!\n");
      validation_pass = false;
    } else {
      printf("Convolution Pass!\n");
      validation_pass = true;
    }
    EXPECT_TRUE(validation_pass);
    free(ref_output);
  }

ret:
  free(input);
  free(weights);
  free(output);
  free(bias);
  free(input_dummy);
  free(weights_dummy);
  free(output_dummy);
  free(bias_dummy);

  return 0;
}

class eltConvTest
    : public ::testing::TestWithParam<
          ::testing::tuple<int, int, int, int, int, int, int, int, int, int>> {
};


INSTANTIATE_TEST_CASE_P(elt_conv_test_common_params, eltConvTest,
                        Combine(Values(5, 6, 7), // tile-size
                                Values(0xa061, 0xa000,
                                       0xa0e0, 0xa0e1),   // execution-mode
                                Values(1, 2, 4),          // pat_i
                                Values(1, 2, 4),          // pat_o
                                Values(nChw16c, nchw),    // input_format
                                Values(OIhw16i16o, oihw), // weights_format
                                Values(nChw16c, nchw),    // output_format
                                Values(1, 2, 4, 8),       // blk_i
                                Values(1, 2, 4, 8),       // blk_o
                                Values(1, 32)             // blk_t
                                // Range(1, 32)              // blk_t
                                ));

TEST_P(eltConvTest, combineTest) {
  int test_tile_size = ::testing::get<0>(GetParam());
  int test_execution_mode = ::testing::get<1>(GetParam());
  int test_pat_i = ::testing::get<2>(GetParam());
  int test_pat_o = ::testing::get<3>(GetParam());

  int test_input_format = ::testing::get<4>(GetParam());
  int test_weights_format = ::testing::get<5>(GetParam());
  int test_output_format = ::testing::get<6>(GetParam());

  int test_blk_i = ::testing::get<7>(GetParam());
  int test_blk_o = ::testing::get<8>(GetParam());
  int test_blk_t = ::testing::get<9>(GetParam());

  int test_mb = rand() % 128 + 1;
  int test_streaming_input = rand() % 3;
  int test_streaming_output = rand() % 3;
  bool test_input_as_blocked = rand() % 2;
  bool test_weights_as_blocked = rand() % 2;
  bool test_output_as_blocked = rand() % 2;
  bool test_with_bias = rand() % 2;
  bool test_with_relu = rand() % 2;
  int ret = test_elt_conv(
      test_tile_size, test_execution_mode, test_pat_i, test_pat_o,
      test_input_format, test_weights_format, test_output_format, test_blk_i,
      test_blk_o, test_blk_t, test_mb, test_streaming_input,
      test_streaming_output, test_input_as_blocked,
      test_weights_as_blocked, test_output_as_blocked, test_with_bias,
      test_with_relu);
  EXPECT_EQ(0, ret);
}
