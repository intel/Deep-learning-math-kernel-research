#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include "elt_utils.hpp"
#include "elt_conv_utils.hpp"
#include "euler.hpp"

bool validate_results = true;
using namespace euler;

int main()
{
  // 1, create convolution desc
  eld_conv_t<float> desc;
  desc.dims = { .input   = { 1, 5, 5, 16 },
                .weights = { 3, 3, 16, 16 },
                .output  = { 1, 5, 5, 16 },
                .bias    = { 16 } };
  desc.formats = { .input = nChw16c, .weights = OIhw16i16o, .output = nChw16c };
  desc.pads = { 1, 1, 1, 1 };
  desc.with_bias = true;
  desc.algorithm = CONV_WINOGRAD;
  desc.tile_size = 5;
  desc.with_relu = false;

  if (desc.setup() != ELD_OK) {
    printf("Fail: Convolution setup error!\n");
    return -1;
  }

  // 2. prepare data
  float *input, *weights, *output, *bias;
  test::prepare_conv_data<float>(desc, &input, &weights, &output, &bias);

  // 3. execute convolution
  int iterations = validate_results ? 1: 100;
  size_t num_ops = test::cal_ops(desc);
  time_start(conv);
  for (int n = 0; n < iterations; ++n) {
    if (ELX_OK != elx_conv<float>(desc, output, input, weights, bias)) {
      printf("Fail: Convolution execlution error!\n");
      test::teardown_conv_data(input, weights, output, bias);
      return -1;
    }
  }
  time_end(conv, iterations, num_ops);

  // 4. cosim
  if (validate_results) {
    float *ref_output = (float *)memalign(64, desc.byte_sizes.output);
    if (test::ref_convolution2d_block16<float>(
            desc, ref_output, input, weights, bias))
      printf("Fail: Convolution ref execution error!\n");
    else if (test::compare_conv_results_block16(desc, output, ref_output))
      printf("Fail: Convolution results not correct!\n");
    else
      printf("Convolution Pass!\n");
    free(ref_output);
  }
  test::teardown_conv_data(input, weights, output, bias);

  return 0;
}
