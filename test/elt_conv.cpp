#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include "elt_utils.hpp"
#include "euler.hpp"

using namespace euler;

size_t cal_ops(eld_conv_t<float> &desc) {

  size_t num_ops = 0;

  for (int oh = 0; oh < desc.dims.output.h; ++oh) {
    for (int ow = 0; ow < desc.dims.output.w; ++ow) {
      for (int kh = 0; kh < desc.dims.weights.h; ++kh) {
        int ih = oh * desc.strides.h - desc.pads.b + kh * desc.dilations.h;
        if (ih < 0 || ih >= desc.dims.input.h)
          continue;
        for (int kw = 0; kw < desc.dims.weights.w; ++kw) {
          int iw = ow * desc.strides.w - desc.pads.l + kw * desc.dilations.w;
          if (iw < 0 || iw >= desc.dims.input.w)
            continue;
          num_ops += 1;
        }
      }
    }
  }
  return num_ops * desc.dims.input.n * desc.dims.weights.i
      * desc.dims.weights.o * 2;
}

int main()
{
  // 1, create convolution desc
  eld_conv_t<float> desc;
  desc.dims = { .input   = { 64, 224, 224, 64 },
                .weights = { 3, 3, 64, 64 },
                .output  = { 64, 224, 224, 64 },
                .bias    = { 64 } };
  desc.formats = { .input = nChw16c, .weights = OIhw16i16o, .output = nChw16c };
  desc.pads = { 1, 1, 1, 1 };
  desc.with_bias = true;
  desc.algorithm = CONV_WINOGRAD;
  desc.tile_size = 5;
  desc.with_relu = false;

  desc.setup();

  // 2. allocate memory
  float *input, *weights, *output, *bias;
  input   = (float *)memalign(64, desc.byte_sizes.input);
  weights = (float *)memalign(64, desc.byte_sizes.weights);
  output  = (float *)memalign(64, desc.byte_sizes.output);
  bias    = (float *)memalign(64, desc.byte_sizes.bias);

#pragma omp parallel for
  for (int i = 0; i < desc.sizes.input; i++) {
    input[i] = i % 18;
  }
#pragma omp parallel for
  for (int i = 0; i < desc.sizes.weights; i++) {
    weights[i] = i % 32;
  }

  // 3. execute convolution
  int iterations = 100;
  size_t num_ops = cal_ops(desc);
  time_start(conv);
  for (int n = 0; n < iterations; ++n) {
    if (ELX_OK != elx_conv<float>(desc, input, weights, output, bias)) {
      printf("Error\n");
    }
  }
  time_end(conv, iterations, num_ops);
}
