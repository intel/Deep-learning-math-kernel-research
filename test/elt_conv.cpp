#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include "elt_utils.hpp"
#include "euler.hpp"

using namespace euler;

int main() {
  // 1, create convolution desc
  eld_conv_t<float> desc;
  desc.dims      = {.input   = {64, 224, 224, 64},
                    .weights = {3, 3, 64, 64},
                    .output  = {64, 224, 224, 64}};
  desc.formats   = {.input   = nChw16c,
                    .weights = OIhw16i16o,
                    .output  = nChw16c};
  desc.pads = {1, 1, 1, 1};
  desc.with_bias = false;
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
  time_start(conv);
  for (int n = 0; n < iterations; ++n) {
    if (ELX_OK != elx_conv<float>(desc, input, weights, output, bias)) {
      // error
      printf("Error\n");
    }
  }
  time_end(conv, iterations);
}
