#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include "elt_utils.hpp"
#include "euler.hpp"

using namespace euler;

namespace euler {

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

void ref_convolution(
    eld_conv_t<float> &desc, float *ouptut, float *input, float *weights)
{
}
}
