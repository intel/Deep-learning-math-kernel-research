#include <stdlib.h>
#include <assert.h>
#include "euler.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elx_conv_wino_prod.hpp"
#include "elx_conv_wino_gemm.hpp"

namespace euler {

template <typename F>
eld_conv_t<F>::eld_conv_t() {
  pads      = {1, 1, 1, 1};
  strides   = {1, 1};
  dilations = {1, 1};
  sizes     = {0, 0, 0, 0};
  algorithm = CONV_DIRECT;
  tile_size = 0;
  with_relu = false;
  with_bias = false;
  xc        = nullptr;
}

template <typename F>
eld_conv_t<F>::~eld_conv_t() {
  if (xc != nullptr) {
    delete xc;
  }
}

template <typename F>
int eld_conv_t<F>::setup() {
  // Dimensions
  if (dims.input.c != dims.weights.i || dims.input.n != dims.output.n ||
      dims.output.c != dims.weights.o) {
    eld_error("Dimension error");
    return ELD_GENERAL_ERROR;
  }

  sizes.input =
      accumulate(dims.input.n, dims.input.c, dims.input.h, dims.input.w);
  sizes.weights = accumulate(dims.weights.o, dims.weights.i, dims.weights.h,
                             dims.weights.w);
  sizes.output =
      accumulate(dims.output.n, dims.output.c, dims.output.h, dims.output.w);
  sizes.bias = dims.bias.c;

  byte_sizes.input = sizeof(F) * sizes.input;
  byte_sizes.weights = sizeof(F) * sizes.weights;
  byte_sizes.output = sizeof(F) * sizes.output;
  byte_sizes.bias = sizeof(F) * sizes.bias;

  // TODO: Check output dimensions
  // TODO: Check formats
  if (formats.input != nChw16c || formats.output != nChw16c ||
      formats.weights != OIhw16i16o) {
    eld_error("Unimplemented");
    return ELD_UNIMPLEMENTED;
  }

  // TODO: Check CPUID
  xc = nullptr;

  // Direct
  if (algorithm == CONV_DIRECT) {
    eld_error("Unimplemented");
    // TODO: Direct
    return ELD_UNIMPLEMENTED;
  } else if (algorithm == CONV_WINOGRAD) {
    // Winograd
    if (dilations.h > 1 || dilations.w > 1 || tile_size != 5 ||
        strides.h != 1 || strides.w != 1 || dims.weights.h != 3 ||
        dims.weights.w != 3) {
      eld_error("Unimplemented");
      return ELD_UNIMPLEMENTED;
    }

    switch (tile_size) {
      case 5:
        xc = new elx_conv_wino_gemm_t<F, 5, 3, 25, 16, ISA_SKX_AVX512>(*this);
        break;
      default:
        xc = new elx_conv_wino_prod_t<F, 5, 3, 16, ISA_GENERIC>(*this);
        break;
    }
  }

  return ELD_OK;
}

}  // namespace euler
