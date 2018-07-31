#include <stdlib.h>
#include <assert.h>
#include "euler.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elx_conv_wino.hpp"
#include "elx_conv_direct_1x1.hpp"

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
  threading = {0, 0};
  blocking  = {0, 0, 0};
  execution_mode = 0;
  partition = {1, 1};
  streaming_hint = {0, 0, 0};
  format_as_blocked = {false, false, false};
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
    el_error("Dimension error");
    return ELD_GENERAL_ERROR;
  }
  if (with_bias && dims.bias.c != dims.output.c) {
    el_error("Dimension error");
    return ELD_GENERAL_ERROR;
  }

  const int V = 16; // TODO: AVX2
  const int fmt_blocked_data = nChw16c;
  const int fmt_blocked_weights = OIhw16i16o;

  bool format_okay
      = any_of(formats.input, nchw, fmt_blocked_data)
      && any_of(formats.weights, oihw, fmt_blocked_weights)
      && any_of(formats.output, nchw, fmt_blocked_data);

  if (!format_okay) {
    el_error("Data format error");
    return ELD_UNIMPLEMENTED;
  }

  int ic = dims.input.c, IC = ALIGNUP(ic, V);
  int oc = dims.output.c, OC = ALIGNUP(oc, V);

  sizes.input = dims.input.n * dims.input.h * dims.input.w;
  sizes.weights = dims.weights.h * dims.weights.w;
  sizes.output = dims.output.n * dims.output.h * dims.output.w;

  sizes.input *= (formats.input == fmt_blocked_data) ? IC : ic;
  sizes.weights
      *= (formats.weights == fmt_blocked_weights) ? OC * IC : oc * ic;
  sizes.output *= (formats.output == fmt_blocked_data) ? OC : oc;
  sizes.bias = (formats.output == fmt_blocked_data) ? OC : oc;

  byte_sizes.input = sizeof(F) * sizes.input;
  byte_sizes.weights = sizeof(F) * sizes.weights;
  byte_sizes.output = sizeof(F) * sizes.output;
  byte_sizes.bias = sizeof(F) * sizes.bias;

  // TODO: Check CPUID
  xc = nullptr;

  if (none_of(prop_kind,
        forward_training,
        forward_inference,
        backward_data,
        backward_weights)) {
    el_error("Propagation kind error");
    return ELD_GENERAL_ERROR;
  }

  if (algorithm == CONV_DEFAULT) {
    if (dims.weights.h == 1 && dims.weights.w == 1) {
      algorithm == CONV_DIRECT_1x1;
    } else if (dims.weights.h == 3 && dims.weights.w == 3 && dilations.h == 1
        && dilations.w == 1 && strides.h == 1 && strides.w == 1 && pads.l == 1
        && pads.r == 1 && pads.t == 1 && pads.b == 1) {
      algorithm = CONV_WINOGRAD;
    } else {
      algorithm = CONV_DIRECT;
    }
  }

  // Direct
  if (algorithm == CONV_DIRECT) {
    el_error("Unimplemented");
    // TODO: Direct
    return ELD_UNIMPLEMENTED;
  } else if (algorithm == CONV_DIRECT_1x1) {
    if (dims.weights.h != 1 || dims.weights.w != 1) {
      el_error("Algorithm CONV_DIRECT_1x1 not supported for this shape.");
      return ELD_GENERAL_ERROR;
    }
    xc = new elx_conv_direct_1x1_t<F, 16, ISA_SKX_AVX512>(*this);
  } else if (algorithm == CONV_WINOGRAD) {
    // Winograd
    if (dilations.h > 1 || dilations.w > 1 ||
        strides.h != 1 || strides.w != 1 ||
        dims.weights.h != 3 || dims.weights.w != 3) {
      el_error("Unimplemented");
      return ELD_UNIMPLEMENTED;
    }

    if (tile_size == 0) {
      // TODO: auto-select tile_size
      el_error("TODO: implement tile size auto-selection");
    } else {
      // TODO: forward, backward_data, backward_weights
      switch (tile_size) {
      case 4:
        xc = new elx_conv_wino_t<F, 4, 3, 16, ISA_SKX_AVX512>(*this);
        break;
      case 5:
        xc = new elx_conv_wino_t<F, 5, 3, 16, ISA_SKX_AVX512>(*this);
        break;
      case 6:
        xc = new elx_conv_wino_t<F, 6, 3, 16, ISA_SKX_AVX512>(*this);
        break;
      case 7:
        xc = new elx_conv_wino_t<F, 7, 3, 16, ISA_SKX_AVX512>(*this);
        break;
      default:
        el_error("Unimplemented tile size");
        break;
      }
    }
  }

  return ELD_OK;
}

}  // namespace euler
