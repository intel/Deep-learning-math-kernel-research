#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include "euler.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elx_conv_wino.hpp"
#include "elx_conv_direct_1x1.hpp"
#include "elx_conv_direct.hpp"

namespace euler {

template <typename UserTypes> eld_conv_t<UserTypes>::eld_conv_t()
{
  pads = { 1, 1, 1, 1 };
  strides = { 1, 1 };
  dilations = { 1, 1 };
  sizes = { 0, 0, 0, 0 };
  algorithm = CONV_DIRECT;
  tile_size = 0;
  with_relu = false;
  with_bias = false;
  with_ip_sum = false;
  with_op_sum = false;
  f16c_opt = false;
  fp_mode = 0;
  xc = nullptr;
  nthreads = 0;
  execution_mode = 0;
  blocking = { 0, 0 };
  flatting = { 0, 0 };
  partition = { 1, 1 };
  streaming_hint = { 0, 0, 0 };
  format_as_blocked = { false, false, false };
  quantization_calibration_min = EL_NO_CALI;
  quantization_calibration_max = EL_NO_CALI;
}

template <typename UserTypes> eld_conv_t<UserTypes>::~eld_conv_t()
{
  if (xc != nullptr) {
    delete xc;
  }
}

template <typename UserTypes> int eld_conv_t<UserTypes>::setup()
{
  // Dimensions
  if (dims.input.c != dims.weights.i || dims.input.n != dims.output.n
      || dims.output.c != dims.weights.o) {
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
      = estl::any_of(formats.input, nchw, nhwc, fmt_blocked_data)
      && estl::any_of(formats.weights, oihw, hwio, fmt_blocked_weights)
      && estl::any_of(formats.output, nchw, nhwc, fmt_blocked_data);

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
  sizes.weights += 4 * V; // for weights pipeline
  sizes.output *= (formats.output == fmt_blocked_data) ? OC : oc;
  sizes.bias = (formats.output == fmt_blocked_data) ? OC : oc;

  byte_sizes.input = sizeof(InputType) * sizes.input;
  byte_sizes.weights = sizeof(WeightsType) * sizes.weights;
  byte_sizes.output = sizeof(OutputType) * sizes.output;
  byte_sizes.bias = sizeof(BiasType) * sizes.bias;

  // Validate padding
  int oh = (dims.input.h + pads.t + pads.b - dims.weights.h) / strides.h + 1; 
  int ow = (dims.input.w + pads.l + pads.r - dims.weights.w) / strides.w + 1;
  if (oh != dims.output.h || ow != dims.output.w) {
    el_error("Padding parameter error");
    return ELX_GENERAL_ERROR;
  }

  // TODO: Check CPUID
  xc = nullptr;

  if (estl::none_of(prop_kind,
        forward_training,
        forward_inference,
        backward_data,
        backward_weights)) {
    el_error("Propagation kind error");
    return ELD_GENERAL_ERROR;
  }

  if (algorithm == CONV_AUTO) {
    if (dims.weights.h == 1 && dims.weights.w == 1) {
      algorithm = CONV_DIRECT_1X1;
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
    if (std::is_same<UserTypes, conv::FP32>::value)
      xc = new elx_conv_direct_t<UserTypes, conv_impl::FP32, 16, ISA_SKX_AVX512>(*this);
    else if (std::is_same<UserTypes, conv::FP16O>::value)
      xc = new elx_conv_direct_t<UserTypes, conv_impl::FP32_F16o, 16, ISA_SKX_AVX512>(*this);
    else
      el_error("TODO: FP16 UserTypes for DIRECT.");
  } else if (algorithm == CONV_DIRECT_1X1) {
    if (dims.weights.h != 1 || dims.weights.w != 1) {
      el_error("Algorithm CONV_DIRECT_1X1 not supported for this shape.");
      return ELD_GENERAL_ERROR;
    }
    if (std::is_same<UserTypes, conv::FP32>::value) {
      if (f16c_opt)
        xc = new elx_conv_direct_1x1_t<UserTypes, conv_impl::FP32_F16w, 16, ISA_SKX_AVX512>(*this);
      else
        xc = new elx_conv_direct_1x1_t<UserTypes, conv_impl::FP32, 16, ISA_SKX_AVX512>(*this);
    } else
      el_error("TODO: FP16 UserTypes for DIRECT 1x1.");
  } else if (algorithm == CONV_WINOGRAD) {
    // Winograd
    if (dilations.h > 1 || dilations.w > 1 ||
        strides.h != 1 || strides.w != 1 ||
        dims.weights.h != 3 || dims.weights.w != 3) {
      el_error("Algorithm CONV_WINOGRAD: data shape not supported");
      return ELD_UNIMPLEMENTED;
    }

    if (tile_size == 0) {
      // TODO: auto-select tile_size
      el_error("TODO: implement tile size auto-selection");
    } else {
      // TODO: forward, backward_data, backward_weights
      if (((execution_mode & 0xF00) != 0x100)
          && std::is_same<UserTypes, conv::FP16>::value) {

        using TarrayTypes = conv_impl::FP32_F16wob;
        switch (tile_size) {
        case 4:
          xc = new elx_conv_wino_t<UserTypes, TarrayTypes, float, 4, 3, 16,
              ISA_SKX_AVX512>(*this);
          break;
        case 5:
          xc = new elx_conv_wino_t<UserTypes, TarrayTypes, float, 5, 3, 16,
              ISA_SKX_AVX512>(*this);
          break;
        case 6:
          xc = new elx_conv_wino_t<UserTypes, TarrayTypes, float, 6, 3, 16,
              ISA_SKX_AVX512>(*this);
          break;
        case 7:
          xc = new elx_conv_wino_t<UserTypes, TarrayTypes, float, 7, 3, 16,
              ISA_SKX_AVX512>(*this);
          break;
        default:
          el_error("Unimplemented tile size");
          break;
        }
      } else if (((execution_mode & 0xF00) != 0x100) && f16c_opt
          && std::is_same<UserTypes, conv::FP32>::value) {
        using TarrayTypes = conv_impl::FP32_F16wo;
        switch (tile_size) {
        case 4:
          xc = new elx_conv_wino_t<UserTypes, TarrayTypes, float, 4, 3, 16,
              ISA_SKX_AVX512>(*this);
          break;
        case 5:
          xc = new elx_conv_wino_t<UserTypes, TarrayTypes, float, 5, 3, 16,
              ISA_SKX_AVX512>(*this);
          break;
        case 6:
          xc = new elx_conv_wino_t<UserTypes, TarrayTypes, float, 6, 3, 16,
              ISA_SKX_AVX512>(*this);
          break;
        case 7:
          xc = new elx_conv_wino_t<UserTypes, TarrayTypes, float, 7, 3, 16,
              ISA_SKX_AVX512>(*this);
          break;
        default:
          el_error("Unimplemented tile size");
          break;
        }
      } else if ((execution_mode & 0xF00) == 0x100 && f16c_opt
          && std::is_same<UserTypes, conv::FP32>::value) {
        using TarrayTypes = conv_impl::FP32_F16o;
        switch (tile_size) {
        case 4:
          xc = new elx_conv_wino_t<UserTypes, TarrayTypes, float, 4, 3, 16,
              ISA_SKX_AVX512>(*this);
          break;
        case 5:
          xc = new elx_conv_wino_t<UserTypes, TarrayTypes, float, 5, 3, 16,
              ISA_SKX_AVX512>(*this);
          break;
        case 6:
          xc = new elx_conv_wino_t<UserTypes, TarrayTypes, float, 6, 3, 16,
              ISA_SKX_AVX512>(*this);
          break;
        default:
          el_error("Unimplemented tile size");
          break;
        }
      } else if ((execution_mode & 0xF00) == 0x100
          && std::is_same<UserTypes, conv::FP16>::value) {
        using TarrayTypes = conv_impl::FP32_F16b;
        switch (tile_size) {
        case 4:
          xc = new elx_conv_wino_t<UserTypes, TarrayTypes, float, 4, 3, 16,
              ISA_SKX_AVX512>(*this);
          break;
        case 5:
          xc = new elx_conv_wino_t<UserTypes, TarrayTypes, float, 5, 3, 16,
              ISA_SKX_AVX512>(*this);
          break;
        case 6:
          xc = new elx_conv_wino_t<UserTypes, TarrayTypes, float, 6, 3, 16,
              ISA_SKX_AVX512>(*this);
          break;
        default:
          el_error("Unimplemented tile size");
          break;
        }
      } else if (!std::is_same<UserTypes, conv::FP16O>::value) {
        using TarrayTypes = conv_impl::FP32;
        switch (tile_size) {
        case 4:
          xc = new elx_conv_wino_t<UserTypes, TarrayTypes, float, 4, 3, 16,
              ISA_SKX_AVX512>(*this);
          break;
        case 5:
          xc = new elx_conv_wino_t<UserTypes, TarrayTypes, float, 5, 3, 16,
              ISA_SKX_AVX512>(*this);
          break;
        case 6:
          xc = new elx_conv_wino_t<UserTypes, TarrayTypes, float, 6, 3, 16,
              ISA_SKX_AVX512>(*this);
          break;
        case 7:
          xc = new elx_conv_wino_t<UserTypes, TarrayTypes, float, 7, 3, 16,
              ISA_SKX_AVX512>(*this);
          break;
        default:
          el_error("Unimplemented tile size");
          break;
        }
      }
    }
  }

  return ELD_OK;
}

template <typename UserTypes>
void eld_conv_t<UserTypes>::preprocess(WeightsType *weights)
{
  this->xc->preprocess(weights);
}

}  // namespace euler
