#include <type_traits>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "el_utils.hpp"
#include "elt_utils.hpp"
#include "euler.hpp"

#ifndef __ELT_CONV_UTILS_HPP__
#define __ELT_CONV_UTILS_HPP__

namespace euler {
namespace test {
  enum {
    FP32 = 0,
    FP16,
    FP16O,
    U8F32F32F32,
    U8F32U8F32,
    U8F32S8F32,
    U8F32F32F32z, // with zero point
    U8F32U8F32z,
    U8F32S8F32z

  };

  union Fp32
  {
    uint32_t u;
    float f;
  };

  static inline float half_2_float(uint16_t value)
  {
    Fp32 out;
    const Fp32 magic = { (254U - 15U) << 23 };
    const Fp32 was_infnan = { (127U + 16U) << 23 };

    out.u = (value & 0x7FFFU) << 13;   /* exponent/mantissa bits */
    out.f *= magic.f;                  /* exponent adjust */
    if (out.f >= was_infnan.f)         /* make sure Inf/NaN survive */
    {
      out.u |= 255U << 23;
    }
    out.u |= (value & 0x8000U) << 16;  /* sign bit */

    return out.f;
  }

  static inline uint16_t float_2_half(float value)
  {
    const Fp32 f32infty = { 255U << 23 };
    const Fp32 f16infty = { 31U << 23 };
    const Fp32 magic = { 15U << 23 };
    const uint32_t sign_mask = 0x80000000U;
    const uint32_t round_mask = ~0xFFFU;

    Fp32 in;
    uint16_t out;

    in.f = value;
    uint32_t sign = in.u & sign_mask;
    in.u ^= sign;

    if (in.u >= f32infty.u) { /* Inf or NaN (all exponent bits set) */
      out = (in.u > f32infty.u) ? 0x7FFFU : 0x7C00U;
    } else { /* (De)normalized number or zero */
      in.u &= round_mask;
      in.f *= magic.f;
      in.u -= round_mask;
      if (in.u > f16infty.u) {
        in.u = f16infty.u; /* Clamp to signed infinity if overflowed */
      }

      out = uint16_t(in.u >> 13); /* Take the bits! */
    }
    out = uint16_t(out | (sign >> 16));

    return out;
  }

  template <typename InputType, typename WeightsType, typename OutputType, typename BiasType>
  void prepare_conv_data(eld_conv_t &desc_ref, eld_conv_t &desc,
      float *input_ref, float *weights_ref, float *output_ref, float *bias_ref,
      InputType **input, WeightsType **weights, OutputType **output, BiasType **bias,
      const char *input_file, const char *weights_file, const char *bias_file,
      int input_format, int weights_format, bool double_buffering = false,
      int data_type_cfg = 0, bool f16c_opt = false, bool validate_results = false);

  int __compare_conv_results_nchw(eld_conv_t &, float *out,
      float *ref, int data_type_cfg, double acc);

  int __compare_conv_results_nhwc(eld_conv_t &, float *out,
      float *ref, int data_type_cfg, double acc);

  int __compare_conv_results_blocked(eld_conv_t &, float *out,
      float *ref, int data_type_cfg, double acc);

  int compare_conv_results(eld_conv_t &, float *out, float *ref,
      int data_type_cfg, bool is_int8_lp = false, bool with_real_data = false);

  void post_process_conv_results(float *ouput_ref, eld_conv_t &desc,
      void *output_res, int data_type_cfg);

  size_t cal_ops(eld_conv_t &desc);
  int cal_iterations(size_t num_ops);

  template <typename InputType, typename WeightsType, typename OutputType, typename BiasType>
  int ref_conv_deconv_2d(eld_conv_t &desc,
      OutputType *output, InputType *input, WeightsType *weights, BiasType *bias);

  template <typename InputType, typename WeightsType, typename OutputType, typename BiasType>
  int ref_convolution2d(eld_conv_t &desc,
      OutputType *output, InputType *input, WeightsType *weights, BiasType *bias);
}
}

#endif // __ELT_CONV_UTILS_HPP__
