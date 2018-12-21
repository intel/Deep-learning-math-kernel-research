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

  template <typename InputType, typename WeightsType, typename OutputType, typename BiasType>
  void prepare_conv_data(eld_conv_t<ConvTypes<InputType, WeightsType, OutputType, BiasType>> &desc,
      InputType **input, WeightsType **weights, OutputType **output,
      BiasType **bias, short **input1, short **weights1, short **output1,
      short **bias1, bool double_buffering = false, bool fp16_mode = false,
      bool f16c_opt = false, bool validate_results = false);

  void teardown_conv_data(void *input, void *weights, void *output, void *bias,
      void *input1, void *weights1, void *output1, void *bias1, bool fp16_mode = false,
      bool validate_results = false);

  template <typename OutputType>
  int __compare_conv_results_nchw(eld_conv_t<conv::FP32> &, OutputType *out,
      float *ref, bool fp16_mode);

  template <typename OutputType>
  int __compare_conv_results_nhwc(eld_conv_t<conv::FP32> &, OutputType *out,
      float *ref, bool fp16_mode);

  template <typename OutputType>
  int __compare_conv_results_blocked(eld_conv_t<conv::FP32> &, OutputType *out,
      float *ref, bool fp16_mode);

  template <typename OutputType>
  int compare_conv_results(eld_conv_t<conv::FP32> &, OutputType *out, float *ref,
      bool fp16_mode);

  size_t cal_ops(eld_conv_t<conv::FP32> &desc);
  int cal_iterations(size_t num_ops);

  template <typename Type, const int dst_fmt, const int src_fmt,
      typename... Args>
  struct reorder {
    reorder(Type *dst, Type *src, Args...)
    {
      assert(dst != nullptr && src != nullptr);
    }
  };

  template <typename Type> struct reorder<Type, nchw, nhwc> {
    reorder(Type *dst, Type *src, int n, int c, int h, int w);
  };

  template <typename Type> struct reorder<Type, nhwc, nchw> {
    reorder(Type *dst, Type *src, int n, int c, int h, int w);
  };

  template <typename Type> struct reorder<Type, nchw, nChw16c> {
    reorder(Type *dst, Type *src, int n, int c, int h, int w);
  };

  template <typename Type> struct reorder<Type, nChw16c, nchw> {
    reorder(Type *dst, Type *src, int n, int c, int h, int w);
  };

  template <typename Type> struct reorder<Type, oihw, OIhw16i16o> {
    reorder(Type *dst, Type *src, int o, int i, int h, int w);
  };

  template <typename Type> struct reorder<Type, OIhw16i16o, oihw> {
    reorder(Type *dst, Type *src, int o, int i, int h, int w);
  };

  template <typename InputType, typename WeightsType, typename OutputType, typename BiasType>
  int ref_convolution2d(eld_conv_t<ConvTypes<InputType, WeightsType, OutputType, BiasType>> &desc,
      OutputType *output, InputType *input, WeightsType *weights, BiasType *bias);

  template <typename InputType, typename WeightsType, typename OutputType, typename BiasType>
  int ref_convolution2d_block16(eld_conv_t<ConvTypes<InputType, WeightsType, OutputType, BiasType>> &desc,
      OutputType *output, InputType *input, WeightsType *weights, BiasType *bias);

}
}

#endif // __ELT_CONV_UTILS_HPP__
