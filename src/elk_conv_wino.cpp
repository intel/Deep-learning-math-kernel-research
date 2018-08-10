#include <assert.h>
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"

namespace euler {

template <typename Type, int ...configs>
template <int T>
void convolution_winograd_kernel<Type, configs...>::gemm(
    elx_conv_t<Type> &xc, Type *toutput, Type *tinput, Type *tweights,
    bool zero_out)
{
  __gemm<T>(xc, toutput, tinput, tweights, zero_out);
}

template <typename Type, int ...configs>
template <int T>
void convolution_winograd_kernel<Type, configs...>::gemm_tail(
    elx_conv_t<Type> &xc, Type *toutput, Type *tinput, Type *tweights,
    bool zero_out)
{
  __gemm_tail<T>(xc, toutput, tinput, tweights, zero_out);
}

template <typename Type, int ...configs>
template <bool is_border>
void convolution_winograd_kernel<Type, configs...>::trans_input(
    elx_conv_t<Type> &xc, Type atinput[A][A][V], Type *input, int _hT_start,
    int _hT_end, int _wT_start, int _wT_end)
{
  __trans_input<is_border>(xc, atinput, input, _hT_start, _hT_end,
      _wT_start, _wT_end);
}

template <typename Type, int ...configs>
template <bool is_border>
void convolution_winograd_kernel<Type, configs...>::trans_inputa(
    elx_conv_t<Type> &xc, Type atinput[A][A][V], Type *input, int _wA,
    int _hT_start, int _hT_end, int _wT_start, int _wT_end)
{
  __trans_inputa<is_border>(xc, atinput, input, _wA, _hT_start, _hT_end,
      _wT_start, _wT_end);
}

template <typename Type, int ...configs>
template <bool ...conditions>
void convolution_winograd_kernel<Type, configs...>::
    trans_output(elx_conv_t<Type> &xc, Type *output,
    Type atoutput[A][A][V], Type *bias, int _hOA_end, int _wOA_end)
{
  __trans_output<conditions...>(
      xc, output, atoutput, bias, _hOA_end, _wOA_end);
}

template <typename Type, int ...configs>
template <bool ...conditions>
void convolution_winograd_kernel<Type, configs...>::trans_outputa_th(
    elx_conv_t<Type> &xc, Type *toutputa,
    Type *toutput, int Tz, bool stream_out)
{
  __trans_outputa_th<conditions...>(
      xc, toutputa, toutput, Tz, stream_out);
}

template <typename Type, int ...configs>
template <bool ...conditions>
void convolution_winograd_kernel<Type, configs...>::trans_outputa_bh(
    elx_conv_t<Type> &xc, Type *output,
    Type atoutputa[A][A - K + 1][V], Type *bias, int _hOA_end, int _wOA_end)
{
  __trans_outputa_bh<conditions...>(
      xc, output, atoutputa, bias, _hOA_end, _wOA_end);
}

template <typename Type, int ...configs>
void convolution_winograd_kernel<Type, configs...>::trans_weights(
    Type atweights[A][A][V][V], Type aweights[K][K][V][V])
{
  __trans_weights(atweights, aweights);
}

}

#define INCLUDE_WINOGRAD_CONVOLUTION_KERNEL

// Gemm kernel
#include "kernel/elk_conv_wino_gemm.hxx"

// F(2x2, 3x3)
#include "kernel/elk_conv_wino_2x2_3x3_input.hxx"
#include "kernel/elk_conv_wino_2x2_3x3_output.hxx"
#include "kernel/elk_conv_wino_2x2_3x3_weights.hxx"

// F(3x3, 3x3)
#include "kernel/elk_conv_wino_3x3_3x3_input.hxx"
#include "kernel/elk_conv_wino_3x3_3x3_output.hxx"
#include "kernel/elk_conv_wino_3x3_3x3_weights.hxx"

// F(5x5, 3x3)
#include "kernel/elk_conv_wino_5x5_3x3_input.hxx"
#include "kernel/elk_conv_wino_5x5_3x3_output.hxx"
#include "kernel/elk_conv_wino_5x5_3x3_weights.hxx"
