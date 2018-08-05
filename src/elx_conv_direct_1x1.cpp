#include <functional>
#include <string.h>
#include <x86intrin.h>
#include "el_utils.hpp"
#include "elx_conv_direct_1x1.hpp"
#include "elk_conv_direct_1x1.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elk_conv_wino.hpp"
#include "elx_conv.hpp"
#include "euler.hpp"

namespace euler {

const unsigned XOPT_MSK = 0xA000;

const unsigned TTM_MSK = 0xF00;
const unsigned TTM_I   = 0x100;
const unsigned TTM_O   = 0x200;
const unsigned TTM_T   = 0x400;

const unsigned FUS_MSK = 0xF0;
const unsigned FUS_I   = 0x10;
const unsigned FUS_O   = 0x20;
const unsigned FUS_T   = 0x40;
const unsigned FUS_A   = 0x80;

const unsigned DUP_MSK = 0xF;
const unsigned DUP_I   = 0x1;
const unsigned DUP_O   = 0x2;
const unsigned DUP_W   = 0x8;

template <typename Type, const int V, const int I>
elx_conv_direct_1x1_t<Type, V, I>::elx_conv_direct_1x1_t(
    eld_conv_t<Type>& dc)
    : elx_conv_t<Type>(dc)
{
  // TODO: error when V!=16 && fmt=OIhw16i16o

  this->IC = ALIGNUP(this->ic, V);
  this->OC = ALIGNUP(this->oc, V);

  this->V = V;
  this->ic2 = this->IC / V;
  this->oc2 = this->OC / V;

  this->ht = this->oh;
  this->wt = this->ow;
  this->nt = this->ht * this->wt;
  this->t = this->nt * this->n;

  // TODO: santize user settings
  if (this->I2 == 0) this->I2 = 4; // TODO: I2 selection
  if (this->O2 == 0) this->O2 = 2; // TODO: O2 selection
  if (this->T == 0)  this->T = 18; // TODO: T selection

  // Tailing
  this->Tr = this->t % this->T ? this->t % this->T : this->T;
  this->Ir = this->ic % V ? this->ic % V : V;
  this->Or = this->oc % V ? this->oc % V : V;

  is_first_run_ = true;
  inference_acc_ = false;
  mthr_ = omp_get_max_threads();

  inference_acc_ = this->prop_kind == forward_inference;

  this->oc4 = this->oc4 == 0 ? 1 : this->oc4;
  this->ic4 = this->ic4 == 0 ? 1 : this->ic4;

  // further divide packed oc/ic
  this->oc3 = (this->oc2 + this->O2 - 1) / this->O2;
  this->ic3 = this->ic2 / this->I2;
  this->O2r = this->oc2 % this->O2;
  if (this->O2r == 0) this->O2r = this->O2;

  xopt_ = this->execution_mode;

  if (xopt_ == 0xb000) {
    this->t3 = this->n;
    this->t2 = (this->nt + this->T - 1) / this->T;
  } else {
    this->t2 = (this->t + this->T - 1) / this->T;
  }

  // In case of Ir != V && blocked-format, assume bias also
  // padded to Vx.

  if (!(xopt_ & XOPT_MSK)) {
    // TODO: deduce xopt
    xopt_ = TTM_O | FUS_T | DUP_I;
  }

  prepare_execute_opt();
  bind_execute_functions();

  // dbg
  printf("T=%d, Tr=%d, t2=%d, t=%d\n", this->T, this->Tr, this->t2, this->t);
  printf("V=%d, Ir=%d, I2=%d, ic3=%d, ic4=%d, IC=%d\n", this->V, this->Ir, this->I2, this->ic3, this->ic4, this->IC);
  printf("V=%d, Or=%d, O2=%d, oc3=%d, oc4=%d, O2r=%d, OC=%d\n", this->V, this->Or, this->O2, this->oc3, this->oc4, this->O2r, this->OC);
}


template <typename Type, const int V, const int I>
int  elx_conv_direct_1x1_t<Type, V, I>::prepare_execute_opt()
{
  size_t tweights_size = 0, tinput_size = 0, toutput_size = 0;
  size_t l1_usage = 0, l2_usage = 0;

  stream_in_ = this->streaming_input
      ? (this->streaming_input == STORE_STREAMING)
      : !(xopt_ & FUS_MSK) ? true : false;
  stream_wei_ = this->streaming_weights
      ? (this->streaming_weights == STORE_STREAMING)
      : !(xopt_ & FUS_MSK) ? true : false;
  stream_out_ = this->streaming_output
      ? (this->streaming_output == STORE_STREAMING)
      : false;

  if (!(xopt_ & TTM_MSK)) {
    this->nthreads = mthr_;
    this->nteams = 1;
  }

  if (xopt_ & TTM_O) {
    if (this->oc3 % this->nteams != 0) {
      // Force single nteams
      this->nthreads = mthr_;
      this->nteams = 1;
    } else {
      // ignore user --pat-o=oc4
      this->oc3 /= this->nteams;
      this->oc4 = this->nteams;
    }
  }
  if (xopt_ & FUS_O) {
    this->oc3 /= this->oc4;
    if (V * this->O2 * this->oc3 * this->oc4 != this->OC) {
      el_error("Config error!");
      return -1;
    }
  }
  if (xopt_ & FUS_I) {
    this->ic3 /= this->ic4;
    if (V * this->I2 * this->ic3 * this->ic4 != this->IC) {
      el_error("Config error!");
      return -1;
    }
  }

  input_is_bfmt_ = this->input_fmt == nchw ? false : true;
  weights_is_bfmt_ = this->weights_fmt == oihw ? false : true;
  output_is_bfmt_ = this->output_fmt == nchw ? false : true;
  input_as_bfmt_ = !input_is_bfmt_ && this->input_as_blocked;
  weights_as_bfmt_ = !weights_is_bfmt_ && this->weights_as_blocked;
  output_as_bfmt_ = !output_is_bfmt_ && this->output_as_blocked;
  is_bfmt_ = input_is_bfmt_ && weights_is_bfmt_ && output_is_bfmt_;

  if (this->ic4 > 1 && this->Ir != V) {
    el_error("Unimplemented: ic4 > 1 for IC % V != 0");
  }
  if (this->oc4 > 1 && this->Or != V
      && (!output_as_bfmt_ || !weights_as_bfmt_)) {
    el_error("Unimplemented: oc4 > 1 for OC % V != 0");
  }

  tweights_ = nullptr;
  tinput_ = nullptr;
  toutput_ = nullptr;

  l1_usage = sizeof(Type)
      * (this->O2 * this->I2 * V * V + this->T * V * (this->I2 + this->O2));

  switch (xopt_) {
  case 0xa000:
    tweights_size = this->IC * this->OC;
    tinput_size = this->IC * this->t;
    toutput_size = this->OC * this->t;
    l2_usage = this->IC * this->OC / this->oc3
        + this->T * (this->IC + this->OC / this->oc3);
    break;
  case 0xa060:
    tweights_size = this->IC * this->OC;
    tinput_size = this->IC * this->t;
    toutput_size = (this->OC / this->oc4) * this->T * mthr_;
    l2_usage = tweights_size / this->oc4
        + this->T * (this->IC + this->OC / this->oc4);
    break;
  case 0xa061:
    tweights_size = this->IC * this->OC;
    tinput_size = this->IC * this->T * mthr_;
    toutput_size = (this->OC / this->oc4) * this->T * mthr_;
    l2_usage = tweights_size / this->oc4
        + this->T * (this->IC + this->OC / this->oc4);
    break;
  case 0xa069:
    tweights_size = this->IC * (this->OC / this->oc4) * mthr_;
    tinput_size = this->IC * this->T * mthr_;
    toutput_size = (this->OC / this->oc4) * this->T * mthr_;
    l2_usage = tweights_size / this->oc4
        + this->T * (this->IC + this->OC / this->oc4);
    break;
  case 0xa201:
    tweights_size = this->IC * this->OC;
    tinput_size = this->IC * this->T * this->t2 * this->nteams;
    toutput_size = this->OC * this->T * this->t2;
    l2_usage = this->IC * this->OC / this->oc3 / this->oc4
        + this->T * (this->IC + this->OC / this->oc3 / this->oc4);
    break;
  case 0xa241:
    tweights_size = this->IC * this->OC;
    tinput_size = this->IC * this->T * mthr_;
    toutput_size = this->OC * this->T * mthr_;
    l2_usage = tweights_size / this->oc4
        + this->T * (this->IC + this->OC / this->oc4);
    break;
  case 0xb000:
    break;
  default:
      el_error("Config error!");
      return -1;
    break;
  }

  l2_usage *= sizeof(Type);

  if (tweights_size > 0)
    MEMALIGN64(&tweights_, (tweights_size + 4 * V) * sizeof(Type)); // weights loading pipeline
  if (tinput_size > 0)
    MEMALIGN64(&tinput_, tinput_size * sizeof(Type));
  if (toutput_size > 0)
    MEMALIGN64(&toutput_, toutput_size * sizeof(Type));

  // dbg
  printf("nteams=%d, nthreads=%d, mthr_=%d\n", this->nteams, this->nthreads, mthr_);
  printf("l2_usage=%ld, l1_usage=%ld\n", l2_usage, l1_usage);

  return 0;
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::bind_execute_functions()
{
#define GEMM_CASE_b(z, T, O)                                                   \
  case T:                                                                      \
    ker_bgemm_ = convolution_direct_1x1_kernel<Type, O, T, V, I, BIAS(true),   \
        RELU(false), SUM(false)>::gemm;                                        \
    break;
#define GEMM0_CASE_b(z, T, O)                                                  \
  case T:                                                                      \
    ker_bgemm0_ = convolution_direct_1x1_kernel<Type, O, T, V, I, BIAS(true),  \
        RELU(false), SUM(false)>::gemm;                                        \
    break;

  switch (this->O2) {
  case 1:
    switch (this->T) {
      BOOST_PP_REPEAT_FROM_TO(1, MAX_FMA_PRL, GEMM_CASE_b, 1);
    default:
      el_error("Convolution_direct_1x1: Unimplemented T in O=1");
      break;
    }
    break;
  case 2:
    switch (this->T) {
      BOOST_PP_REPEAT_FROM_TO(1, 15, GEMM_CASE_b, 2);
    default:
      el_error("Convolution_direct_1x1: Unimplemented T in O=2");
      break;
    }
    break;
  case 3:
    switch (this->T) {
      BOOST_PP_REPEAT_FROM_TO(1, 9, GEMM_CASE_b, 3);
    default:
      el_error("Convolution_direct_1x1: Unimplemented T in O=3");
      break;
    }
    break;
  case 4:
    switch (this->T) {
      BOOST_PP_REPEAT_FROM_TO(1, 7, GEMM_CASE_b, 4);
    default:
      el_error("Convolution_direct_1x1: Unimplemented T in O=4");
      break;
    }
    break;
  default:
    el_error("O2 > 4 unsupported");
  }

  switch (this->O2r) {
  case 1:
    switch (this->T) {
      BOOST_PP_REPEAT_FROM_TO(1, MAX_FMA_PRL, GEMM0_CASE_b, 1);
    default:
      el_error("Convolution_direct_1x1: Unimplemented T in O=1");
      break;
    }
    break;
  case 2:
    switch (this->T) {
      BOOST_PP_REPEAT_FROM_TO(1, 15, GEMM0_CASE_b, 2);
    default:
      el_error("Convolution_direct_1x1: Unimplemented T in O=2");
      break;
    }
    break;
  case 3:
    switch (this->T) {
      BOOST_PP_REPEAT_FROM_TO(1, 9, GEMM0_CASE_b, 3);
    default:
      el_error("Convolution_direct_1x1: Unimplemented T in O=3");
      break;
    }
    break;
  case 4:
    switch (this->T) {
      BOOST_PP_REPEAT_FROM_TO(1, 7, GEMM0_CASE_b, 4);
    default:
      el_error("Convolution_direct_1x1: Unimplemented T in O=4");
      break;
    }
    break;
  default:
    el_error("O2 > 4 unsupported");
  }


  if (this->Ir != V) {
#define GEMM_TAIL_CASE(z, n, data)                                             \
  case n:                                                                      \
    ker_gemm_tail_                                                             \
        = convolution_winograd_kernel<S_GEMM(Type, n, V, I)>::gemm_tail;       \
    break;

    switch (this->T) {
      BOOST_PP_REPEAT_FROM_TO(1, MAX_FMA_PRL, GEMM_TAIL_CASE, nil)
    default:
      el_error("Unimplemented");
      break;
    }

#define GEMM_CASE0_TAIL(z, n, data)                                            \
  case n:                                                                      \
    ker_gemm0_tail_                                                            \
        = convolution_winograd_kernel<S_GEMM(Type, n, V, I)>::gemm_tail;       \
    break;

    switch (this->Tr) {
      BOOST_PP_REPEAT_FROM_TO(1, MAX_FMA_PRL, GEMM_CASE0_TAIL, nil);
    default:
      el_error("Unimplemented");
      break;
    }
  } else {
    ker_gemm_tail_ = ker_gemm_;
    ker_gemm0_tail_ = ker_gemm0_;
  }

#define EXECUTE_CASE(n)                                                        \
  case 0x##n:                                                                  \
    printf("execute_opt=" #n "\n");                                            \
    execute_opt_ = &elx_conv_direct_1x1_t<Type, V, I>::__execute_##n;          \
    break

  switch (xopt_) {
    EXECUTE_CASE(a000);
    EXECUTE_CASE(b000);
    EXECUTE_CASE(a060);
    EXECUTE_CASE(a061);
    EXECUTE_CASE(a069);
  default:
    el_error("Unimplemented");
    break;
  }
}

template <typename Type, const int V, const int I>
elx_conv_direct_1x1_t<Type, V, I>::~elx_conv_direct_1x1_t()
{
  if (tweights_ != nullptr) {
    free(tweights_);
    tweights_ = nullptr;
  }
  if (tinput_ != nullptr) {
    free(tinput_);
    tinput_ = nullptr;
  }
  if (toutput_ != nullptr) {
    free(toutput_);
    toutput_ = nullptr;
  }
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__trans_input_blocked(
    Type *tinput, Type *input)
{
  MD5(Type, ainput, input, this->n, this->ic3, this->I2, this->ih * this->iw, V);
  MD2(Type, atinput2, tinput, this->t2, this->ic3 * this->I2 * this->T * V);

#pragma omp parallel
#pragma omp for nowait collapse(3) schedule(static)
  for_each (_t2, this->t2) {
    for_each (_ic3, this->ic3) {
      for_each (_I2, this->I2) {
        int Tz = _t2 == this->t2 - 1 ? this->Tr : this->T;
        MD4(Type, atinput4, &md2(atinput2, _t2, 0), this->ic3, this->I2, Tz, V);
        for_each (_T, Tz) {
          int _n = (_t2 * this->T + _T) / (this->ih * this->iw);
          int _hw = (_t2 * this->T + _T) % (this->ih * this->iw);
          if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
            if (stream_in_)
              _mm512_stream_ps(&md4(atinput4, _ic3, _I2, _T, 0),
                  *(__m512 *)&md5(ainput, _n, _ic3, _I2, _hw, 0));
            else
              _mm512_store_ps(&md4(atinput4, _ic3, _I2, _T, 0),
                  *(__m512 *)&md5(ainput, _n, _ic3, _I2, _hw, 0));

          } else {
#pragma omp simd
            for_each (_V, V) {
              md4(atinput4, _ic3, _I2, _T, _V)
                = md5(ainput, _n, _ic3, _I2, _hw, _V);
            }
          }
        }
      }
    }
  }
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__trans_input_plain(
    Type *tinput, Type *input)
{
  // TODO
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::trans_input(Type *tinput, Type *input)
{
  if (input_is_bfmt_ || input_as_bfmt_)
    __trans_input_blocked(tinput, input);
  else
    __trans_input_plain(tinput, input);
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__trans_input_blocked(
    Type *tinput, Type *input, int _t2, int Tz)
{
  MD5(Type, ainput, input, this->n, this->ic3, this->I2, this->ih * this->iw, V);
  MD4(Type, atinput, tinput, this->ic3, this->I2, Tz, V);

  for_each (_ic3, this->ic3) {
    for_each (_I2, this->I2) {
      for_each (_T, Tz) {
        int _n = (_t2 * this->T + _T) / (this->ih * this->iw);
        int _hw = (_t2 * this->T + _T) % (this->ih * this->iw);
        if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
          if (stream_in_)
            _mm512_stream_ps(&md4(atinput, _ic3, _I2, _T, 0),
                *(__m512 *)&md5(ainput, _n, _ic3, _I2, _hw, 0));
          else
            _mm512_store_ps(&md4(atinput, _ic3, _I2, _T, 0),
                *(__m512 *)&md5(ainput, _n, _ic3, _I2, _hw, 0));
        } else {
#pragma omp simd
          for_each (_V, V) {
            md4(atinput, _ic3, _I2, _T, _V)
                = md5(ainput, _n, _ic3, _I2, _hw, _V);
          }
        }
      }
    }
  }
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__trans_input_plain(
    Type *tinput, Type *input, int _t2, int Tz)
{
  // TODO
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::trans_input(Type *tinput, Type *input, int _t2, int Tz)
{
  if (input_is_bfmt_ || input_as_bfmt_)
    __trans_input_blocked(tinput, input, _t2, Tz);
  else
    __trans_input_plain(tinput, input, _t2, Tz);
}


template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__trans_output_blocked(
    Type *output, Type *toutput, Type *bias)
{
  MD5(Type, aoutput, output, this->n, this->oc3, this->O2, this->oh * this->ow, V);
  MD2(Type, atoutput2, toutput, this->t2, this->oc3 * this->O2 * this->T * V);

#pragma omp parallel
#pragma omp  for nowait collapse(3) schedule(static)
  for_each (_t2, this->t2) {
    for_each (_oc3, this->oc3) {
      for_each (_O2, this->O2) {
        int Tz = _t2 == this->t2 - 1 ? this->Tr : this->T;
        MD4(Type, atoutput4, &md2(atoutput2, _t2, 0), this->oc3, this->O2, Tz, V);
        for_each (_T, Tz) {
          int _n = (_t2 * this->T + _T) / (this->oh * this->ow);
          int _hw = (_t2 * this->T + _T) % (this->oh * this->ow);
          if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
            if (stream_out_)
              _mm512_stream_ps(&md5(aoutput, _n, _oc3, _O2, _hw, 0),
                  *(__m512 *)&md4(atoutput4, _oc3, _O2, _T, 0));
            else
              _mm512_store_ps(&md5(aoutput, _n, _oc3, _O2, _hw, 0),
                  *(__m512 *)&md4(atoutput4, _oc3, _O2, _T, 0));
          } else {
#pragma omp simd
            for_each (_V, V) {
              md5(aoutput, _n, _oc3, _O2, _hw, _V)
                  = md4(atoutput4, _oc3, _O2, _T, _V);
            }
          }
        }
      }
    }
  }
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__trans_output_plain(
    Type *toutput, Type *output, Type *bias)
{
  // TODO
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::trans_output(
    Type *output, Type *toutput, Type *bias)
{
  if (output_is_bfmt_ || output_as_bfmt_)
    __trans_output_blocked(output, toutput, bias);
  else
    __trans_output_plain(output, toutput, bias);
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__trans_output_blocked(
    Type *output, Type *toutput, Type *bias, int _t2, int Tz)
{
  MD6(Type, aoutput, output, this->n, this->oc4, this->oc3, this->O2, this->oh * this->ow, V);
  MD4(Type, atoutput, toutput, this->oc3, this->O2, Tz, V);

  for_each (_oc3, this->oc3) {
    for_each (_O2, this->O2) {
      for_each (_T, Tz) {
        int _n = (_t2 * this->T + _T) / (this->oh * this->ow);
        int _hw = (_t2 * this->T + _T) % (this->oh * this->ow);
        if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
          if (stream_out_)
            _mm512_stream_ps(&md6(aoutput, _n, 0, _oc3, _O2, _hw, 0),
                *(__m512 *)&md4(atoutput, _oc3, _O2, _T, 0));
          else
            _mm512_store_ps(&md6(aoutput, _n, 0, _oc3, _O2, _hw, 0),
                *(__m512 *)&md4(atoutput, _oc3, _O2, _T, 0));
        } else {
#pragma omp simd
          for_each (_V, V) {
            md6(aoutput, _n, 0, _oc3, _O2, _hw, _V)
                = md4(atoutput, _oc3, _O2, _T, _V);
          }
        }
      }
    }
  }
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__trans_output_plain(
    Type *toutput, Type *output, Type *bias, int _t2, int Tz)
{
  // TODO
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::trans_output(
    Type *output, Type *toutput, Type *bias, int _t2, int Tz)
{
  if (output_is_bfmt_ || output_as_bfmt_)
    __trans_output_blocked(output, toutput, bias, _t2, Tz);
  else
    __trans_output_plain(output, toutput, bias, _t2, Tz);
}

// input:   t2, ic3, I2, T, V
// weights: oc3, ic3, O2, I2, V, V
// output:  t2, oc3, O2, T, V
template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::gemm(
    Type *toutput, Type *tinput, Type *tweights)
{
  MD2(Type, atinput2, tinput, this->t2, this->ic3 * this->I2 * this->T * V);
  MD3(Type, atweights, tweights, this->oc3, this->ic3, this->O2 * this->I2 * V * V);
  MD2(Type, atoutput2, toutput, this->t2, this->oc3 * this->O2 * this->T * V);
  //MD2(Type, abias, bias, this->oc3, this->O2 * V);

#pragma omp parallel for collapse(2)
  for_each (_t2, this->t2) {
    for_each (_oc3, this->oc3) {
      int Tz = _t2 == this->t2 - 1 ? this->Tr : this->T;
      auto ker_gemm = (_t2 == this->t2 - 1) ? ker_gemm0_ : ker_gemm_;
      MD2(Type, atinput, &md2(atinput2, _t2, 0), this->ic3, this->I2 * Tz * V);
      MD2(Type, atoutput, &md2(atoutput2, _t2, 0), this->oc3, this->O2 * Tz * V);
      for_each (_ic3, this->ic3) {
        // TODO
        ker_gemm(*this, &md2(atoutput, _oc3, 0), &md2(atinput, _ic3, 0),
            &md3(atweights, _oc3, _ic3, 0), _ic3 == 0);
      }
    }
  }
}

// input:   ic3, I2, T, V
// weights: oc3, ic3, O2, I2, V, V
// output:  oc3, O2, T, V
template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::gemm(
    Type *toutput, Type *tinput, Type *tweights, int _t2, int Tz)
{
  MD2(Type, atinput, tinput, this->ic3, this->I2 * Tz * V);
  MD3(Type, atweights, tweights, this->oc3, this->ic3, this->O2 * this->I2 * V * V);
  MD2(Type, atoutput, toutput, this->oc3, this->O2 * Tz * V);

  auto ker_gemm = (_t2 == this->t2 - 1) ? ker_gemm0_ : ker_gemm_;

  for_each (_oc3, this->oc3) {
    for_each (_ic3, this->ic3) {
      ker_gemm(*this, &md2(atoutput, _oc3, 0), &md2(atinput, _ic3, 0),
          &md3(atweights, _oc3, _ic3, 0), _ic3 == 0);
    }
  }
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__trans_weights_blocked(
    Type *tweights, Type *weights)
{
  MD7(Type, aweights, weights, this->oc4, this->oc3, this->O2, this->ic3, this->I2, V, V);
  MD7(Type, atweights, tweights, this->oc4, this->oc3, this->ic3, this->O2, this->I2, V,  V);

#pragma omp parallel
#pragma omp for nowait collapse(4) schedule(static)
  for_each (_oc4, this->oc4) {
    for_each (_oc3, this->oc3) {
      for_each (_ic3, this->ic3) {
        for_each (_O2, this->O2) {
          for_each (_I2, this->I2) {
            for_each (_iV, V) {
              if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
                if (stream_wei_)
                  _mm512_stream_ps(
                      &md7(atweights, _oc4, _oc3, _ic3, _O2, _I2, _iV, 0),
                      *(__m512 *)&md7(
                          aweights, _oc4, _oc3, _O2, _ic3, _I2, _iV, 0));
                else
                  _mm512_store_ps(
                      &md7(atweights, _oc4, _oc3, _ic3, _O2, _I2, _iV, 0),
                      *(__m512 *)&md7(
                          aweights, _oc4, _oc3, _O2, _ic3, _I2, _iV, 0));
              } else {
#pragma omp simd
                for_each (_oV, V) {
                  md7(atweights, _oc4, _oc3, _ic3, _O2, _I2, _iV, _oV)
                      = md7(aweights, _oc4, _oc3, _O2, _ic3, _I2, _iV, _oV);
                  ;
                }
              }
            }
          }
        }
      }
    }
  }
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__trans_weights_plain(
    Type *tweights, Type *weights)
{
  // TODO
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::trans_weights(
    Type *tweights, Type *weights)
{
  if (weights_is_bfmt_ || weights_as_bfmt_)
    __trans_weights_blocked(tweights, weights);
  else
    __trans_weights_plain(tweights, weights);
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__trans_weights_blocked(
    Type *tweights, Type *weights, int _oc4)
{
  MD7(Type, aweights, weights, this->oc4, this->oc3, this->O2, this->ic3, this->I2, V, V);
  MD6(Type, atweights, tweights, this->oc3, this->ic3, this->O2, this->I2, V,  V);

  for_each (_oc3, this->oc3) {
    for_each (_ic3, this->ic3) {
      for_each (_O2, this->O2) {
        for_each (_I2, this->I2) {
          for_each (_iV, V) {
            if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
              if (stream_wei_)
                _mm512_stream_ps(
                    &md6(atweights, _oc3, _ic3, _O2, _I2, _iV, 0),
                    *(__m512 *)&md7(
                        aweights, _oc4, _oc3, _O2, _ic3, _I2, _iV, 0));
              else
                _mm512_store_ps(
                    &md6(atweights, _oc3, _ic3, _O2, _I2, _iV, 0),
                    *(__m512 *)&md7(
                        aweights, _oc4, _oc3, _O2, _ic3, _I2, _iV, 0));
            } else {
#pragma omp simd
              for_each (_oV, V) {
                md6(atweights, _oc3, _ic3, _O2, _I2, _iV, _oV)
                    = md7(aweights, _oc4, _oc3, _O2, _ic3, _I2, _iV, _oV);
                ;
              }
            }
          }
        }
      }
    }
  }
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__trans_weights_plain(
    Type *tweights, Type *weights, int _oc4)
{
  // TODO
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::trans_weights(
    Type *tweights, Type *weights, int _oc4)
{
  if (weights_is_bfmt_ || weights_as_bfmt_)
    __trans_weights_blocked(tweights, weights, _oc4);
  else
    __trans_weights_plain(tweights, weights, _oc4);
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__execute_a000(
    Type *output, Type *input, Type *weights, Type *bias)
{
  if (is_first_run_)
    trans_weights(tweights_, weights);

  trans_input(tinput_, input);
  gemm(toutput_, tinput_, tweights_);
  trans_output(output, toutput_, bias);

  if (inference_acc_) is_first_run_ = false;
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__execute_a060(
    Type *output, Type *input, Type *weights, Type *bias)
{
  MD2(Type, atinput2, tinput_, this->t2, this->T * this->IC);
  MD2(Type, atoutput2, toutput_, mthr_, this->T * this->oc3 * this->O2 * V);
  MD2(Type, atweights2, tweights_, this->oc4, this->IC * this->oc3 * this->O2 * V);

  MD3(Type, aoutput, output, this->n, this->oc4, this->oh * this->ow * this->oc3 * this->O2 * V);
  MD2(Type, abias, bias, this->oc4, this->oc3 * this->O2 * V);

  if (is_first_run_) {
    trans_weights(tweights_, weights);
  }
  trans_input(tinput_, input);

#pragma omp parallel num_threads(mthr_) proc_bind(close)
  {
#pragma omp for nowait collapse(2)
    for_each (_t2, this->t2) {
      for_each (_oc4, this->oc4) {
        int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
        size_t ithr = omp_get_thread_num();

        gemm(&md2(atoutput2, ithr, 0), &md2(atinput2, _t2, 0),
            &md2(atweights2, _oc4, 0), _t2, Tz);
        trans_output(&md3(aoutput, 0, _oc4, 0), &md2(atoutput2, ithr, 0),
            &md2(abias, _oc4, 0), _t2, Tz);
      }
    }
  }
  if (inference_acc_)
    is_first_run_ = false;
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__execute_a061(
    Type *output, Type *input, Type *weights, Type *bias)
{
  MD2(Type, atinput2, tinput_, mthr_, this->T * this->IC);
  MD2(Type, atoutput2, toutput_, mthr_, this->T * this->oc3 * this->O2 * V);
  MD2(Type, atweights2, tweights_, this->oc4, this->IC * this->oc3 * this->O2 * V);

  MD3(Type, aoutput, output, this->n, this->oc4, this->oh * this->ow * this->oc3 * this->O2 * V);
  MD2(Type, abias, bias, this->oc4, this->oc3 * this->O2 * V);

  if (is_first_run_) {
    trans_weights(tweights_, weights);
  }
#pragma omp parallel num_threads(mthr_) proc_bind(close)
  {
#pragma omp for nowait collapse(2)
    for_each (_t2, this->t2) {
      for_each (_oc4, this->oc4) {
        int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
        size_t ithr = omp_get_thread_num();

        trans_input(&md2(atinput2, ithr, 0), input, _t2, Tz);
        gemm(&md2(atoutput2, ithr, 0), &md2(atinput2, ithr, 0),
            &md2(atweights2, _oc4, 0), _t2, Tz);
        trans_output(&md3(aoutput, 0, _oc4, 0), &md2(atoutput2, ithr, 0),
            &md2(abias, _oc4, 0), _t2, Tz);
      }
    }
  }
  if (inference_acc_)
    is_first_run_ = false;
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__execute_a069(
    Type *output, Type *input, Type *weights, Type *bias)
{
  MD2(Type, atinput2, tinput_, mthr_, this->T * this->IC);
  MD2(Type, atoutput2, toutput_, mthr_, this->T * this->oc3 * this->O2 * V);
  MD2(Type, atweights2, tweights_, mthr_, this->IC * this->oc3 * this->O2 * V);

  MD3(Type, aoutput, output, this->n, this->oc4, this->oh * this->ow * this->oc3 * this->O2 * V);
  MD2(Type, abias, bias, this->oc4, this->oc3 * this->O2 * V);

#pragma omp parallel num_threads(mthr_) proc_bind(close)
  {
#pragma omp for nowait collapse(2)
    for_each (_t2, this->t2) {
      for_each (_oc4, this->oc4) {
        int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
        size_t ithr = omp_get_thread_num();

        trans_weights(&md2(atweights2, ithr, 0), weights, _oc4);
        trans_input(&md2(atinput2, ithr, 0), input, _t2, Tz);
        gemm(&md2(atoutput2, ithr, 0), &md2(atinput2, ithr, 0),
            &md2(atweights2, ithr, 0), _t2, Tz);
        trans_output(&md3(aoutput, 0, _oc4, 0), &md2(atoutput2, ithr, 0),
            &md2(abias, _oc4, 0), _t2, Tz);
      }
    }
  }
}

// hw = t2 * T
template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__execute_b000(
    Type *output, Type *input, Type *weights, Type *bias)
{
  // weights: oc3*, O2, ic3, I2, V, V
  // input:   t3*, ic3, I2, t2*, T, V
  // output:  t3*, oc3*, O2, t2*, T, V
  MD2(Type, aweights, weights, this->oc3, this->O2 * this->IC * V);
  MD4(Type, ainput, input, this->t3, this->ic2, this->t2, this->T * V);
  MD2(Type, aoutput, output, this->t3, this->OC * this->t2 * this->T);
  MD2(Type, abias, bias, this->oc3, this->O2 * V);

#pragma omp parallel num_threads(mthr_) proc_bind(close)
#pragma omp for nowait collapse(3)
  for_each (_t3, this->t3) {
    for_each (_oc3, this->oc3) {
      for_each (_t2, this->t2) {

        MD4(Type, aoutput2, &md2(aoutput, _t3, 0), this->oc3, this->O2, this->t2, this->T * V);
        if (_oc3 == this->oc3 - 1)
          ker_bgemm0_(*this, &md4(aoutput2, _oc3, 0, _t2, 0),
            &md4(ainput, _t3, 0, _t2, 0), &md2(aweights, _oc3, 0),
            &md2(abias, _oc3, 0));
        else
          ker_bgemm_(*this, &md4(aoutput2, _oc3, 0, _t2, 0),
            &md4(ainput, _t3, 0, _t2, 0), &md2(aweights, _oc3, 0),
            &md2(abias, _oc3, 0));
      }
    }
  }
}

// hw = (t2 - 1) * T + Tr

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::execute(
    Type *output, Type *input, Type *weights, Type *bias)
{
  if (is_bfmt_)
    return (this->*execute_opt_)(output, input, weights, bias);
  else {
    // TODO
  }
}

} // namespace euler
