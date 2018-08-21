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

// XOPT
// kernel options:
//   - a: CCC, s1
//   - b: CCS, s1
//   - c: SSS: s1
//   - d: SSS: s2
// teaming: same as winograd
// fusion:  same as winograd
// dup:     same as winograd
//
// ------+-----+---------+--------+-----+--------------------------------------
//       | ker | teaming | fusion | dup |             notes
// ------+-----+---------+--------+-----+--------------------------------------
//  a061 |  a  |    -    |   t+o  |  I  | plain, stride>=1, padded
// ------+-----+---------+--------+-----+--------------------------------------
//  b061 |  b  |    -    |   t+o  |  I  | blocked, stride>=1, large batch
// ------+-----+---------+--------+-----+--------------------------------------
//  c060 |  c  |    -    |   t+o  |  -  | Tr, Or, blocked, stride=1
// ------+-----+---------+--------+-----+--------------------------------------
//  d060 |  d  |    -    |   t+o  |  -  | Or, blocked, stride>=1, small batch
// ------+-----+---------+--------+-----+--------------------------------------
//
namespace euler {

template <typename Type, const int V, const int I>
elx_conv_direct_1x1_t<Type, V, I>::elx_conv_direct_1x1_t(
    eld_conv_t<Type>& dc)
    : elx_conv_t<Type>(dc)
{
  // user input
  xopt_ = this->execution_mode;

  this->IC = ALIGNUP(this->ic, V);
  this->OC = ALIGNUP(this->oc, V);

  if (this->I2 == 0) this->I2 = this->ic2;
  if (this->O2 == 0) this->O2 = 3;
  if (this->T == 0)  this->T = 1;
  this->oc4 = this->oc4 == 0 ? 1 : this->oc4;
  this->ic4 = this->ic4 == 0 ? 1 : this->ic4;

  this->V = V;
  this->ic2 = this->IC / V;
  this->oc2 = this->OC / V;

  // t3, t2, (T, Tr)
  if (xopt_ == 0xc060) {
    bool shape_ok = this->hs == 1 && this->ws == 1 && this->lp == 0
        && this->rp == 0 && this->tp == 0 && this->bp == 0;
    if (!shape_ok)
      el_error("Shape not supported by c060");

    this->t3 = this->n;
    this->ht = this->oh;
    this->wt = this->ow;
    this->nt = this->ht * this->wt;
    this->t = this->nt * this->n;
    this->t2 = (this->nt + this->T - 1) / this->T;
    this->Tr = this->nt % this->T ? this->nt % this->T : this->T;
  } else if (xopt_ == 0xd060 || xopt_ == 0xb061 || xopt_ == 0xa061) {
    if (xopt_ == 0xd060) {
      bool shape_ok = this->hs == 2 && this->ws == 2 && this->lp == 0
          && this->rp == 0 && this->tp == 0 && this->bp == 0;
      if (!shape_ok)
        el_error("Shape not supported by d060");
    }
    this->t3 = this->n;
    this->ht = this->oh;
    this->wt = this->ow / this->T;
    this->nt = this->oh * this->ow;
    this->t2 = this->nt / this->T;
    this->Tr = this->T; // No Tr support
    this->t = this->nt * this->n;

    if (this->ht * this->hs != this->ih
        || this->wt * this->ws * this->T != this->iw) {
      el_error("Unimplemented non-unitride shape or blocking");
    }
  }

  // TODO: Ir, Or Tailing
  this->Ir = this->ic % V ? this->ic % V : V;
  this->Or = this->oc % V ? this->oc % V : V;

  // oc4, (oc3, oc3r), (O2, O2r) 
  this->oc34 = (this->oc2 + this->O2 - 1) / this->O2;
  this->O2r = this->oc2 % this->O2;
  if (this->O2r == 0) this->O2r = this->O2;
  this->oc3 = this->oc4; // FIXME, swap order
  this->oc4 = (this->oc34 + this->oc3 - 1) / this->oc3;
  this->oc3r = this->oc34 % this->oc3;
  if (this->oc3r == 0) this->oc3r = this->oc3;

  if ((xopt_ == 0xa061 || xopt_ == 0xb061)
      && (this->O2r != this->O2 || this->oc3r != this->oc3)) {
    el_error("No oc tailing for 0xa061, 0xb061");
  }

  // ic4, ic3, I3
  this->ic34 = this->ic2 / this->I2;
  this->ic3 = this->ic34 / this->ic4;
  if (this->ic4 * this->ic3 * this->I2 * V != this->IC)
    el_error("IC blocking error");

  if (xopt_ == 0xa061 && this->ic4 != 1) {
    el_error("ic4 != 1 not support in 0xa061");
  }

  is_first_run_ = true;
  inference_acc_ = false;
  mthr_ = omp_get_max_threads();
  inference_acc_ = this->prop_kind == forward_inference;

  prepare_execute_opt();
  bind_execute_functions();

  // dbg
  printf("T=%d, Tr=%d, t2=%d, ht=%d, wt=%d, t=%d\n", this->T, this->Tr, this->t2, this->ht, this->wt, this->t);
  printf("V=%d, Ir=%d, I2=%d, ic3=%d, ic4=%d, IC=%d\n", this->V, this->Ir, this->I2, this->ic3, this->ic4, this->IC);
  printf("V=%d, Or=%d, O2=%d, oc3=%d, oc4=%d, O2r=%d, oc3r=%d, OC=%d\n", this->V, this->Or, this->O2, this->oc3, this->oc4, this->O2r, this->oc3r, this->OC);
}


template <typename Type, const int V, const int I>
int  elx_conv_direct_1x1_t<Type, V, I>::prepare_execute_opt()
{
  size_t tweights_size = 0, tinput_size = 0, toutput_size = 0;
  size_t l1_usage = 0, l2_usage = 0;

  stream_in_ = this->streaming_input
      ? (this->streaming_input == STORE_STREAMING) : false;
  stream_wei_ = this->streaming_weights
      ? (this->streaming_weights == STORE_STREAMING) : false;
  stream_out_ = this->streaming_output
      ? (this->streaming_output == STORE_STREAMING) : false;

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
  case 0xa061:
    toutput_size = mthr_ * this->oc3 * this->O2 * this->T * V;
  case 0xb061:
    tinput_size = mthr_ * this->ic3 * this->I2 * this->T * V;
    tweights_size = this->IC * this->OC;
    break;
  case 0xc060:
  case 0xd060:
    l2_usage = this->IC * this->OC / this->oc4 + this->IC * this->T
        + this->OC / this->oc4 * this->T;
    break;
  default:
      el_error("Unknown xopt!");
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
#define GEMM_CASE(z, T_, O_)                                                   \
  case T_:                                                                     \
    if (this->hs == 1 && xopt_ == 0xc060) {                                    \
      if (this->with_bias)                                                     \
        *func = convolution_direct_1x1_kernel<Type, 1, O_, T_, SSS, V, I,      \
            BIAS(true), RELU(false), SUM(false)>::gemm;                        \
      else                                                                     \
        *func = convolution_direct_1x1_kernel<Type, 1, O_, T_, SSS, V, I,      \
            BIAS(false), RELU(false), SUM(false)>::gemm;                       \
    } else if (this->hs == 2 && xopt_ == 0xd060) {                             \
      if (this->with_bias)                                                     \
        *func = convolution_direct_1x1_kernel<Type, 2, O_, T_, SSS, V, I,      \
            BIAS(true), RELU(false), SUM(false)>::gemm;                        \
      else                                                                     \
        *func = convolution_direct_1x1_kernel<Type, 2, O_, T_, SSS, V, I,      \
            BIAS(false), RELU(false), SUM(false)>::gemm;                       \
    } else if (xopt_ == 0xb061) {                                              \
      if (this->with_bias)                                                     \
        *func = convolution_direct_1x1_kernel<Type, 1, O_, T_, CCS, V, I,      \
            BIAS(true), RELU(false), SUM(false)>::gemm;                        \
      else                                                                     \
        *func = convolution_direct_1x1_kernel<Type, 1, O_, T_, CCS, V, I,      \
            BIAS(false), RELU(false), SUM(false)>::gemm;                       \
    } else if (xopt_ == 0xa061) {                                              \
      if (this->with_bias)                                                     \
        *func = convolution_direct_1x1_kernel<Type, 1, O_, T_, CCC, V, I,      \
            BIAS(true), RELU(false), SUM(false)>::gemm;                        \
      else                                                                     \
        *func = convolution_direct_1x1_kernel<Type, 1, O_, T_, CCC, V, I,      \
            BIAS(false), RELU(false), SUM(false)>::gemm;                       \
    }                                                                          \
    break;

  auto bind_kernel = [&](int O2_, int T_,
                  decltype(convolution_direct_1x1_kernel<Type, 1, 1, 1, 0, V, I,
                           false, false, false>::gemm) **func) {
    switch (O2_) {
    case 1:
      switch (T_) {
        BOOST_PP_REPEAT_FROM_TO(1, 33, GEMM_CASE, 1);
      default:
        el_error("Convolution_direct_1x1: Unimplemented T>32 in O=1");
        break;
      }
      break;
    case 2:
      switch (T_) {
        BOOST_PP_REPEAT_FROM_TO(1, 15, GEMM_CASE, 2);
      default:
        el_error("Convolution_direct_1x1: Unimplemented T>14 in O=2");
        break;
      }
      break;
    case 3:
      switch (T_) {
        BOOST_PP_REPEAT_FROM_TO(1, 15, GEMM_CASE, 3);
      default:
        el_error("Convolution_direct_1x1: Unimplemented T>14 in O=3");
        break;
      }
      break;
    case 4:
      switch (T_) {
        BOOST_PP_REPEAT_FROM_TO(1, 15, GEMM_CASE, 4);
      default:
        el_error("Convolution_direct_1x1: Unimplemented T>14 in O=4");
        break;
      }
      break;
    case 5:
      switch (T_) {
        BOOST_PP_REPEAT_FROM_TO(1, 6, GEMM_CASE, 5);
      default:
        el_error("Convolution_direct_1x1: Unimplemented T>5 in O=5");
        break;
      }
      break;
    case 6:
      switch (T_) {
        BOOST_PP_REPEAT_FROM_TO(1, 5, GEMM_CASE, 6);
      default:
        el_error("Convolution_direct_1x1: Unimplemented T>4 in O=6");
        break;
      }
      break;
    case 7:
      switch (T_) {
        BOOST_PP_REPEAT_FROM_TO(1, 4, GEMM_CASE, 7);
      default:
        el_error("Convolution_direct_1x1: Unimplemented T>3 in O=7");
        break;
      }
      break;
    case 8:
      switch (T_) {
        BOOST_PP_REPEAT_FROM_TO(1, 9, GEMM_CASE, 8);
      default:
        el_error("Convolution_direct_1x1: Unimplemented T>8 in O=8");
        break;
      }
      break;

    default:
      el_error("O2 > 8 unsupported");
    }
  };

  bind_kernel(this->O2, this->T, &ker_gemm_O_T_);
  bind_kernel(this->O2, this->Tr, &ker_gemm_O_Tr_);
  bind_kernel(this->O2r, this->T, &ker_gemm_Or_T_);
  bind_kernel(this->O2r, this->Tr, &ker_gemm_Or_Tr_);

#define EXECUTE_CASE(n)                                                        \
  case 0x##n:                                                                  \
    printf("execute_opt=" #n "\n");                                            \
    execute_opt_ = &elx_conv_direct_1x1_t<Type, V, I>::__execute_##n;          \
    break

  switch (xopt_) {
    EXECUTE_CASE(a061);
    EXECUTE_CASE(b061);
    EXECUTE_CASE(c060);
    EXECUTE_CASE(d060);
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
void elx_conv_direct_1x1_t<Type, V, I>::gemm_c060(Type *output, Type *input,
    Type *weights, Type *bias, int _ic4, int _oc4, int _t2)
{
  // weights: oc3*, O2(O2r), ic4*, ic3*, I2, V, V
  // input:   ic3*, I2, t2*, T(Tr), V
  // output:  oc3*, O2(O2r), t2*, T(Tr), V
  MD2(Type, ainput, input, this->ic3, this->I2 * this->ih * this->iw * V);
  MD2(Type, aoutput, output, this->oc3, this->O2 * this->oh * this->ow * V);
  MD5(Type, aweights, weights, this->oc3, this->O2, this->ic4, this->ic3,
      this->I2 * V * V);
  MD2(Type, abias, bias, this->oc3, this->O2 * V);

  iter_each (_ic3, this->ic3) {
    bool reset = _ic4 == 0 && _ic3 == 0;
    int oc3 = _oc4 == this->oc4 - 1 ? this->oc3r : this->oc3;
    MD2(Type, ainput2, &md2(ainput, _ic3, 0), this->t2, this->T * V);

    iter_each (_oc3, oc3) {
      MD2(Type, aoutput2, &md2(aoutput, _oc3, 0), this->t2, this->T * V);

      if (_oc4 == this->oc4 - 1 && _oc3 == oc3 - 1) {
        if (_t2 == this->t2 - 1)
          ker_gemm_Or_Tr_(*this, &md2(aoutput2, _t2, 0), &md2(ainput2, _t2, 0),
              &md5(aweights, _oc3, 0, _ic4, _ic3, 0), &md2(abias, _oc3, 0),
              reset);
        else
          ker_gemm_Or_T_(*this, &md2(aoutput2, _t2, 0), &md2(ainput2, _t2, 0),
              &md5(aweights, _oc3, 0, _ic4, _ic3, 0), &md2(abias, _oc3, 0),
              reset);
      } else {
        if (_t2 == this->t2 - 1)
          ker_gemm_O_Tr_(*this, &md2(aoutput2, _t2, 0), &md2(ainput2, _t2, 0),
              &md5(aweights, _oc3, 0, _ic4, _ic3, 0), &md2(abias, _oc3, 0),
              reset);
        else
          ker_gemm_O_T_(*this, &md2(aoutput2, _t2, 0), &md2(ainput2, _t2, 0),
              &md5(aweights, _oc3, 0, _ic4, _ic3, 0), &md2(abias, _oc3, 0),
              reset);
      }
    }
  }
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__execute_c060(
    Type *output, Type *input, Type *weights, Type *bias)
{
  // weights: oc4*, oc3, O2(O2r), ic4*, ic3, I2, V, V
  // input:   t3*, ic4*, ic3, I2, t2*, T(Tr), V
  // output:  t3*, oc4*, oc3, O2(O2r), t2*, T(Tr), V
  MD2(Type, aweights, weights, this->oc4, this->oc3 * this->O2 * this->IC * V);
  MD3(Type, ainput, input, this->t3, this->ic4, this->ic3 * this->I2 * this->ih * this->iw * V);
  MD2(Type, aoutput, output, this->t3, this->OC * this->oh * this->ow);
  MD2(Type, abias, bias, this->oc4, this->oc3 * this->O2 * V);

  iter_each (_ic4, this->ic4) {
#pragma omp parallel num_threads(mthr_) proc_bind(close)
#pragma omp for nowait collapse(3)
    iter_each (_t3, this->t3) {
      iter_each (_oc4, this->oc4) {
        iter_each (_t2, this->t2) {
          MD2(Type, aoutput2, &md2(aoutput, _t3, 0), this->oc4,
              this->oc3 * this->O2 * this->oh * this->ow * V);

          gemm_c060(&md2(aoutput2, _oc4, 0), &md3(ainput, _t3, _ic4, 0),
              &md2(aweights, _oc4, 0), &md2(abias, _oc4, 0), _ic4, _oc4, _t2);
        }
      }
    }
  }
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__trans_weights_blocked(
    Type *tweights, Type *weights)
{
  // oc4, (oc3, oc3r), (O2, O2r), ic4, ic3, I2, V, V -> ic4, oc4, ic3, (oc3, oc3r), I2, V, (O2, O2r), V
  MD8(Type, aweights, weights, this->oc4, this->oc3, this->O2, this->ic4, this->ic3, this->I2, V, V);
  MD8(Type, atweights, tweights, this->oc4, this->ic4, this->oc3, this->ic3, this->I2, V, this->O2, V);

#pragma omp parallel
#pragma omp for nowait collapse(4) schedule(static)
  iter_each (_oc4, this->oc4) {
    iter_each (_ic4, this->ic4) {
      iter_each (_oc3, this->oc3) {
        iter_each (_ic3, this->ic3) {
          iter_each (_I2, this->I2) {
            iter_each (_iV, V) {
              iter_each (_O2, this->O2) {
                if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
                  if (stream_wei_)
                    _mm512_stream_ps(&md8(atweights, _oc4, _ic4, _oc3, _ic3,
                                         _I2, _iV, _O2, 0),
                        *(__m512 *)&md8(aweights, _oc4, _oc3, _O2, _ic4, _ic3,
                            _I2, _iV, 0));
                  else
                    _mm512_store_ps(&md8(atweights, _oc4, _ic4, _oc3, _ic3, _I2,
                                        _iV, _O2, 0),
                        *(__m512 *)&md8(aweights, _oc4, _oc3, _O2, _ic4, _ic3,
                            _I2, _iV, 0));
                } else {
#pragma omp simd
                  iter_each (_oV, V) {
                    md8(atweights, _oc4, _ic4, _oc3, _ic3, _I2, _iV, _O2, _oV)
                        = md8(aweights, _oc4, _oc3, _O2, _ic4, _ic3, _I2, _iV,
                            _oV);
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
void elx_conv_direct_1x1_t<Type, V, I>::__trans_input_blocked(
    Type *tinput, Type *input)
{
  // ic3, I2, ht, hs, wt, T, ws, V -> ht, wt | ic3, I2, T, V
  MD8(Type, ainput, input, this->ic3, this->I2, this->ht, this->hs, this->wt, this->T, this->ws, V);
  MD4(Type, atinput, tinput, this->ic3, this->I2, this->T, V);

  iter_each (_ic3, this->ic3) {
    iter_each (_I2, this->I2) {
      iter_each (_T, this->T) {
#pragma omp simd
        iter_each (_V, V) {
          md4(atinput, _ic3, _I2, _T, _V)
              = md8(ainput, _ic3, _I2, 0, 0, 0, _T, 0, _V);
        }
      }
    }
  }
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__trans_input_plain(
    Type *tinput, Type *input)
{
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::trans_input(
    Type *tinput, Type *input)
{
  if (input_is_bfmt_ || input_as_bfmt_)
    __trans_input_blocked(tinput, input);
  else
    __trans_input_plain(tinput, input);
}


template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__trans_output_blocked(
    Type *output, Type *toutput)
{
  // oc3, O2, T, V => n, oc4 | oc3, O2, ht, wt, T, V
  MD4(Type, atoutput, toutput, this->oc3, this->O2, this->T, V);
  MD6(Type, aoutput, output, this->oc3, this->O2, this->ht, this->wt, this->T, V);

  iter_each (_oc3, this->oc3) {
    iter_each (_O2, this->O2) {
      iter_each (_T, this->T) {
#pragma omp simd
        iter_each (_V, V) {
          md6(aoutput, _oc3, _O2, 0, 0, _T, _V)
              = md4(atoutput, _oc3, _O2, _T, _V);
        }
      }
    }
  }
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__trans_output_plain(
    Type *output, Type *toutput)
{
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::trans_output(
    Type *output, Type *toutput)
{
  if (output_is_bfmt_ || output_as_bfmt_)
    __trans_output_blocked(output, toutput);
  else
    __trans_output_plain(output, toutput);
}


template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::gemm_b061(Type *output, Type *input,
    Type *weights, Type *bias, int _ic4, int _oc4)
{
  // weights: oc3*, ic3*, O2, I2, V, V
  // input:   ic3*, I2, T, V
  // output:  oc3*, O2, ht*, wt*, T, V
  MD2(Type, ainput, input, this->ic3, this->I2 * this->T * V);
  MD5(Type, aoutput, output, this->oc3, this->O2, this->ht, this->wt, this->T * V);
  MD3(Type, aweights, weights, this->oc3, this->ic3, this->O2 * this->I2 * V * V);
  MD2(Type, abias, bias, this->oc3, this->O2 * V);

  iter_each (_ic3, this->ic3) {
    bool reset = _ic4 == 0 && _ic3 == 0;
    iter_each (_oc3, this->oc3) {
      ker_gemm_O_T_(*this, &md5(aoutput, _oc3, 0, 0, 0, 0),
          &md2(ainput, _ic3, 0), &md3(aweights, _oc3, _ic3, 0),
          &md2(abias, _oc3, 0), reset);
    }
  }
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::gemm_a061(Type *output, Type *input,
    Type *weights, Type *bias, int _ic4, int _oc4)
{
  // weights: oc3*, ic3*, O2, I2, V, V
  // input:   ic3*, I2, T, V
  // output:  oc3*, O2, T, V
  MD2(Type, ainput, input, this->ic3, this->I2 * this->T * V);
  MD2(Type, aoutput, output, this->oc3, this->O2 * this->T * V);
  MD3(Type, aweights, weights, this->oc3, this->ic3, this->O2 * this->I2 * V * V);
  MD2(Type, abias, bias, this->oc3, this->O2 * V);

  iter_each (_ic3, this->ic3) {
    bool reset = _ic4 == 0 && _ic3 == 0;
    iter_each (_oc3, this->oc3) {
      ker_gemm_O_T_(*this, &md2(aoutput, _oc3, 0),
          &md2(ainput, _ic3, 0), &md3(aweights, _oc3, _ic3, 0),
          &md2(abias, _oc3, 0), reset);
    }
  }
}


template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__execute_b061(
    Type *output, Type *input, Type *weights, Type *bias)
{
  // weights: oc4*, oc3, O2(O2r), ic4*, ic3, I2, V, V
  // input:   t3*, ic4*, ic3, I2, t2*, T(Tr), V
  // output:  t3*, oc4*, oc3, O2(O2r), t2*, T(Tr), V
  MD10(Type, ainput, input, this->t3, this->ic4, this->ic3, this->I2, this->ht, this->hs, this->wt, this->T, this->ws, V);
  MD2(Type, aoutput, output, this->t3, this->OC * this->oh * this->ow);
  MD2(Type, abias, bias, this->oc4, this->oc3 * this->O2 * V);

  MD2(Type, atinput, tinput_, mthr_, this->ic3 * this->I2 * this->T * V);
  MD3(Type, atweights, tweights_, this->oc4, this->ic4, this->oc3 * this->ic3 * this->O2 * this->I2 * V * V);

  if (is_first_run_) {
    trans_weights(tweights_, weights);
  }

  iter_each (_ic4, this->ic4) {
#pragma omp parallel num_threads(mthr_) proc_bind(close)
#pragma omp for nowait collapse(4)
    iter_each (_t3, this->t3) {
      iter_each (_oc4, this->oc4) {
        iter_each (_ht, this->ht) {
          iter_each (_wt, this->wt) {
            size_t ithr = omp_get_thread_num();
            MD5(Type, aoutput2, &md2(aoutput, _t3, 0), this->oc4,
                this->oc3 * this->O2, this->ht, this->wt, this->T * V);

            trans_input(&md2(atinput, ithr, 0),
                &md10(ainput, _t3, _ic4, 0, 0, _ht, 0, _wt, 0, 0, 0));
            gemm_b061(&md5(aoutput2, _oc4, 0, _ht, _wt, 0), &md2(atinput, ithr, 0),
                &md3(atweights, _oc4, _ic4, 0), &md2(abias, _oc4, 0), _ic4,
                _oc4);
          }
        }
      }
    }
  }

  if (inference_acc_) is_first_run_ = false;
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__execute_a061(
    Type *output, Type *input, Type *weights, Type *bias)
{
  MD10(Type, ainput, input, this->t3, this->ic4, this->ic3, this->I2, this->ht, this->hs, this->wt, this->T, this->ws, V);
  MD8(Type, aoutput, output, this->t3, this->oc4, this->oc3, this->O2, this->ht, this->wt, this->T, V);
  MD2(Type, abias, bias, this->oc4, this->oc3 * this->O2 * V);

  MD2(Type, atoutput, toutput_, mthr_, this->oc3 * this->O2 * this->T * V);
  MD2(Type, atinput, tinput_, mthr_, this->ic3 * this->I2 * this->T * V);
  MD3(Type, atweights, tweights_, this->oc4, this->ic4, this->oc3 * this->ic3 * this->O2 * this->I2 * V * V);

  if (is_first_run_) {
    trans_weights(tweights_, weights);
  }

#pragma omp parallel num_threads(mthr_) proc_bind(close)
#pragma omp for nowait collapse(4)
  iter_each (_t3, this->t3) {
    iter_each (_oc4, this->oc4) {
      iter_each (_ht, this->ht) {
        iter_each (_wt, this->wt) {
          size_t ithr = omp_get_thread_num();
          MD5(Type, aoutput2, &md2(aoutput, _t3, 0), this->oc4,
              this->oc3 * this->O2, this->ht, this->wt, this->T * V);

          trans_input(&md2(atinput, ithr, 0),
              &md10(ainput, _t3, 0, 0, 0, _ht, 0, _wt, 0, 0, 0));
          gemm_a061(&md2(atoutput, ithr, 0), &md2(atinput, ithr, 0),
              &md3(atweights, _oc4, 0, 0), &md2(abias, _oc4, 0), 0, _oc4);
          trans_output(&md8(aoutput, _t3, _oc4, 0, 0, _ht, _wt, 0, 0),
              &md2(atoutput, ithr, 0));
        }
      }
    }
  }

  if (inference_acc_)
    is_first_run_ = false;
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::gemm_d060(Type *output, Type *input,
    Type *weights, Type *bias, int _ic4, int _oc4, int _ht, int _wt)
{
  // weights: oc3*, O2, ic4*, ic3*, I2, V, V
  // input:   ic3*, I2, ht*, hs*, wt*, T, ws, V
  // output:  oc3*, O2(O2r), ht*, wt*, T, V
  MD5(Type, ainput, input, this->ic3, this->I2, this->ih, this->wt, this->T * this->ws * V);
  MD5(Type, aoutput, output, this->oc3, this->O2, this->ht, this->wt,  this->T * V);
  MD5(Type, aweights, weights, this->oc3, this->O2, this->ic4, this->ic3,
      this->I2 * V * V);
  MD2(Type, abias, bias, this->oc3, this->O2 * V);

  iter_each (_ic3, this->ic3) {
    bool reset = _ic4 == 0 && _ic3 == 0;
    int oc3 = _oc4 == this->oc4 - 1 ? this->oc3r : this->oc3;

    iter_each (_oc3, oc3) {
      if (_oc4 == this->oc4 - 1 && _oc3 == oc3 - 1) {
        ker_gemm_Or_T_(*this, &md5(aoutput, _oc3, 0, 0, 0, 0),
            &md5(ainput, _ic3, 0, 0, 0, 0),
            &md5(aweights, _oc3, 0, _ic4, _ic3, 0), &md2(abias, _oc3, 0),
            reset);
      } else {
        ker_gemm_O_T_(*this, &md5(aoutput, _oc3, 0, 0, 0, 0),
            &md5(ainput, _ic3, 0, 0, 0, 0),
            &md5(aweights, _oc3, 0, _ic4, _ic3, 0), &md2(abias, _oc3, 0),
            reset);
      }
    }
  }
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__execute_d060(
    Type *output, Type *input, Type *weights, Type *bias)
{
  // weights: oc4*, oc3, O2(O2r), ic4*, ic3, I2, V, V
  // input:   t3*, ic4*, ic3, I2, ht*, S, wt*, T, S, V
  // output:  t3*, oc4*, oc3, O2(O2r), ht*wt*, T, V
  MD2(Type, aweights, weights, this->oc4, this->oc3 * this->O2 * this->IC * V);
  MD7(Type, ainput, input, this->t3, this->ic4, this->ic3 * this->I2, this->ht, this->hs, this->wt, this->T * this->ws * V);
  MD2(Type, aoutput, output, this->t3, this->OC * this->oh * this->ow);
  MD2(Type, abias, bias, this->oc4, this->oc3 * this->O2 * V);

  iter_each (_ic4, this->ic4) {
#pragma omp parallel num_threads(mthr_) proc_bind(close)
#pragma omp for nowait collapse(4)
    iter_each (_t3, this->t3) {
      iter_each (_oc4, this->oc4) {
        iter_each (_ht, this->ht) {
          iter_each (_wt, this->wt) {
            MD5(Type, aoutput2, &md2(aoutput, _t3, 0), this->oc4,
                this->oc3 * this->O2, this->ht, this->wt, this->T * V);

            gemm_d060(&md5(aoutput2, _oc4, 0, _ht, _wt, 0),
                &md7(ainput, _t3, _ic4, 0, _ht, 0, _wt, 0),
                &md2(aweights, _oc4, 0), &md2(abias, _oc4, 0), _ic4, _oc4, _ht,
                _wt);
          }
        }
      }
    }
  }
}

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
