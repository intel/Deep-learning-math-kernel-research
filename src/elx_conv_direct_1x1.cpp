#include <string.h>
#include "el_intrin.hpp"
#include "el_utils.hpp"
#include "elx_conv_direct_1x1.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "euler.hpp"

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
  if (this->T == 0)  this->T = 1;
  if (this->O == 0)  this->O = 1;
  if (this->O1 == 0) this->O1 = 1;
  this->O2 = this->O * this->O1;

  this->oc4 = this->oc4 == 0 ? 1 : this->oc4;
  this->ic4 = this->ic4 == 0 ? 1 : this->ic4;

  this->V = V;
  this->ic2 = this->IC / V;
  this->oc2 = this->OC / V;

  no_pad_ = this->lp == 0 && this->rp == 0 && this->tp == 0 && this->bp == 0;
  if (!no_pad_) {
    if (xopt_ != 0xa061)
      el_error("Only 0xa061 support padding");
    bool shape_ok =
      (this->oh == (this->ih - 1 + this->tp + this->bp) / this->hs + 1) &&
      (this->ow == (this->iw - 1 + this->lp + this->rp) / this->ws + 1);
    if (!shape_ok)
      el_error("Unmatched paddding shape not supported by a061");
  }

  // t3, t2, (T, Tr)
  if (xopt_ == 0xc060 || xopt_ == 0xf061) {
    bool shape_ok = this->hs == 1 && this->ws == 1 && no_pad_;
    if (!shape_ok)
      el_error("Shape not supported by c060");

    this->t3 = this->n;
    this->ht = this->oh;
    this->wt = this->ow;
    this->nt = this->ht * this->wt;
    this->t = this->nt * this->n;
    this->t2 = (this->nt + this->T - 1) / this->T;
    this->Tr = this->nt % this->T ? this->nt % this->T : this->T;
  } else if (xopt_ == 0xd060 || xopt_ == 0xa061 ||
      xopt_ == 0xb061 || xopt_ == 0xe060) {
    if (xopt_ == 0xd060) {
      bool shape_ok = this->hs == 2 && this->ws == 2 && no_pad_;
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

    if (no_pad_ && (this->ht * this->hs != this->ih
        || this->wt * this->ws * this->T != this->iw)) {
      el_error("Unimplemented non-unitride shape or blocking");
    }
  }

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

  if ((xopt_ == 0xa061 || xopt_ == 0xb061 || xopt_ == 0xe060 || xopt_ == 0xf061)
      && (this->O2r != this->O2 || this->oc3r != this->oc3)) {
    el_error("No oc tailing for 0xa061, 0xb061, 0xe060, 0xf061");
  }

  // ic4, ic3, I3
  this->ic34 = this->ic2 / this->I2;
  this->ic3 = this->ic34 / this->ic4;
  if (this->ic4 * this->ic3 * this->I2 * V != this->IC)
    el_error("IC blocking error");

  if ((xopt_ == 0xa061 || xopt_ == 0xf061) && this->ic4 != 1) {
    el_error("ic4 != 1 not support in 0xa061 and 0xf061");
  }

  attr_ = 0x0;
  is_first_run_ = true;
  inference_acc_ = false;
  mthr_ = omp_get_max_threads();
  inference_acc_ = this->prop_kind == forward_inference;

  attr_ = this->with_bias ? set_attr(attr_, bias_idx) : attr_;
  if (xopt_ == 0xb061 || xopt_ == 0xe060 || xopt_ == 0xc060 || xopt_ == 0xd060) {
    attr_ = this->with_ip_sum ? set_attr(attr_, ip_sum_idx) : attr_;
  }

  prepare_execute_opt();
  bind_execute_functions();

  // dbg
  printf("T=%d, Tr=%d, t2=%d, ht=%d, wt=%d, t=%d\n",
      this->T, this->Tr, this->t2, this->ht, this->wt, this->t);
  printf("V=%d, Ir=%d, I2=%d, ic3=%d, ic4=%d, IC=%d\n",
      this->V, this->Ir, this->I2, this->ic3, this->ic4, this->IC);
  printf("V=%d, Or=%d, O2=%d (O=%d, O1=%d), oc3=%d, oc4=%d, O2r=%d, oc3r=%d, OC=%d\n",
      this->V, this->Or, this->O2, this->O, this->O1,
      this->oc3, this->oc4, this->O2r, this->oc3r, this->OC);
}


template <typename Type, const int V, const int I>
int  elx_conv_direct_1x1_t<Type, V, I>::prepare_execute_opt()
{
  size_t tweights_size = 0, tinput_size = 0, toutput_size = 0;
  size_t binput_size = 0, bweights_size = 0, boutput_size = 0;

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

  if (this->with_ip_sum && this->with_relu && !output_is_bfmt_) {
    el_error("Unimplemented: fuse sum (plain format) and relu together");
  }

  if (this->ic4 > 1 && this->Ir != V) {
    el_error("Unimplemented: ic4 > 1 for IC % V != 0");
  }
  if (this->oc4 > 1 && this->Or != V) {
    el_error("Unimplemented: oc4 > 1 for OC % V != 0");
  }

  if (!is_bfmt_ && (xopt_ != 0xa061 && xopt_ != 0xf061)) {
    el_error("Unimplemented: only a061, f061 mode support plain format\n");
  }

  if (input_as_bfmt_)
    binput_size = this->n * this->IC * this->ih * this->iw;
  if (weights_as_bfmt_)
    bweights_size = this->OC * this->IC;
  if (output_as_bfmt_)
    boutput_size = this->n * this->OC * this->oh * this->ow;

  tweights_ = nullptr;
  tinput_ = nullptr;
  toutput_ = nullptr;
  tinput_msk_ = nullptr;
  binput_ = nullptr;
  bweights_ = nullptr;
  boutput_ = nullptr;

  switch (xopt_) {
  case 0xa061:
    toutput_size = mthr_ * this->oc3 * this->O2 * this->T * V;
  case 0xb061:
    tinput_msk_ = (unsigned char *)malloc(mthr_ * this->ht * this->wt);
    tinput_size = mthr_ * this->ic3 * this->I2 * V * this->ht * this->wt * this->T;
    tweights_size = this->IC * this->OC;
    break;
  case 0xe060:
    tweights_size = this->IC * this->OC;
    break;
  case 0xf061:
    tinput_msk_ = (unsigned char *)malloc(mthr_ * this->t2);
    toutput_size = mthr_ * this->oc3 * this->O2 * this->T * V;
    tinput_size = mthr_ * this->ic3 * this->I2 * this->T * V * this->t2;
    tweights_size = this->IC * this->OC;
    break;
  case 0xc060:
  case 0xd060:
    break;
  default:
      el_error("Unknown xopt!");
      return -1;
    break;
  }

  const size_t align = PAGE_SIZE / sizeof(Type);
#define WEIGHTS_MAX_PRELOAD 4
  if (tweights_size > 0)
    tweights_size += WEIGHTS_MAX_PRELOAD * V;

  tweights_size_ = tweights_size > 0 ? alignup(tweights_size, align) : 0;
  tinput_size_ = tinput_size > 0 ? alignup(tinput_size, align) : 0;
  toutput_size_ = toutput_size > 0 ? alignup(toutput_size, align) : 0;
  binput_size_ = binput_size > 0 ? alignup(binput_size, align) : 0;
  bweights_size_ = bweights_size > 0 ? alignup(bweights_size, align) : 0;
  boutput_size_ = boutput_size > 0 ? alignup(boutput_size, align) : 0;

  scratch_ = nullptr;
  size_t workspace_size = tweights_size_;
  size_t scratch_size = tinput_size_ + tweights_size_ + toutput_size_
      + binput_size_ + bweights_size_ + boutput_size_;
  // TODO: user provided buffer
  if (scratch_size != 0)
    scratch_ = (Type *)galloc::acquire(scratch_size * sizeof(Type));

  set_trans_buffers();

  // dbg
  printf("nthreads=%d, mthr_=%d\n", this->nthreads, mthr_);
  return 0;
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::set_trans_buffers()
{
  tinput_ = (Type *)galloc::get();
  tweights_ = tinput_ + tinput_size_;
  toutput_ = tweights_ + tweights_size_;
  binput_ = toutput_ + toutput_size_;
  bweights_ = binput_ + binput_size_;
  boutput_ = bweights_ + bweights_size_;
}

template <typename Type, const int V, const int I>
elx_conv_direct_1x1_t<Type, V, I>::~elx_conv_direct_1x1_t()
{
  if (tinput_msk_ != nullptr) {
    free(tinput_msk_);
    tinput_msk_ = nullptr;
  }

  galloc::release();
}

// n, ic, ih, iw => n, ic2, ih, iw, V
template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::trans_input_2_blocked(
    Type *binput, Type *input)
{
  MD4(Type, abinput4, binput, this->n, this->ic2, this->ih * this->iw, V);
  SET_EPI32(this->ih * this->iw)

  if (this->Ir == V) {
    MD4(Type, ainput4, input, this->n, this->ic2, V, this->ih * this->iw);
#pragma omp parallel num_threads(mthr_) proc_bind(close)
#pragma omp for nowait collapse(3)
    iter_each(_n, this->n) {
    iter_each(_ic2, this->ic2) {
    iter_each(_t, this->ih * this->iw) {
      if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
         constexpr int scale = sizeof(Type);
         __m<V> ain = _mm<V>::i32gather_ps(vindex,
             &md4(ainput4, _n, _ic2, 0, _t), scale);
         _mm<V>::store_ps(&md4(abinput4, _n, _ic2, _t, 0), ain);
      } else {
#pragma omp simd
        iter_each (_iv, V) {
          md4(abinput4, _n, _ic2, _t, _iv) = md4(ainput4, _n, _ic2, _iv, _t);
        }
      }
    }}}
  } else {
    MD3(Type, ainput3, input, this->n, this->ic, this->ih * this->iw);
#pragma omp parallel num_threads(mthr_) proc_bind(close)
#pragma omp for nowait collapse(3)
    iter_each(_n, this->n) {
    iter_each(_ic2, this->ic2) {
    iter_each(_t, this->ih * this->iw) {
      bool is_Ir = _ic2 == this->ic2 - 1;
      if (is_Ir) {
#pragma omp simd
        iter_each (_iv, this->Ir) {
          md4(abinput4, _n, _ic2, _t, _iv)
              = md3(ainput3, _n, (this->ic2 - 1) * V + _iv, _t);
        }
      } else {
        if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
           constexpr int scale = sizeof(Type);
           __m<V> ain = _mm<V>::i32gather_ps(vindex,
               &md3(ainput3, _n, _ic2 * V , _t), scale);
           _mm<V>::store_ps(&md4(abinput4, _n, _ic2, _t, 0), ain);
        } else {
#pragma omp simd
          iter_each (_iv, V) {
            md4(abinput4, _n, _ic2, _t, _iv)
                = md3(ainput3, _n, _ic2 * V + _iv, _t);
          }
        }
      }
    }}}
  }
}

// oc, ic => oc2, ic2, V, V
template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::trans_weights_2_blocked(
    Type *bweights, Type *weights)
{
  MD4(Type, abweights4, bweights, this->oc2, this->ic2, V, V);
  SET_EPI32(this->ic)

  if (this->Ir == V && this->Or == V) {
    MD4(Type, aweights4, weights, this->oc2, V, this->ic2, V);
#pragma omp parallel num_threads(mthr_) proc_bind(close)
#pragma omp for nowait collapse(3)
    iter_each(_oc2, this->oc2) {
    iter_each(_ic2, this->ic2) {
    iter_each(_iv, V) {
      if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
         constexpr int scale = sizeof(Type);
         __m<V> t = _mm<V>::i32gather_ps(vindex,
             &md4(aweights4, _oc2, 0, _ic2, _iv), scale);
         _mm<V>::store_ps(&md4(abweights4, _oc2, _ic2, _iv, 0), t);
      } else {
#pragma omp simd
        iter_each (_ov, V) {
          md4(abweights4, _oc2, _ic2, _iv, _ov)
              = md4(aweights4, _oc2, _ov, _ic2, _iv);
        }
      }
    }}}
  } else {
    MD2(Type, aweights2, weights, this->oc, this->ic);
#pragma omp parallel num_threads(mthr_) proc_bind(close)
#pragma omp for nowait collapse(2)
    iter_each(_oc2, this->oc2) {
    iter_each(_ic2, this->ic2) {
      bool is_Or = _oc2 == this->oc2 - 1;
      bool is_Ir = _ic2 == this->ic2 - 1;
      int iV = is_Ir ? this->Ir : V;
      if (is_Or) {
        iter_each(_iv, iV) {
#pragma omp simd
          iter_each (_ov, this->Or) {
            md4(abweights4, _oc2, _ic2, _iv, _ov)
                = md2(aweights2, _oc2 * V + _ov, _ic2 * V + _iv);
          }
        }
      } else {
        iter_each(_iv, iV) {
          if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
             constexpr int scale = sizeof(Type);
             __m<V> t = _mm<V>::i32gather_ps(vindex,
                 &md2(aweights2, _oc2 * V, _ic2 * V + _iv), scale);
             _mm<V>::store_ps(&md4(abweights4, _oc2, _ic2, _iv, 0), t);
          } else {
#pragma omp simd
            iter_each (_ov, V) {
              md4(abweights4, _oc2, _ic2, _iv, _ov)
                  = md2(aweights2, _oc2 * V + _ov, _ic2 * V + _iv);
            }
          }
        }
      }
    }}
  }
}

// n, oc2, oh, ow, V => n, oc, oh, ow
template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::trans_output_2_plain(
    Type *output, Type *boutput)
{
  MD5(Type, aboutput, boutput, this->n, this->oc2, this->oh, this->ow, V);
  MD4(Type, aoutput, output, this->n, this->oc, this->oh, this->ow);
  if (this->with_ip_sum) {
#pragma omp parallel for collapse(3)
    iter_each (_n, this->n) {
    iter_each (_oc2, this->oc2) {
    iter_each (_oh, this->oh) {
      int v = _oc2 == this->oc2 - 1 ? this->Or : V;
      iter_each (_V, v) {
      iter_each (_ow, this->ow) {
        md4(aoutput, _n, _oc2 * V + _V, _oh, _ow)
          += md5(aboutput, _n, _oc2, _oh, _ow, _V);
      }}
    }}}
  } else {
#pragma omp parallel for collapse(3)
    iter_each (_n, this->n) {
    iter_each (_oc2, this->oc2) {
    iter_each (_oh, this->oh) {
      int v = _oc2 == this->oc2 - 1 ? this->Or : V;
      iter_each (_V, v) {
      iter_each (_ow, this->ow) {
        md4(aoutput, _n, _oc2 * V + _V, _oh, _ow)
          = md5(aboutput, _n, _oc2, _oh, _ow, _V);
      }}
    }}}
  }
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__trans_weights_blocked(
    Type *tweights, Type *weights)
{
  // oc4, (oc3, oc3r), (O2, O2r), ic4, ic3, I2, V, V ->
  // ic4, oc4, ic3, (oc3, oc3r), I2, V, (O2, O2r), V
  MD8(Type, aweights, weights, this->oc4, this->oc3, this->O2, this->ic4,
      this->ic3, this->I2, V, V);
  MD8(Type, atweights, tweights, this->oc4, this->ic4, this->oc3, this->ic3,
      this->I2, V, this->O2, V);

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
          _mm<V>::stream_ps(&md8(atweights, _oc4, _ic4, _oc3, _ic3, _I2, _iV, _O2, 0),
              *(__m<V> *)&md8(aweights, _oc4, _oc3, _O2, _ic4, _ic3, _I2, _iV, 0));
        else
          _mm<V>::store_ps(&md8(atweights, _oc4, _ic4, _oc3, _ic3, _I2, _iV, _O2, 0),
              *(__m<V> *)&md8(aweights, _oc4, _oc3, _O2, _ic4, _ic3, _I2, _iV, 0));
      } else {
#pragma omp simd
        iter_each (_oV, V) {
          md8(atweights, _oc4, _ic4, _oc3, _ic3, _I2, _iV, _O2, _oV)
              = md8(aweights, _oc4, _oc3, _O2, _ic4, _ic3, _I2, _iV, _oV);
        }
      }
    }}}
  }}}}
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__trans_weights_plain(
    Type *tweights, Type *weights)
{
  // oc4, (oc3, oc3r), (O2, O2r), V, ic4, ic3, I2, V ->
  // ic4, oc4, ic3, (oc3, oc3r), I2, V, (O2, O2r), V
  MD8(Type, atweights, tweights, this->oc4, this->ic4, this->oc3,
      this->ic3, this->I2, V, this->O2, V);
  MD8(Type, aweights, weights, this->oc4, this->oc3, this->O2, V,
      this->ic4, this->ic3, this->I2, V);
  SET_EPI32(this->ic)

  if (this->Ir == V && this->Or == V) {
#pragma omp parallel
#pragma omp for nowait collapse(5) schedule(static)
    iter_each (_oc4, this->oc4) {
    iter_each (_ic4, this->ic4) {
    iter_each (_oc3, this->oc3) {
    iter_each (_ic3, this->ic3) {
    iter_each (_I2, this->I2) {
      iter_each (_iV, V) {
      iter_each (_O2, this->O2) {
        if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
          constexpr int scale = sizeof(Type);
         __m<V> t = _mm<V>::i32gather_ps(vindex,
             &md8(aweights, _oc4, _oc3, _O2, 0, _ic4, _ic3, _I2, _iV), scale);
         _mm<V>::store_ps(&md8(atweights, _oc4, _ic4, _oc3, _ic3, _I2, _iV, _O2,
             0), t);
        } else {
#pragma omp simd
          iter_each (_oV, V) {
            md8(atweights, _oc4, _ic4, _oc3, _ic3, _I2, _iV, _O2, _oV)
                = md8(aweights, _oc4, _oc3, _O2, _oV, _ic4, _ic3, _I2, _iV);
          }
        }
      }}
    }}}}}
  } else {
    auto readin_v = [&](Type *atwei, Type *wei) {
      MD3(Type, awei, wei, V, this->ic2, V);
      if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
        constexpr auto scale = sizeof(Type);
        auto t = _mm<V>::i32gather_ps(vindex, &md3(awei, 0, 0, 0), scale);
        _mm<V>::store_ps(atwei, t);
      } else {
        iter_each(_oV, V)
          atwei[_oV] = md3(awei, _oV, 0, 0);
      }
    };

    auto readin_r = [&](Type *atwei, int _oc4, int _oc3, int _O2,
            int _ic4, int _ic3, int _I2, int _iV, bool is_Ir, bool is_Or) {
      MD2(Type, aweights2, weights, this->oc, this->ic);
      int _oc2 = _oc4 * this->oc3 * this->O2 + _oc3 * this->O2 + _O2;
      int _ic2 = _ic4 * this->ic3 * this->I2 + _ic3 * this->I2 + _I2;

      if (is_Or) {
#pragma omp simd
        iter_each (_oV, this->Or) {
          atwei[_oV] = md2(aweights2, _oc2 * V + _oV, _ic2 * V + _iV);
        }
      } else {
        if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
          constexpr auto scale = sizeof(Type);
          auto t = _mm<V>::i32gather_ps(vindex,
              &md2(aweights2, _oc2 *V, _ic2 * V + _iV), scale);
          _mm<V>::store_ps(atwei, t);
        } else {
#pragma omp simd
          iter_each (_oV, this->Or) {
            atwei[_oV] = md2(aweights2, _oc2 * V + _oV, _ic2 * V + _iV);
          }
        }
      }
    };

#pragma omp parallel
#pragma omp for nowait collapse(5) schedule(static)
    iter_each (_oc4, this->oc4) {
    iter_each (_ic4, this->ic4) {
    iter_each (_oc3, this->oc3) {
    iter_each (_ic3, this->ic3) {
    iter_each (_I2, this->I2) {
      bool is_Ir = (_ic4 == this->ic4 - 1) && (_ic3 == this->ic3 -1)
          && (_I2 == this->I2 - 1);
      int iV = is_Ir ? this->Ir : V;
      iter_each (_iV, iV) {
      iter_each (_O2, this->O2) {
        bool is_Or = (_oc4 == this->oc4 - 1) && (_oc3 == this->oc3 - 1)
            && (_O2 == this->O2 - 1);
        if (this->Ir != V || is_Ir || is_Or)
          readin_r(&md8(atweights, _oc4, _ic4, _oc3, _ic3, _I2, _iV, _O2, 0),
              _oc4, _oc3, _O2, _ic4, _ic3, _I2, _iV, is_Ir, is_Or);
        else
          readin_v(&md8(atweights, _oc4, _ic4, _oc3, _ic3, _I2, _iV, _O2, 0),
              &md8(aweights, _oc4, _oc3, _O2, 0, _ic4, _ic3, _I2, _iV));
      }}
    }}}}}
  }
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
void elx_conv_direct_1x1_t<Type, V, I>::__trans_pad_input_blocked(
    Type *tinput, Type *input, int _ht, int _wt)
{
  MD4(Type, atinput, tinput, this->ic3, this->I2, this->T, V);
  MD5(Type, ainput, input, this->ic3, this->I2, this->ih, this->iw, V);

  int _ih = _ht * this->hs - this->tp;
  iter_each (_ic3, this->ic3) {
  iter_each (_I2, this->I2) {
  iter_each (_T, this->T) {
    int _iw = _wt * (this->ws * this->T) + _T * this->ws - this->lp;
    if (_ih < 0 || _ih >= this->ih || _iw < 0 || _iw >= this->iw) {
#pragma omp simd
      iter_each (_V, V) {
        md4(atinput, _ic3, _I2, _T, _V) = 0.0f;
      }
    } else {
      if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
        if (stream_in_)
          _mm<V>::stream_ps(&md4(atinput, _ic3, _I2, _T,0),
             *((__m<V> *)&md5(ainput, _ic3, _I2, _ih, _iw, 0)));
        else
          _mm<V>::store_ps(&md4(atinput, _ic3, _I2, _T,0),
             *((__m<V> *)&md5(ainput, _ic3, _I2, _ih, _iw, 0)));
      } else {
#pragma omp simd
        iter_each (_V, V) {
          md4(atinput, _ic3, _I2, _T, _V)
            = md5(ainput, _ic3, _I2, _ih, _iw, _V);
        }
      }
    }
  }}}
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__trans_input_blocked(
    Type *tinput, Type *input, int _ht, int _wt)
{
  // ic3, I2, ht, hs, wt, T, ws, V -> ht, wt | ic3, I2, T, V
  MD8(Type, ainput, input, this->ic3, this->I2, this->ht, this->hs,
      this->wt, this->T, this->ws, V);
  MD4(Type, atinput, tinput, this->ic3, this->I2, this->T, V);

  iter_each (_ic3, this->ic3) {
  iter_each (_I2, this->I2) {
  iter_each (_T, this->T) {
    if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
      if (stream_in_)
        _mm<V>::stream_ps(&md4(atinput, _ic3, _I2, _T, 0),
             *((__m<V> *)&md8(ainput, _ic3, _I2, _ht, 0, _wt, _T, 0, 0)));
      else
        _mm<V>::store_ps(&md4(atinput, _ic3, _I2, _T, 0),
             *((__m<V> *)&md8(ainput, _ic3, _I2, _ht, 0, _wt, _T, 0, 0)));
    } else {
#pragma omp simd
      iter_each (_V, V) {
        md4(atinput, _ic3, _I2, _T, _V)
            = md8(ainput, _ic3, _I2, _ht, 0, _wt, _T, 0, _V);
      }
    }
  }}}
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__trans_pad_input_plain(
    Type *tinput, Type *input, int _ht, int _wt)
{
  MD3(Type, atinput, tinput, this->ic3 * this->I2, this->T, V);
  SET_EPI32(this->ih * this->iw)
  if (this->Ir == V) {
    MD4(Type, ainput, input, this->ic3 * this->I2, V, this->ih, this->iw);

    int _ih = _ht * this->hs - this->tp;
    iter_each (_ic2, this->ic3 * this->I2) {
    iter_each (_T, this->T) {
      int _iw = _wt * (this->ws * this->T) + _T * this->ws - this->lp;
      if (_ih < 0 || _ih >= this->ih || _iw < 0 || _iw >= this->iw) {
#pragma omp simd
        iter_each (_V, V) {
          md3(atinput, _ic2, _T, _V) = 0.0f;
        }
      } else {
        if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
           constexpr int scale = sizeof(Type);
           __m<V> ain = _mm<V>::i32gather_ps(vindex,
               &md4(ainput, _ic2, 0, _ih, _iw), scale);
           _mm<V>::store_ps(&md3(atinput, _ic2, _T, 0), ain);
        } else {
#pragma omp simd
          iter_each (_V, V) {
            md3(atinput, _ic2, _T, _V)
                = md4(ainput, _ic2, _V, _ih, _iw);
          }
        }
      }
    }}
  } else {
    MD3(Type, ainput, input, this->ic, this->ih, this->iw);

    int _ih = _ht * this->hs - this->tp;
    iter_each (_ic2, this->ic3 * this->I2) {
    iter_each (_T, this->T) {
      int _iw = _wt * (this->ws * this->T) + _T * this->ws - this->lp;
      bool is_Ir = _ic2 == this->ic2 - 1;
      if (is_Ir) {
        if (_ih < 0 || _ih >= this->ih || _iw < 0 || _iw >= this->iw) {
#pragma omp simd
          iter_each (_V, V) {
            md3(atinput, _ic2, _T, _V) = 0.0f;
          }
        } else {
#pragma omp simd
          iter_each (_V, this->Ir) {
            md3(atinput, _ic2, _T, _V)
                = md3(ainput, (this->ic2 - 1) * V + _V, _ih, _iw);
          }
        }
      } else {
        if (_ih < 0 || _ih >= this->ih || _iw < 0 || _iw >= this->iw) {
#pragma omp simd
          iter_each (_V, V) {
            md3(atinput, _ic2, _T, _V) = 0.0f;
          }
        } else {
          if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
            constexpr int scale = sizeof(Type);
            __m<V> ain = _mm<V>::i32gather_ps(vindex,
                &md3(ainput, _ic2 * V, _ih, _iw), scale);
            _mm<V>::store_ps(&md3(atinput, _ic2, _T, 0), ain);
          } else {
#pragma omp simd
            iter_each (_V, V) {
              md3(atinput, _ic2, _T, _V)
                  = md3(ainput, _ic2 * V + _V, _ih, _iw);
            }
          }
        }
      }
    }}
  }
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__trans_input_plain(
    Type *tinput, Type *input, int _ht, int _wt)
{
  // ic3, I2, V, ht, hs, wt, T, ws -> ht, wt | ic3, I2, T, V
  MD3(Type, atinput, tinput, this->ic3 * this->I2, this->T, V);
  SET_EPI32(this->ih * this->iw)
  if (this->Ir == V) {
    MD7(Type, ainput, input, this->ic3 * this->I2, V, this->ht,
        this->hs, this->wt, this->T, this->ws);
    iter_each (_ic2, this->ic3 * this->I2) {
    iter_each (_T, this->T) {
      if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
         constexpr int scale = sizeof(Type);
         __m<V> ain = _mm<V>::i32gather_ps(vindex,
             &md7(ainput, _ic2, 0, _ht, 0, _wt, _T, 0), scale);
         _mm<V>::store_ps(&md3(atinput, _ic2, _T, 0), ain);
      } else {
#pragma omp simd
        iter_each (_V, V) {
          md3(atinput, _ic2, _T, _V)
              = md7(ainput, _ic2, _V, _ht, 0, _wt, _T, 0);
        }
      }
    }}
  } else {
    MD6(Type, ainput6, input, this->ic, this->ht, this->hs,
        this->wt, this->T, this->ws);
    iter_each (_ic2, this->ic3 * this->I2) {
    iter_each (_T, this->T) {
      bool is_Ir = _ic2 == this->ic2 - 1;
      if (is_Ir) {
#pragma omp simd
        iter_each (_V, this->Ir) {
          md3(atinput, _ic2, _T, _V)
              = md6(ainput6, (this->ic2 - 1) * V + _V, _ht, 0, _wt, _T, 0);
        }
      } else {
        if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
          constexpr int scale = sizeof(Type);
          __m<V> ain = _mm<V>::i32gather_ps(vindex,
              &md6(ainput6, _ic2 * V, _ht, 0, _wt, _T, 0), scale);
          _mm<V>::store_ps(&md3(atinput, _ic2, _T, 0), ain);
        } else {
#pragma omp simd
          iter_each (_V, V) {
            md3(atinput, _ic2, _T, _V)
                = md6(ainput6, _ic2 * V + _V, _ht, 0, _wt, _T, 0);
          }
        }
      }
    }}
  }
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::trans_input(
    Type *tinput, Type *input, int _ht, int _wt)
{
  if (no_pad_) {
    if (input_is_bfmt_ || input_as_bfmt_)
      __trans_input_blocked(tinput, input, _ht, _wt);
    else
      __trans_input_plain(tinput, input, _ht, _wt);
  } else {
    if (input_is_bfmt_ || input_as_bfmt_)
      __trans_pad_input_blocked(tinput, input, _ht, _wt);
    else
      __trans_pad_input_plain(tinput, input, _ht, _wt);
  }
}


template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__trans_output_blocked(
    Type *output, Type *toutput, int _oc4, int _ht, int _wt)
{
  // oc3, O2, T, V => n, oc4 | oc3, O2, ht, wt, T, V
  MD4(Type, atoutput, toutput, this->oc3, this->O2, this->T, V);
  MD7(Type, aoutput, output, this->oc4, this->oc3, this->O2,
      this->ht, this->wt, this->T, V);

  iter_each (_oc3, this->oc3) {
  iter_each (_O2, this->O2) {
  iter_each (_T, this->T) {
    if (this->with_ip_sum && !output_as_bfmt_) {
#pragma omp simd
      iter_each (_V, V) {
        md7(aoutput, _oc4, _oc3, _O2, _ht, _wt, _T, _V)
            += md4(atoutput, _oc3, _O2, _T, _V);
      }
    } else if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
      if (stream_out_)
        _mm<V>::stream_ps(&md7(aoutput, _oc4, _oc3, _O2, _ht, _wt, _T, 0),
             *((__m<V> *)&md4(atoutput, _oc3, _O2, _T, 0)));
      else
        _mm<V>::store_ps(&md7(aoutput, _oc4, _oc3, _O2, _ht, _wt, _T, 0),
             *((__m<V> *)&md4(atoutput, _oc3, _O2, _T, 0)));
    } else {
#pragma omp simd
      iter_each (_V, V) {
        md7(aoutput, _oc4, _oc3, _O2, _ht, _wt, _T, _V)
            = md4(atoutput, _oc3, _O2, _T, _V);
      }
    }
  }}}
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__trans_output_plain(
    Type *output, Type *toutput, int _oc4, int _ht, int _wt)
{
  // oc3, O2, T, V => n, oc4 | oc3, O2, V, ht, wt, T
  SET_EPI32(this->oh * this->ow)
  if (this->Or == V) {
    MD4(Type, atoutput, toutput, this->oc3, this->O2, this->T, V);
    MD7(Type, aoutput, output, this->oc4, this->oc3, this->O2, V,
        this->ht, this->wt, this->T);
    iter_each (_oc3, this->oc3) {
    iter_each (_O2, this->O2) {
    iter_each (_T, this->T) {
      if (this->with_ip_sum && !output_as_bfmt_) {
#pragma omp simd
        iter_each (_V, V) {
          md7(aoutput, _oc4, _oc3, _O2, _V, _ht, _wt, _T)
              += md4(atoutput, _oc3, _O2, _T, _V);
        }
      } else if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
        __m<V> t = _mm<V>::load_ps(&md4(atoutput, _oc3, _O2, _T, 0));
        constexpr int scale = sizeof(Type);
        _mm<V>::i32scatter_ps(&md7(aoutput, _oc4, _oc3, _O2, 0, _ht, _wt, _T),
            vindex, t, scale);
      } else {
#pragma omp simd
        iter_each (_V, V) {
          md7(aoutput, _oc4, _oc3, _O2, _V, _ht, _wt, _T)
              = md4(atoutput, _oc3, _O2, _T, _V);
        }
      }
    }}}
  } else {
    MD4(Type, atoutput, toutput, this->oc3, this->O2, this->T, V);
    MD4(Type, aoutput, output, this->oc, this->ht, this->wt, this->T);
    iter_each (_oc3, this->oc3) {
    iter_each (_O2, this->O2) {
    iter_each (_T, this->T) {
      int _oc2 = _oc4 * this->oc3 * this->O2 + _oc3 * this->O2 + _O2;
      bool is_Or = (_oc4 == this->oc4 - 1) && (_oc3 == this->oc3 - 1)
          && (_O2 == this->O2 - 1);
      if (is_Or) {
        if (this->with_ip_sum && !output_as_bfmt_) {
#pragma omp simd
          iter_each(_ov, this->Or) {
            md4(aoutput, (this->oc2 - 1) * V + _ov, _ht, _wt, _T)
                += md4(atoutput, _oc3, _O2, _T, _ov);
          }
        } else {
#pragma omp simd
          iter_each(_ov, this->Or) {
            md4(aoutput, (this->oc2 - 1) * V + _ov, _ht, _wt, _T)
                = md4(atoutput, _oc3, _O2, _T, _ov);
          }
        }
      } else {
        if (this->with_ip_sum && !output_as_bfmt_) {
#pragma omp simd
          iter_each(_V, V) {
            md4(aoutput, _oc2 * V + _V, _ht, _wt, _T)
                += md4(atoutput, _oc3, _O2, _T, _V);
          }
        } else if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
          __m<V> t = _mm<V>::load_ps(&md4(atoutput, _oc3, _O2, _T, 0));
          constexpr int scale = sizeof(Type);
          _mm<V>::i32scatter_ps(&md4(aoutput, _oc2 * V, _ht, _wt, _T), vindex,
              t, scale);
        } else {
#pragma omp simd
          iter_each(_V, V) {
            md4(aoutput, _oc2 * V + _V, _ht, _wt, _T)
                = md4(atoutput, _oc3, _O2, _T, _V);
          }
        }
      }
    }}}
  }
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::trans_output(
    Type *output, Type *toutput, int _oc4, int _ht, int _wt)
{
  if (output_is_bfmt_ || output_as_bfmt_)
    __trans_output_blocked(output, toutput, _oc4, _ht, _wt);
  else
    __trans_output_plain(output, toutput, _oc4, _ht, _wt);
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__trans_input_plain2(
    Type *tinput, Type *input, int _t2, int Tz)
{
  MD3(Type, atinput, tinput, this->ic3 * this->I2, Tz, V);
  SET_EPI32(this->ih * this->iw)
  if (this->Ir == V) {
    MD3(Type, ainput, input, this->ic3 * this->I2, V, this->ih * this->iw);
    iter_each (_ic2, this->ic3 * this->I2) {
    iter_each (_T, Tz) {
      if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
         constexpr int scale = sizeof(Type);
         __m<V> ain = _mm<V>::i32gather_ps(vindex,
             &md3(ainput, _ic2, 0, _t2 * this->T + _T), scale);
         _mm<V>::store_ps(&md3(atinput, _ic2, _T, 0), ain);
      } else {
#pragma omp simd
        iter_each (_V, V) {
          md3(atinput, _ic2, _T, _V)
              = md3(ainput, _ic2, _V, _t2 * this->T + _T);
        }
      }
    }}
  } else {
    MD2(Type, ainput2, input, this->ic, this->ih * this->iw);
    iter_each (_ic2, this->ic3 * this->I2) {
    iter_each (_T, Tz) {
      bool is_Ir = _ic2 == this->ic2 - 1;
      if (is_Ir) {
#pragma omp simd
        iter_each (_V, this->Ir) {
          md3(atinput, _ic2, _T, _V)
              = md2(ainput2, (this->ic2 - 1) * V + _V, _t2 * this->T + _T);
        }
      } else {
        if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
          constexpr int scale = sizeof(Type);
          __m<V> ain = _mm<V>::i32gather_ps(vindex,
              &md2(ainput2, _ic2 * V, _t2 * this->T + _T), scale);
          _mm<V>::store_ps(&md3(atinput, _ic2, _T, 0), ain);
        } else {
#pragma omp simd
          iter_each (_V, V) {
            md3(atinput, _ic2, _T, _V)
                = md2(ainput2, _ic2 * V + _V, _t2 * this->T + _T);
          }
        }
      }
    }}
  }
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__trans_input_blocked2(
    Type *tinput, Type *input, int _t2, int Tz)
{
  MD4(Type, ainput, input, this->ic3, this->I2, this->ih * this->iw, V);
  MD4(Type, atinput, tinput, this->ic3, this->I2, Tz, V);
  iter_each (_ic3, this->ic3) {
  iter_each (_I2, this->I2) {
  iter_each (_T, Tz) {
    if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
      if (stream_in_)
        _mm<V>::stream_ps(&md4(atinput, _ic3, _I2, _T, 0),
             *((__m<V> *)&md4(ainput, _ic3, _I2, _t2 * this->T + _T, 0)));
      else
        _mm<V>::store_ps(&md4(atinput, _ic3, _I2, _T, 0),
             *((__m<V> *)&md4(ainput, _ic3, _I2, _t2 * this->T + _T, 0)));
    } else {
#pragma omp simd
      iter_each (_V, V) {
        md4(atinput, _ic3, _I2, _T, _V)
            = md4(ainput, _ic3, _I2, _t2 * this->T + _T, _V);
      }
    }
  }}}
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::trans_input2(
    Type *tinput, Type *input, int _t2, int Tz)
{
  if (input_is_bfmt_ || input_as_bfmt_)
    __trans_input_blocked2(tinput, input, _t2, Tz);
  else
    __trans_input_plain2(tinput, input, _t2, Tz);
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__trans_output_plain2(
    Type *output, Type *toutput, int _oc4, int _t2, int Tz)
{
  SET_EPI32(this->oh * this->ow)

  if (this->Or == V) {
    MD4(Type, atoutput, toutput, this->oc3, this->O2, Tz, V);
    MD5(Type, aoutput, output, this->oc4, this->oc3, this->O2, V,
        this->oh * this->ow);
    iter_each (_oc3, this->oc3) {
    iter_each (_O2, this->O2) {
    iter_each (_T, Tz) {
      if (this->with_ip_sum && !output_as_bfmt_) {
#pragma omp simd
        iter_each (_V, V) {
          md5(aoutput, _oc4, _oc3, _O2, _V, _t2 * this->T + _T)
              += md4(atoutput, _oc3, _O2, _T, _V);
        }
      } else if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
        __m<V> t = _mm<V>::load_ps(&md4(atoutput, _oc3, _O2, _T, 0));
        constexpr int scale = sizeof(Type);
        _mm<V>::i32scatter_ps(&md5(aoutput, _oc4, _oc3, _O2, 0, _t2 * this->T + _T),
            vindex, t, scale);
      } else {
#pragma omp simd
        iter_each (_V, V) {
          md5(aoutput, _oc4, _oc3, _O2, _V, _t2 * this->T + _T)
              = md4(atoutput, _oc3, _O2, _T, _V);
        }
      }
    }}}
  } else {
    MD4(Type, atoutput, toutput, this->oc3, this->O2, Tz, V);
    MD2(Type, aoutput, output, this->oc, this->oh * this->ow);
    iter_each (_oc3, this->oc3) {
    iter_each (_O2, this->O2) {
    iter_each (_T, Tz) {
      int _oc2 = _oc4 * this->oc3 * this->O2 + _oc3 * this->O2 + _O2;
      bool is_Or = (_oc4 == this->oc4 - 1) && (_oc3 == this->oc3 - 1)
          && (_O2 == this->O2 - 1);
      if (is_Or) {
        if (this->with_ip_sum && !output_as_bfmt_) {
#pragma omp simd
          iter_each(_ov, this->Or) {
            md2(aoutput, (this->oc2 - 1) * V + _ov, _t2 * this->T + _T)
                += md4(atoutput, _oc3, _O2, _T, _ov);
          }
        } else {
#pragma omp simd
          iter_each(_ov, this->Or) {
            md2(aoutput, (this->oc2 - 1) * V + _ov, _t2 * this->T + _T)
                = md4(atoutput, _oc3, _O2, _T, _ov);
          }
        }
      } else {
        if (this->with_ip_sum && !output_as_bfmt_) {
#pragma omp simd
          iter_each(_V, V) {
            md2(aoutput, _oc2 * V + _V, _t2 * this->T + _T)
                += md4(atoutput, _oc3, _O2, _T, _V);
          }
        } else if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
          __m<V> t = _mm<V>::load_ps(&md4(atoutput, _oc3, _O2, _T, 0));
          constexpr int scale = sizeof(Type);
          _mm<V>::i32scatter_ps(&md2(aoutput, _oc2 * V, _t2 * this->T + _T),
              vindex, t, scale);
        } else {
#pragma omp simd
          iter_each(_V, V) {
            md2(aoutput, _oc2 * V + _V, _t2 * this->T + _T)
                = md4(atoutput, _oc3, _O2, _T, _V);
          }
        }
      }
    }}}
  }
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__trans_output_blocked2(
    Type *output, Type *toutput, int _oc4, int _t2, int Tz)
{
  // oc3, O2, T, V => n, oc4 | oc3, O2, ht, wt, T, V
  MD4(Type, atoutput, toutput, this->oc3, this->O2, Tz, V);
  MD5(Type, aoutput, output, this->oc4, this->oc3, this->O2,
      this->oh * this->ow, V);

  iter_each (_oc3, this->oc3) {
  iter_each (_O2, this->O2) {
  iter_each (_T, Tz) {
    if (this->with_ip_sum && !output_as_bfmt_) {
#pragma omp simd
      iter_each (_V, V) {
        md5(aoutput, _oc4, _oc3, _O2, _t2 * this->T + _T, _V)
            += md4(atoutput, _oc3, _O2, _T, _V);
      }
    } else if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
      if (stream_out_)
        _mm<V>::stream_ps(&md5(aoutput, _oc4, _oc3, _O2, _t2 * this->T + _T, 0),
             *((__m<V> *)&md4(atoutput, _oc3, _O2, _T, 0)));
      else
        _mm<V>::store_ps(&md5(aoutput, _oc4, _oc3, _O2, _t2 * this->T + _T, 0),
             *((__m<V> *)&md4(atoutput, _oc3, _O2, _T, 0)));
    } else {
#pragma omp simd
      iter_each (_V, V) {
        md5(aoutput, _oc4, _oc3, _O2, _t2 * this->T + _T, _V)
            = md4(atoutput, _oc3, _O2, _T, _V);
      }
    }
  }}}
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::trans_output2(
    Type *output, Type *toutput, int _oc4, int _t2, int Tz)
{
  if (output_is_bfmt_ || output_as_bfmt_)
    __trans_output_blocked2(output, toutput, _oc4, _t2, Tz);
  else
    __trans_output_plain2(output, toutput, _oc4, _t2, Tz);
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::gemm_a061(Type *output, Type *input,
    Type *weights, Type *bias, int _ic4)
{
  // weights: oc3*, ic3*, O2, I2, V, V
  // input:   ic3*, I2, T, V
  // output:  oc3*, O2, T, V
  MD2(Type, ainput, input, this->ic3, this->I2 * this->T * V);
  MD2(Type, aoutput, output, this->oc3, this->O2 * this->T * V);
  MD3(Type, aweights, weights, this->oc3, this->ic3,
      this->O2 * this->I2 * V * V);
  MD2(Type, abias, bias, this->oc3, this->O2 * V);

  iter_each (_ic3, this->ic3 - 1) {
    int attr = _ic4 == 0 && _ic3 == 0 ? set_attr(attr_, r_output_idx) : attr_;
    iter_each (_oc3, this->oc3) {
      ker_gemm_I_O_T_(*this, &md2(aoutput, _oc3, 0),
          &md2(ainput, _ic3, 0), &md3(aweights, _oc3, _ic3, 0),
          &md2(abias, _oc3, 0), attr, 0, nullptr, nullptr);
    }
  }
  int attr = _ic4 == 0 && this->ic3 == 1 ?
      set_attr(attr_, r_output_idx) : attr_;
  attr = this->with_relu && _ic4 == this->ic4 - 1 ?
      (set_attr(attr, relu_idx)) : attr;
  iter_each (_oc3, this->oc3) {
    ker_gemm_IrO_T_(*this, &md2(aoutput, _oc3, 0),
        &md2(ainput, this->ic3 - 1, 0), &md3(aweights, _oc3, this->ic3 - 1, 0),
        &md2(abias, _oc3, 0), attr, 0, nullptr, nullptr);
  }
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::gemm_b061(Type *output, Type *input,
    Type *weights, Type *bias, int _ic4)
{
  // weights: oc3*, ic3*, O2, I2, V, V
  // input:   ic3*, I2, T, V
  // output:  oc3*, O2, ht*, wt*, T, V
  MD2(Type, ainput, input, this->ic3, this->I2 * this->T * V);
  MD5(Type, aoutput, output, this->oc3, this->O2,
      this->ht, this->wt, this->T * V);
  MD3(Type, aweights, weights, this->oc3, this->ic3,
      this->O2 * this->I2 * V * V);
  MD2(Type, abias, bias, this->oc3, this->O2 * V);

  iter_each (_ic3, this->ic3) {
    int attr = _ic4 == 0 && _ic3 == 0 ? set_attr(attr_, r_output_idx) : attr_;
    attr = this->with_relu && _ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1 ?
        set_attr(attr, relu_idx) : attr;
    iter_each (_oc3, this->oc3) {
      ker_gemm_I_O_T_(*this, &md5(aoutput, _oc3, 0, 0, 0, 0),
          &md2(ainput, _ic3, 0), &md3(aweights, _oc3, _ic3, 0),
          &md2(abias, _oc3, 0), attr, 0, nullptr, nullptr);
    }
  }
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::gemm_f061(Type *output, Type *input,
    Type *weights, Type *bias, int _t2, int Tz)
{
  MD2(Type, ainput, input, this->ic3, this->I2 * Tz * V);
  MD2(Type, aoutput, output, this->oc3, this->O2 * Tz * V);
  MD3(Type, aweights, weights, this->oc3, this->ic3,
      this->O2 * this->I2 * V * V);
  MD2(Type, abias, bias, this->oc3, this->O2 * V);

  auto ker_gemm = (_t2 == this->t2 - 1) ? ker_gemm_I_O_Tr_ : ker_gemm_I_O_T_;
  auto ker_gemm_tail = (_t2 == this->t2 - 1) ?
      ker_gemm_IrO_Tr_ : ker_gemm_IrO_T_;

  iter_each (_ic3, this->ic3 - 1) {
    int attr = _ic3 == 0 ? set_attr(attr_, r_output_idx) : attr_;
    iter_each (_oc3, this->oc3) {
      ker_gemm(*this, &md2(aoutput, _oc3, 0),
          &md2(ainput, _ic3, 0), &md3(aweights, _oc3, _ic3, 0),
          &md2(abias, _oc3, 0), attr, 0, nullptr, nullptr);
    }
  }
  int attr = this->ic3 == 1 ? set_attr(attr_, r_output_idx) : attr_;
  attr = this->with_relu ? set_attr(attr, relu_idx) : attr;
  iter_each(_oc3, this->oc3) {
    ker_gemm_tail(*this, &md2(aoutput, _oc3, 0),
        &md2(ainput, this->ic3 - 1, 0), &md3(aweights, _oc3, this->ic3 - 1, 0),
        &md2(abias, _oc3, 0), attr, 0, nullptr, nullptr);
  }
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::gemm_c060(Type *output, Type *input,
    Type *weights, Type *bias, int _ic4, int _oc4, int _t2)
{
  // weights: oc3, O2(O2r), ic4*, ic3, I2, V(Ir), V(Or)
  // input:   ic3, I2, t2*, T(Tr), V(Ir)
  // output:  oc3, O2(O2r), t2*, T(Tr), V(Or)
  MD2(Type, ainput, input, this->ic3, this->I2 * this->ih * this->iw * V);
  MD2(Type, aoutput, output, this->oc3, this->O2 * this->oh * this->ow * V);
  MD5(Type, aweights, weights, this->oc3, this->O2, this->ic4, this->ic3,
      this->I2 * V * V);
  MD2(Type, abias, bias, this->oc3, this->O2 * V);

  int oc3 = _oc4 == this->oc4 - 1 ? this->oc3r : this->oc3;
  iter_each (_ic3, this->ic3) {
    int attr = _ic4 == 0 && _ic3 == 0 ? set_attr(attr_, r_output_idx) : attr_;
    attr  = this->with_relu && _ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1 ?
        set_attr(attr, relu_idx) : attr;
    MD2(Type, ainput2, &md2(ainput, _ic3, 0), this->t2, this->T * V);
    iter_each (_oc3, oc3) {
      MD2(Type, aoutput2, &md2(aoutput, _oc3, 0), this->t2, this->T * V);
      if (_oc4 == this->oc4 - 1 && _oc3 == oc3 - 1) {
        if (_t2 == this->t2 - 1)
          ker_gemm_I_OrTr_(*this, &md2(aoutput2, _t2, 0), &md2(ainput2, _t2, 0),
              &md5(aweights, _oc3, 0, _ic4, _ic3, 0), &md2(abias, _oc3, 0),
              attr, 0, nullptr, nullptr);
        else
          ker_gemm_I_OrT_(*this, &md2(aoutput2, _t2, 0), &md2(ainput2, _t2, 0),
              &md5(aweights, _oc3, 0, _ic4, _ic3, 0), &md2(abias, _oc3, 0),
              attr, 0, nullptr, nullptr);
      } else {
        if (_t2 == this->t2 - 1)
          ker_gemm_I_O_Tr_(*this, &md2(aoutput2, _t2, 0), &md2(ainput2, _t2, 0),
              &md5(aweights, _oc3, 0, _ic4, _ic3, 0), &md2(abias, _oc3, 0),
              attr, 0, nullptr, nullptr);
        else
          ker_gemm_I_O_T_(*this, &md2(aoutput2, _t2, 0), &md2(ainput2, _t2, 0),
              &md5(aweights, _oc3, 0, _ic4, _ic3, 0), &md2(abias, _oc3, 0),
              attr, 0, nullptr, nullptr);
      }
    }
  }
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::gemm_d060(Type *output, Type *input,
    Type *weights, Type *bias, int _ic4, int _oc4, int _ht, int _wt)
{
  // weights: oc3*, O2, ic4*, ic3*, I2, V, V
  // input:   ic3*, I2, ht*, hs*, wt*, T, ws, V
  // output:  oc3*, O2(O2r), ht*, wt*, T, V
  MD5(Type, ainput, input, this->ic3, this->I2, this->ih, this->wt,
      this->T * this->ws * V);
  MD5(Type, aoutput, output, this->oc3, this->O2, this->ht, this->wt,
      this->T * V);
  MD5(Type, aweights, weights, this->oc3, this->O2, this->ic4, this->ic3,
      this->I2 * V * V);
  MD2(Type, abias, bias, this->oc3, this->O2 * V);

  iter_each (_ic3, this->ic3) {
    int attr = _ic4 == 0 && _ic3 == 0 ? set_attr(attr_, r_output_idx) : attr_;
    attr = this->with_relu && _ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1 ?
        set_attr(attr, relu_idx) : attr;
    int oc3 = _oc4 == this->oc4 - 1 ? this->oc3r : this->oc3;

    iter_each (_oc3, oc3) {
      if (_oc4 == this->oc4 - 1 && _oc3 == oc3 - 1) {
        ker_gemm_I_OrT_(*this, &md5(aoutput, _oc3, 0, 0, 0, 0),
            &md5(ainput, _ic3, 0, 0, 0, 0),
            &md5(aweights, _oc3, 0, _ic4, _ic3, 0), &md2(abias, _oc3, 0),
            attr, 0, nullptr, nullptr);
      } else {
        ker_gemm_I_O_T_(*this, &md5(aoutput, _oc3, 0, 0, 0, 0),
            &md5(ainput, _ic3, 0, 0, 0, 0),
            &md5(aweights, _oc3, 0, _ic4, _ic3, 0), &md2(abias, _oc3, 0),
            attr, 0, nullptr, nullptr);
      }
    }
  }
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::gemm_e060(Type *output, Type *input,
    Type *weights, Type *bias, int _ic4)
{
  // weights: oc3*, ic3*, O2, I2, V, V
  // input:   ic3*, I2, ht*, hs*, wt*, T, ws, V
  // output:  oc3*, O2, ht*, wt*, T, V
  MD5(Type, ainput, input, this->ic3, this->I2, this->ih, this->wt,
      this->T * this->ws * V);
  MD5(Type, aoutput, output, this->oc3, this->O2, this->ht, this->wt,
      this->T * V);
  MD3(Type, aweights, weights, this->oc3, this->ic3,
      this->O2 * this->I2 * V * V);
  MD2(Type, abias, bias, this->oc3, this->O2 * V);


  iter_each (_ic3, this->ic3) {
    int attr = _ic4 == 0 && _ic3 == 0 ? set_attr(attr_, r_output_idx) : attr_;
    attr = this->with_relu && _ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1 ?
        set_attr(attr, relu_idx) : attr;
    iter_each (_oc3, this->oc3) {
      ker_gemm_I_O_T_(*this, &md5(aoutput, _oc3, 0, 0, 0, 0),
          &md5(ainput, _ic3, 0, 0, 0, 0), &md3(aweights, _oc3, _ic3, 0),
          &md2(abias, _oc3, 0), attr, 0, nullptr, nullptr);
    }
  }
}

} // namespace euler
