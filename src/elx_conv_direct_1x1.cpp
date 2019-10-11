#include "elx_conv_direct_1x1.hpp"
#include "el_parallel.hpp"

namespace euler {

Template_elx_conv_direct_1x1_t
Instance_elx_conv_direct_1x1_t::elx_conv_direct_1x1_t(eld_conv_t &dc)
    : elx_conv_t(dc)
{
  // user input
  xopt_ = this->execution_mode;

  this->Vx = 1;
  this->V1 = V / this->Vx;
  this->IC = ALIGNUP(this->ic, V);
  this->OC = ALIGNUP(this->oc, V);

  if (this->I2 == 0) this->I2 = this->ic2;
  if (this->T == 0)  this->T = 1;
  if (this->O == 0)  this->O = 1;
  if (this->O1 == 0) this->O1 = 1;
  if (this->O1 != 1) el_error("blk-o != 1 is not supported");
  this->O2 = this->O * this->O1;

  this->O4 = this->O4 == 0 ? 1 : this->O4;
  this->I4 = this->I4 == 0 ? 1 : this->I4;

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

  // n, t2, (T, Tr)
  if (xopt_ == 0xc060 || xopt_ == 0xf061) {
    bool shape_ok = this->hs == 1 && this->ws == 1 && no_pad_;
    if (!shape_ok)
      el_error("Shape not supported by c060");

    this->ht = this->oh;
    this->wt = this->ow;
    this->nt = this->ht * this->wt;
    this->t = this->nt * this->n;
    this->t2 = (this->nt + this->T - 1) / this->T;
    this->Tr = this->nt % this->T ? this->nt % this->T : this->T;
  } else if (xopt_ == 0xa061 || xopt_ == 0xb061) {
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

  // O4, (O3, O3r), (O2, O2r)
  this->oc34 = (this->oc2 + this->O2 - 1) / this->O2;
  this->O2r = this->oc2 % this->O2;
  if (this->O2r == 0) this->O2r = this->O2;
  this->O3 = this->O4; // FIXME, swap order
  this->O4 = (this->oc34 + this->O3 - 1) / this->O3;
  this->O3r = this->oc34 % this->O3;
  if (this->O3r == 0) this->O3r = this->O3;

  if ((xopt_ == 0xa061 || xopt_ == 0xb061 || xopt_ == 0xc060 || xopt_ == 0xf061)
      && (this->O2r != this->O2 || this->O3r != this->O3)) {
    el_error("No oc tailing for 0xa061, 0xb061, 0xe060, 0xf061");
  }

  // I4, I3, I3
  this->ic34 = this->ic2 / this->I2;
  this->I3 = this->ic34 / this->I4;
  if (this->I4 * this->I3 * this->I2 * V != this->IC)
    el_error("IC blocking error");

  if ((xopt_ == 0xa061 || xopt_ == 0xf061) && this->I4 != 1) {
    el_error("I4 != 1 not support in 0xa061 and 0xf061");
  }

  attr_ = 0x0;
  is_first_run_ = true;
  inference_acc_ = false;
  mthr_ = estl::max_concurrency();
  inference_acc_ = this->prop_kind == forward_inference;

  attr_ = this->with_bias ? set_bit(attr_, AT_BIAS_MASK) : attr_;
  if (xopt_ == 0xb061 || xopt_ == 0xc060) {
    attr_ = this->with_ip_sum ? set_bit(attr_, AT_INP_SUM_MASK) : attr_;
  }

  prepare_execute_opt();
  bind_execute_functions();

  // dbg
  el_log(DEBUG, "T=%d, Tr=%d, t2=%d, ht=%d, wt=%d, t=%d",
         this->T, this->Tr, this->t2, this->ht, this->wt, this->t);
  el_log(DEBUG, "V=%d, Ir=%d, I2=%d, I3=%d, I4=%d, IC=%d",
         V, this->Ir, this->I2, this->I3, this->I4, this->IC);
  el_log(DEBUG, "V=%d, Or=%d, O2=%d (O=%d, O1=%d), O3=%d, O4=%d, O2r=%d, O3r=%d, OC=%d",
         V, this->Or, this->O2, this->O, this->O1,
         this->O3, this->O4, this->O2r, this->O3r, this->OC);
}

Template_elx_conv_direct_1x1_t
int  Instance_elx_conv_direct_1x1_t::prepare_execute_opt()
{
  size_t tweights_size = 0, tinput_size = 0, toutput_size = 0;
  size_t binput_size = 0, bweights_size = 0, boutput_size = 0;

  stream_in_ = this->streaming_input
      ? (this->streaming_input == STORE_STREAMING) : false;
  stream_out_ = this->streaming_output
      ? (this->streaming_output == STORE_STREAMING) : false;

  input_is_bfmt_ = this->input_fmt == nChw16c; // nChw8c
  weights_is_bfmt_ = this->weights_fmt == OIhw16i16o;
  output_is_bfmt_ = this->output_fmt == nChw16c;
  input_as_bfmt_ = this->input_fmt == nchw && this->input_as_blocked;
  weights_as_bfmt_ = this->input_fmt == oihw && this->weights_as_blocked;
  output_as_bfmt_ = this->output_fmt == nchw && this->output_as_blocked;
  is_bfmt_ = input_is_bfmt_ && weights_is_bfmt_ && output_is_bfmt_;

  if (this->with_ip_sum && this->with_relu && !output_is_bfmt_) {
    el_error("Unimplemented: fuse sum (plain format) and relu together");
  }

  if (this->I4 > 1 && this->Ir != V) {
    el_error("Unimplemented: I4 > 1 for IC % V != 0");
  }
  if (this->O4 > 1 && this->Or != V) {
    el_error("Unimplemented: O4 > 1 for OC % V != 0");
  }

  if (!is_bfmt_ && (xopt_ != 0xa061 && xopt_ != 0xf061)) {
    el_error("Unimplemented: only a061, f061 mode support plain format\n");
  }

  if (input_as_bfmt_)
    binput_size = this->n * this->IC * this->ih * this->iw * sizeof(InputType);
  if (weights_as_bfmt_)
    bweights_size = this->OC * this->IC * sizeof(WeightsType);
  if (output_as_bfmt_)
    boutput_size = this->n * this->OC * this->oh * this->ow * sizeof(OutputType);

  tweights_ = nullptr;
  tinput_ = nullptr;
  toutput_ = nullptr;
  tinput_msk_ = nullptr;
  binput_ = nullptr;
  bweights_ = nullptr;
  boutput_ = nullptr;

  switch (xopt_) {
  case 0xa061:
    toutput_size = mthr_ * this->O3 * this->O2 * this->T * V * sizeof(ToutputType);
  case 0xb061:
    tinput_msk_ = (unsigned char *)malloc(mthr_ * this->I4 * this->ht * this->wt);
    tinput_size = mthr_ * this->I3 * this->I2 * V * this->ht * this->wt * this->T * sizeof(TinputType);
    tweights_size = this->IC * this->OC * sizeof(TweightsType);
    break;
  case 0xf061:
    tinput_msk_ = (unsigned char *)malloc(mthr_ * this->t2);
    toutput_size = mthr_ * this->O3 * this->O2 * this->T * V * sizeof(ToutputType);
    tinput_size = mthr_ * this->I3 * this->I2 * this->T * V * this->t2 * sizeof(TinputType);
    tweights_size = this->IC * this->OC * sizeof(TweightsType);
    break;
  case 0xc060:
    tweights_size = this->IC * this->OC * sizeof(TweightsType);
    break;
  default:
      el_error("Unknown xopt!");
      return -1;
    break;
  }

  const size_t align = PAGE_SIZE;
#define WEIGHTS_MAX_PRELOAD 4
  if (tweights_size > 0)
    tweights_size += WEIGHTS_MAX_PRELOAD * V;

  tweights_size_ = tweights_size > 0 ? alignup(tweights_size, align) : 0;
  tinput_size_ = tinput_size > 0 ? alignup(tinput_size, align) : 0;
  toutput_size_ = toutput_size > 0 ? alignup(toutput_size, align) : 0;
  binput_size_ = binput_size > 0 ? alignup(binput_size, align) : 0;
  bweights_size_ = bweights_size > 0 ? alignup(bweights_size, align) : 0;
  boutput_size_ = boutput_size > 0 ? alignup(boutput_size, align) : 0;

  workspace_size_ = tweights_size_;
  scratch_size_ = tinput_size_ + toutput_size_
    + binput_size_ + bweights_size_ + boutput_size_;

  return 0;
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::set_scratch_buffers(void *base)
{
  if (base != nullptr) {
    tinput_ = (TinputType *)base;
    toutput_ = (ToutputType *)((char *)tinput_ + tinput_size_);
    binput_ = (InputType *)((char *)toutput_ + toutput_size_);
    bweights_ = (WeightsType *)((char *)binput_ + binput_size_);
    boutput_ = (OutputType *)((char *)bweights_ + bweights_size_);
  }
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::set_workspace_buffers(void *base)
{
  if (base != nullptr) {
    tweights_ = (TweightsType *)base;
  }
}

Template_elx_conv_direct_1x1_t
Instance_elx_conv_direct_1x1_t::~elx_conv_direct_1x1_t()
{
  if (tinput_msk_ != nullptr) {
    ::free(tinput_msk_);
    tinput_msk_ = nullptr;
  }
}

// n, ic, ih, iw => n, ic2, ih, iw, V
Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::trans_input_2_blocked(
    InputType *binput, InputType *input)
{
  const __i<V> vindex = _mm<V>::set_epi32(SET_VINDEX_16(this->ih * this->iw));

  if (this->Ir == V) {
    estl::parallel_for<3>(mthr_, [&](int _n, int _ic2, int _t) {
      MD4(InputType, abinput4, binput, this->n, this->ic2, this->ih * this->iw, V);
      MD4(InputType, ainput4, input, this->n, this->ic2, V, this->ih * this->iw);
      if (I == ISA_AVX512 && std::is_same<InputType, float>::value) {
         constexpr int scale = sizeof(InputType);
         __m<V> ain = _mm<V>::i32gather_ps(vindex,
             &md4(ainput4, _n, _ic2, 0, _t), scale);
         _mm<V>::store_ps(&md4(abinput4, _n, _ic2, _t, 0), ain);
      } else {
        #pragma omp simd
        iter_each (_iv, V) {
          md4(abinput4, _n, _ic2, _t, _iv) = md4(ainput4, _n, _ic2, _iv, _t);
        }
      }
    }, this->n, this->ic2, this->ih * this->iw);
  } else {
    estl::parallel_for<3>(mthr_, [&](int _n, int _ic2, int _t) {
      MD4(InputType, abinput4, binput, this->n, this->ic2, this->ih * this->iw, V);
      MD3(InputType, ainput3, input, this->n, this->ic, this->ih * this->iw);
      bool is_Ir = _ic2 == this->ic2 - 1;
      if (is_Ir) {
        #pragma omp simd
        iter_each (_iv, this->Ir) {
          md4(abinput4, _n, _ic2, _t, _iv)
              = md3(ainput3, _n, (this->ic2 - 1) * V + _iv, _t);
        }
      } else {
        if (I == ISA_AVX512 && std::is_same<InputType, float>::value) {
           constexpr int scale = sizeof(InputType);
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
    }, this->n, this->ic2, this->ih * this->iw);
  }
}

// oc, ic => oc2, ic2, V, V
Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::trans_weights_2_blocked(
    WeightsType *bweights, WeightsType *weights)
{
  const __i<V> vindex = _mm<V>::set_epi32(SET_VINDEX_16(this->ic));

  if (this->Ir == V && this->Or == V) {
    estl::parallel_for<3>(mthr_, [&](int _oc2, int _ic2, int _iV) {
      MD4(WeightsType, abweights4, bweights, this->oc2, this->ic2, V, V);
      MD4(WeightsType, aweights4, weights, this->oc2, V, this->ic2, V);
      if (I == ISA_AVX512 && std::is_same<WeightsType, float>::value) {
         constexpr int scale = sizeof(WeightsType);
         __m<V> t = _mm<V>::i32gather_ps(vindex,
             &md4(aweights4, _oc2, 0, _ic2, _iV), scale);
         _mm<V>::store_ps(&md4(abweights4, _oc2, _ic2, _iV, 0), t);
      } else {
        #pragma omp simd
        iter_each (_ov, V) {
          md4(abweights4, _oc2, _ic2, _iV, _ov)
              = md4(aweights4, _oc2, _ov, _ic2, _iV);
        }
      }
    }, this->oc2, this->ic2, V);
  } else {
    estl::parallel_for<2>(mthr_, [&](int _oc2, int _ic2) {
      MD2(WeightsType, aweights2, weights, this->oc, this->ic);
      MD4(WeightsType, abweights4, bweights, this->oc2, this->ic2, V, V);
      bool is_Or = _oc2 == this->oc2 - 1;
      bool is_Ir = _ic2 == this->ic2 - 1;
      int iV = is_Ir ? this->Ir : V;
      if (is_Or) {
        iter_each(_iV, iV) {
          #pragma omp simd
          iter_each (_ov, this->Or) {
            md4(abweights4, _oc2, _ic2, _iV, _ov)
                = md2(aweights2, _oc2 * V + _ov, _ic2 * V + _iV);
          }
        }
      } else {
        iter_each(_iV, iV) {
          if (I == ISA_AVX512 && std::is_same<WeightsType, float>::value) {
             constexpr int scale = sizeof(WeightsType);
             __m<V> t = _mm<V>::i32gather_ps(vindex,
                 &md2(aweights2, _oc2 * V, _ic2 * V + _iV), scale);
             _mm<V>::store_ps(&md4(abweights4, _oc2, _ic2, _iV, 0), t);
          } else {
            #pragma omp simd
            iter_each (_ov, V) {
              md4(abweights4, _oc2, _ic2, _iV, _ov)
                  = md2(aweights2, _oc2 * V + _ov, _ic2 * V + _iV);
            }
          }
        }
      }
    }, this->oc2, this->ic2);
  }
}

// n, oc2, oh, ow, V => n, oc, oh, ow
Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::trans_output_2_plain(
    OutputType *output, OutputType *boutput)
{
  if (this->with_ip_sum) {
    estl::parallel_for<3>(mthr_, [&](int _n, int _oc2, int _oh) {
      MD5(OutputType, aboutput, boutput, this->n, this->oc2, this->oh, this->ow, V);
      MD4(OutputType, aoutput, output, this->n, this->oc, this->oh, this->ow);
      int v = _oc2 == this->oc2 - 1 ? this->Or : V;
      iter_each (_V, v) {
      iter_each (_ow, this->ow) {
        md4(aoutput, _n, _oc2 * V + _V, _oh, _ow)
          += md5(aboutput, _n, _oc2, _oh, _ow, _V);
      }}
    }, this->n, this->oc2, this->oh);
  } else {
    estl::parallel_for<3>(mthr_, [&](int _n, int _oc2, int _oh) {
      MD5(OutputType, aboutput, boutput, this->n, this->oc2, this->oh, this->ow, V);
      MD4(OutputType, aoutput, output, this->n, this->oc, this->oh, this->ow);
      int v = _oc2 == this->oc2 - 1 ? this->Or : V;
      iter_each (_V, v) {
      iter_each (_ow, this->ow) {
        md4(aoutput, _n, _oc2 * V + _V, _oh, _ow)
          = md5(aboutput, _n, _oc2, _oh, _ow, _V);
      }}
    }, this->n, this->oc2, this->oh);
  }
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::__trans_weights_post(WeightsType *aweights,
    TweightsType *tweights, int _O4, int _I4, int _O3, int _I3, int _I2, int _iV, int _O2)
{
  MD8(TweightsType, atweights, tweights, this->O4, this->I4, this->O3, this->I3,
      this->I2, V, this->O2, V);

  if (I == ISA_AVX512 && std::is_same<WeightsType, float>::value) {
    if (std::is_same<TweightsType, float>::value) {
      _mm<V>::store_ps(&md8(atweights, _O4, _I4, _O3, _I3, _I2, _iV, _O2, 0),
                       *(__m<V> *)aweights);
    } else {
      if (this->O == 2) { // fp32->bf16
        auto mask = _mm<V>::set1_epi32(0xFFFF0000);
        if (_O2 == 0) {
          auto si512 = _mm<V>::load_si512(aweights);
          auto w0 = _mm<V>::and_epi32(si512, mask);
          _mm<V>::store_si512((__i<V> *)&md8(atweights, _O4, _I4, _O3,
                                             _I3, _I2, _iV, _O2, 0), w0);
        } else {
          auto si512 = _mm<V>::load_si512(aweights);
          auto w1 = _mm<V>::and_epi32(si512, mask);
          auto sr_w1 = _mm<V>::bsrli_epi128(w1, 2);

          auto w0 = _mm<V>::load_si512(
              &md8(atweights, _O4, _I4, _O3, _I3, _I2, _iV, 0, 0));

          auto w0w1 = _mm<V>::or_epi32(w0, sr_w1);
          _mm<V>::store_si512((__i<V> *)&md8(atweights, _O4, _I4, _O3,
                                             _I3, _I2, _iV, 0, 0), w0w1);
        }
      } else {            // fp32->fp16
        auto fp16v = _mm<V>::cvtps_ph(*(__m<V> *)aweights,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm<V/2>::store_si256(
            (__i<V/2> *)&md8(atweights, _O4, _I4, _O3, _I3, _I2, _iV, _O2, 0), fp16v);
      }
    }
  } else {
    #pragma omp simd
    iter_each (_oV, V) {
      md8(atweights, _O4, _I4, _O3, _I3, _I2, _iV, _O2, _oV) = aweights[_oV];
    }
  }
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::__trans_weights_Or_post(WeightsType *aweights,
    TweightsType *tweights, int _O4, int _I4, int _O3, int _I3, int _I2, int _iV, int _O2)
{
  MD8(TweightsType, atweights, tweights, this->O4, this->I4, this->O3, this->I3,
      this->I2, V, this->O2, V);

  if (I == ISA_AVX512 && std::is_same<WeightsType, float>::value) {
    __mmask16 k = _mm512_int2mask(this->ormask);
    if (std::is_same<TweightsType, float>::value) {
      auto w = _mm<V>::maskz_load_ps(k, aweights);
      _mm<V>::store_ps(
          &md8(atweights, _O4, _I4, _O3, _I3, _I2, _iV, this->O - 1, 0), w);
    } else {
      if (this->O == 2) { // fp32 -> bf16
        // _O index in this path is 1
        auto mask = _mm<V>::set1_epi32(0xFFFF0000);
        auto si512 = _mm<V>::maskz_load_epi32(k, aweights);
        auto w1 = _mm<V>::and_epi32(si512, mask);
        auto sr_w1 = _mm<V>::bsrli_epi128(w1, 2);

        auto w0 = _mm<V>::load_si512(&md8(atweights,
            _O4, _I4, _O3, _I3, _I2, _iV, 0, 0));
        auto w0w1 = _mm<V>::or_epi32(w0, sr_w1);
        _mm<V>::store_si512((__i<V> *)&md8(atweights, _O4, _I4, _O3,
                                           _I3, _I2, _iV, 0, 0), w0w1);
      } else {            // fp32 -> fp16
        auto t = _mm<V>::maskz_load_ps(k, aweights);
        auto fp16v = _mm<V>::cvtps_ph(t,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm<V/2>::store_si256(
            (__i<V/2> *)&md8(atweights, _O4, _I4, _O3, _I3, _I2, _iV, _O2, 0), fp16v);
      }
    }
  } else {
    #pragma omp simd
    iter_each (_oV, this->Or) {
      md8(atweights, _O4, _I4, _O3, _I3, _I2, _iV, _O2, _oV)
        = aweights[_oV];
    }
  }
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::__trans_weights_blocked(
    TweightsType *tweights, WeightsType *weights)
{
  // O4, (O3, O3r), (O2, O2r), I4, I3, I2, V, V ->
  // I4, O4, I3, (O3, O3r), I2, V, (O2, O2r), V

  estl::parallel_for<4>(mthr_, [&](int _O4, int _I4, int _O3, int _I3) {
    iter_each (_I2, this->I2) {
    iter_each (_iV, V) {
    iter_each (_O2, this->O2) {
      MD8(WeightsType, aweights, weights, this->O4, this->O3, this->O2,
          this->I4, this->I3, this->I2, V, V);
      __trans_weights_post(&md8(aweights, _O4, _O3, _O2, _I4, _I3, _I2, _iV, 0),
          tweights, _O4, _I4, _O3, _I3, _I2, _iV, _O2);
    }}}
  }, this->O4, this->I4, this->O3, this->I3);
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::__trans_weights_oihw(
    TweightsType *tweights, WeightsType *weights)
{
  // O4, (O3, O3r), (O2, O2r), V, I4, I3, I2, V ->
  // I4, O4, I3, (O3, O3r), I2, V, (O2, O2r), V
  const __i<V> vindex = _mm<V>::set_epi32(SET_VINDEX_16(this->ic));

  if (this->Ir == V && this->Or == V) {
    estl::parallel_for<5>(mthr_, [&](int _O4, int _I4, int _O3, int _I3, int _I2) {
      iter_each (_iV, V) {
      iter_each (_O2, this->O2) {
        MD8(WeightsType, aweights, weights, this->O4, this->O3, this->O2, V,
            this->I4, this->I3, this->I2, V);
        constexpr int scale = sizeof(WeightsType);
        auto awei = _mm<V>::i32gather_ps(vindex,
            &md8(aweights, _O4, _O3, _O2, 0, _I4, _I3, _I2, _iV), scale);
        __trans_weights_post((WeightsType *)&awei,
            tweights, _O4, _I4, _O3, _I3, _I2, _iV, _O2);
      }}
    }, this->O4, this->I4, this->O3, this->I3, this->I2);
  } else {
    auto readin_v = [&](TweightsType *tweights, WeightsType *weights,
        int _O4, int _O3, int _O2, int _I4, int _I3, int _I2, int _iV) {
      MD8(WeightsType, aweights, weights, this->O4, this->O3, this->O2, V,
          this->I4, this->I3, this->I2, V);
      constexpr auto scale = sizeof(WeightsType);
      auto awei = _mm<V>::i32gather_ps(vindex,
          &md8(aweights, _O4, _O3, _O2, 0, _I4, _I3, _I2, _iV), scale);
      __trans_weights_post((WeightsType *)&awei,
          tweights, _O4, _I4, _O3, _I3, _I2, _iV, _O2);
    };

    auto readin_r = [&](TweightsType *tweights, WeightsType *weights,
        int _O4, int _O3, int _O2, int _I4, int _I3, int _I2, int _iV, bool is_Or) {
      MD2(WeightsType, aweights2, weights, this->oc, this->ic);
      int _oc2 = _O4 * this->O3 * this->O2 + _O3 * this->O2 + _O2;
      int _ic2 = _I4 * this->I3 * this->I2 + _I3 * this->I2 + _I2;
      constexpr auto scale = sizeof(WeightsType);

      if (is_Or) {
        auto zero = _mm<V>::setzero_epi32();
        __mmask16 k = _mm512_int2mask(this->ormask);
        auto awei = _mm<V>::mask_i32gather_epi32(zero, k, vindex,
            &md2(aweights2, _oc2 *V, _ic2 * V + _iV), scale);
        __trans_weights_Or_post((WeightsType *)&awei,
            tweights, _O4, _I4, _O3, _I3, _I2, _iV, _O2);
      } else {
        auto awei = _mm<V>::i32gather_ps(vindex,
            &md2(aweights2, _oc2 *V, _ic2 * V + _iV), scale);
        __trans_weights_post((WeightsType *)&awei,
            tweights, _O4, _I4, _O3, _I3, _I2, _iV, _O2);
      }
    };

    estl::parallel_for<5>(mthr_, [&](int _O4, int _I4, int _O3, int _I3, int _I2) {
      bool is_Ir = (_I4 == this->I4 - 1) && (_I3 == this->I3 -1)
          && (_I2 == this->I2 - 1);
      int iV = is_Ir ? this->Ir : V;
      iter_each (_iV, iV) {
      iter_each (_O2, this->O2) {
        bool is_Or = (_O4 == this->O4 - 1) && (_O3 == this->O3 - 1)
            && (_O2 == this->O2 - 1);
        if (this->Ir != V || is_Ir || is_Or)
          readin_r(tweights, weights, _O4, _O3, _O2, _I4, _I3, _I2, _iV, is_Or);
        else
          readin_v(tweights, weights, _O4, _O3, _O2, _I4, _I3, _I2, _iV);
      }}
    }, this->O4, this->I4, this->O3, this->I3, this->I2);
  }
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::__trans_weights_hwio(
    TweightsType *tweights, WeightsType *weights)
{
  if (this->Ir == V && this->Or == V) {
    estl::parallel_for<5>(mthr_, [&](int _O4, int _I4, int _O3, int _I3, int _I2) {
      MD8(TweightsType, atweights, tweights, this->O4, this->I4, this->O3,
          this->I3, this->I2, V, this->O2, V);
      MD8(WeightsType, aweights, weights, this->I4, this->I3, this->I2, V,
          this->O4, this->O3, this->O2, V);
      iter_each (_iV, V) {
      iter_each (_O2, this->O2) {
        __trans_weights_post(&md8(aweights, _I4, _I3, _I2, _iV, _O4, _O3, _O2, 0),
            tweights, _O4, _I4, _O3, _I3, _I2, _iV, _O2);
      }}
    }, this->O4, this->I4, this->O3, this->I3, this->I2);
  } else {
    auto readin = [&](int _O4, int _I4, int _O3, int _I3,
                      int _I2, int _iV, int _O2) {
      MD2(WeightsType, aweights2, weights, this->ic, this->oc);
      int _ic2 = _I4 * this->I3 * this->I2 + _I3 * this->I2 + _I2;
      int _oc2 = _O4 * this->O3 * this->O2 + _O3 * this->O2 + _O2;
      bool is_Or = this->Or != V && _O4 == this->O4 - 1
          && _O3 == this->O3 - 1 && _O2 == this->O2 - 1;

      if (is_Or)
        __trans_weights_Or_post(&md2(aweights2, _ic2 * V + _iV, _oc2 * V),
            tweights, _O4, _I4, _O3, _I3, _I2, _iV, _O2);
      else
        __trans_weights_post(&md2(aweights2, _ic2 * V + _iV, _oc2 * V),
            tweights, _O4, _I4, _O3, _I3, _I2, _iV, _O2);
    };

    estl::parallel_for<5>(mthr_, [&](int _O4, int _I4, int _O3, int _I3, int _I2) {
      MD8(TweightsType, atweights, tweights, this->O4, this->I4, this->O3,
          this->I3, this->I2, V, this->O2, V);
      MD8(WeightsType, aweights, weights, this->I4, this->I3, this->I2, V,
          this->O4, this->O3, this->O2, V);
      int iV = (this->Ir != V && _I4 == this->I4 - 1
          && _I3 == this->I3 - 1 && _I2 == this->I2 - 1)
          ? this->Ir : V;
      iter_each (_iV, iV) {
      iter_each (_O2, this->O2) {
        readin(_O4, _I4, _O3, _I3, _I2, _iV, _O2);
      }}
    }, this->O4, this->I4, this->O3, this->I3, this->I2);
  }
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::trans_weights(
    TweightsType *tweights, WeightsType *weights)
{
  if (weights_is_bfmt_ || weights_as_bfmt_)
    __trans_weights_blocked(tweights, weights);
  else if (this->weights_fmt == hwio)
    __trans_weights_hwio(tweights, weights);
  else
    __trans_weights_oihw(tweights, weights);
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::__trans_pad_input_blocked(
    TinputType *tinput, InputType *input, int _ht, int _wt)
{
  MD4(TinputType, atinput, tinput, this->I3, this->I2, this->T, V);
  MD5(InputType, ainput, input, this->I3, this->I2, this->ih, this->iw, V);

  int _ih = _ht * this->hs - this->tp;
  iter_each (_I3, this->I3) {
  iter_each (_I2, this->I2) {
  iter_each (_T, this->T) {
    int _iw = _wt * (this->ws * this->T) + _T * this->ws - this->lp;
    if (_ih < 0 || _ih >= this->ih || _iw < 0 || _iw >= this->iw) {
#pragma omp simd
      iter_each (_V, V) {
        md4(atinput, _I3, _I2, _T, _V) = 0.0f;
      }
    } else {
      if (I == ISA_AVX512 && std::is_same<InputType, float>::value) {
        if (stream_in_)
          _mm<V>::stream_ps(&md4(atinput, _I3, _I2, _T,0),
             *((__m<V> *)&md5(ainput, _I3, _I2, _ih, _iw, 0)));
        else
          _mm<V>::store_ps(&md4(atinput, _I3, _I2, _T,0),
             *((__m<V> *)&md5(ainput, _I3, _I2, _ih, _iw, 0)));
      } else {
#pragma omp simd
        iter_each (_V, V) {
          md4(atinput, _I3, _I2, _T, _V)
            = md5(ainput, _I3, _I2, _ih, _iw, _V);
        }
      }
    }
  }}}
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::__trans_input_blocked(
    TinputType *tinput, InputType *input, int _ht, int _wt)
{
  // I3, I2, ht, hs, wt, T, ws, V -> ht, wt | I3, I2, T, V
  MD8(InputType, ainput, input, this->I3, this->I2, this->ht, this->hs,
      this->wt, this->T, this->ws, V);
  MD4(TinputType, atinput, tinput, this->I3, this->I2, this->T, V);

  iter_each (_I3, this->I3) {
  iter_each (_I2, this->I2) {
  iter_each (_T, this->T) {
    if (I == ISA_AVX512 && std::is_same<InputType, float>::value) {
      if (stream_in_)
        _mm<V>::stream_ps(&md4(atinput, _I3, _I2, _T, 0),
             *((__m<V> *)&md8(ainput, _I3, _I2, _ht, 0, _wt, _T, 0, 0)));
      else
        _mm<V>::store_ps(&md4(atinput, _I3, _I2, _T, 0),
             *((__m<V> *)&md8(ainput, _I3, _I2, _ht, 0, _wt, _T, 0, 0)));
    } else {
      #pragma omp simd
      iter_each (_V, V) {
        md4(atinput, _I3, _I2, _T, _V)
            = md8(ainput, _I3, _I2, _ht, 0, _wt, _T, 0, _V);
      }
    }
  }}}
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::__trans_pad_input_plain(
    TinputType *tinput, InputType *input, int _ht, int _wt)
{
  MD3(TinputType, atinput, tinput, this->I3 * this->I2, this->T, V);
  const __i<V> vindex = _mm<V>::set_epi32(SET_VINDEX_16(this->ih * this->iw));

  if (this->Ir == V) {
    MD4(InputType, ainput, input, this->I3 * this->I2, V, this->ih, this->iw);

    int _ih = _ht * this->hs - this->tp;
    iter_each (_ic2, this->I3 * this->I2) {
    iter_each (_T, this->T) {
      int _iw = _wt * (this->ws * this->T) + _T * this->ws - this->lp;
      if (_ih < 0 || _ih >= this->ih || _iw < 0 || _iw >= this->iw) {
        #pragma omp simd
        iter_each (_V, V) {
          md3(atinput, _ic2, _T, _V) = 0.0f;
        }
      } else {
        if (I == ISA_AVX512 && std::is_same<InputType, float>::value) {
           constexpr int scale = sizeof(InputType);
           __m<V> ain = _mm<V>::i32gather_ps(vindex,
               &md4(ainput, _ic2, 0, _ih, _iw), scale);
           _mm<V>::store_ps(&md3(atinput, _ic2, _T, 0), ain);
        } else {
          #pragma omp simd
          iter_each (_V, V) {
            md3(atinput, _ic2, _T, _V) = md4(ainput, _ic2, _V, _ih, _iw);
          }
        }
      }
    }}
  } else {
    MD3(InputType, ainput, input, this->ic, this->ih, this->iw);

    int _ih = _ht * this->hs - this->tp;
    iter_each (_ic2, this->I3 * this->I2) {
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
          if (I == ISA_AVX512 && std::is_same<InputType, float>::value) {
            constexpr int scale = sizeof(InputType);
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

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::__trans_input_nchw(
    TinputType *tinput, InputType *input, int _ht, int _wt)
{
  // I3, I2, V, ht, hs, wt, T, ws -> ht, wt | I3, I2, T, V
  MD3(TinputType, atinput, tinput, this->I3 * this->I2, this->T, V);
  const __i<V> vindex = _mm<V>::set_epi32(SET_VINDEX_16(this->ih * this->iw));
  if (this->Ir == V) {
    MD7(InputType, ainput, input, this->I3 * this->I2, V, this->ht,
        this->hs, this->wt, this->T, this->ws);
    iter_each (_ic2, this->I3 * this->I2) {
    iter_each (_T, this->T) {
      if (I == ISA_AVX512 && std::is_same<InputType, float>::value) {
         constexpr int scale = sizeof(InputType);
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
    MD6(InputType, ainput6, input, this->ic, this->ht, this->hs,
        this->wt, this->T, this->ws);
    iter_each (_ic2, this->I3 * this->I2) {
    iter_each (_T, this->T) {
      bool is_Ir = _ic2 == this->ic2 - 1;
      if (is_Ir) {
        #pragma omp simd
        iter_each (_V, this->Ir) {
          md3(atinput, _ic2, _T, _V)
              = md6(ainput6, (this->ic2 - 1) * V + _V, _ht, 0, _wt, _T, 0);
        }
      } else {
        if (I == ISA_AVX512 && std::is_same<InputType, float>::value) {
          constexpr int scale = sizeof(InputType);
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

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::trans_input(
    TinputType *tinput, InputType *input, int _ht, int _wt)
{
  if (no_pad_) {
    if (input_is_bfmt_ || input_as_bfmt_)
      __trans_input_blocked(tinput, input, _ht, _wt);
    else
      __trans_input_nchw(tinput, input, _ht, _wt);
  } else {
    if (input_is_bfmt_ || input_as_bfmt_)
      __trans_pad_input_blocked(tinput, input, _ht, _wt);
    else
      __trans_pad_input_plain(tinput, input, _ht, _wt);
  }
}


Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::__trans_output_blocked(
    OutputType *output, ToutputType *toutput, int _O4, int _ht, int _wt)
{
  // O3, O2, T, V => n, O4 | O3, O2, ht, wt, T, V
  MD4(ToutputType, atoutput, toutput, this->O3, this->O2, this->T, V);
  MD7(OutputType, aoutput, output, this->O4, this->O3, this->O2,
      this->ht, this->wt, this->T, V);

  iter_each (_O3, this->O3) {
  iter_each (_O2, this->O2) {
  iter_each (_T, this->T) {
    if (this->with_ip_sum && !output_as_bfmt_) {
      #pragma omp simd
      iter_each (_V, V) {
        md7(aoutput, _O4, _O3, _O2, _ht, _wt, _T, _V)
            += md4(atoutput, _O3, _O2, _T, _V);
      }
    } else if (I == ISA_AVX512 && std::is_same<OutputType, float>::value) {
      if (stream_out_)
        _mm<V>::stream_ps(&md7(aoutput, _O4, _O3, _O2, _ht, _wt, _T, 0),
             *((__m<V> *)&md4(atoutput, _O3, _O2, _T, 0)));
      else
        _mm<V>::store_ps(&md7(aoutput, _O4, _O3, _O2, _ht, _wt, _T, 0),
             *((__m<V> *)&md4(atoutput, _O3, _O2, _T, 0)));
    } else {
      #pragma omp simd
      iter_each (_V, V) {
        md7(aoutput, _O4, _O3, _O2, _ht, _wt, _T, _V)
            = md4(atoutput, _O3, _O2, _T, _V);
      }
    }
  }}}
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::__trans_output_nchw(
    OutputType *output, ToutputType *toutput, int _O4, int _ht, int _wt)
{
  // O3, O2, T, V => n, O4 | O3, O2, V, ht, wt, T
  const __i<V> vindex = _mm<V>::set_epi32(SET_VINDEX_16(this->oh * this->ow));
  if (this->Or == V) {
    MD4(ToutputType, atoutput, toutput, this->O3, this->O2, this->T, V);
    MD7(OutputType, aoutput, output, this->O4, this->O3, this->O2, V,
        this->ht, this->wt, this->T);
    iter_each (_O3, this->O3) {
    iter_each (_O2, this->O2) {
    iter_each (_T, this->T) {
      if (this->with_ip_sum && !output_as_bfmt_) {
        #pragma omp simd
        iter_each (_V, V) {
          md7(aoutput, _O4, _O3, _O2, _V, _ht, _wt, _T)
              += md4(atoutput, _O3, _O2, _T, _V);
        }
      } else if (I == ISA_AVX512 && std::is_same<OutputType, float>::value) {
        __m<V> t = _mm<V>::load_ps(&md4(atoutput, _O3, _O2, _T, 0));
        constexpr int scale = sizeof(OutputType);
        _mm<V>::i32scatter_ps(&md7(aoutput, _O4, _O3, _O2, 0, _ht, _wt, _T),
            vindex, t, scale);
      } else {
        #pragma omp simd
        iter_each (_V, V) {
          md7(aoutput, _O4, _O3, _O2, _V, _ht, _wt, _T)
              = md4(atoutput, _O3, _O2, _T, _V);
        }
      }
    }}}
  } else {
    MD4(OutputType, atoutput, toutput, this->O3, this->O2, this->T, V);
    MD4(OutputType, aoutput, output, this->oc, this->ht, this->wt, this->T);
    iter_each (_O3, this->O3) {
    iter_each (_O2, this->O2) {
    iter_each (_T, this->T) {
      int _oc2 = _O4 * this->O3 * this->O2 + _O3 * this->O2 + _O2;
      bool is_Or = (_O4 == this->O4 - 1) && (_O3 == this->O3 - 1)
          && (_O2 == this->O2 - 1);
      if (is_Or) {
        if (this->with_ip_sum && !output_as_bfmt_) {
          #pragma omp simd
          iter_each(_ov, this->Or) {
            md4(aoutput, (this->oc2 - 1) * V + _ov, _ht, _wt, _T)
                += md4(atoutput, _O3, _O2, _T, _ov);
          }
        } else {
          #pragma omp simd
          iter_each(_ov, this->Or) {
            md4(aoutput, (this->oc2 - 1) * V + _ov, _ht, _wt, _T)
                = md4(atoutput, _O3, _O2, _T, _ov);
          }
        }
      } else {
        if (this->with_ip_sum && !output_as_bfmt_) {
          #pragma omp simd
          iter_each(_V, V) {
            md4(aoutput, _oc2 * V + _V, _ht, _wt, _T)
                += md4(atoutput, _O3, _O2, _T, _V);
          }
        } else if (I == ISA_AVX512 && std::is_same<OutputType, float>::value) {
          __m<V> t = _mm<V>::load_ps(&md4(atoutput, _O3, _O2, _T, 0));
          constexpr int scale = sizeof(OutputType);
          _mm<V>::i32scatter_ps(&md4(aoutput, _oc2 * V, _ht, _wt, _T), vindex,
              t, scale);
        } else {
          #pragma omp simd
          iter_each(_V, V) {
            md4(aoutput, _oc2 * V + _V, _ht, _wt, _T)
                = md4(atoutput, _O3, _O2, _T, _V);
          }
        }
      }
    }}}
  }
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::trans_output(
    OutputType *output, ToutputType *toutput, int _O4, int _ht, int _wt)
{
  if (output_is_bfmt_ || output_as_bfmt_)
    __trans_output_blocked(output, toutput, _O4, _ht, _wt);
  else
    __trans_output_nchw(output, toutput, _O4, _ht, _wt);
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::__trans_input_plain2(
    TinputType *tinput, InputType *input, int _t2, int Tz)
{
  MD3(TinputType, atinput, tinput, this->I3 * this->I2, Tz, V);
  const __i<V> vindex = _mm<V>::set_epi32(SET_VINDEX_16(this->ih * this->iw));
  if (this->Ir == V) {
    MD3(InputType, ainput, input, this->I3 * this->I2, V, this->ih * this->iw);
    iter_each (_ic2, this->I3 * this->I2) {
    iter_each (_T, Tz) {
      if (I == ISA_AVX512 && std::is_same<InputType, float>::value) {
         constexpr int scale = sizeof(InputType);
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
    MD2(InputType, ainput2, input, this->ic, this->ih * this->iw);
    iter_each (_ic2, this->I3 * this->I2) {
    iter_each (_T, Tz) {
      bool is_Ir = _ic2 == this->ic2 - 1;
      if (is_Ir) {
        #pragma omp simd
        iter_each (_V, this->Ir) {
          md3(atinput, _ic2, _T, _V)
              = md2(ainput2, (this->ic2 - 1) * V + _V, _t2 * this->T + _T);
        }
      } else {
        if (I == ISA_AVX512 && std::is_same<InputType, float>::value) {
          constexpr int scale = sizeof(InputType);
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

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::__trans_input_blocked2(
    TinputType *tinput, InputType *input, int _t2, int Tz)
{
  MD4(InputType, ainput, input, this->I3, this->I2, this->ih * this->iw, V);
  MD4(TinputType, atinput, tinput, this->I3, this->I2, Tz, V);
  iter_each (_I3, this->I3) {
  iter_each (_I2, this->I2) {
  iter_each (_T, Tz) {
    if (I == ISA_AVX512 && std::is_same<InputType, float>::value) {
      if (stream_in_)
        _mm<V>::stream_ps(&md4(atinput, _I3, _I2, _T, 0),
             *((__m<V> *)&md4(ainput, _I3, _I2, _t2 * this->T + _T, 0)));
      else
        _mm<V>::store_ps(&md4(atinput, _I3, _I2, _T, 0),
             *((__m<V> *)&md4(ainput, _I3, _I2, _t2 * this->T + _T, 0)));
    } else {
      #pragma omp simd
      iter_each (_V, V) {
        md4(atinput, _I3, _I2, _T, _V)
            = md4(ainput, _I3, _I2, _t2 * this->T + _T, _V);
      }
    }
  }}}
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::trans_input2(
    TinputType *tinput, InputType *input, int _t2, int Tz)
{
  if (input_is_bfmt_ || input_as_bfmt_)
    __trans_input_blocked2(tinput, input, _t2, Tz);
  else
    __trans_input_plain2(tinput, input, _t2, Tz);
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::__trans_output_plain2(
    OutputType *output, ToutputType *toutput, int _O4, int _t2, int Tz)
{
  const __i<V> vindex = _mm<V>::set_epi32(SET_VINDEX_16(this->oh * this->ow));

  if (this->Or == V) {
    MD4(ToutputType, atoutput, toutput, this->O3, this->O2, Tz, V);
    MD5(OutputType, aoutput, output, this->O4, this->O3, this->O2, V,
        this->oh * this->ow);
    iter_each (_O3, this->O3) {
    iter_each (_O2, this->O2) {
    iter_each (_T, Tz) {
      if (this->with_ip_sum && !output_as_bfmt_) {
        #pragma omp simd
        iter_each (_V, V) {
          md5(aoutput, _O4, _O3, _O2, _V, _t2 * this->T + _T)
              += md4(atoutput, _O3, _O2, _T, _V);
        }
      } else if (I == ISA_AVX512 && std::is_same<OutputType, float>::value) {
        __m<V> t = _mm<V>::load_ps(&md4(atoutput, _O3, _O2, _T, 0));
        constexpr int scale = sizeof(OutputType);
        _mm<V>::i32scatter_ps(&md5(aoutput, _O4, _O3, _O2, 0, _t2 * this->T + _T),
            vindex, t, scale);
      } else {
        #pragma omp simd
        iter_each (_V, V) {
          md5(aoutput, _O4, _O3, _O2, _V, _t2 * this->T + _T)
              = md4(atoutput, _O3, _O2, _T, _V);
        }
      }
    }}}
  } else {
    MD4(ToutputType, atoutput, toutput, this->O3, this->O2, Tz, V);
    MD2(OutputType, aoutput, output, this->oc, this->oh * this->ow);
    iter_each (_O3, this->O3) {
    iter_each (_O2, this->O2) {
    iter_each (_T, Tz) {
      int _oc2 = _O4 * this->O3 * this->O2 + _O3 * this->O2 + _O2;
      bool is_Or = (_O4 == this->O4 - 1) && (_O3 == this->O3 - 1)
          && (_O2 == this->O2 - 1);
      if (is_Or) {
        if (this->with_ip_sum && !output_as_bfmt_) {
          #pragma omp simd
          iter_each(_ov, this->Or) {
            md2(aoutput, (this->oc2 - 1) * V + _ov, _t2 * this->T + _T)
                += md4(atoutput, _O3, _O2, _T, _ov);
          }
        } else {
          #pragma omp simd
          iter_each(_ov, this->Or) {
            md2(aoutput, (this->oc2 - 1) * V + _ov, _t2 * this->T + _T)
                = md4(atoutput, _O3, _O2, _T, _ov);
          }
        }
      } else {
        if (this->with_ip_sum && !output_as_bfmt_) {
          #pragma omp simd
          iter_each(_V, V) {
            md2(aoutput, _oc2 * V + _V, _t2 * this->T + _T)
                += md4(atoutput, _O3, _O2, _T, _V);
          }
        } else if (I == ISA_AVX512 && std::is_same<OutputType, float>::value) {
          __m<V> t = _mm<V>::load_ps(&md4(atoutput, _O3, _O2, _T, 0));
          constexpr int scale = sizeof(OutputType);
          _mm<V>::i32scatter_ps(&md2(aoutput, _oc2 * V, _t2 * this->T + _T),
              vindex, t, scale);
        } else {
          #pragma omp simd
          iter_each(_V, V) {
            md2(aoutput, _oc2 * V + _V, _t2 * this->T + _T)
                = md4(atoutput, _O3, _O2, _T, _V);
          }
        }
      }
    }}}
  }
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::__trans_output_blocked2(
    OutputType *output, ToutputType *toutput, int _O4, int _t2, int Tz)
{
  // O3, O2, T, V => n, O4 | O3, O2, ht, wt, T, V
  MD4(ToutputType, atoutput, toutput, this->O3, this->O2, Tz, V);
  MD5(OutputType, aoutput, output, this->O4, this->O3, this->O2,
      this->oh * this->ow, V);

  iter_each (_O3, this->O3) {
  iter_each (_O2, this->O2) {
  iter_each (_T, Tz) {
    if (this->with_ip_sum && !output_as_bfmt_) {
      #pragma omp simd
      iter_each (_V, V) {
        md5(aoutput, _O4, _O3, _O2, _t2 * this->T + _T, _V)
            += md4(atoutput, _O3, _O2, _T, _V);
      }
    } else if (I == ISA_AVX512 && std::is_same<OutputType, float>::value) {
      if (stream_out_)
        _mm<V>::stream_ps(&md5(aoutput, _O4, _O3, _O2, _t2 * this->T + _T, 0),
             *((__m<V> *)&md4(atoutput, _O3, _O2, _T, 0)));
      else
        _mm<V>::store_ps(&md5(aoutput, _O4, _O3, _O2, _t2 * this->T + _T, 0),
             *((__m<V> *)&md4(atoutput, _O3, _O2, _T, 0)));
    } else {
      #pragma omp simd
      iter_each (_V, V) {
        md5(aoutput, _O4, _O3, _O2, _t2 * this->T + _T, _V)
            = md4(atoutput, _O3, _O2, _T, _V);
      }
    }
  }}}
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::trans_output2(
    OutputType *output, ToutputType *toutput, int _O4, int _t2, int Tz)
{
  if (output_is_bfmt_ || output_as_bfmt_)
    __trans_output_blocked2(output, toutput, _O4, _t2, Tz);
  else
    __trans_output_plain2(output, toutput, _O4, _t2, Tz);
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::gemm_a061(ToutputType *output,
    TinputType *input, TweightsType *weights, BiasType *bias, int _I4)
{
  // weights: O3*, I3*, O2, I2, V, V
  // input:   I3*, I2, [T], V
  // output:  O3*, O2, [T], V
  MD3(TweightsType, aweights, weights, this->O3, this->I3,
      this->O2 * this->I2 * V * V);
  MD2(BiasType, abias, bias, this->O3, this->O2 * V);

  if (this->input_fmt == nhwc) {
    MD2(TinputType, ainput, input, this->I3, this->I2 * V);
    MD2(ToutputType, aoutput, output, this->O3, this->O2 * V);
    iter_each (_I3, this->I3 - 1) {
      int attr
          = _I4 == 0 && _I3 == 0
          ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK)
          : attr_;
      iter_each (_O3, this->O3) {
        ker_gemm_I_O_T_(
            *this,
            &md2(aoutput, _O3, 0),
            &md2(ainput, _I3, 0),
            &md3(aweights, _O3, _I3, 0),
            &md2(abias, _O3, 0), attr);
      }
    }
    int attr
        = _I4 == 0 && this->I3 == 1
        ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK)
        : attr_;
    if (_I4 == this->I4 - 1) {
      if (this->Ir != V) attr = set_bit(attr, AT_Ir_MASK);
      if (this->with_relu) attr = set_bit(attr, AT_RELU_MASK);
    }
    iter_each (_O3, this->O3) {
      ker_gemm_I_O_T_(
          *this,
          &md2(aoutput, _O3, 0),
          &md2(ainput, this->I3 - 1, 0),
          &md3(aweights, _O3, this->I3 - 1, 0),
          &md2(abias, _O3, 0), attr);
    }
  } else { // nchw
    MD2(TinputType, ainput, input, this->I3, this->I2 * this->T * V);
    MD2(ToutputType, aoutput, output, this->O3, this->O2 * this->T * V);
    iter_each (_I3, this->I3 - 1) {
      int attr
          = _I4 == 0 && _I3 == 0
          ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK)
          : attr_;
      iter_each (_O3, this->O3) {
        ker_gemm_I_O_T_(
            *this,
            &md2(aoutput, _O3, 0),
            &md2(ainput, _I3, 0),
            &md3(aweights, _O3, _I3, 0),
            &md2(abias, _O3, 0), attr);
      }
    }
    int attr
        = _I4 == 0 && this->I3 == 1
        ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK)
        : attr_;
    if (_I4 == this->I4 - 1) {
      if (this->Ir != V) attr = set_bit(attr, AT_Ir_MASK);
      if (this->with_relu) attr = set_bit(attr, AT_RELU_MASK);
    }
    iter_each (_O3, this->O3) {
      ker_gemm_I_O_T_(
          *this,
          &md2(aoutput, _O3, 0),
          &md2(ainput, this->I3 - 1, 0),
          &md3(aweights, _O3, this->I3 - 1, 0),
          &md2(abias, _O3, 0), attr);
    }
  }
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::gemm_b061(OutputType *output,
    TinputType *input, TweightsType *weights, BiasType *bias, int _I4)
{
  // weights: O3*, I3*, O2, I2, V, V
  // input:   I3*, I2, T, V
  // output:  O3*, O2, ht*, wt*, T, V
  MD2(TinputType, ainput, input, this->I3, this->I2 * this->T * V);
  MD5(OutputType, aoutput, output, this->O3, this->O2,
      this->ht, this->wt, this->T * V);
  MD3(TweightsType, aweights, weights, this->O3, this->I3,
      this->O2 * this->I2 * V * V);
  MD2(BiasType, abias, bias, this->O3, this->O2 * V);

  iter_each (_I3, this->I3) {
    int attr
        = _I4 == 0 && _I3 == 0
        ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK)
        : attr_;
    attr
        = this->with_relu && _I4 == this->I4 - 1 && _I3 == this->I3 - 1
        ? set_bit(attr, AT_RELU_MASK)
        : attr;
    iter_each (_O3, this->O3) {
      ker_gemm_I_O_T_(
          *this,
          &md5(aoutput, _O3, 0, 0, 0, 0),
          &md2(ainput, _I3, 0),
          &md3(aweights, _O3, _I3, 0),
          &md2(abias, _O3, 0), attr);
    }
  }
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::gemm_f061(ToutputType *output,
    TinputType *input, TweightsType *weights, BiasType *bias, int _t2, int Tz)
{
  MD3(TweightsType, aweights, weights, this->O3, this->I3,
      this->O2 * this->I2 * V * V);
  MD2(BiasType, abias, bias, this->O3, this->O2 * V);

  auto ker_gemm = (_t2 == this->t2 - 1) ? ker_gemm_I_O_Tr_ : ker_gemm_I_O_T_;

  if (this->input_fmt == nhwc) {
    MD2(TinputType, ainput, input, this->I3, this->I2 * V);
    MD2(ToutputType, aoutput, output, this->O3, this->O2 * V);
    iter_each (_I3, this->I3 - 1) {
      int attr = _I3 == 0 ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK) : attr_;
      iter_each (_O3, this->O3) {
        ker_gemm(
            *this,
            &md2(aoutput, _O3, 0),
            &md2(ainput, _I3, 0),
            &md3(aweights, _O3, _I3, 0),
            &md2(abias, _O3, 0), attr);
      }
    }
    int attr = this->I3 == 1 ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK) : attr_;
    if (this->Ir != V) attr = set_bit(attr, AT_Ir_MASK);
    if (this->with_relu) attr = set_bit(attr, AT_RELU_MASK);
    iter_each(_O3, this->O3) {
      ker_gemm(
          *this,
          &md2(aoutput, _O3, 0),
          &md2(ainput, this->I3 - 1, 0),
          &md3(aweights, _O3, this->I3 - 1, 0),
          &md2(abias, _O3, 0), attr);
    }
  } else { // nchw
    MD2(TinputType, ainput, input, this->I3, this->I2 * Tz * V);
    MD2(ToutputType, aoutput, output, this->O3, this->O2 * Tz * V);
    iter_each (_I3, this->I3 - 1) {
      int attr = _I3 == 0 ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK) : attr_;
      iter_each (_O3, this->O3) {
        ker_gemm(
            *this,
            &md2(aoutput, _O3, 0),
            &md2(ainput, _I3, 0),
            &md3(aweights, _O3, _I3, 0),
            &md2(abias, _O3, 0), attr);
      }
    }
    int attr = this->I3 == 1 ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK) : attr_;
    if (this->Ir != V) attr = set_bit(attr, AT_Ir_MASK);
    if (this->with_relu) attr = set_bit(attr, AT_RELU_MASK);
    iter_each(_O3, this->O3) {
      ker_gemm(
          *this,
          &md2(aoutput, _O3, 0),
          &md2(ainput, this->I3 - 1, 0),
          &md3(aweights, _O3, this->I3 - 1, 0),
          &md2(abias, _O3, 0), attr);
    }
  }
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::gemm_c060(OutputType *output,
    InputType *input, TweightsType *weights, BiasType *bias,
    int _I4, int _O4, int _t2)
{
  // weights: O3, O2, I4*, I3, I2, V, V
  // input:   I3, I2, t2*, T(Tr), V
  // output:  O3, O2, t2*, T(Tr), V
  MD2(InputType, ainput, input, this->I3, this->I2 * this->ih * this->iw * V);
  MD2(OutputType, aoutput, output, this->O3, this->O2 * this->oh * this->ow * V);
  MD3(TweightsType, aweights, weights, this->O3, this->I3,
      this->O2 * this->I2 * V * V);
  MD2(BiasType, abias, bias, this->O3, this->O2 * V);

  auto ker_gemm = (_t2 == this->t2 - 1) ? ker_gemm_I_O_Tr_ : ker_gemm_I_O_T_;

  iter_each (_I3, this->I3) {
    int attr
        = _I4 == 0 && _I3 == 0
        ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK)
        : attr_;
    attr
        = this->with_relu && _I4 == this->I4 - 1 && _I3 == this->I3 - 1
        ? set_bit(attr, AT_RELU_MASK)
        : attr;
    MD2(InputType, ainput2, &md2(ainput, _I3, 0), this->t2, this->T * V);
    iter_each (_O3, this->O3) {
      MD2(OutputType, aoutput2, &md2(aoutput, _O3, 0), this->t2, this->T * V);
      ker_gemm(
          *this,
          &md2(aoutput2, _t2, 0),
          &md2(ainput2, _t2, 0),
          &md3(aweights, _O3, _I3, 0),
          &md2(abias, _O3, 0), attr);
    }
  }
}

} // namespace euler
