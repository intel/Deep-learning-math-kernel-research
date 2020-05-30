#include "elx_conv_direct_1x1.hpp"
#include "elx_conv_direct_1x1_bind.hpp"
#include "elx_conv_direct_1x1_xopt.hpp"
#include "el_parallel.hpp"

namespace euler {

Template_elx_conv_direct_1x1_t
Instance_elx_conv_direct_1x1_t::elx_conv_direct_1x1_t(eld_conv_t &dc)
    : elx_conv_t(dc)
{
  // user input
  xopt_ = ep.execution_mode;
  if (xopt_ == 0) {
    if (ep.input_fmt == nChw16c) {
      xopt_ = ep.ws == 1 ? a060 : a061;
    } else { // plain
      xopt_ = ep.ws == 1 ? a061p1 : a061p2;
    }
  }

  ep.Vx = 1;
  ep.V1 = V / ep.Vx;
  ep.IC = ALIGNUP(ep.ic, V);
  ep.OC = ALIGNUP(ep.oc, V);

  if (ep.I2 == 0) ep.I2 = ep.ic2;
  if (ep.T == 0)  ep.T = 1;
  if (ep.O == 0)  ep.O = 1;
  if (ep.O1 == 0) ep.O1 = 1;
  if (ep.O1 != 1) el_error("blk-o != 1 is not supported");
  ep.O2 = ep.O * ep.O1;

  ep.O4 = ep.O4 == 0 ? 1 : ep.O4;
  ep.I4 = ep.I4 == 0 ? 1 : ep.I4;

  ep.ic2 = ep.IC / V;
  ep.oc2 = ep.OC / V;

  no_pad_ = ep.lp == 0 && ep.rp == 0 && ep.tp == 0 && ep.bp == 0;
  if (!no_pad_) {
    if (xopt_ != a061p2)
      el_error("Only a061p2 support padding");
    bool shape_ok =
      (ep.oh == (ep.ih - 1 + ep.tp + ep.bp) / ep.hs + 1) &&
      (ep.ow == (ep.iw - 1 + ep.lp + ep.rp) / ep.ws + 1);
    if (!shape_ok)
      el_error("Unmatched paddding shape not supported by a061p2");
  }

  // n, t2, (T, Tr)
  if (xopt_ == a060 || xopt_ == a061p1) {
    bool shape_ok = ep.hs == 1 && ep.ws == 1 && no_pad_;
    if (!shape_ok)
      el_error("Shape not supported by a060");

    ep.ht = ep.oh;
    ep.wt = ep.ow;
    ep.nt = ep.ht * ep.wt;
    ep.t = ep.nt * ep.n;
    ep.t2 = (ep.nt + ep.T - 1) / ep.T;
    ep.Tr = ep.nt % ep.T ? ep.nt % ep.T : ep.T;
  } else if (xopt_ == a061 || xopt_ == a061p2) {
    ep.ht = ep.oh;
    ep.wt = ep.ow / ep.T;
    ep.nt = ep.oh * ep.ow;
    ep.t2 = ep.nt / ep.T;
    ep.Tr = ep.T; // No Tr support
    ep.t = ep.nt * ep.n;

    if (no_pad_ && (ep.ht * ep.hs != ep.ih
        || ep.wt * ep.ws * ep.T != ep.iw)) {
      el_error("Unimplemented non-unitride shape or blocking");
    }
  }

  ep.Ir = ep.ic % V ? ep.ic % V : V;
  ep.Or = ep.oc % V ? ep.oc % V : V;

  // O4, (O3, O3r), (O2, O2r)
  ep.oc34 = (ep.oc2 + ep.O2 - 1) / ep.O2;
  ep.O2r = ep.oc2 % ep.O2;
  if (ep.O2r == 0) ep.O2r = ep.O2;
  ep.O3 = ep.O4; // FIXME, swap order
  ep.O4 = (ep.oc34 + ep.O3 - 1) / ep.O3;
  ep.O3r = ep.oc34 % ep.O3;
  if (ep.O3r == 0) ep.O3r = ep.O3;

  if ((xopt_ == a061 || xopt_ == a061p1 || xopt_ == a061p2)
      && (ep.O2r != ep.O2 || ep.O3r != ep.O3)) {
    el_error("No oc tailing for a061");
  }

  // I4, I3, I3
  ep.ic34 = ep.ic2 / ep.I2;
  ep.I3 = ep.ic34 / ep.I4;
  if (ep.I4 * ep.I3 * ep.I2 * V != ep.IC)
    el_error("IC blocking error");

  if ((xopt_ == a061p1 || xopt_ == a061p2) && ep.I4 != 1) {
    el_error("I4 != 1 not support in a061p1 and a061p2");
  }

  attr_ = 0x0;
  is_first_run_ = true;
  inference_acc_ = false;
  mthr_ = estl::max_concurrency();
  inference_acc_ = ep.prop_kind == forward_inference;

  attr_ = ep.with_bias ? set_bit(attr_, AT_BIAS_MASK) : attr_;
  if (xopt_ == a061 || xopt_ == a060) {
    attr_ = ep.with_ip_sum ? set_bit(attr_, AT_INP_SUM_MASK) : attr_;
  }

  prepare_execute_opt();
  bind_execute_functions();

  // dbg
  el_log(__DEBUG, "T=%d, Tr=%d, t2=%d, ht=%d, wt=%d, t=%d",
         ep.T, ep.Tr, ep.t2, ep.ht, ep.wt, ep.t);
  el_log(__DEBUG, "V=%d, Ir=%d, I2=%d, I3=%d, I4=%d, IC=%d",
         V, ep.Ir, ep.I2, ep.I3, ep.I4, ep.IC);
  el_log(__DEBUG, "V=%d, Or=%d, O2=%d (O=%d, O1=%d), O3=%d, O4=%d, O2r=%d, O3r=%d, OC=%d",
         V, ep.Or, ep.O2, ep.O, ep.O1,
         ep.O3, ep.O4, ep.O2r, ep.O3r, ep.OC);
}

Template_elx_conv_direct_1x1_t
int  Instance_elx_conv_direct_1x1_t::prepare_execute_opt()
{
  size_t tweights_size = 0, tinput_size = 0, toutput_size = 0;
  size_t binput_size = 0, bweights_size = 0, boutput_size = 0;

  stream_in_ = ep.streaming_input
      ? (ep.streaming_input == STORE_STREAMING) : false;
  stream_out_ = ep.streaming_output
      ? (ep.streaming_output == STORE_STREAMING) : false;

  input_is_bfmt_ = ep.input_fmt == nChw16c; // nChw8c
  weights_is_bfmt_ = ep.weights_fmt == OIhw16i16o;
  output_is_bfmt_ = ep.output_fmt == nChw16c;
  input_as_bfmt_ = ep.input_fmt == nchw && ep.input_as_blocked;
  weights_as_bfmt_ = ep.input_fmt == oihw && ep.weights_as_blocked;
  output_as_bfmt_ = ep.output_fmt == nchw && ep.output_as_blocked;
  is_bfmt_ = input_is_bfmt_ && weights_is_bfmt_ && output_is_bfmt_;

  if (ep.with_ip_sum && ep.with_relu && !output_is_bfmt_) {
    el_error("Unimplemented: fuse sum (plain format) and relu together");
  }

  if (ep.I4 > 1 && ep.Ir != V) {
    el_error("Unimplemented: I4 > 1 for IC % V != 0");
  }
  if (ep.O4 > 1 && ep.Or != V) {
    el_error("Unimplemented: O4 > 1 for OC % V != 0");
  }

  if (!is_bfmt_ && (xopt_ != a061p1 && xopt_ != a061p2)) {
    el_error("Unimplemented: only a061p1, a061p2 mode support plain format\n");
  }

  if (input_as_bfmt_)
    binput_size = ep.n * ep.IC * ep.ih * ep.iw * sizeof(InputType);
  if (weights_as_bfmt_)
    bweights_size = ep.OC * ep.IC * sizeof(WeightsType);
  if (output_as_bfmt_)
    boutput_size = ep.n * ep.OC * ep.oh * ep.ow * sizeof(OutputType);

  tweights_ = nullptr;
  tinput_ = nullptr;
  toutput_ = nullptr;
  tinput_msk_ = nullptr;
  binput_ = nullptr;
  bweights_ = nullptr;
  boutput_ = nullptr;

  switch (xopt_) {
  case a061p1:
    tinput_msk_ = (unsigned char *)aligned_alloc(64, mthr_ * ep.t2);
    toutput_size = mthr_ * ep.O3 * ep.O2 * ep.T * V * sizeof(ToutputType);
    tinput_size = mthr_ * ep.I3 * ep.I2 * ep.T * V * ep.t2 * sizeof(TinputType);
    tweights_size = ep.IC * ep.OC * sizeof(TweightsType);
    break;
  case a061p2:
    toutput_size = mthr_ * ep.O3 * ep.O2 * ep.T * V * sizeof(ToutputType);
  case a061:
    tinput_msk_ = (unsigned char *)aligned_alloc(64, mthr_ * ep.I4 * ep.ht * ep.wt);
    tinput_size = mthr_ * ep.I3 * ep.I2 * V * ep.ht * ep.wt * ep.T * sizeof(TinputType);
    tweights_size = ep.IC * ep.OC * sizeof(TweightsType);
    break;
  case a060:
    tweights_size = ep.IC * ep.OC * sizeof(TweightsType);
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
  const __i<V> vindex = _mm<V>::set_epi32(SET_VINDEX_16(ep.ih * ep.iw));

  if (ep.Ir == V) {
    estl::parallel_for<3>([&](int _n, int _ic2, int _t) {
      MD4(InputType, abinput4, binput, ep.n, ep.ic2, ep.ih * ep.iw, V);
      MD4(InputType, ainput4, input, ep.n, ep.ic2, V, ep.ih * ep.iw);
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
    }, ep.n, ep.ic2, ep.ih * ep.iw);
  } else {
    estl::parallel_for<3>([&](int _n, int _ic2, int _t) {
      MD4(InputType, abinput4, binput, ep.n, ep.ic2, ep.ih * ep.iw, V);
      MD3(InputType, ainput3, input, ep.n, ep.ic, ep.ih * ep.iw);
      bool is_Ir = _ic2 == ep.ic2 - 1;
      if (is_Ir) {
        #pragma omp simd
        iter_each (_iv, ep.Ir) {
          md4(abinput4, _n, _ic2, _t, _iv)
              = md3(ainput3, _n, (ep.ic2 - 1) * V + _iv, _t);
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
    }, ep.n, ep.ic2, ep.ih * ep.iw);
  }
}

// oc, ic => oc2, ic2, V, V
Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::trans_weights_2_blocked(
    WeightsType *bweights, WeightsType *weights)
{
  const __i<V> vindex = _mm<V>::set_epi32(SET_VINDEX_16(ep.ic));

  if (ep.Ir == V && ep.Or == V) {
    estl::parallel_for<3>([&](int _oc2, int _ic2, int _iV) {
      MD4(WeightsType, abweights4, bweights, ep.oc2, ep.ic2, V, V);
      MD4(WeightsType, aweights4, weights, ep.oc2, V, ep.ic2, V);
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
    }, ep.oc2, ep.ic2, V);
  } else {
    estl::parallel_for<2>([&](int _oc2, int _ic2) {
      MD2(WeightsType, aweights2, weights, ep.oc, ep.ic);
      MD4(WeightsType, abweights4, bweights, ep.oc2, ep.ic2, V, V);
      bool is_Or = _oc2 == ep.oc2 - 1;
      bool is_Ir = _ic2 == ep.ic2 - 1;
      int iV = is_Ir ? ep.Ir : V;
      if (is_Or) {
        iter_each(_iV, iV) {
          #pragma omp simd
          iter_each (_ov, ep.Or) {
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
    }, ep.oc2, ep.ic2);
  }
}

// n, oc2, oh, ow, V => n, oc, oh, ow
Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::trans_output_2_plain(
    OutputType *output, OutputType *boutput)
{
  if (ep.with_ip_sum) {
    estl::parallel_for<3>([&](int _n, int _oc2, int _oh) {
      MD5(OutputType, aboutput, boutput, ep.n, ep.oc2, ep.oh, ep.ow, V);
      MD4(OutputType, aoutput, output, ep.n, ep.oc, ep.oh, ep.ow);
      int v = _oc2 == ep.oc2 - 1 ? ep.Or : V;
      iter_each (_V, v) {
      iter_each (_ow, ep.ow) {
        md4(aoutput, _n, _oc2 * V + _V, _oh, _ow)
          += md5(aboutput, _n, _oc2, _oh, _ow, _V);
      }}
    }, ep.n, ep.oc2, ep.oh);
  } else {
    estl::parallel_for<3>([&](int _n, int _oc2, int _oh) {
      MD5(OutputType, aboutput, boutput, ep.n, ep.oc2, ep.oh, ep.ow, V);
      MD4(OutputType, aoutput, output, ep.n, ep.oc, ep.oh, ep.ow);
      int v = _oc2 == ep.oc2 - 1 ? ep.Or : V;
      iter_each (_V, v) {
      iter_each (_ow, ep.ow) {
        md4(aoutput, _n, _oc2 * V + _V, _oh, _ow)
          = md5(aboutput, _n, _oc2, _oh, _ow, _V);
      }}
    }, ep.n, ep.oc2, ep.oh);
  }
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::__trans_weights_post(WeightsType *aweights,
    TweightsType *tweights, int _O4, int _I4, int _O3, int _I3, int _I2, int _iV, int _O2)
{
  MD8(TweightsType, atweights, tweights, ep.O4, ep.I4, ep.O3, ep.I3,
      ep.I2, V, ep.O2, V);

  if (I == ISA_AVX512 && std::is_same<WeightsType, float>::value) {
    if (std::is_same<TweightsType, float>::value) {
      _mm<V>::store_ps(&md8(atweights, _O4, _I4, _O3, _I3, _I2, _iV, _O2, 0),
                       *(__m<V> *)aweights);
    } else {
      if (ep.O == 2) { // fp32->bf16
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
  MD8(TweightsType, atweights, tweights, ep.O4, ep.I4, ep.O3, ep.I3,
      ep.I2, V, ep.O2, V);

  if (I == ISA_AVX512 && std::is_same<WeightsType, float>::value) {
    __mmask16 k = _mm512_int2mask(ep.ormask);
    if (std::is_same<TweightsType, float>::value) {
      auto w = _mm<V>::maskz_load_ps(k, aweights);
      _mm<V>::store_ps(
          &md8(atweights, _O4, _I4, _O3, _I3, _I2, _iV, ep.O - 1, 0), w);
    } else {
      if (ep.O == 2) { // fp32 -> bf16
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
    iter_each (_oV, ep.Or) {
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

  estl::parallel_for<4>([&](int _O4, int _I4, int _O3, int _I3) {
    iter_each (_I2, ep.I2) {
    iter_each (_iV, V) {
    iter_each (_O2, ep.O2) {
      MD8(WeightsType, aweights, weights, ep.O4, ep.O3, ep.O2,
          ep.I4, ep.I3, ep.I2, V, V);
      __trans_weights_post(&md8(aweights, _O4, _O3, _O2, _I4, _I3, _I2, _iV, 0),
          tweights, _O4, _I4, _O3, _I3, _I2, _iV, _O2);
    }}}
  }, ep.O4, ep.I4, ep.O3, ep.I3);
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::__trans_weights_oihw(
    TweightsType *tweights, WeightsType *weights)
{
  // O4, (O3, O3r), (O2, O2r), V, I4, I3, I2, V ->
  // I4, O4, I3, (O3, O3r), I2, V, (O2, O2r), V
  const __i<V> vindex = _mm<V>::set_epi32(SET_VINDEX_16(ep.ic));

  if (ep.Ir == V && ep.Or == V) {
    estl::parallel_for<5>([&](int _O4, int _I4, int _O3, int _I3, int _I2) {
      iter_each (_iV, V) {
      iter_each (_O2, ep.O2) {
        MD8(WeightsType, aweights, weights, ep.O4, ep.O3, ep.O2, V,
            ep.I4, ep.I3, ep.I2, V);
        constexpr int scale = sizeof(WeightsType);
        auto awei = _mm<V>::i32gather_ps(vindex,
            &md8(aweights, _O4, _O3, _O2, 0, _I4, _I3, _I2, _iV), scale);
        __trans_weights_post((WeightsType *)&awei,
            tweights, _O4, _I4, _O3, _I3, _I2, _iV, _O2);
      }}
    }, ep.O4, ep.I4, ep.O3, ep.I3, ep.I2);
  } else {
    auto readin_v = [&](TweightsType *tweights, WeightsType *weights,
        int _O4, int _O3, int _O2, int _I4, int _I3, int _I2, int _iV) {
      MD8(WeightsType, aweights, weights, ep.O4, ep.O3, ep.O2, V,
          ep.I4, ep.I3, ep.I2, V);
      constexpr auto scale = sizeof(WeightsType);
      auto awei = _mm<V>::i32gather_ps(vindex,
          &md8(aweights, _O4, _O3, _O2, 0, _I4, _I3, _I2, _iV), scale);
      __trans_weights_post((WeightsType *)&awei,
          tweights, _O4, _I4, _O3, _I3, _I2, _iV, _O2);
    };

    auto readin_r = [&](TweightsType *tweights, WeightsType *weights,
        int _O4, int _O3, int _O2, int _I4, int _I3, int _I2, int _iV, bool is_Or) {
      MD2(WeightsType, aweights2, weights, ep.oc, ep.ic);
      int _oc2 = _O4 * ep.O3 * ep.O2 + _O3 * ep.O2 + _O2;
      int _ic2 = _I4 * ep.I3 * ep.I2 + _I3 * ep.I2 + _I2;
      constexpr auto scale = sizeof(WeightsType);

      if (is_Or) {
        auto zero = _mm<V>::setzero_epi32();
        __mmask16 k = _mm512_int2mask(ep.ormask);
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

    estl::parallel_for<5>([&](int _O4, int _I4, int _O3, int _I3, int _I2) {
      bool is_Ir = (_I4 == ep.I4 - 1) && (_I3 == ep.I3 -1)
          && (_I2 == ep.I2 - 1);
      int iV = is_Ir ? ep.Ir : V;
      iter_each (_iV, iV) {
      iter_each (_O2, ep.O2) {
        bool is_Or = (_O4 == ep.O4 - 1) && (_O3 == ep.O3 - 1)
            && (_O2 == ep.O2 - 1);
        if (ep.Ir != V || is_Ir || is_Or)
          readin_r(tweights, weights, _O4, _O3, _O2, _I4, _I3, _I2, _iV, is_Or);
        else
          readin_v(tweights, weights, _O4, _O3, _O2, _I4, _I3, _I2, _iV);
      }}
    }, ep.O4, ep.I4, ep.O3, ep.I3, ep.I2);
  }
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::__trans_weights_hwio(
    TweightsType *tweights, WeightsType *weights)
{
  if (ep.Ir == V && ep.Or == V) {
    estl::parallel_for<5>([&](int _O4, int _I4, int _O3, int _I3, int _I2) {
      MD8(TweightsType, atweights, tweights, ep.O4, ep.I4, ep.O3,
          ep.I3, ep.I2, V, ep.O2, V);
      MD8(WeightsType, aweights, weights, ep.I4, ep.I3, ep.I2, V,
          ep.O4, ep.O3, ep.O2, V);
      iter_each (_iV, V) {
      iter_each (_O2, ep.O2) {
        __trans_weights_post(&md8(aweights, _I4, _I3, _I2, _iV, _O4, _O3, _O2, 0),
            tweights, _O4, _I4, _O3, _I3, _I2, _iV, _O2);
      }}
    }, ep.O4, ep.I4, ep.O3, ep.I3, ep.I2);
  } else {
    auto readin = [&](int _O4, int _I4, int _O3, int _I3,
                      int _I2, int _iV, int _O2) {
      MD2(WeightsType, aweights2, weights, ep.ic, ep.oc);
      int _ic2 = _I4 * ep.I3 * ep.I2 + _I3 * ep.I2 + _I2;
      int _oc2 = _O4 * ep.O3 * ep.O2 + _O3 * ep.O2 + _O2;
      bool is_Or = ep.Or != V && _O4 == ep.O4 - 1
          && _O3 == ep.O3 - 1 && _O2 == ep.O2 - 1;

      if (is_Or)
        __trans_weights_Or_post(&md2(aweights2, _ic2 * V + _iV, _oc2 * V),
            tweights, _O4, _I4, _O3, _I3, _I2, _iV, _O2);
      else
        __trans_weights_post(&md2(aweights2, _ic2 * V + _iV, _oc2 * V),
            tweights, _O4, _I4, _O3, _I3, _I2, _iV, _O2);
    };

    estl::parallel_for<5>([&](int _O4, int _I4, int _O3, int _I3, int _I2) {
      MD8(TweightsType, atweights, tweights, ep.O4, ep.I4, ep.O3,
          ep.I3, ep.I2, V, ep.O2, V);
      MD8(WeightsType, aweights, weights, ep.I4, ep.I3, ep.I2, V,
          ep.O4, ep.O3, ep.O2, V);
      int iV = (ep.Ir != V && _I4 == ep.I4 - 1
          && _I3 == ep.I3 - 1 && _I2 == ep.I2 - 1)
          ? ep.Ir : V;
      iter_each (_iV, iV) {
      iter_each (_O2, ep.O2) {
        readin(_O4, _I4, _O3, _I3, _I2, _iV, _O2);
      }}
    }, ep.O4, ep.I4, ep.O3, ep.I3, ep.I2);
  }
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::trans_weights(
    TweightsType *tweights, WeightsType *weights)
{
  if (weights_is_bfmt_ || weights_as_bfmt_)
    __trans_weights_blocked(tweights, weights);
  else if (ep.weights_fmt == hwio)
    __trans_weights_hwio(tweights, weights);
  else
    __trans_weights_oihw(tweights, weights);
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::__trans_pad_input_blocked(
    TinputType *tinput, InputType *input, int _ht, int _wt)
{
  MD4(TinputType, atinput, tinput, ep.I3, ep.I2, ep.T, V);
  MD5(InputType, ainput, input, ep.I3, ep.I2, ep.ih, ep.iw, V);

  int _ih = _ht * ep.hs - ep.tp;
  iter_each (_I3, ep.I3) {
  iter_each (_I2, ep.I2) {
  iter_each (_T, ep.T) {
    int _iw = _wt * (ep.ws * ep.T) + _T * ep.ws - ep.lp;
    if (_ih < 0 || _ih >= ep.ih || _iw < 0 || _iw >= ep.iw) {
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
  MD8(InputType, ainput, input, ep.I3, ep.I2, ep.ht, ep.hs,
      ep.wt, ep.T, ep.ws, V);
  MD4(TinputType, atinput, tinput, ep.I3, ep.I2, ep.T, V);

  iter_each (_I3, ep.I3) {
  iter_each (_I2, ep.I2) {
  iter_each (_T, ep.T) {
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
  MD3(TinputType, atinput, tinput, ep.I3 * ep.I2, ep.T, V);
  const __i<V> vindex = _mm<V>::set_epi32(SET_VINDEX_16(ep.ih * ep.iw));

  if (ep.Ir == V) {
    MD4(InputType, ainput, input, ep.I3 * ep.I2, V, ep.ih, ep.iw);

    int _ih = _ht * ep.hs - ep.tp;
    iter_each (_ic2, ep.I3 * ep.I2) {
    iter_each (_T, ep.T) {
      int _iw = _wt * (ep.ws * ep.T) + _T * ep.ws - ep.lp;
      if (_ih < 0 || _ih >= ep.ih || _iw < 0 || _iw >= ep.iw) {
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
    MD3(InputType, ainput, input, ep.ic, ep.ih, ep.iw);

    int _ih = _ht * ep.hs - ep.tp;
    iter_each (_ic2, ep.I3 * ep.I2) {
    iter_each (_T, ep.T) {
      int _iw = _wt * (ep.ws * ep.T) + _T * ep.ws - ep.lp;
      bool is_Ir = _ic2 == ep.ic2 - 1;
      if (is_Ir) {
        if (_ih < 0 || _ih >= ep.ih || _iw < 0 || _iw >= ep.iw) {
          #pragma omp simd
          iter_each (_V, V) {
            md3(atinput, _ic2, _T, _V) = 0.0f;
          }
        } else {
          #pragma omp simd
          iter_each (_V, ep.Ir) {
            md3(atinput, _ic2, _T, _V)
                = md3(ainput, (ep.ic2 - 1) * V + _V, _ih, _iw);
          }
        }
      } else {
        if (_ih < 0 || _ih >= ep.ih || _iw < 0 || _iw >= ep.iw) {
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
  MD3(TinputType, atinput, tinput, ep.I3 * ep.I2, ep.T, V);
  const __i<V> vindex = _mm<V>::set_epi32(SET_VINDEX_16(ep.ih * ep.iw));
  if (ep.Ir == V) {
    MD7(InputType, ainput, input, ep.I3 * ep.I2, V, ep.ht,
        ep.hs, ep.wt, ep.T, ep.ws);
    iter_each (_ic2, ep.I3 * ep.I2) {
    iter_each (_T, ep.T) {
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
    MD6(InputType, ainput6, input, ep.ic, ep.ht, ep.hs,
        ep.wt, ep.T, ep.ws);
    iter_each (_ic2, ep.I3 * ep.I2) {
    iter_each (_T, ep.T) {
      bool is_Ir = _ic2 == ep.ic2 - 1;
      if (is_Ir) {
        #pragma omp simd
        iter_each (_V, ep.Ir) {
          md3(atinput, _ic2, _T, _V)
              = md6(ainput6, (ep.ic2 - 1) * V + _V, _ht, 0, _wt, _T, 0);
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
  MD4(ToutputType, atoutput, toutput, ep.O3, ep.O2, ep.T, V);
  MD7(OutputType, aoutput, output, ep.O4, ep.O3, ep.O2,
      ep.ht, ep.wt, ep.T, V);

  iter_each (_O3, ep.O3) {
  iter_each (_O2, ep.O2) {
  iter_each (_T, ep.T) {
    if (ep.with_ip_sum && !output_as_bfmt_) {
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
  const __i<V> vindex = _mm<V>::set_epi32(SET_VINDEX_16(ep.oh * ep.ow));
  if (ep.Or == V) {
    MD4(ToutputType, atoutput, toutput, ep.O3, ep.O2, ep.T, V);
    MD7(OutputType, aoutput, output, ep.O4, ep.O3, ep.O2, V,
        ep.ht, ep.wt, ep.T);
    iter_each (_O3, ep.O3) {
    iter_each (_O2, ep.O2) {
    iter_each (_T, ep.T) {
      if (ep.with_ip_sum && !output_as_bfmt_) {
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
    MD4(OutputType, atoutput, toutput, ep.O3, ep.O2, ep.T, V);
    MD4(OutputType, aoutput, output, ep.oc, ep.ht, ep.wt, ep.T);
    iter_each (_O3, ep.O3) {
    iter_each (_O2, ep.O2) {
    iter_each (_T, ep.T) {
      int _oc2 = _O4 * ep.O3 * ep.O2 + _O3 * ep.O2 + _O2;
      bool is_Or = (_O4 == ep.O4 - 1) && (_O3 == ep.O3 - 1)
          && (_O2 == ep.O2 - 1);
      if (is_Or) {
        if (ep.with_ip_sum && !output_as_bfmt_) {
          #pragma omp simd
          iter_each(_ov, ep.Or) {
            md4(aoutput, (ep.oc2 - 1) * V + _ov, _ht, _wt, _T)
                += md4(atoutput, _O3, _O2, _T, _ov);
          }
        } else {
          #pragma omp simd
          iter_each(_ov, ep.Or) {
            md4(aoutput, (ep.oc2 - 1) * V + _ov, _ht, _wt, _T)
                = md4(atoutput, _O3, _O2, _T, _ov);
          }
        }
      } else {
        if (ep.with_ip_sum && !output_as_bfmt_) {
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
  MD3(TinputType, atinput, tinput, ep.I3 * ep.I2, Tz, V);
  const __i<V> vindex = _mm<V>::set_epi32(SET_VINDEX_16(ep.ih * ep.iw));
  if (ep.Ir == V) {
    MD3(InputType, ainput, input, ep.I3 * ep.I2, V, ep.ih * ep.iw);
    iter_each (_ic2, ep.I3 * ep.I2) {
    iter_each (_T, Tz) {
      if (I == ISA_AVX512 && std::is_same<InputType, float>::value) {
         constexpr int scale = sizeof(InputType);
         __m<V> ain = _mm<V>::i32gather_ps(vindex,
             &md3(ainput, _ic2, 0, _t2 * ep.T + _T), scale);
         _mm<V>::store_ps(&md3(atinput, _ic2, _T, 0), ain);
      } else {
        #pragma omp simd
        iter_each (_V, V) {
          md3(atinput, _ic2, _T, _V)
              = md3(ainput, _ic2, _V, _t2 * ep.T + _T);
        }
      }
    }}
  } else {
    MD2(InputType, ainput2, input, ep.ic, ep.ih * ep.iw);
    iter_each (_ic2, ep.I3 * ep.I2) {
    iter_each (_T, Tz) {
      bool is_Ir = _ic2 == ep.ic2 - 1;
      if (is_Ir) {
        #pragma omp simd
        iter_each (_V, ep.Ir) {
          md3(atinput, _ic2, _T, _V)
              = md2(ainput2, (ep.ic2 - 1) * V + _V, _t2 * ep.T + _T);
        }
      } else {
        if (I == ISA_AVX512 && std::is_same<InputType, float>::value) {
          constexpr int scale = sizeof(InputType);
          __m<V> ain = _mm<V>::i32gather_ps(vindex,
              &md2(ainput2, _ic2 * V, _t2 * ep.T + _T), scale);
          _mm<V>::store_ps(&md3(atinput, _ic2, _T, 0), ain);
        } else {
          #pragma omp simd
          iter_each (_V, V) {
            md3(atinput, _ic2, _T, _V)
                = md2(ainput2, _ic2 * V + _V, _t2 * ep.T + _T);
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
  MD4(InputType, ainput, input, ep.I3, ep.I2, ep.ih * ep.iw, V);
  MD4(TinputType, atinput, tinput, ep.I3, ep.I2, Tz, V);
  iter_each (_I3, ep.I3) {
  iter_each (_I2, ep.I2) {
  iter_each (_T, Tz) {
    if (I == ISA_AVX512 && std::is_same<InputType, float>::value) {
      if (stream_in_)
        _mm<V>::stream_ps(&md4(atinput, _I3, _I2, _T, 0),
             *((__m<V> *)&md4(ainput, _I3, _I2, _t2 * ep.T + _T, 0)));
      else
        _mm<V>::store_ps(&md4(atinput, _I3, _I2, _T, 0),
             *((__m<V> *)&md4(ainput, _I3, _I2, _t2 * ep.T + _T, 0)));
    } else {
      #pragma omp simd
      iter_each (_V, V) {
        md4(atinput, _I3, _I2, _T, _V)
            = md4(ainput, _I3, _I2, _t2 * ep.T + _T, _V);
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
  const __i<V> vindex = _mm<V>::set_epi32(SET_VINDEX_16(ep.oh * ep.ow));

  if (ep.Or == V) {
    MD4(ToutputType, atoutput, toutput, ep.O3, ep.O2, Tz, V);
    MD5(OutputType, aoutput, output, ep.O4, ep.O3, ep.O2, V,
        ep.oh * ep.ow);
    iter_each (_O3, ep.O3) {
    iter_each (_O2, ep.O2) {
    iter_each (_T, Tz) {
      if (ep.with_ip_sum && !output_as_bfmt_) {
        #pragma omp simd
        iter_each (_V, V) {
          md5(aoutput, _O4, _O3, _O2, _V, _t2 * ep.T + _T)
              += md4(atoutput, _O3, _O2, _T, _V);
        }
      } else if (I == ISA_AVX512 && std::is_same<OutputType, float>::value) {
        __m<V> t = _mm<V>::load_ps(&md4(atoutput, _O3, _O2, _T, 0));
        constexpr int scale = sizeof(OutputType);
        _mm<V>::i32scatter_ps(&md5(aoutput, _O4, _O3, _O2, 0, _t2 * ep.T + _T),
            vindex, t, scale);
      } else {
        #pragma omp simd
        iter_each (_V, V) {
          md5(aoutput, _O4, _O3, _O2, _V, _t2 * ep.T + _T)
              = md4(atoutput, _O3, _O2, _T, _V);
        }
      }
    }}}
  } else {
    MD4(ToutputType, atoutput, toutput, ep.O3, ep.O2, Tz, V);
    MD2(OutputType, aoutput, output, ep.oc, ep.oh * ep.ow);
    iter_each (_O3, ep.O3) {
    iter_each (_O2, ep.O2) {
    iter_each (_T, Tz) {
      int _oc2 = _O4 * ep.O3 * ep.O2 + _O3 * ep.O2 + _O2;
      bool is_Or = (_O4 == ep.O4 - 1) && (_O3 == ep.O3 - 1)
          && (_O2 == ep.O2 - 1);
      if (is_Or) {
        if (ep.with_ip_sum && !output_as_bfmt_) {
          #pragma omp simd
          iter_each(_ov, ep.Or) {
            md2(aoutput, (ep.oc2 - 1) * V + _ov, _t2 * ep.T + _T)
                += md4(atoutput, _O3, _O2, _T, _ov);
          }
        } else {
          #pragma omp simd
          iter_each(_ov, ep.Or) {
            md2(aoutput, (ep.oc2 - 1) * V + _ov, _t2 * ep.T + _T)
                = md4(atoutput, _O3, _O2, _T, _ov);
          }
        }
      } else {
        if (ep.with_ip_sum && !output_as_bfmt_) {
          #pragma omp simd
          iter_each(_V, V) {
            md2(aoutput, _oc2 * V + _V, _t2 * ep.T + _T)
                += md4(atoutput, _O3, _O2, _T, _V);
          }
        } else if (I == ISA_AVX512 && std::is_same<OutputType, float>::value) {
          __m<V> t = _mm<V>::load_ps(&md4(atoutput, _O3, _O2, _T, 0));
          constexpr int scale = sizeof(OutputType);
          _mm<V>::i32scatter_ps(&md2(aoutput, _oc2 * V, _t2 * ep.T + _T),
              vindex, t, scale);
        } else {
          #pragma omp simd
          iter_each(_V, V) {
            md2(aoutput, _oc2 * V + _V, _t2 * ep.T + _T)
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
  MD4(ToutputType, atoutput, toutput, ep.O3, ep.O2, Tz, V);
  MD5(OutputType, aoutput, output, ep.O4, ep.O3, ep.O2,
      ep.oh * ep.ow, V);

  iter_each (_O3, ep.O3) {
  iter_each (_O2, ep.O2) {
  iter_each (_T, Tz) {
    if (ep.with_ip_sum && !output_as_bfmt_) {
      #pragma omp simd
      iter_each (_V, V) {
        md5(aoutput, _O4, _O3, _O2, _t2 * ep.T + _T, _V)
            += md4(atoutput, _O3, _O2, _T, _V);
      }
    } else if (I == ISA_AVX512 && std::is_same<OutputType, float>::value) {
      if (stream_out_)
        _mm<V>::stream_ps(&md5(aoutput, _O4, _O3, _O2, _t2 * ep.T + _T, 0),
             *((__m<V> *)&md4(atoutput, _O3, _O2, _T, 0)));
      else
        _mm<V>::store_ps(&md5(aoutput, _O4, _O3, _O2, _t2 * ep.T + _T, 0),
             *((__m<V> *)&md4(atoutput, _O3, _O2, _T, 0)));
    } else {
      #pragma omp simd
      iter_each (_V, V) {
        md5(aoutput, _O4, _O3, _O2, _t2 * ep.T + _T, _V)
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
void Instance_elx_conv_direct_1x1_t::gemm_a061p2(ToutputType *output,
    TinputType *input, TweightsType *weights, BiasType *bias, int _I4)
{
  // weights: O3*, I3*, O2, I2, V, V
  // input:   I3*, I2, [T], V
  // output:  O3*, O2, [T], V
  MD3(TweightsType, aweights, weights, ep.O3, ep.I3,
      ep.O2 * ep.I2 * V * V);
  MD2(BiasType, abias, bias, ep.O3, ep.O2 * V);

  if (ep.input_fmt == nhwc) {
    MD2(TinputType, ainput, input, ep.I3, ep.I2 * V);
    MD2(ToutputType, aoutput, output, ep.O3, ep.O2 * V);
    iter_each (_I3, ep.I3 - 1) {
      int attr
          = _I4 == 0 && _I3 == 0
          ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK)
          : attr_;
      iter_each (_O3, ep.O3) {
        ker_gemm_I_O_T_(
            ep,
            &md2(aoutput, _O3, 0),
            &md2(ainput, _I3, 0),
            &md3(aweights, _O3, _I3, 0),
            &md2(abias, _O3, 0), attr);
      }
    }
    int attr
        = _I4 == 0 && ep.I3 == 1
        ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK)
        : attr_;
    if (_I4 == ep.I4 - 1) {
      if (ep.Ir != V) attr = set_bit(attr, AT_Ir_MASK);
      if (ep.with_relu) attr = set_bit(attr, AT_RELU_MASK);
    }
    iter_each (_O3, ep.O3) {
      ker_gemm_I_O_T_(
          ep,
          &md2(aoutput, _O3, 0),
          &md2(ainput, ep.I3 - 1, 0),
          &md3(aweights, _O3, ep.I3 - 1, 0),
          &md2(abias, _O3, 0), attr);
    }
  } else { // nchw
    MD2(TinputType, ainput, input, ep.I3, ep.I2 * ep.T * V);
    MD2(ToutputType, aoutput, output, ep.O3, ep.O2 * ep.T * V);
    iter_each (_I3, ep.I3 - 1) {
      int attr
          = _I4 == 0 && _I3 == 0
          ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK)
          : attr_;
      iter_each (_O3, ep.O3) {
        ker_gemm_I_O_T_(
            ep,
            &md2(aoutput, _O3, 0),
            &md2(ainput, _I3, 0),
            &md3(aweights, _O3, _I3, 0),
            &md2(abias, _O3, 0), attr);
      }
    }
    int attr
        = _I4 == 0 && ep.I3 == 1
        ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK)
        : attr_;
    if (_I4 == ep.I4 - 1) {
      if (ep.Ir != V) attr = set_bit(attr, AT_Ir_MASK);
      if (ep.with_relu) attr = set_bit(attr, AT_RELU_MASK);
    }
    iter_each (_O3, ep.O3) {
      ker_gemm_I_O_T_(
          ep,
          &md2(aoutput, _O3, 0),
          &md2(ainput, ep.I3 - 1, 0),
          &md3(aweights, _O3, ep.I3 - 1, 0),
          &md2(abias, _O3, 0), attr);
    }
  }
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::gemm_a061(OutputType *output,
    TinputType *input, TweightsType *weights, BiasType *bias, int _I4)
{
  // weights: O3*, I3*, O2, I2, V, V
  // input:   I3*, I2, T, V
  // output:  O3*, O2, ht*, wt*, T, V
  MD2(TinputType, ainput, input, ep.I3, ep.I2 * ep.T * V);
  MD5(OutputType, aoutput, output, ep.O3, ep.O2,
      ep.ht, ep.wt, ep.T * V);
  MD3(TweightsType, aweights, weights, ep.O3, ep.I3,
      ep.O2 * ep.I2 * V * V);
  MD2(BiasType, abias, bias, ep.O3, ep.O2 * V);

  iter_each (_I3, ep.I3) {
    bool last_ic3 = _I4 == ep.I4 - 1 && _I3 == ep.I3 - 1;
    int attr = _I4 == 0 && _I3 == 0
        ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK)
        : attr_;
    attr = ep.with_relu && last_ic3
        ? set_bit(attr, AT_RELU_MASK)
        : attr;
    attr = ep.Ir != V && last_ic3 ? set_bit(attr, AT_Ir_MASK) : attr;

    iter_each (_O3, ep.O3) {
      ker_gemm_I_O_T_(
          ep,
          &md5(aoutput, _O3, 0, 0, 0, 0),
          &md2(ainput, _I3, 0),
          &md3(aweights, _O3, _I3, 0),
          &md2(abias, _O3, 0), attr);
    }
  }
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::gemm_a061p1(ToutputType *output,
    TinputType *input, TweightsType *weights, BiasType *bias, int _t2, int Tz)
{
  MD3(TweightsType, aweights, weights, ep.O3, ep.I3,
      ep.O2 * ep.I2 * V * V);
  MD2(BiasType, abias, bias, ep.O3, ep.O2 * V);

  auto ker_gemm = (_t2 == ep.t2 - 1) ? ker_gemm_I_O_Tr_ : ker_gemm_I_O_T_;

  if (ep.input_fmt == nhwc) {
    MD2(TinputType, ainput, input, ep.I3, ep.I2 * V);
    MD2(ToutputType, aoutput, output, ep.O3, ep.O2 * V);
    iter_each (_I3, ep.I3 - 1) {
      int attr = _I3 == 0 ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK) : attr_;
      iter_each (_O3, ep.O3) {
        ker_gemm(
            ep,
            &md2(aoutput, _O3, 0),
            &md2(ainput, _I3, 0),
            &md3(aweights, _O3, _I3, 0),
            &md2(abias, _O3, 0), attr);
      }
    }
    int attr = ep.I3 == 1 ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK) : attr_;
    if (ep.Ir != V) attr = set_bit(attr, AT_Ir_MASK);
    if (ep.with_relu) attr = set_bit(attr, AT_RELU_MASK);
    iter_each(_O3, ep.O3) {
      ker_gemm(
          ep,
          &md2(aoutput, _O3, 0),
          &md2(ainput, ep.I3 - 1, 0),
          &md3(aweights, _O3, ep.I3 - 1, 0),
          &md2(abias, _O3, 0), attr);
    }
  } else { // nchw
    MD2(TinputType, ainput, input, ep.I3, ep.I2 * Tz * V);
    MD2(ToutputType, aoutput, output, ep.O3, ep.O2 * Tz * V);
    iter_each (_I3, ep.I3 - 1) {
      int attr = _I3 == 0 ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK) : attr_;
      iter_each (_O3, ep.O3) {
        ker_gemm(
            ep,
            &md2(aoutput, _O3, 0),
            &md2(ainput, _I3, 0),
            &md3(aweights, _O3, _I3, 0),
            &md2(abias, _O3, 0), attr);
      }
    }
    int attr = ep.I3 == 1 ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK) : attr_;
    if (ep.Ir != V) attr = set_bit(attr, AT_Ir_MASK);
    if (ep.with_relu) attr = set_bit(attr, AT_RELU_MASK);
    iter_each(_O3, ep.O3) {
      ker_gemm(
          ep,
          &md2(aoutput, _O3, 0),
          &md2(ainput, ep.I3 - 1, 0),
          &md3(aweights, _O3, ep.I3 - 1, 0),
          &md2(abias, _O3, 0), attr);
    }
  }
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::gemm_a060(OutputType *output,
    InputType *input, TweightsType *weights, BiasType *bias,
    int _I4, int _O4, int _t2)
{
  // weights: O3, O2, I4*, I3, I2, V, V
  // input:   I3, I2, t2*, T(Tr), V
  // output:  O3, O2, t2*, T(Tr), V
  MD2(InputType, ainput, input, ep.I3, ep.I2 * ep.ih * ep.iw * V);
  MD2(OutputType, aoutput, output, ep.O3, ep.O2 * ep.oh * ep.ow * V);
  MD3(TweightsType, aweights, weights, ep.O3, ep.I3,
      ep.O2 * ep.I2 * V * V);
  MD2(BiasType, abias, bias, ep.O3, ep.O2 * V);

  auto ker_gemm = (_t2 == ep.t2 - 1) ? ker_gemm_I_O_Tr_ : ker_gemm_I_O_T_;

  iter_each (_I3, ep.I3) {
    bool last_ic3 = _I4 == ep.I4 - 1 && _I3 == ep.I3 - 1;
    int attr = _I4 == 0 && _I3 == 0
        ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK)
        : attr_;
    attr = ep.with_relu && last_ic3
        ? set_bit(attr, AT_RELU_MASK)
        : attr;
    attr = ep.Ir != V && last_ic3 ? set_bit(attr, AT_Ir_MASK) : attr;

    MD2(InputType, ainput2, &md2(ainput, _I3, 0), ep.t2, ep.T * V);
    iter_each (_O3, ep.O3) {
      MD2(OutputType, aoutput2, &md2(aoutput, _O3, 0), ep.t2, ep.T * V);
      ker_gemm(
          ep,
          &md2(aoutput2, _t2, 0),
          &md2(ainput2, _t2, 0),
          &md3(aweights, _O3, _I3, 0),
          &md2(abias, _O3, 0), attr);
    }
  }
}

//fp32-f32f32f32
template class elx_conv_direct_1x1_t<conv::FP32, conv_impl::FP32, 16, ISA_AVX512>;

//fp32-f32f16f32
template class elx_conv_direct_1x1_t<conv::FP32, conv_impl::FP32_F16w, 16, ISA_AVX512>;

} // namespace euler
