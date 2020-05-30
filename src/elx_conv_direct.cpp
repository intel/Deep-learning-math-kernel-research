#include "el_intrin.hpp"
#include "el_stl.hpp"
#include "el_utils.hpp"
#include "el_parallel.hpp"
#include "elx_conv_direct.hpp"
#include "elx_conv_direct_bind.hpp"
#include "elx_conv_direct_xopt.hpp"

namespace euler {

Template_elx_conv_direct_t
Instance_elx_conv_direct_t::elx_conv_direct_t(eld_conv_t &dc)
    : elx_conv_t(dc)
{
  // user input
  xopt_ = ep.execution_mode;
  if (xopt_ == 0) {
    if (estl::any_of(ep.kw, 3, 5, 7))
      xopt_ = 0xc060; // conv kernel
    else
      xopt_ = 0xa060; // gemm kernel
  }
  mthr_ = estl::max_concurrency();

  ep.vmg = 1;
  ep.Vx = 1;
  ep.V1 = V / ep.Vx;
  ep.ocg = ep.oc / ep.g;
  ep.icg = ep.ic / ep.g;

  if (ep.g > 1) {
    // nhwc only
    if ((ep.icg % V != 0 || ep.ocg % V != 0) &&
        (ep.input_fmt != nhwc || ep.output_fmt != nhwc))
      el_error("Unimplemented: group conv with tail ic/oc needs nhwc input/output");

    ep.ic /= ep.g;
    ep.oc /= ep.g;
    if (xopt_ != 0xc060 && xopt_ != 0xa060)
      el_error("Unimplemented: group conv support only 0xc060|0xa060");
  }
  ep.IC = ALIGNUP(ep.ic, V);
  ep.OC = ALIGNUP(ep.oc, V);

  if (ep.I2 == 0) ep.I2 = 1;
  if (ep.T == 0)  ep.T = 1;
  if (ep.O == 0)  ep.O = 1;
  if (ep.O1 == 0) ep.O1 = 1;
  ep.O2 = ep.O * ep.O1;

  ep.O4 = ep.O4 == 0 ? 1 : ep.O4;
  ep.I4 = ep.I4 == 0 ? 1 : ep.I4;

  ep.ic2 = ep.IC / V;
  ep.oc2 = ep.OC / V;

  // n, t2, (T, Tr)
  if (xopt_ == 0xc060 || xopt_ == 0xc070 || xopt_ == 0xa060) {
    ep.ht = ep.oh;
    ep.wt = (ep.ow + ep.T - 1)/ ep.T;
    ep.Tr = ep.ow % ep.T ? ep.ow % ep.T : ep.T;
    ep.nt = ep.oh * ep.ow;
    ep.t2 = ep.nt / ep.T;
    ep.t = ep.nt * ep.n;

    if (ep.T < ep.lp || ep.Tr < ep.rp) {
      el_error("Unimplemented T: (T,Tr) < (lp,rp)");
    }
    bool format_ok =
        estl::any_of(ep.weights_fmt, hwio, ghwio, OIhw16i16o, gOIhw16i16o) &&
        (((ep.input_fmt == nhwc) && (ep.output_fmt == nhwc)) ||
         (V == 16 && xopt_ == 0xc060 &&
          (estl::any_of(ep.input_fmt, nchw, nChw16c)) &&
          (ep.output_fmt == nChw16c)) ||
         (V == 16 && xopt_ == 0xc070 && ep.g == 1 &&
          (ep.input_fmt == nChw16c) && (ep.output_fmt == nChw16c)) ||
         (V == 16 && xopt_ == 0xa060 && (ep.input_fmt == nChw16c) &&
          (ep.output_fmt == nChw16c)));
    if (!format_ok) {
      el_error("direct: format not supported");
    }

    if (xopt_ == 0xc060 || xopt_ == 0xc070) {
      bool shape_ok = estl::any_of(ep.kh, 3, 5, 7)
          && estl::any_of(ep.kw, 3, 5, 7)
          && (ep.ws == 1 || ep.ws == 2)
          && estl::any_of(ep.lp, ep.kw / 2 - 1, ep.kw / 2)
          && estl::any_of(ep.rp, ep.kw / 2 - 1, ep.kw / 2)
          && estl::any_of(ep.tp, ep.kh / 2 - 1, ep.kh / 2)
          && estl::any_of(ep.bp, ep.kh / 2 - 1, ep.kh / 2);
      if (!shape_ok) {
        el_error("direct: c060: shape not supported");
      }
    }

    if (ep.g == 1 && ep.ic < V) {
      bool ok = ep.input_fmt == nchw
          && ep.weights_fmt == hwio
          && xopt_ == 0xc060;
      if (!ok) {
        el_error("direct: first-conv: support only g=1, xopt=c060 with nchw/hwio");
      }
    }
  }

  ep.Ir = ep.ic % V ? ep.ic % V : V;
  ep.Or = ep.oc % V ? ep.oc % V : V;
  ep.ormask = (1 << ep.Or) - 1;

  // O4, (O3, O3r), (O2, O2r)
  ep.oc34 = (ep.oc2 + ep.O2 - 1) / ep.O2;
  ep.O2r = ep.oc2 % ep.O2;
  if (ep.O2r == 0) ep.O2r = ep.O2;
  ep.O3 = ep.O4; // FIXME, swap order
  ep.O4 = (ep.oc34 + ep.O3 - 1) / ep.O3;
  ep.O3r = ep.oc34 % ep.O3;
  if (ep.O3r == 0) ep.O3r = ep.O3;

  if (ep.O2r != ep.O2 || ep.O3r != ep.O3) {
    el_error("No oc tailing support");
  }

  // I4, I3, I3
  ep.ic34 = ep.ic2 / ep.I2;
  ep.I3 = ep.ic34 / ep.I4;
  if (ep.I4 * ep.I3 * ep.I2 * V != ep.IC) {
    el_error("IC blocking error");
  }

  attr_ = 0x0;
  is_first_run_ = true;
  inference_acc_ = false;
  inference_acc_ = ep.prop_kind == forward_inference;

  attr_ = ep.with_bias ? set_bit(attr_, AT_BIAS_MASK) : attr_;
  attr_ = ep.with_ip_sum ? set_bit(attr_, AT_INP_SUM_MASK) : attr_;

  prepare_execute_opt();
  bind_execute_functions();

  // dbg
  el_log(__DEBUG, "T=%d, Tr=%d, t2=%d, ht=%d, wt=%d, t=%d",
         ep.T, ep.Tr, ep.t2, ep.ht, ep.wt, ep.t);
  el_log(__DEBUG, "V=%d, Ir=%d, I2=%d, I3=%d, I4=%d, IC=%d, g=%d",
         V, ep.Ir, ep.I2, ep.I3, ep.I4, ep.IC, ep.g);
  el_log(__DEBUG, "V=%d, Or=%d, O2=%d (O=%d, O1=%d), O3=%d, O4=%d, O2r=%d, O3r=%d, OC=%d, g=%d",
         V, ep.Or, ep.O2, ep.O, ep.O1,
         ep.O3, ep.O4, ep.O2r, ep.O3r, ep.OC, ep.g);
}

Template_elx_conv_direct_t
int Instance_elx_conv_direct_t::prepare_execute_opt()
{
  if (ep.with_ip_sum && ep.with_relu && ep.output_fmt != nChw16c) {
    el_error("Unimplemented: fuse sum (plain format) and relu together");
  }

  toutput_size_ = 0;
  tweights_size_ = 0;
  tweights_ = nullptr;
  toutput_ = nullptr;

  switch (xopt_) {
  case 0xc070:
    toutput_size_ = ep.I4 * ep.n * ep.g * ep.OC * ep.oh * ep.ow * sizeof(ToutputType);
  case 0xc060:
  case 0xa060:
    tweights_size_ = ep.g * ep.kh * ep.kw * ep.IC * ep.OC * sizeof(TweightsType);
    break;
  default:
    el_error("Unknown xopt!");
    return -1;
    break;
  }

#define WEIGHTS_MAX_PRELOAD 4
  if (tweights_size_ > 0)
    tweights_size_ += WEIGHTS_MAX_PRELOAD * V;
  workspace_size_ = tweights_size_;
  scratch_size_ = toutput_size_;

  return 0;
}

Template_elx_conv_direct_t
void Instance_elx_conv_direct_t::set_workspace_buffers(void *base)
{
  if (base != nullptr)
    tweights_ = (TweightsType *)base;
}

Template_elx_conv_direct_t
void Instance_elx_conv_direct_t::set_scratch_buffers(void *base)
{
  if (base != nullptr)
    toutput_ = (ToutputType *)base;
}

Template_elx_conv_direct_t
Instance_elx_conv_direct_t::~elx_conv_direct_t()
{
}

Template_elx_conv_direct_t
void Instance_elx_conv_direct_t::__trans_weights_post(WeightsType *aweights,
    TweightsType *tweights, int _g, int _O4, int _I4, int _O3, int _I3,
    int _kh, int _kw, int _O1, int _I2, int _iV, int _O)
{
  int Vr = (ep.g == 1) && (ep.ic < V) ? ep.Ir : V;
  MD12(TweightsType, atweights, tweights, ep.g, ep.O4, ep.I4, ep.O3,
       ep.I3, ep.kh, ep.kw, ep.O1, ep.I2, Vr, ep.O, V);

  if (I == ISA_AVX512 && std::is_same<WeightsType, float>::value) {
    if (std::is_same<TweightsType, float>::value) {
      _mm<V>::store_ps(&md12(atweights, _g, _O4, _I4, _O3, _I3, _kh, _kw,
                             _O1, _I2, _iV, _O, 0), *(__m<V> *)aweights);
    } else {
      if (ep.O == 2) { // fp32 -> bf16
        auto mask = _mm<V>::set1_epi32(0xFFFF0000);
        if (_O == 0) {
          auto si512 = _mm<V>::load_si512(aweights);
          auto w0 = _mm<V>::and_epi32(si512, mask);
          _mm<V>::store_si512((__i<V> *)&md12(atweights,
              _g, _O4, _I4, _O3, _I3, _kh, _kw, _O1, _I2, _iV, 0, 0), w0);
        } else {
          auto si512 = _mm<V>::load_si512(aweights);
          auto w1 = _mm<V>::and_epi32(si512, mask);
          auto sr_w1 = _mm<V>::bsrli_epi128(w1, 2);

          auto w0 = _mm<V>::load_si512(&md12(atweights,
              _g, _O4, _I4, _O3, _I3, _kh, _kw, _O1, _I2, _iV, 0, 0));
          auto w0w1 = _mm<V>::or_epi32(w0, sr_w1);
          _mm<V>::store_si512((__i<V> *)&md12(atweights,
              _g, _O4, _I4, _O3, _I3, _kh, _kw, _O1, _I2, _iV, 0, 0), w0w1);
        }
      } else {            // fp32 -> fp16
        auto fp16v = _mm<V>::cvtps_ph(*(__m<V> *)aweights,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm<V/2>::store_si256((__m256i *)&md12(atweights,
            _g, _O4, _I4, _O3, _I3, _kh, _kw, _O1, _I2, _iV, _O, 0), fp16v);
      }
    }
  } else {
    #pragma omp simd
    iter_each (_oV, V) {
      md12(atweights, _g, _O4, _I4, _O3, _I3, _kh, _kw, _O1, _I2, _iV, _O,
           _oV) = aweights[_oV];
    }
  }
}

Template_elx_conv_direct_t
void Instance_elx_conv_direct_t::__trans_weights_Or_post(WeightsType *aweights,
    TweightsType *tweights, int _g, int _O4, int _I4, int _O3, int _I3,
    int _kh, int _kw, int _O1, int _I2, int _iV, int _O)
{
  int Vr = (ep.g == 1) && (ep.ic < V) ? ep.Ir : V;
  MD12(TweightsType, atweights, tweights, ep.g, ep.O4, ep.I4, ep.O3,
       ep.I3, ep.kh, ep.kw, ep.O1, ep.I2, Vr, ep.O, V);

  if (I == ISA_AVX512 && std::is_same<WeightsType, float>::value) {
    __mmask16 k = _mm512_int2mask(ep.ormask);
    if (std::is_same<TweightsType, float>::value) {
      auto w = _mm<V>::maskz_load_ps(k, aweights);
      _mm<V>::store_ps(&md12(atweights, _g, _O4, _I4, _O3, _I3,
                       _kh, _kw, _O1, _I2, _iV, _O, 0), w);
    } else {
      if (ep.O == 2) { // fp32 -> bf16
        // _O index in this path is 1
        auto mask = _mm<V>::set1_epi32(0xFFFF0000);
        auto si512 = _mm<V>::maskz_load_epi32(k, aweights);
        auto w1 = _mm<V>::and_epi32(si512, mask);
        auto sr_w1 = _mm<V>::bsrli_epi128(w1, 2);

        auto w0 = _mm<V>::load_si512(&md12(atweights,
            _g, _O4, _I4, _O3, _I3, _kh, _kw, _O1, _I2, _iV, 0, 0));
        auto w0w1 = _mm<V>::or_epi32(w0, sr_w1);
        _mm<V>::store_si512((__i<V> *)&md12(atweights,
            _g, _O4, _I4, _O3, _I3, _kh, _kw, _O1, _I2, _iV, 0, 0), w0w1);
      } else {            // fp32 -> fp16
        auto w = _mm<V>::maskz_load_ps(k, aweights);
        auto fp16v = _mm<V>::cvtps_ph(w, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm<V / 2>::store_si256((__m256i *)&md12(atweights,
            _g, _O4, _I4, _O3, _I3, _kh, _kw, _O1, _I2, _iV, _O, 0), fp16v);
      }
    }
  } else {
    #pragma omp simd
    iter_each(_oV, ep.Or) {
      md12(atweights, _g, _O4, _I4, _O3, _I3, _kh, _kw, _O1, _I2, _iV, _O,
           _oV) = aweights[_oV];
    }
  }
}

// weights (hwio): kh, kw, ic, oc
// weights (blocked): oc2, ic2, kh, kw, V, V
// tweights: O4, I4, O3, _I3, kh, kw, O1, I2, V, O, V
Template_elx_conv_direct_t
void Instance_elx_conv_direct_t::trans_weights_to_compact(
    TweightsType *tweights, WeightsType *weights)
{
  // clang-format off
  if (ep.weights_fmt == OIhw16i16o || ep.weights_fmt == gOIhw16i16o) {
    // skip O for tasks allocation, as O == 2 will be optimized by BF16 type
    estl::parallel_for<8, 4>([&](int _g, int _O4, int _O3, int _O1,
                                 int _O, int _I4, int _I3, int _I2) {
      MD12(WeightsType, aweights, weights, ep.g, ep.O4, ep.O3, ep.O1,
           ep.O, ep.I4, ep.I3, ep.I2, ep.kh, ep.kw, V, V);
      iter_each (_kh, ep.kh) {
      iter_each (_kw, ep.kw) {
      iter_each (_iV, V) {
        __trans_weights_post(
            &md12(aweights, _g, _O4, _O3, _O1, _O, _I4, _I3, _I2, _kh, _kw, _iV, 0),
            tweights, _g, _O4, _I4, _O3, _I3, _kh, _kw, _O1, _I2, _iV, _O);
      }}}
    }, ep.g, ep.O4, ep.O3, ep.O1, ep.O, ep.I4, ep.I3, ep.I2);
  } else if (ep.weights_fmt == hwio || ep.weights_fmt == ghwio) {
    estl::parallel_for<6>([&](int _g, int _kh, int _kw,
                              int _I4, int _I3, int _I2) {
      MD5(WeightsType, aweights0, weights, ep.g, ep.kh, ep.kw, ep.ic, ep.oc);
      auto Ir = _I4 == ep.I4 - 1 && _I3 == ep.I3 - 1
           && _I2 == ep.I2 - 1 ? ep.Ir : V;
      iter_each (_iV, Ir) {
      iter_each (_O4, ep.O4) {
      iter_each (_O3, ep.O3) {
      iter_each (_O1, ep.O1) {
        // handling ic/oc != 16x
        MD5(WeightsType, aweights1, &md5(aweights0, _g, _kh, _kw, 0, 0),
            ep.I4, ep.I3, ep.I2, V, ep.oc);
        MD5(WeightsType, aweights2, &md5(aweights1, _I4, _I3, _I2, _iV, 0),
            ep.O4, ep.O3, ep.O1, ep.O, V);

        bool is_Or = ep.Or != V && _O4 == ep.O4 - 1
            && _O3 == ep.O3 - 1 && _O1 == ep.O1 - 1;
        auto O = is_Or ? ep.O - 1: ep.O;
        iter_each(_O, O) {
          __trans_weights_post(&md5(aweights2, _O4, _O3, _O1, _O, 0),
              tweights, _g, _O4, _I4, _O3, _I3, _kh, _kw, _O1, _I2, _iV, _O);
        }

        // handling Or
        if (is_Or) {
          __trans_weights_Or_post(&md5(aweights2, _O4, _O3, _O1, ep.O - 1, 0),
              tweights, _g, _O4, _I4, _O3, _I3, _kh, _kw, _O1, _I2, _iV, ep.O - 1);
        }
      }}}}
    }, ep.g, ep.kh, ep.kw, ep.I4, ep.I3, ep.I2);
  } else {
    el_error("Unimplemented weights format\n");
  }
  // clang-format on
}

// kh,kw=odd, lp=rp=standard, ih=oh*hs, iw=ow*ws, hs=ws=1
Template_elx_conv_direct_t void
Instance_elx_conv_direct_t::conv_c060(OutputType *output,
    InputType *input, TweightsType *weights, BiasType *bias, int _I4, int _O4,
    int _ht, int _wt)
{
  // input:   I3*, I2, V, ht*, hs*, wt*, T, ws
  // output:  O3*, O2, ht*, wt*, T, V
  int Vr = (ep.g == 1 && ep.ic < V) ? ep.Ir : V;
  MD3(TweightsType, aweights, weights, ep.O3, ep.I3,
      ep.kh * ep.kw * ep.O2 * ep.I2 * V * Vr);
  MD2(BiasType, abias, bias, ep.O3, ep.O2 * V);

  auto ker_conv = _wt == ep.wt - 1 ? ker_conv_Tr_ : ker_conv_;

  int khs = estl::max(0, ep.tp - ep.hs * _ht);
  int khe = estl::min(ep.kh, ep.ih + ep.tp - ep.hs * _ht);
  int kws = _wt == 0 ? ep.lp : 0;
  int kwe = _wt == ep.wt - 1 ? ep.kw - ep.rp : ep.kw;
  assert(ep.T > ep.lp && ep.Tr > ep.rp);

  auto _ih = _ht * ep.hs + (ep.kh / 2) - ep.tp;
  auto _iw = _wt * ep.T * ep.ws + (ep.kw / 2) - ep.lp;
  int pad_l = _wt == 0 ? ep.lp : 0;
  int pad_r = _wt == ep.wt - 1 ? ep.rp : 0;

  if (ep.input_fmt == nhwc) {
    MD4(InputType, ainput0, input, ep.ih, ep.iw, ep.g, ep.ic);
    MD3(InputType, ainput1, &md4(ainput0, _ih, _iw, 0, 0), ep.I4, ep.I3, ep.I2 * V);
    MD2(OutputType, aoutput, output, ep.O3, ep.O2 * V);

    iter_each(_O3, ep.O3) {
    iter_each(_I3, ep.I3) {
      int attr = (_I4 == 0 && _I3 == 0) ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK) : attr_;
      if (_I4 == ep.I4 - 1 && _I3 == ep.I3 - 1) {
        if (ep.Ir != V) attr = set_bit(attr, AT_Ir_MASK);
        if (ep.with_relu) attr = set_bit(attr, AT_RELU_MASK);
      }
      if (ep.Or != V && _O4 == ep.O4 - 1 && _O3 == ep.O3 - 1) {
        attr = set_bit(attr, AT_Or_MASK);
      }
      ker_conv(ep, &md2(aoutput, _O3, 0),
          &md3(ainput1, 0, _I3, 0), &md3(aweights, _O3, _I3, 0),
          &md2(abias, _O3, 0), khs, khe, kws, kwe, pad_l, pad_r, attr);
    }}
  } else if (ep.input_fmt == nchw) {
    MD4(InputType, ainput, input, ep.I3, ep.I2 * V, ep.ih, ep.iw);
    MD2(OutputType, aoutput, output, ep.O3, ep.O2 * ep.ht * ep.ow * V);

    iter_each(_O3, ep.O3) {
    iter_each(_I3, ep.I3) {
      int attr = (_I4 == 0 && _I3 == 0) ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK) : attr_;
      if (_I4 == ep.I4 - 1 && _I3 == ep.I3 - 1) {
        if (ep.Ir != V) attr = set_bit(attr, AT_Ir_MASK);
        if (ep.with_relu) attr = set_bit(attr, AT_RELU_MASK);
      }
      ker_conv(ep, &md2(aoutput, _O3, 0),
          &md4(ainput, _I3, 0, _ih, _iw), &md3(aweights, _O3, _I3, 0),
          &md2(abias, _O3, 0), khs, khe, kws, kwe, pad_l, pad_r, attr);
    }}
  } else { // blocked
    MD5(InputType, ainput, input, ep.I3, ep.I2, ep.ih, ep.iw, V);
    MD2(OutputType, aoutput, output, ep.O3, ep.O2 * ep.ht * ep.ow * V);

    iter_each(_O3, ep.O3) {
    iter_each(_I3, ep.I3) {
      int attr = (_I4 == 0 && _I3 == 0) ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK) : attr_;
      if (_I4 == ep.I4 - 1 && _I3 == ep.I3 - 1) {
        if (ep.Ir != V) attr = set_bit(attr, AT_Ir_MASK);
        if (ep.with_relu) attr = set_bit(attr, AT_RELU_MASK);
      }
      ker_conv(ep, &md2(aoutput, _O3, 0),
          &md5(ainput, _I3, 0, _ih, _iw, 0), &md3(aweights, _O3, _I3, 0),
          &md2(abias, _O3, 0), khs, khe, kws, kwe, pad_l, pad_r, attr);
    }}
  }
}

// kh,kw=odd, lp=rp=standard, ih=oh*hs, iw=ow*ws, hs=ws=1
Template_elx_conv_direct_t void
Instance_elx_conv_direct_t::conv_c070(OutputType *output,
    InputType *input, TweightsType *weights, BiasType *bias, int _I4, int _I3,
    int _O4, int _ht, int _wt)
{
  // input:   I3*, I2, V, ht*, hs*, wt*, T, ws
  // output:  O3*, O2, ht*, wt*, T, V
  MD3(TweightsType, aweights, weights, ep.O3, ep.I3,
      ep.kh * ep.kw * ep.O2 * ep.I2 * V * V);
  MD2(BiasType, abias, bias, ep.O3, ep.O2 * V);

  auto ker_conv = _wt == ep.wt - 1 ? ker_conv_Tr_ : ker_conv_;

  int khs = estl::max(0, ep.tp - ep.hs * _ht);
  int khe = estl::min(ep.kh, ep.ih + ep.tp - ep.hs * _ht);
  int kws = _wt == 0 ? ep.lp : 0;
  int kwe = _wt == ep.wt - 1 ? ep.kw - ep.rp : ep.kw;
  assert(ep.T > ep.lp && ep.Tr > ep.rp);

  auto _ih = _ht * ep.hs + (ep.kh / 2) - ep.tp;
  auto _iw = _wt * ep.T * ep.ws + (ep.kw / 2) - ep.lp;
  int pad_l = _wt == 0 ? ep.lp : 0;
  int pad_r = _wt == ep.wt - 1 ? ep.rp : 0;

  MD2(OutputType, aoutput_nhwc, output, ep.O3, ep.O2 * V);
  MD2(OutputType, aoutput_blocked, output, ep.O3, ep.O2 * ep.ht * ep.ow * V);
  MD3(InputType, ainput_nhwc, input, ep.ih, ep.iw, ep.ic);
  MD4(InputType, ainput_blocked, input, ep.I2, ep.ih, ep.iw, V);

  iter_each(_O3, ep.O3) {
    OutputType *aout = ep.output_fmt == nhwc
                           ? &md2(aoutput_nhwc, _O3, 0)
                           : &md2(aoutput_blocked, _O3, 0);
    InputType *ain   = ep.input_fmt == nhwc
                           ? &md3(ainput_nhwc, _ih, _iw, 0)
                           : &md4(ainput_blocked, 0, _ih, _iw, 0);
    int attr = 0;
    if (_I3 == 0) {
      attr = (_I4 == 0) ? attr_ : attr;
      attr = set_bit(attr, AT_CLEAR_OUTPUT_MASK);
    }
    if (_I4 == ep.I4 - 1 && _I3 == ep.I3 - 1) {
      if (ep.Ir != V) attr = set_bit(attr, AT_Ir_MASK);
    }
    if (ep.output_fmt == nhwc && ep.Or != V && _O4 == ep.O4 - 1 &&
        _O3 == ep.O3 - 1) {
      attr = set_bit(attr, AT_Or_MASK);
    }
    ker_conv(ep, aout, ain, &md3(aweights, _O3, 0, 0),
             &md2(abias, _O3, 0), khs, khe, kws, kwe, pad_l, pad_r, attr);
  }
}

// slow path
Template_elx_conv_direct_t
void Instance_elx_conv_direct_t::gemm_a060(OutputType *output, InputType *input,
    TweightsType *weights, BiasType *bias, int _I4, int _O4, int _ht, int _wt)
{
  // input:   I3*, I2, ht*, hs*, wt*, T, ws, V
  // output:  O3*, O2, ht*, wt*, T, V
  MD5(TweightsType, aweights, weights, ep.O3, ep.I3, ep.kh, ep.kw,
      ep.O2 * ep.I2 * V * V);
  MD3(BiasType, abias, bias, ep.O3, ep.O2, V);

  int Tz = _wt == ep.wt - 1 ? ep.Tr : ep.T;
  int ows0 = _wt * ep.T;
  int khs = estl::max(0, ep.tp - ep.hs * _ht);
  int khe = estl::min(ep.kh, ep.ih + ep.tp - ep.hs * _ht);
  assert(ep.T > ep.lp);
  assert(ep.Tr > ep.rp);

  if (ep.input_fmt == nhwc) {
    MD4(InputType, ainput0, input, ep.ih, ep.iw, ep.g, ep.ic);
    MD4(OutputType, aoutput0, output, ep.ht, ep.ow, ep.g, ep.oc);

    iter_each (_O3, ep.O3) {
    iter_each (_I3, ep.I3) {
      bool oc3_has_Or
          = ep.Or != V && _O4 == ep.O4 - 1 && _O3 == ep.O3 - 1;
      if (_I4 == 0 && _I3 == 0) {
        iter_each (_O2, ep.O2) {
          bool O2_has_Or = oc3_has_Or && (_O2 == ep.O2 - 1);
          __m<V> s = ep.with_bias ? *(__m<V> *)&md3(abias, _O3, _O2, 0)
                                     : _mm<V>::setzero_ps();
          if (I == ISA_AVX512 && std::is_same<OutputType, float>::value) {
            __mmask16 k = _mm512_int2mask(O2_has_Or ? ep.ormask : 0xFFFF);
            iter_each (_T, Tz) {
              MD4(OutputType, aoutput1, &md4(aoutput0, _ht, ows0 + _T, 0, 0),
                  ep.O4, ep.O3, ep.O2, V);
              _mm512_mask_store_ps(&md4(aoutput1, 0, _O3, _O2, 0), k, s);
            }
          } else el_error("direct: a060: unimplemented");
        }
      }
      int attr = attr_;
      if (_I4 == ep.I4 - 1 && _I3 == ep.I3 - 1) {
        if (ep.Ir != V) attr = set_bit(attr, AT_Ir_MASK);
      }
      if (oc3_has_Or) {
        attr = set_bit(attr, AT_Or_MASK);
      }

      for (int _kh = khs; _kh < khe; ++_kh) {
        auto _ih = ep.hs * _ht + _kh - ep.tp;
        for (int _kw = 0; _kw < ep.kw; ++_kw) {
          auto _iws = ep.ws * ows0 + _kw - ep.lp;
          while (_iws < 0) _iws += ep.ws;
          auto _ows = (_iws + ep.lp - _kw) / ep.ws;

          MD4(InputType, ainput1, &md4(ainput0, _ih, _iws, 0, 0), ep.I4,
              ep.I3, ep.I2, V);
          MD4(OutputType, aoutput1, &md4(aoutput0, _ht, _ows, 0, 0), ep.O4,
              ep.O3, ep.O2, V);
          ker_gemm_[_wt][_kw](
              ep, &md4(aoutput1, 0, _O3, 0, 0), &md4(ainput1, 0, _I3, 0, 0),
              &md5(aweights, _O3, _I3, _kh, _kw, 0), &md3(abias, _O3, 0, 0),
              attr);
        }
      }

      if (_I4 == ep.I4 - 1 && _I3 == ep.I3 - 1) {
        if (ep.with_relu) {
          iter_each (_O2, ep.O2) {
            bool O2_has_Or = oc3_has_Or && (_O2 == ep.O2 - 1);
            __mmask16 k = _mm512_int2mask(O2_has_Or ? ep.ormask : 0xFFFF);
            if (I == ISA_AVX512 && std::is_same<OutputType, float>::value) {
              iter_each (_T, Tz) {
                MD4(OutputType, aoutput1, &md4(aoutput0, _ht, ows0 + _T, 0, 0),
                    ep.O4, ep.O3, ep.O2, V);
                auto s = *(__m<V> *)&md4(aoutput1, 0, _O3, _O2, 0);
                auto lower = *(__m<V> *)(ep.relu_bound_lower_vec);
                auto upper = *(__m<V> *)(ep.relu_bound_upper_vec);
                s = _mm<V>::max_ps(s, lower);
                s = _mm<V>::min_ps(s, upper);
                _mm512_mask_store_ps(&md4(aoutput1, 0, _O3, _O2, 0), k, s);
              }
            } else el_error("direct: a060: unimplemented");
          }
        }
      }
    }}
  } else {
    MD5(InputType, ainput, input, ep.I3, ep.I2, ep.ih, ep.iw, V);
    MD5(OutputType, aoutput, output, ep.O3, ep.O2, ep.ht, ep.ow, V);

    iter_each (_O3, ep.O3) {
    iter_each (_I3, ep.I3) {
      if (_I4 == 0 && _I3 == 0) {
        iter_each (_O2, ep.O2) {
          __m<V> s = ep.with_bias ? *(__m<V> *)&md3(abias, _O3, _O2, 0)
                                     : _mm<V>::setzero_ps();
          iter_each (_T, Tz) {
            if (I == ISA_AVX512 && std::is_same<OutputType, float>::value)
              _mm<V>::store_ps(&md5(aoutput, _O3, _O2, _ht, ows0 + _T, 0), s);
            else
              el_error("direct: a060: unimplemented");
          }
        }
      }

      for (int _kh = khs; _kh < khe; ++_kh) {
        auto _ih = ep.hs * _ht + _kh - ep.tp;
        for (int _kw = 0; _kw < ep.kw; ++_kw) {
          auto _iws = ep.ws * ows0 + _kw - ep.lp;
          while (_iws < 0) _iws += ep.ws;
          auto _ows = (_iws + ep.lp - _kw) / ep.ws;
          ker_gemm_[_wt][_kw](ep, &md5(aoutput, _O3, 0, _ht, _ows, 0),
              &md5(ainput, _I3, 0, _ih, _iws, 0),
              &md5(aweights, _O3, _I3, _kh, _kw, 0), &md3(abias, _O3, 0, 0),
              attr_);
        }
      }

      if (_I4 == ep.I4 - 1 && _I3 == ep.I3 - 1) {
        if (ep.with_relu) {
          iter_each (_O2, ep.O2) {
          iter_each (_T, Tz) {
            if (I == ISA_AVX512 && std::is_same<OutputType, float>::value) {
              auto s = *(__m<V> *)&md5(aoutput, _O3, _O2, _ht, ows0 + _T, 0);
              auto lower = *(__m<V> *)(ep.relu_bound_lower_vec);
              auto upper = *(__m<V> *)(ep.relu_bound_upper_vec);
              s = _mm<V>::max_ps(s, lower);
              s = _mm<V>::min_ps(s, upper);
              _mm<V>::store_ps(&md5(aoutput, _O3, _O2, _ht, ows0 + _T, 0), s);
            } else
              el_error("direct: a060: unimplemented");
          }}
        }
      }
    }}
  }
}

// fp32-f32f32f32
template class elx_conv_direct_t<conv::FP32, conv_impl::FP32, 16, ISA_AVX512>;
// fp32-f32f16f32
template class elx_conv_direct_t<conv::FP32, conv_impl::FP32_F16w, 16, ISA_AVX512>;

#ifdef ENABLE_USER_FP16
// fp16o-f32f32f16
template class elx_conv_direct_t<conv::FP16O, conv_impl::FP32_F16o, 16, ISA_AVX512>;
#endif

} // namespace euler
