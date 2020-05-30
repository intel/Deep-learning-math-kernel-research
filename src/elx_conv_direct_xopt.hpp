#pragma once

#include "elx_conv_direct.hpp"
#include "el_parallel.hpp"

// XOPT
//
// fusion:  same as winograd
// dup:     same as winograd
// ------+-----+--------+-----+------------------------------------------------
//       | ker | fusion | dup |             notes
// ------+-----+--------+-----+------------------------------------------------
//  c060 |conv |   t+o  |  -  | nhwc|blocked|nchw-input, Ir/Tr/Or, K=3,5,7 S=1,2, group
// ------+-----+--------+-----+------------------------------------------------
//  c070 |conv |  t+o+i |  -  | nhwc|blocked, Ir/Tr/Or, K=3,5,7 S=1,2 small spatial, group=1
// ------+-----+--------+-----+------------------------------------------------
//  a060 |gemm |   t+o  |  -  | nhwc|blocked, Ir/Tr/Or, group
// ------+-----+--------+-----+------------------------------------------------
//
namespace euler {

Template_elx_conv_direct_t
void Instance_elx_conv_direct_t::__execute_c060(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  // input (blocked): n*, I4*, I3, I2, ht*, S, wt*, T, S, V(Ir)
  // input (nchw): n*, I4*, I3, I2, V(Ir), ht*, S, wt*, T, S
  // input (nhwc): n*, ht*, S, wt*, T, S, I4*, I3, I2, V(Ir)
  // weights: O4*, O3, O2, I4*, I3, I2, V(Ir), V
  // output (blocked):  n*, O4*, O3, O2(O2r), ht*wt*, T, V
  // output (nhwc):  n*, ht*wt*, T, O4*, O3, O2(O2r), V
  if (is_first_run_) {
    setup_workspace([&]() {
      trans_weights_to_compact(tweights_, weights);
    });
  }

  if (ep.input_fmt == nchw) { // nchw => blocked
    estl::parallel_for<5, 1>([&](int _n, int _I4, int _O4, int _ht, int _wt) {
      int Vr = ep.ic < V ? ep.Ir : V;
      MD2(BiasType, abias, bias, ep.O4, ep.O3 * ep.O2 * V);
      MD3(TweightsType, atweights, tweights_, ep.O4, ep.I4,
          V * Vr * ep.kh * ep.kw * ep.I3 * ep.O3 * ep.I2
              * ep.O2);
      MD2(InputType, ainput0, input, ep.n, ep.ic * ep.ih * ep.iw);
      MD3(InputType, ainput1, &md2(ainput0, _n, 0), ep.I4,
          ep.I3 * ep.I2 * V, ep.ih * ep.iw);
      MD5(OutputType, aoutput0, output, ep.n, ep.O4,
          ep.O3 * ep.O2, ep.ht, ep.ow * V);
      MD3(OutputType, aoutput1, &md5(aoutput0, _n, _O4, 0, _ht, 0), ep.wt,
          ep.T, V);
      conv_c060(&md3(aoutput1, _wt, 0, 0), &md3(ainput1, _I4, 0, 0),
          &md3(atweights, _O4, _I4, 0), &md2(abias, _O4, 0), _I4, _O4, _ht,
          _wt);
    }, ep.n, ep.I4, ep.O4, ep.ht, ep.wt);
  } else if (ep.input_fmt == nhwc) { // nhwc => nhwc
    estl::parallel_for<6, 2>([&](int _n, int _g, int _I4, int _O4, int _ht, int _wt) {
      MD2(BiasType, abias0, bias, ep.g, ep.oc);
      MD2(BiasType, abias1, &md2(abias0, _g, 0), ep.O4, ep.O3 * ep.O2 * V);
      MD4(TweightsType, atweights, tweights_, ep.g, ep.O4, ep.I4,
          V * V * ep.kh * ep.kw * ep.I3 * ep.O3 * ep.I2
              * ep.O2);
      MD5(InputType, ainput0, input, ep.n, ep.ih, ep.iw, ep.g,
          ep.ic);
      MD2(InputType, ainput1, &md5(ainput0, _n, 0, 0, _g, 0), ep.I4,
          ep.I3 * ep.I2 * V);
      MD4(OutputType, aoutput0, output, ep.n, ep.ht, ep.ow, ep.g * ep.oc);
      MD4(OutputType, aoutput1, &md4(aoutput0, _n, _ht, 0, 0), ep.wt,
          ep.T, ep.g, ep.oc);
      MD2(OutputType, aoutput2, &md4(aoutput1, _wt, 0, _g, 0), ep.O4,
          ep.O3 * ep.O2 * V);
      conv_c060(&md2(aoutput2, _O4, 0), &md2(ainput1, _I4, 0),
          &md4(atweights, _g, _O4, _I4, 0), &md2(abias1, _O4, 0),
          _I4, _O4, _ht, _wt);
    },  ep.n, ep.g, ep.I4, ep.O4, ep.ht, ep.wt);
  } else { // blocked => blocked
    estl::parallel_for<6, 2>([&](int _n, int _g, int _I4, int _O4, int _ht, int _wt) {
      MD2(BiasType, abias0, bias, ep.g, ep.oc);
      MD2(BiasType, abias1, &md2(abias0, _g, 0), ep.O4, ep.O3 * ep.O2 * V);
      MD4(TweightsType, atweights, tweights_, ep.g, ep.O4, ep.I4,
          V * V * ep.kh * ep.kw * ep.I3 * ep.O3 * ep.I2
              * ep.O2);
      MD5(InputType, ainput, input, ep.n, ep.g, ep.I4,
          ep.I3 * ep.I2, ep.ih * ep.iw * V);
      MD6(OutputType, aoutput0, output, ep.n, ep.g, ep.O4,
          ep.O3 * ep.O2, ep.ht, ep.ow * V);
      MD3(OutputType, aoutput1, &md6(aoutput0, _n, _g, _O4, 0, _ht, 0),
          ep.wt, ep.T, V);
      conv_c060(&md3(aoutput1, _wt, 0, 0), &md5(ainput, _n, _g, _I4, 0, 0),
          &md4(atweights, _g, _O4, _I4, 0), &md2(abias1, _O4, 0),
          _I4, _O4, _ht, _wt);
    }, ep.n, ep.g, ep.I4, ep.O4, ep.ht, ep.wt);
  }

  if (is_first_run_ && inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_direct_t
void Instance_elx_conv_direct_t::__execute_c070(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  // input (blocked): n*, I4*, I3, I2, ht*, S, wt*, T, S, V(Ir)
  // input (nhwc): n*, ht*, S, wt*, T, S, I4*, I3, I2, V(Ir)
  // weights: O4*, O3, O2, I4*, I3, I2, V(Ir), V
  // output (blocked):  n*, O4*, O3, O2(O2r), ht*wt*, T, V
  // output (nhwc):  n*, ht*wt*, T, O4*, O3, O2(O2r), V
  if (is_first_run_) {
    setup_workspace([&]() {
      trans_weights_to_compact(tweights_, weights);
    });
  }

  THREAD_PARALLEL()
  {
    int ithr = estl::current_thread_index();
    if (ep.input_fmt == nhwc) { // nhwc => nhwc
      THREAD_FOR2(6, 2, mthr_, ithr, [&](int _I4, int _n, int _I3,
                                         int _O4, int _ht, int _wt) {
        MD2(BiasType, abias, bias, ep.O4, ep.O3 * ep.O2 * V);
        MD5(TweightsType, atweights, tweights_, ep.O4, ep.I4, ep.O3,
            ep.I3, V * V * ep.kh * ep.kw * ep.I2 * ep.O2);
        MD4(InputType, ainput0, input, ep.n, ep.ih, ep.iw, ep.ic);
        MD3(InputType, ainput1, &md4(ainput0, _n, 0, 0, 0), ep.I4,
            ep.I3, ep.I2 * V);
        MD5(OutputType, atoutput0, toutput_, ep.I4, ep.n, ep.ht,
            ep.ow, ep.oc);
        MD3(OutputType, atoutput1, &md5(atoutput0, _I4, _n, _ht, 0, 0),
            ep.wt, ep.T, ep.oc);
        MD2(OutputType, atoutput2, &md3(atoutput1, _wt, 0, 0), ep.O4,
            ep.O3 * ep.O2 * V);
        conv_c070(&md2(atoutput2, _O4, 0), &md3(ainput1, _I4, _I3, 0),
            &md5(atweights, _O4, _I4, 0, _I3, 0), &md2(abias, _O4, 0),
            _I4, _I3, _O4, _ht, _wt);
      }, ep.I4, ep.n, ep.I3, ep.O4, ep.ht, ep.wt);

      THREAD_BARRIER()
      THREAD_FOR(4, mthr_, ithr, [&](int _n, int _oh, int _ow, int _oc2) {
        MD5(ToutputType, atoutput0, toutput_, ep.I4, ep.n, ep.oh,
            ep.ow, ep.oc);
        MD4(OutputType, aoutput0, output, ep.n, ep.oh, ep.ow, ep.oc);
        MD2(OutputType, aoutput1, &md4(aoutput0, _n, _oh, _ow, 0), ep.oc2, V);
        if (std::is_same<OutputType, float>::value) {
          __m<V> zero = _mm<V>::setzero_ps();
          __m<V> out = zero;
          for (int _I4 = 0; _I4 < ep.I4; ++_I4) {
            MD2(ToutputType, atoutput1, &md5(atoutput0, _I4, _n, _oh, _ow, 0),
                ep.oc2, V);
            out += *(__m<V> *)&md2(atoutput1, _oc2, 0);
          }
          if (ep.with_relu) {
            auto lower = *(__m<V> *)(ep.relu_bound_lower_vec);
            auto upper = *(__m<V> *)(ep.relu_bound_upper_vec);
            out = _mm<V>::max_ps(out, lower);
            out = _mm<V>::min_ps(out, upper);
          }
          if (ep.Or != V && _oc2 == ep.oc2 - 1) {
            iter_each (_V, ep.Or) {
              md2(aoutput1, _oc2, _V) = out[_V];
            }
          } else {
            *(__m<V> *)&md2(aoutput1, _oc2, 0) = out;
          }
        } else {
          el_error("Unsupported data type");
        }
      }, ep.n, ep.oh, ep.ow, ep.oc2);
    } else { // blocked => blocked
      THREAD_FOR2(6, 2, mthr_, ithr, [&](int _I4, int _n, int _I3,
                                         int _O4, int _ht, int _wt) {
        MD2(BiasType, abias, bias, ep.O4, ep.O3 * ep.O2 * V);
        MD5(TweightsType, atweights, tweights_, ep.O4, ep.I4, ep.O3,
            ep.I3, V * V * ep.kh * ep.kw * ep.I2 * ep.O2);
        MD5(InputType, ainput, input, ep.n, ep.I4, ep.I3, ep.I2,
            ep.ih * ep.iw * V);
        MD6(OutputType, atoutput0, toutput_, ep.I4, ep.n, ep.O4,
            ep.O3 * ep.O2, ep.ht, ep.ow * V);
        MD3(OutputType, atoutput1, &md6(atoutput0, _I4, _n, _O4, 0, _ht, 0),
            ep.wt, ep.T, V);
        conv_c070(&md3(atoutput1, _wt, 0, 0), &md5(ainput, _n, _I4, _I3, 0, 0),
            &md5(atweights, _O4, _I4, 0, _I3, 0), &md2(abias, _O4, 0), _I4,
            _I3, _O4, _ht, _wt);
      }, ep.I4, ep.n, ep.I3, ep.O4, ep.ht, ep.wt);
      THREAD_BARRIER()
      THREAD_FOR(1, mthr_, ithr, [&](int o) {
        MD3(ToutputType, atoutput, toutput_, ep.I4,
            ep.n * ep.oc2 * ep.oh * ep.ow, V);
        MD2(OutputType, aoutput, output,
            ep.n * ep.oc2 * ep.oh * ep.ow, V);
        if (std::is_same<OutputType, float>::value) {
          __m<V> zero = _mm<V>::setzero_ps();
          __m<V> out = zero;
          for (int _I4 = 0; _I4 < ep.I4; ++_I4) {
            out += *(__m<V> *)&md3(atoutput, _I4, o, 0);
          }
          if (ep.with_relu) {
            auto lower = *(__m<V> *)(ep.relu_bound_lower_vec);
            auto upper = *(__m<V> *)(ep.relu_bound_upper_vec);
            out = _mm<V>::max_ps(out, lower);
            out = _mm<V>::min_ps(out, upper);
          }
          *(__m<V> *)&md2(aoutput, o, 0) = out;
        } else {
          el_error("Unsupported data type");
        }
      }, ep.n * ep.oc2 * ep.oh * ep.ow);
    }
  }

  if (is_first_run_ && inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_direct_t
void Instance_elx_conv_direct_t::__execute_a060(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  // input (blocked): n*, I4*, I3, I2, ih, iw, V(Ir)
  // weights: O4*, O3, O2, I4*, I3, I2, V(Ir), V
  // output:  n*, O4*, O3, O2(O2r), ht*wt*, T(Tr), V
  if (is_first_run_) {
    setup_workspace([&]() {
      trans_weights_to_compact(tweights_, weights);
    });
  }

  if (ep.input_fmt == nhwc) { // nhwc -> nhwc
    estl::parallel_for<6, 2>([&](int _n, int _g, int _I4, int _O4, int _ht, int _wt) {
      MD2(BiasType, abias0, bias, ep.g, ep.oc);
      MD2(BiasType, abias1, &md2(abias0, _g, 0), ep.O4, ep.O3 * ep.O2 * V);
      MD4(TweightsType, atweights, tweights_, ep.g, ep.O4, ep.I4,
          V * V * ep.kh * ep.kw * ep.I3 * ep.O3 * ep.I2
              * ep.O2);
      MD4(InputType, ainput0, input, ep.n, ep.ih * ep.iw, ep.g, ep.ic);
      MD2(InputType, ainput1, &md4(ainput0, _n, 0, _g, 0), ep.I4,
          ep.I3 * ep.I2 * V);
      MD4(OutputType, aoutput0, output, ep.n, ep.ht * ep.ow, ep.g,
          ep.oc);
      MD2(OutputType, aoutput1, &md4(aoutput0, _n, 0, _g, 0), ep.O4,
          ep.O3 * ep.O2 * V);
      gemm_a060(&md2(aoutput1, _O4, 0), &md2(ainput1, _I4, 0),
          &md4(atweights, _g, _O4, _I4, 0), &md2(abias1, _O4, 0),
          _I4, _O4, _ht, _wt);
    }, ep.n, ep.g, ep.I4, ep.O4, ep.ht, ep.wt);
  } else { // blocked -> blocked
    estl::parallel_for<6, 2>([&](int _n, int _g, int _I4, int _O4, int _ht, int _wt) {
      MD2(BiasType, abias0, bias, ep.g, ep.oc);
      MD2(BiasType, abias1, &md2(abias0, _g, 0), ep.O4, ep.O3 * ep.O2 * V);
      MD4(TweightsType, atweights, tweights_, ep.g, ep.O4, ep.I4,
          V * V * ep.kh * ep.kw * ep.I3 * ep.O3 * ep.I2
              * ep.O2);
      MD5(InputType, ainput, input, ep.n, ep.g, ep.I4,
          ep.I3 * ep.I2, ep.ih * ep.iw * V);
      MD5(OutputType, aoutput, output, ep.n, ep.g, ep.O4,
          ep.O3 * ep.O2, ep.ht * ep.ow * V);
      gemm_a060(&md5(aoutput, _n, _g, _O4, 0, 0),
                &md5(ainput, _n, _g, _I4, 0, 0),
                &md4(atweights, _g, _O4, _I4, 0),
                &md2(abias1, _O4, 0), _I4, _O4, _ht, _wt);
    }, ep.n, ep.g, ep.I4, ep.O4, ep.ht, ep.wt);
  }

  if (is_first_run_ && inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_direct_t
void Instance_elx_conv_direct_t::execute(
    void *output, void *input, void *weights, void *bias)
{
  (this->*execute_opt_)((OutputType *)output,
      (InputType *)input, (WeightsType *)weights, (BiasType *)bias);
}

} // namespace euler
