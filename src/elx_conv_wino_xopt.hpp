#pragma once

#include "el_parallel.hpp"
#include "elx_conv_wino.hpp"

namespace euler {

//
// -------------+------------+--------------+-------------
//  execute-opt | gemm dtype | fusion-along | duplication
// -------------+------------+--------------+-------------
//     A000     |   FP32     |      _       |    _
// -------------+------------+--------------+-------------
//     A033     |   FP32     |    i + o     |  I + O
// -------------+------------+--------------+-------------
//     A061     |   FP32     |    t + o     |    I
// -------------+------------+--------------+-------------
//     A071     |   FP32     |  i + t + o   |    I
// -------------+------------+--------------+-------------
//     A073     |   FP32     |  i + t + o   |  I + O
// -------------+------------+--------------+-------------
//


// tweights:     O4 | O3, I3, A, A, O2, I2, V, V
// tinputs:  t2      | A, A, I3, I2, T, V
// toutput:  t2, O4 | A, A, O3, O2, T, V

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__execute_a061(
    OutputType * __restrict output, InputType * __restrict input,
    WeightsType * __restrict weights, BiasType * __restrict bias)
{
  if (is_first_run_) {
    setup_workspace([&](){
      trans_weights(tweights_, weights, ep.O4);
    });
  }
  auto t2_history = -1;

  estl::parallel_for<2>([&, t2_history](int _t2, int _O4) mutable {
    MD2(TinputType, atinput2, tinput_, mthr_,
        A * A * ep.T * ep.IC);
    MD2(ToutputType, atoutput2, toutput_, mthr_,
        A * A * ep.T * ep.O3 * ep.O2 * V);
    MD2(TweightsType, atweights2, tweights_, ep.O4,
        A * A * ep.IC * ep.O3 * ep.O2 * V);
    MD2(BiasType, abias, bias, ep.O4, ep.O3 * ep.O2 * V);

    int Tz = _t2 == (ep.t2 - 1) ? ep.Tr : ep.T;
    int ithr = estl::current_thread_index();

    if (t2_history != _t2) {
      trans_input(&md2(atinput2, ithr, 0), input, Tz, _t2, 0);
      t2_history = _t2;
    }
    gemm.execute(
        &md2(atoutput2, ithr, 0),
        &md2(atinput2, ithr, 0),
        &md2(atweights2, _O4, 0),
        _t2, Tz);
    trans_output(output, &md2(atoutput2, ithr, 0),
        &md2(abias, _O4, 0), Tz, _t2, _O4, 0);
  }, ep.t2, ep.O4);

  if (is_first_run_ && inference_acc_)
    is_first_run_ = false;
}

// tweights:     O4, I4 | O3, I3, A, A, O2, I2, V, V
// tinputs:  t2,      I4 | A, A, I3, I2, T, V
// toutput:  t2, O4      | A, A, O3, O2, T, V
Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__execute_a071(
    OutputType * __restrict output, InputType * __restrict input,
    WeightsType * __restrict weights, BiasType * __restrict bias)
{
  if (is_first_run_) {
    setup_workspace([&](){
      trans_weights(tweights_, weights, ep.O4);
    });
  }
  int last_I4 = -1, last_t2 = -1;

  estl::parallel_for<3, 2>([&, last_I4, last_t2]
                           (int _t2, int _O4, int _I4) mutable {
    MD2(TinputType, atinput2, tinput_, mthr_,
        A * A * ep.T * ep.I3 * ep.I2 * V);
    MD2(ToutputType, atoutput2, toutput_, ep.t2,
        ep.O4 * A * A * ep.T * ep.O3 * ep.O2 * V);
    MD3(TweightsType, atweights3, tweights_, ep.O4, ep.I4,
        A * A * ep.I3 * ep.I2 * V * ep.O3 * ep.O2 * V);
    MD2(BiasType, abias, bias, ep.O4, ep.O3 * ep.O2 * V);

    int Tz = _t2 == (ep.t2 - 1) ? ep.Tr : ep.T;
    int ithr = estl::current_thread_index();

    MD2(ToutputType, atoutput3, &md2(atoutput2, _t2, 0),
        ep.O4, A * A * Tz * ep.O3 * ep.O2 * V);

    if (last_I4 != _I4 || last_t2 != _t2) {
      trans_input(&md2(atinput2, ithr, 0), input, Tz, _t2, _I4);
      last_t2 = _t2;
      last_I4 = _I4;
    }
    gemm.execute(
        &md2(atoutput3, _O4, 0),
        &md2(atinput2, ithr, 0),
        &md3(atweights3, _O4, _I4, 0),
        _t2, Tz, _I4);
    if (_I4 == ep.I4 - 1) {
      trans_output(output, &md2(atoutput3, _O4, 0),
          &md2(abias, _O4, 0), Tz, _t2, _O4, _I4);
    }
  }, ep.t2, ep.O4, ep.I4);

  if (is_first_run_ && inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__execute_a073(
    OutputType * __restrict output, InputType * __restrict input,
    WeightsType * __restrict weights, BiasType * __restrict bias)
{
  if (is_first_run_) {
    setup_workspace([&](){
      trans_weights(tweights_, weights, ep.O4);
    });
  }

  int last_I4 = -1, last_t2 = -1;

  estl::parallel_for<3, 2>([&, last_I4, last_t2]
                           (int _t2, int _O4, int _I4) mutable {
    MD2(TinputType, atinput2, tinput_, mthr_,
        A * A * ep.T * ep.I3 * ep.I2 * V);
    MD2(ToutputType, atoutput2, toutput_, mthr_,
        A * A * ep.T * ep.O3 * ep.O2 * V);
    MD3(TweightsType, atweights3, tweights_, ep.O4, ep.I4,
        A * A * ep.I3 * ep.I2 * V * ep.O3 * ep.O2 * V);
    MD2(BiasType, abias, bias, ep.O4, ep.O3 * ep.O2 * V);

    int Tz = _t2 == (ep.t2 - 1) ? ep.Tr : ep.T;
    int ithr = estl::current_thread_index();

    if (last_I4 != _I4 || last_t2 != _t2) {
      trans_input(&md2(atinput2, ithr, 0), input, Tz, _t2, _I4);
      last_t2 = _t2;
      last_I4 = _I4;
    }
    gemm.execute_na(
        &md2(atoutput2, ithr, 0),
        &md2(atinput2, ithr, 0),
        &md3(atweights3, _O4, _I4, 0),
        _t2, Tz, _I4);
    trans_output(output, &md2(atoutput2, ithr, 0),
        &md2(abias, _O4, 0), Tz, _t2, _O4, _I4);
  }, ep.t2, ep.O4, ep.I4);

  if (is_first_run_ && inference_acc_)
    is_first_run_ = false;
}

// Flat mode
Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__execute_a000(
    OutputType * __restrict output, InputType * __restrict input,
    WeightsType * __restrict weights, BiasType * __restrict bias)
{
  if (is_first_run_) {
    setup_workspace([&](){
      trans_weights(tweights_, weights);
    });
  }

  THREAD_PARALLEL()
  {
    trans_input(tinput_, input, 0);
    THREAD_BARRIER();
    gemm.execute(toutput_, tinput_, tweights_);
    THREAD_BARRIER();
    trans_output(output, toutput_, bias, 0, 0);
  }
  if (is_first_run_ && inference_acc_) is_first_run_ = false;
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__execute_a033(
    OutputType * __restrict output, InputType * __restrict input,
    WeightsType * __restrict weights, BiasType * __restrict bias)
{
  if (is_first_run_) {
    setup_workspace([&](){
      trans_weights(tweights_, weights, ep.O4);
    });
  }

  TinputType *_tinput = ep.use_scratch_pad
      ? (TinputType *)ep.scratch_pad : tinput_;

  THREAD_PARALLEL()
  {
    int last_I4 = -1;
    MD3(TweightsType, atweights, tweights_, ep.O4, ep.I4,
        A * A * ep.I3 * ep.I2 * V * ep.O3 * ep.O2 * V);
    MD2(BiasType, abias, bias, ep.O4, ep.O3 * ep.O2 * V);

    iter_each(_I4, ep.I4) {
    iter_each(_O4, ep.O4) {
      if (_I4 != last_I4) {
        trans_input(_tinput, input, _I4);
        last_I4 = _I4;
      }
      THREAD_BARRIER()
      gemm.execute_na(toutput_, _tinput, &md3(atweights, _O4, _I4, 0), _I4);
      THREAD_BARRIER()
      trans_output(
          output, toutput_, &md2(abias, _O4, 0), _O4, _I4);
    }}
  }

  if (is_first_run_ && inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::execute(
    void * __restrict output, void * __restrict input,
    void * __restrict weights, void * __restrict bias)
{
  if (is_bfmt_)
    return (this->*execute_opt_)((OutputType *)output,
        (InputType *)input, (WeightsType *)weights, (BiasType *)bias);
  else {
    InputType *in = (InputType *)input;
    WeightsType *wei = (WeightsType *)weights;
    OutputType *out = output_as_bfmt_ ? boutput_ : (OutputType *)output;

    if (input_as_bfmt_) {
      estl::parallel_for<3>([&](int _n, int _ic2, int _ih) {
        MD5(InputType, abinput, binput_, ep.n, ep.ic2, ep.ih, ep.iw, V);
        MD4(InputType, ainput, (InputType *)input, ep.n, ep.ic, ep.ih, ep.iw);

        int v = _ic2 == ep.ic2 - 1 ? ep.Ir : V;
        iter_each (_iw, ep.iw) {
          #pragma omp simd
          iter_each (_v, v)
            md5(abinput, _n, _ic2, _ih, _iw, _v)
                = md4(ainput, _n, _ic2 * V + _v, _ih, _iw);
        }
      }, ep.n, ep.ic2, ep.ih);
      in = binput_;
    }

    if (weights_as_bfmt_) {
      estl::parallel_for<3>([&](int _oc2, int _ic2, int _kh) {
        MD6(WeightsType, abweights, bweights_, ep.oc2, ep.ic2,
            ep.kh, ep.kw, V, V);
        MD4(WeightsType, aweights, (WeightsType *)weights, ep.oc, ep.ic,
            ep.kh, ep.kw);
        int iv = _ic2 == ep.ic2 - 1 ? ep.Ir : V;
        int ov = _oc2 == ep.oc2 - 1 ? ep.Or : V;
        iter_each (_kw, ep.kw) {
        iter_each (_iv, iv) {
        #pragma omp simd
          iter_each (_ov, ov) {
            md6(abweights, _oc2, _ic2, _kh, _kw, _iv, _ov)
              = md4(aweights, _oc2 * V + _ov, _ic2 * V + _iv, _kh, _kw);
          }
        }}
      }, ep.oc2, ep.ic2, ep.kh);
      wei = bweights_;
    }

    // TODO: padding bias

    (this->*execute_opt_)((OutputType *)out,
        (InputType *)in, (WeightsType *)wei, (BiasType *)bias);

    if (output_as_bfmt_) {
      estl::parallel_for<3>([&](int _n, int _oc2, int _oh) {
        MD5(OutputType, aboutput, boutput_, ep.n, ep.oc2, ep.oh, ep.ow, V);
        MD4(OutputType, aoutput, (OutputType *)output, ep.n, ep.oc, ep.oh, ep.ow);

        int v = _oc2 == ep.oc2 - 1 ? ep.Or : V;
        if (ep.with_ip_sum) {
          iter_each (_V, v) {
          iter_each (_ow, ep.ow) {
            md4(aoutput, _n, _oc2 * V + _V, _oh, _ow)
              += md5(aboutput, _n, _oc2, _oh, _ow, _V);
          }}
        } else {
          iter_each (_V, v) {
          iter_each (_ow, ep.ow) {
            md4(aoutput, _n, _oc2 * V + _V, _oh, _ow)
              = md5(aboutput, _n, _oc2, _oh, _ow, _V);
          }}
        }
      }, ep.n, ep.oc2, ep.oh);
    }
  } // !is_bfmt_
}

} // namespace euler
