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
//     A079     |   FP32     |  i + t + o   |  I + W
// -------------+------------+--------------+-------------
//     A07b     |   FP32     |  i + t + o   |  I + W + O
// -------------+------------+--------------+-------------
//


// tweights:     oc4 | oc3, ic3, A, A, O2, I2, V, V
// tinputs:  t2      | A, A, ic3, I2, T, V
// toutput:  t2, oc4 | A, A, oc3, O2, T, V

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__execute_a061(
    OutputType * __restrict output, InputType * __restrict input,
    WeightsType * __restrict weights, BiasType * __restrict bias)
{
  if (is_first_run_) {
    setup_workspace([&](){
      trans_weights(tweights_, weights, this->oc4);
    });
  }
  auto t2_history = -1;

  parallel_for<2>(mthr_, [&, t2_history](int _t2, int _oc4) mutable {
    MD2(TinputType, atinput2, tinput_, mthr_,
        A * A * this->T * this->IC);
    MD2(ToutputType, atoutput2, toutput_, mthr_,
        A * A * this->T * this->oc3 * this->O2 * V);
    MD2(TweightsType, atweights2, tweights_, this->oc4,
        A * A * this->IC * this->oc3 * this->O2 * V);
    MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);

    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    int ithr = el_get_thread_num();

    if (t2_history != _t2) {
      trans_input(&md2(atinput2, ithr, 0), input, Tz, _t2, 0);
      t2_history = _t2;
    }
    gemm.execute(
        &md2(atoutput2, ithr, 0),
        &md2(atinput2, ithr, 0),
        &md2(atweights2, _oc4, 0),
        _t2, Tz);
    trans_output(output, &md2(atoutput2, ithr, 0),
        &md2(abias, _oc4, 0), Tz, _t2, _oc4, 0);
  }, this->t2, this->oc4);

  if (inference_acc_)
    is_first_run_ = false;
}

// tweights:     oc4, ic4 | oc3, ic3, A, A, O2, I2, V, V
// tinputs:  t2,      ic4 | A, A, ic3, I2, T, V
// toutput:  t2, oc4      | A, A, oc3, O2, T, V
Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__execute_a071(
    OutputType * __restrict output, InputType * __restrict input,
    WeightsType * __restrict weights, BiasType * __restrict bias)
{
  if (is_first_run_) {
    setup_workspace([&](){
      trans_weights(tweights_, weights, this->oc4);
    });
  }
  int last_ic4 = -1, last_t2 = -1;

  parallel_for<3, 2>(mthr_, [&, last_ic4, last_t2]
                     (int _t2, int _oc4, int _ic4) mutable {
    MD2(TinputType, atinput2, tinput_, mthr_,
        A * A * this->T * this->ic3 * this->I2 * V);
    MD2(ToutputType, atoutput2, toutput_, this->t2,
        this->oc4 * A * A * this->T * this->oc3 * this->O2 * V);
    MD3(TweightsType, atweights3, tweights_, this->oc4, this->ic4,
        A * A * this->ic3 * this->I2 * V * this->oc3 * this->O2 * V);
    MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);

    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    int ithr = el_get_thread_num();

    MD2(ToutputType, atoutput3, &md2(atoutput2, _t2, 0),
        this->oc4, A * A * Tz * this->oc3 * this->O2 * V);

    if (last_ic4 != _ic4 || last_t2 != _t2) {
      trans_input(&md2(atinput2, ithr, 0), input, Tz, _t2, _ic4);
      last_t2 = _t2;
      last_ic4 = _ic4;
    }
    gemm.execute(
        &md2(atoutput3, _oc4, 0),
        &md2(atinput2, ithr, 0),
        &md3(atweights3, _oc4, _ic4, 0),
        _t2, Tz, _ic4);
    if (_ic4 == this->ic4 - 1) {
      trans_output(output, &md2(atoutput3, _oc4, 0),
          &md2(abias, _oc4, 0), Tz, _t2, _oc4, _ic4);
    }
  }, this->t2, this->oc4, this->ic4);

  if (inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__execute_a073(
    OutputType * __restrict output, InputType * __restrict input,
    WeightsType * __restrict weights, BiasType * __restrict bias)
{
  if (is_first_run_) {
    setup_workspace([&](){
      trans_weights(tweights_, weights, this->oc4);
    });
  }

  int last_ic4 = -1, last_t2 = -1;

  parallel_for<3, 2>(mthr_, [&, last_ic4, last_t2]
                     (int _t2, int _oc4, int _ic4) mutable {
    MD2(TinputType, atinput2, tinput_, mthr_,
        A * A * this->T * this->ic3 * this->I2 * V);
    MD2(ToutputType, atoutput2, toutput_, mthr_,
        A * A * this->T * this->oc3 * this->O2 * V);
    MD3(TweightsType, atweights3, tweights_, this->oc4, this->ic4,
        A * A * this->ic3 * this->I2 * V * this->oc3 * this->O2 * V);
    MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);

    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    int ithr = el_get_thread_num();

    if (last_ic4 != _ic4 || last_t2 != _t2) {
      trans_input(&md2(atinput2, ithr, 0), input, Tz, _t2, _ic4);
      last_t2 = _t2;
      last_ic4 = _ic4;
    }
    gemm.execute_na(
        &md2(atoutput2, ithr, 0),
        &md2(atinput2, ithr, 0),
        &md3(atweights3, _oc4, _ic4, 0),
        _t2, Tz, _ic4);
    trans_output(output, &md2(atoutput2, ithr, 0),
        &md2(abias, _oc4, 0), Tz, _t2, _oc4, _ic4);
  }, this->t2, this->oc4, this->ic4);

  if (inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__execute_a07b(
    OutputType * __restrict output, InputType * __restrict input,
    WeightsType * __restrict weights, BiasType * __restrict bias)
{
  int last_ic4 = -1, last_t2 = -1, last_oc4 = -1;

  parallel_for<3, 1>(mthr_, [&, last_ic4, last_t2, last_oc4]
                     (int _oc4, int _ic4, int _t2) mutable {
    MD2(TinputType, atinput2, tinput_, mthr_,
        A * A * this->T * this->ic3 * this->I2 * V);
    MD2(ToutputType, atoutput2, toutput_, mthr_,
        A * A * this->T * this->oc3 * this->O2 * V);
    MD2(TweightsType, atweights2, tweights_, mthr_,
        A * A * this->ic3 * this->I2 * V * this->oc3 * this->O2 * V);
    MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);

    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    int ithr = el_get_thread_num();

    if (last_ic4 != _ic4 || last_oc4 != _oc4) {
      trans_weights(&md2(atweights2, ithr, 0), weights, _ic4, _oc4);
    }
    if (last_ic4 != _ic4 || last_t2 != _t2) {
      trans_input(&md2(atinput2, ithr, 0), input, Tz, _t2, _ic4);
    }
    gemm.execute_na(
        &md2(atoutput2, ithr, 0),
        &md2(atinput2, ithr, 0),
        &md2(atweights2, ithr, 0),
        _t2, Tz, _ic4);
    trans_output(output, &md2(atoutput2, ithr, 0),
                 &md2(abias, _oc4, 0), Tz, _t2, _oc4, _ic4);
    last_oc4 = _oc4; last_ic4 = _ic4; last_t2 = _t2;
  }, this->oc4, this->ic4, this->t2);
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__execute_a079(
    OutputType * __restrict output, InputType * __restrict input,
    WeightsType * __restrict weights, BiasType * __restrict bias)
{
  int last_ic4 = -1, last_t2 = -1, last_oc4 = -1;

  parallel_for<3, 1>(mthr_, [&, last_ic4, last_t2, last_oc4]
                     (int _oc4, int _ic4, int _t2) mutable {
    MD2(TinputType, atinput2, tinput_, mthr_,
        A * A * this->T * this->ic3 * this->I2 * V);
    MD2(ToutputType, atoutput2, toutput_, this->t2,
        this->oc4 * A * A * this->T * this->oc3 * this->O2 * V);
    MD2(TweightsType, atweights2, tweights_, mthr_,
        A * A * this->ic3 * this->I2 * V * this->oc3 * this->O2 * V);
    MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);

    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    int ithr = el_get_thread_num();

    MD2(ToutputType, atoutput3, &md2(atoutput2, _t2, 0),
        this->oc4, A * A * Tz * this->oc3 * this->O2 * V);

    if (last_ic4 != _ic4 || last_oc4 != _oc4) {
      trans_weights(&md2(atweights2, ithr, 0), weights, _ic4, _oc4);
    }
    if (last_ic4 != _ic4 || last_t2 != _t2) {
      trans_input(&md2(atinput2, ithr, 0), input, Tz, _t2, _ic4);
    }
    gemm.execute(
         &md2(atoutput3, _oc4, 0),
         &md2(atinput2, ithr, 0),
         &md2(atweights2, ithr, 0),
         _t2, Tz, _ic4);
    if (_ic4 == this->ic4 - 1) {
      trans_output(output, &md2(atoutput3, _oc4, 0),
                   &md2(abias, _oc4, 0), Tz, _t2, _oc4, _ic4);
    }
    last_oc4 = _oc4; last_ic4 = _ic4; last_t2 = _t2;
  }, this->oc4, this->ic4, this->t2);
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
  if (inference_acc_) is_first_run_ = false;
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__execute_a033(
    OutputType * __restrict output, InputType * __restrict input,
    WeightsType * __restrict weights, BiasType * __restrict bias)
{
  MD3(TweightsType, atweights, tweights_, this->oc4, this->ic4,
      A * A * this->ic3 * this->I2 * V * this->oc3 * this->O2 * V);

  MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);

  if (is_first_run_) {
    setup_workspace([&](){
      trans_weights(tweights_, weights, this->oc4);
    });
  }

  TinputType *_tinput = this->use_scratch_pad
      ? (TinputType *)this->scratch_pad : tinput_;

  THREAD_PARALLEL()
  {
    int last_ic4 = -1;
    iter_each(_ic4, this->ic4) {
    iter_each(_oc4, this->oc4) {
      if (_ic4 != last_ic4) {
        trans_input(_tinput, input, _ic4);
        last_ic4 = _ic4;
      }
      THREAD_BARRIER()
      gemm.execute_na(toutput_, _tinput, &md3(atweights, _oc4, _ic4, 0), _ic4);
      THREAD_BARRIER()
      trans_output(
          output, toutput_, &md2(abias, _oc4, 0), _oc4, _ic4);
    }}
  }

  if (inference_acc_)
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
      parallel_for<3>(mthr_, [&](int _n, int _ic2, int _ih) {
        MD5(InputType, abinput, binput_, this->n, this->ic2, this->ih, this->iw, V);
        MD4(InputType, ainput, (InputType *)input, this->n, this->ic, this->ih, this->iw);

        int v = _ic2 == this->ic2 - 1 ? this->Ir : V;
        iter_each (_iw, this->iw) {
          #pragma omp simd
          iter_each (_v, v)
            md5(abinput, _n, _ic2, _ih, _iw, _v)
                = md4(ainput, _n, _ic2 * V + _v, _ih, _iw);
        }
      }, this->n, this->ic2, this->ih);
      in = binput_;
    }

    if (weights_as_bfmt_) {
      parallel_for<3>(mthr_, [&](int _oc2, int _ic2, int _kh) {
        MD6(WeightsType, abweights, bweights_, this->oc2, this->ic2,
            this->kh, this->kw, V, V);
        MD4(WeightsType, aweights, (WeightsType *)weights, this->oc, this->ic,
            this->kh, this->kw);
        int iv = _ic2 == this->ic2 - 1 ? this->Ir : V;
        int ov = _oc2 == this->oc2 - 1 ? this->Or : V;
        iter_each (_kw, this->kw) {
        iter_each (_iv, iv) {
        #pragma omp simd
          iter_each (_ov, ov) {
            md6(abweights, _oc2, _ic2, _kh, _kw, _iv, _ov)
              = md4(aweights, _oc2 * V + _ov, _ic2 * V + _iv, _kh, _kw);
          }
        }}
      }, this->oc2, this->ic2, this->kh);
      wei = bweights_;
    }

    // TODO: padding bias

    (this->*execute_opt_)((OutputType *)out,
        (InputType *)in, (WeightsType *)wei, (BiasType *)bias);

    if (output_as_bfmt_) {
      parallel_for<3>(mthr_, [&](int _n, int _oc2, int _oh) {
        MD5(OutputType, aboutput, boutput_, this->n, this->oc2, this->oh, this->ow, V);
        MD4(OutputType, aoutput, (OutputType *)output, this->n, this->oc, this->oh, this->ow);

        int v = _oc2 == this->oc2 - 1 ? this->Or : V;
        if (this->with_ip_sum) {
          iter_each (_V, v) {
          iter_each (_ow, this->ow) {
            md4(aoutput, _n, _oc2 * V + _V, _oh, _ow)
              += md5(aboutput, _n, _oc2, _oh, _ow, _V);
          }}
        } else {
          iter_each (_V, v) {
          iter_each (_ow, this->ow) {
            md4(aoutput, _n, _oc2 * V + _V, _oh, _ow)
              = md5(aboutput, _n, _oc2, _oh, _ow, _V);
          }}
        }
      }, this->n, this->oc2, this->oh);
    }
  } // !is_bfmt_
}

} // namespace euler
