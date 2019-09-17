#include "elx_conv_direct.hpp"
#include "el_parallel.hpp"

// XOPT
//
// fusion:  same as winograd
// dup:     same as winograd
// ------+-----+--------+-----+------------------------------------------------
//       | ker | fusion | dup |             notes
// ------+-----+--------+-----+------------------------------------------------
//  a060 |conv |   t+o  |  -  | nhwc|blocked|nchw-input, Ir/Tr/Or, K=3,5,7 S=1,2, group
// ------+-----+--------+-----+------------------------------------------------
//  b060 |conv |   t+o  |  -  | nhwc|blocked, Ir/Tr/Or, K=3,5,7 S=1,2 small spatial, group=1
// ------+-----+--------+-----+------------------------------------------------
//  d060 |gemm |   t+o  |  -  | nhwc|blocked, Ir/Tr/Or, group
// ------+-----+--------+-----+------------------------------------------------
//
namespace euler {

Template_elx_conv_direct_t
void Instance_elx_conv_direct_t::__execute_a060(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  // input (blocked): t3*, ic4*, ic3, I2, ht*, S, wt*, T, S, V(Ir)
  // input (nchw): t3*, ic4*, ic3, I2, V(Ir), ht*, S, wt*, T, S
  // input (nhwc): t3*, ht*, S, wt*, T, S, ic4*, ic3, I2, V(Ir)
  // weights: oc4*, oc3, O2, ic4*, ic3, I2, V(Ir), V
  // output (blocked):  t3*, oc4*, oc3, O2(O2r), ht*wt*, T, V
  // output (nhwc):  t3*, ht*wt*, T, oc4*, oc3, O2(O2r), V
  if (is_first_run_) {
    setup_workspace([&]() {
      trans_weights_to_compact(tweights_, weights);
    });
  }

  if (this->input_fmt == nchw) { // nchw => blocked
    parallel_for<5, 1>(mthr_, [&](int _t3, int _ic4, int _oc4, int _ht, int _wt) {
      int Vr = this->ic < V ? this->Ir : V;
      MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);
      MD3(TweightsType, atweights, tweights_, this->oc4, this->ic4,
          V * Vr * this->kh * this->kw * this->ic3 * this->oc3 * this->I2
              * this->O2);
      MD2(InputType, ainput0, input, this->t3, this->ic * this->ih * this->iw);
      MD3(InputType, ainput1, &md2(ainput0, _t3, 0), this->ic4,
          this->ic3 * this->I2 * V, this->ih * this->iw);
      MD5(OutputType, aoutput0, output, this->t3, this->oc4,
          this->oc3 * this->O2, this->ht, this->ow * V);
      MD3(OutputType, aoutput1, &md5(aoutput0, _t3, _oc4, 0, _ht, 0), this->wt,
          this->T, V);
      conv_a060(&md3(aoutput1, _wt, 0, 0), &md3(ainput1, _ic4, 0, 0),
          &md3(atweights, _oc4, _ic4, 0), &md2(abias, _oc4, 0), _ic4, _oc4, _ht,
          _wt);
    }, this->t3, this->ic4, this->oc4, this->ht, this->wt);
  } else if (this->input_fmt == nhwc) { // nhwc => nhwc
    parallel_for<6, 2>(mthr_, [&](int _t3, int _g, int _ic4, int _oc4, int _ht, int _wt) {
      MD2(BiasType, abias0, bias, this->g, this->oc);
      MD2(BiasType, abias1, &md2(abias0, _g, 0), this->oc4, this->oc3 * this->O2 * V);
      MD4(TweightsType, atweights, tweights_, this->g, this->oc4, this->ic4,
          V * V * this->kh * this->kw * this->ic3 * this->oc3 * this->I2
              * this->O2);
      MD5(InputType, ainput0, input, this->t3, this->ih, this->iw, this->g,
          this->ic);
      MD2(InputType, ainput1, &md5(ainput0, _t3, 0, 0, _g, 0), this->ic4,
          this->ic3 * this->I2 * V);
      MD4(OutputType, aoutput0, output, this->t3, this->ht, this->ow, this->g * this->oc);
      MD4(OutputType, aoutput1, &md4(aoutput0, _t3, _ht, 0, 0), this->wt,
          this->T, this->g, this->oc);
      MD2(OutputType, aoutput2, &md4(aoutput1, _wt, 0, _g, 0), this->oc4,
          this->oc3 * this->O2 * V);
      conv_a060(&md2(aoutput2, _oc4, 0), &md2(ainput1, _ic4, 0),
          &md4(atweights, _g, _oc4, _ic4, 0), &md2(abias1, _oc4, 0),
          _ic4, _oc4, _ht, _wt);
    },  this->t3, this->g, this->ic4, this->oc4, this->ht, this->wt);
  } else { // blocked => blocked
    parallel_for<6, 2>(mthr_, [&](int _t3, int _g, int _ic4, int _oc4, int _ht, int _wt) {
      MD2(BiasType, abias0, bias, this->g, this->oc);
      MD2(BiasType, abias1, &md2(abias0, _g, 0), this->oc4, this->oc3 * this->O2 * V);
      MD4(TweightsType, atweights, tweights_, this->g, this->oc4, this->ic4,
          V * V * this->kh * this->kw * this->ic3 * this->oc3 * this->I2
              * this->O2);
      MD5(InputType, ainput, input, this->t3, this->g, this->ic4,
          this->ic3 * this->I2, this->ih * this->iw * V);
      MD6(OutputType, aoutput0, output, this->t3, this->g, this->oc4,
          this->oc3 * this->O2, this->ht, this->ow * V);
      MD3(OutputType, aoutput1, &md6(aoutput0, _t3, _g, _oc4, 0, _ht, 0),
          this->wt, this->T, V);
      conv_a060(&md3(aoutput1, _wt, 0, 0), &md5(ainput, _t3, _g, _ic4, 0, 0),
          &md4(atweights, _g, _oc4, _ic4, 0), &md2(abias1, _oc4, 0),
          _ic4, _oc4, _ht, _wt);
    }, this->t3, this->g, this->ic4, this->oc4, this->ht, this->wt);
  }

  if (inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_direct_t
void Instance_elx_conv_direct_t::__execute_b060(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  // input (blocked): t3*, ic4*, ic3, I2, ht*, S, wt*, T, S, V(Ir)
  // input (nhwc): t3*, ht*, S, wt*, T, S, ic4*, ic3, I2, V(Ir)
  // weights: oc4*, oc3, O2, ic4*, ic3, I2, V(Ir), V
  // output (blocked):  t3*, oc4*, oc3, O2(O2r), ht*wt*, T, V
  // output (nhwc):  t3*, ht*wt*, T, oc4*, oc3, O2(O2r), V
  if (is_first_run_) {
    setup_workspace([&]() {
      trans_weights_to_compact(tweights_, weights);
    });
  }

  THREAD_PARALLEL()
  {
    int ithr = el_get_thread_num();
    if (this->input_fmt == nhwc) { // nhwc => nhwc
      THREAD_FOR2(6, 2, mthr_, ithr, [&](int _ic4, int _t3, int _ic3,
                                         int _oc4, int _ht, int _wt) {
        MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);
        MD5(TweightsType, atweights, tweights_, this->oc4, this->ic4, this->oc3,
            this->ic3, V * V * this->kh * this->kw * this->I2 * this->O2);
        MD4(InputType, ainput0, input, this->t3, this->ih, this->iw, this->ic);
        MD3(InputType, ainput1, &md4(ainput0, _t3, 0, 0, 0), this->ic4,
            this->ic3, this->I2 * V);
        MD5(OutputType, atoutput0, toutput_, this->ic4, this->t3, this->ht,
            this->ow, this->oc);
        MD3(OutputType, atoutput1, &md5(atoutput0, _ic4, _t3, _ht, 0, 0),
            this->wt, this->T, this->oc);
        MD2(OutputType, atoutput2, &md3(atoutput1, _wt, 0, 0), this->oc4,
            this->oc3 * this->O2 * V);
        conv_b060(&md2(atoutput2, _oc4, 0), &md3(ainput1, _ic4, _ic3, 0),
            &md5(atweights, _oc4, _ic4, 0, _ic3, 0), &md2(abias, _oc4, 0),
            _ic4, _ic3, _oc4, _ht, _wt);
      }, this->ic4, this->t3, this->ic3, this->oc4, this->ht, this->wt);

      THREAD_BARRIER()
      THREAD_FOR(4, mthr_, ithr, [&](int _n, int _oh, int _ow, int _oc2) {
        MD5(ToutputType, atoutput0, toutput_, this->ic4, this->n, this->oh,
            this->ow, this->oc);
        MD4(OutputType, aoutput0, output, this->n, this->oh, this->ow, this->oc);
        MD2(OutputType, aoutput1, &md4(aoutput0, _n, _oh, _ow, 0), this->oc2, V);
        if (std::is_same<OutputType, float>::value) {
          __m<V> zero = _mm<V>::setzero_ps();
          __m<V> out = zero;
          for (int _ic4 = 0; _ic4 < this->ic4; ++_ic4) {
            MD2(ToutputType, atoutput1, &md5(atoutput0, _ic4, _n, _oh, _ow, 0),
                this->oc2, V);
            out += *(__m<V> *)&md2(atoutput1, _oc2, 0);
          }
          if (this->with_relu) {
            auto lower = *(__m<V> *)(this->relu_bound_lower_vec);
            auto upper = *(__m<V> *)(this->relu_bound_upper_vec);
            out = _mm<V>::max_ps(out, lower);
            out = _mm<V>::min_ps(out, upper);
          }
          if (this->Or != V && _oc2 == this->oc2 - 1) {
            iter_each (_V, this->Or) {
              md2(aoutput1, _oc2, _V) = out[_V];
            }
          } else {
            *(__m<V> *)&md2(aoutput1, _oc2, 0) = out;
          }
        } else {
          el_error("Unsupported data type");
        }
      }, this->t3, this->oh, this->ow, this->oc2);
    } else { // blocked => blocked
      THREAD_FOR2(6, 2, mthr_, ithr, [&](int _ic4, int _t3, int _ic3,
                                         int _oc4, int _ht, int _wt) {
        MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);
        MD5(TweightsType, atweights, tweights_, this->oc4, this->ic4, this->oc3,
            this->ic3, V * V * this->kh * this->kw * this->I2 * this->O2);
        MD5(InputType, ainput, input, this->t3, this->ic4, this->ic3, this->I2,
            this->ih * this->iw * V);
        MD6(OutputType, atoutput0, toutput_, this->ic4, this->t3, this->oc4,
            this->oc3 * this->O2, this->ht, this->ow * V);
        MD3(OutputType, atoutput1, &md6(atoutput0, _ic4, _t3, _oc4, 0, _ht, 0),
            this->wt, this->T, V);
        conv_b060(&md3(atoutput1, _wt, 0, 0), &md5(ainput, _t3, _ic4, _ic3, 0, 0),
            &md5(atweights, _oc4, _ic4, 0, _ic3, 0), &md2(abias, _oc4, 0), _ic4,
            _ic3, _oc4, _ht, _wt);
      }, this->ic4, this->t3, this->ic3, this->oc4, this->ht, this->wt);
      THREAD_BARRIER()
      THREAD_FOR(1, mthr_, ithr, [&](int o) {
        MD3(ToutputType, atoutput, toutput_, this->ic4,
            this->n * this->oc2 * this->oh * this->ow, V);
        MD2(OutputType, aoutput, output,
            this->n * this->oc2 * this->oh * this->ow, V);
        if (std::is_same<OutputType, float>::value) {
          __m<V> zero = _mm<V>::setzero_ps();
          __m<V> out = zero;
          for (int _ic4 = 0; _ic4 < this->ic4; ++_ic4) {
            out += *(__m<V> *)&md3(atoutput, _ic4, o, 0);
          }
          if (this->with_relu) {
            auto lower = *(__m<V> *)(this->relu_bound_lower_vec);
            auto upper = *(__m<V> *)(this->relu_bound_upper_vec);
            out = _mm<V>::max_ps(out, lower);
            out = _mm<V>::min_ps(out, upper);
          }
          *(__m<V> *)&md2(aoutput, o, 0) = out;
        } else {
          el_error("Unsupported data type");
        }
      }, this->t3 * this->oc2 * this->oh * this->ow);
    }
  }

  if (inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_direct_t
void Instance_elx_conv_direct_t::__execute_d060(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  // input (blocked): t3*, ic4*, ic3, I2, ih, iw, V(Ir)
  // weights: oc4*, oc3, O2, ic4*, ic3, I2, V(Ir), V
  // output:  t3*, oc4*, oc3, O2(O2r), ht*wt*, T(Tr), V
  if (is_first_run_) {
    setup_workspace([&]() {
      trans_weights_to_compact(tweights_, weights);
    });
  }

  if (this->input_fmt == nhwc) { // nhwc -> nhwc
    parallel_for<6, 2>(mthr_, [&](int _t3, int _g, int _ic4, int _oc4, int _ht, int _wt) {
      MD2(BiasType, abias0, bias, this->g, this->oc);
      MD2(BiasType, abias1, &md2(abias0, _g, 0), this->oc4, this->oc3 * this->O2 * V);
      MD4(TweightsType, atweights, tweights_, this->g, this->oc4, this->ic4,
          V * V * this->kh * this->kw * this->ic3 * this->oc3 * this->I2
              * this->O2);
      MD4(InputType, ainput0, input, this->t3, this->ih * this->iw, this->g, this->ic);
      MD2(InputType, ainput1, &md4(ainput0, _t3, 0, _g, 0), this->ic4,
          this->ic3 * this->I2 * V);
      MD4(OutputType, aoutput0, output, this->t3, this->ht * this->ow, this->g,
          this->oc);
      MD2(OutputType, aoutput1, &md4(aoutput0, _t3, 0, _g, 0), this->oc4,
          this->oc3 * this->O2 * V);
      gemm_d060(&md2(aoutput1, _oc4, 0), &md2(ainput1, _ic4, 0),
          &md4(atweights, _g, _oc4, _ic4, 0), &md2(abias1, _oc4, 0),
          _ic4, _oc4, _ht, _wt);
    }, this->t3, this->g, this->ic4, this->oc4, this->ht, this->wt);
  } else { // blocked -> blocked
    parallel_for<6, 2>(mthr_, [&](int _t3, int _g, int _ic4, int _oc4, int _ht, int _wt) {
      MD2(BiasType, abias0, bias, this->g, this->oc);
      MD2(BiasType, abias1, &md2(abias0, _g, 0), this->oc4, this->oc3 * this->O2 * V);
      MD4(TweightsType, atweights, tweights_, this->g, this->oc4, this->ic4,
          V * V * this->kh * this->kw * this->ic3 * this->oc3 * this->I2
              * this->O2);
      MD5(InputType, ainput, input, this->t3, this->g, this->ic4,
          this->ic3 * this->I2, this->ih * this->iw * V);
      MD5(OutputType, aoutput, output, this->t3, this->g, this->oc4,
          this->oc3 * this->O2, this->ht * this->ow * V);
      gemm_d060(&md5(aoutput, _t3, _g, _oc4, 0, 0),
                &md5(ainput, _t3, _g, _ic4, 0, 0),
                &md4(atweights, _g, _oc4, _ic4, 0),
                &md2(abias1, _oc4, 0), _ic4, _oc4, _ht, _wt);
    }, this->t3, this->g, this->ic4, this->oc4, this->ht, this->wt);
  }

  if (inference_acc_)
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
