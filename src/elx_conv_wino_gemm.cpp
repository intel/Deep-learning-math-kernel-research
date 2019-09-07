#include "el_parallel.hpp"
#include "elx_conv_wino_gemm.hpp"

namespace euler {

template <typename GarrayTypes, const int A, const int V, const int I>
void elx_conv_wino_gemm_t<GarrayTypes, A, V, I>::setup(elx_conv_params_t *conv_xc)
{
  attr_ = 0x0;
  xc    = conv_xc;
  mthr_ = xc->nthreads;

  bind_kernel_functions();
}

template <typename GarrayTypes, const int A, const int V, const int I>
void elx_conv_wino_gemm_t<GarrayTypes, A, V, I>::bind_kernel_functions()
{
  gemm_kernel_binder::bind<1, GKF_CCC>(xc->O, xc->T, &ker_gemm_);
  gemm_kernel_binder::bind<1, GKF_CCC>(xc->O, xc->Tr, &ker_gemm0_);
}

// tweights:     oc4 | oc3, ic3, A, A, O2, I2, V, V
// tinputs:       t2 | A, A, ic3, I2, T, V
// toutput:  t2, oc4 | A, A, oc3, O2, T, V
template <typename GarrayTypes, const int A, const int V, const int I>
void elx_conv_wino_gemm_t<GarrayTypes, A, V, I>::execute(
    ToutputType *toutput, TinputType *tinput, TweightsType *tweights,
    int _t2, int Tz, int _ic4)
{
  auto ker_gemm = (_t2 == xc->t2 - 1) ? ker_gemm0_ : ker_gemm_;

  MD6(TinputType, atinput, tinput, A, A, xc->ic3, xc->I2, Tz, V);
  MD6(ToutputType, atoutput, toutput, A, A, xc->oc3, xc->O2, Tz, V);
  MD5(TweightsType, atweights, tweights, xc->oc3, xc->ic3, A, A, xc->O2 * xc->I2 * V * V);

  bool scramble = (xc->T == xc->Tr) || (xc->t2 >= 2 * mthr_);
  if (scramble) {
    int it_start = el_get_thread_num();
    iter_each(i, A * A) {
      int n = (it_start + i) % (A * A);
      int _wA = n % A;
      int _hA = n / A;
      iter_each(_oc3, xc->oc3) {
        bool last_ic4 = _ic4 == xc->ic4 - 1;
        int ic3 = last_ic4 ? xc->ic3 - 1 : xc->ic3;
        iter_each(_ic3, ic3) {
          int attr = _ic3 == 0 && _ic4 == 0
              ? set_attr(attr_, r_output_idx) : attr_;
          ker_gemm(*xc,
              &md6(atoutput, _hA, _wA, _oc3, 0, 0, 0),
              &md6(atinput, _hA, _wA, _ic3, 0, 0, 0),
              &md5(atweights, _oc3, _ic3, _hA, _wA, 0),
              nullptr, attr);
        }
        if (last_ic4) {
          auto attr = xc->ic3 == 1 && xc->ic4 == 1
                          ? set_attr(attr_, r_output_idx)
                          : attr_;
         if (xc->Ir != V)
           attr = set_attr(attr, has_Ir_idx);
         ker_gemm(*xc,
             &md6(atoutput, _hA, _wA, _oc3, 0, 0, 0),
             &md6(atinput, _hA, _wA, xc->ic3 - 1, 0, 0, 0),
             &md5(atweights, _oc3, xc->ic3 - 1, _hA, _wA, 0),
             nullptr, attr);
        }
      }
    }
  } else {
    iter_each(_hA, A) {
    iter_each(_wA, A) {
    iter_each(_oc3, xc->oc3) {
      bool last_ic4 = _ic4 == xc->ic4 - 1;
      int ic3 = last_ic4 ? xc->ic3 - 1 : xc->ic3;
      iter_each(_ic3, ic3) {
        int attr =
            _ic3 == 0 && _ic4 == 0 ? set_attr(attr_, r_output_idx) : attr_;
        ker_gemm(*xc,
            &md6(atoutput, _hA, _wA, _oc3, 0, 0, 0),
            &md6(atinput, _hA, _wA, _ic3, 0, 0, 0),
            &md5(atweights, _oc3, _ic3, _hA, _wA, 0),
            nullptr, attr);
      }
      if (last_ic4) {
        auto attr = xc->ic3 == 1 && xc->ic4 == 1
                        ? set_attr(attr_, r_output_idx)
                        : attr_;
        if (xc->Ir != V)
          attr = set_attr(attr, has_Ir_idx);
        ker_gemm(*xc,
            &md6(atoutput, _hA, _wA, _oc3, 0, 0, 0),
            &md6(atinput, _hA, _wA, xc->ic3 - 1, 0, 0, 0),
            &md5(atweights, _oc3, xc->ic3 - 1, _hA, _wA, 0),
            nullptr, attr);
      }
    }}}
  }
}

template <typename GarrayTypes, const int A, const int V, const int I>
void elx_conv_wino_gemm_t<GarrayTypes, A, V, I>::execute_na(
    ToutputType *toutput, TinputType *tinput, TweightsType *tweights,
    int _t2, int Tz, int _ic4)
{
  auto ker_gemm = (_t2 == xc->t2 - 1) ? ker_gemm0_ : ker_gemm_;

  MD6(TinputType, atinput, tinput, A, A, xc->ic3, xc->I2, Tz, V);
  MD6(ToutputType, atoutput, toutput, A, A, xc->oc3, xc->O2, Tz, V);
  MD5(TweightsType, atweights, tweights, xc->oc3, xc->ic3, A, A,
      xc->O2 * xc->I2 * V * V);

  bool scramble = (xc->T == xc->Tr) || (xc->t2 >= 2 * mthr_);
  if (scramble) {
    int it_start = el_get_thread_num();
    iter_each(i, A * A) {
      int n = (it_start + i) % (A * A);
      int _wA = n % A;
      int _hA = n / A;
      iter_each(_oc3, xc->oc3) {
        bool last_ic4 = _ic4 == xc->ic4 - 1;
        int ic3 = last_ic4 ? xc->ic3 - 1 : xc->ic3;
        iter_each(_ic3, ic3) {
          int attr = _ic3 == 0 ? set_attr(attr_, r_output_idx) : attr_;
          ker_gemm(*xc,
              &md6(atoutput, _hA, _wA, _oc3, 0, 0, 0),
              &md6(atinput, _hA, _wA, _ic3, 0, 0, 0),
              &md5(atweights, _oc3, _ic3, _hA, _wA, 0),
              nullptr, attr);
        }
        if (last_ic4) {
          auto attr = xc->ic3 == 1 ? set_attr(attr_, r_output_idx) : attr_;
          if (xc->Ir != V)
            attr = set_attr(attr, has_Ir_idx);
          ker_gemm(*xc,
              &md6(atoutput, _hA, _wA, _oc3, 0, 0, 0),
              &md6(atinput, _hA, _wA, xc->ic3 - 1, 0, 0, 0),
              &md5(atweights, _oc3, xc->ic3 - 1, _hA, _wA, 0),
              nullptr, attr);
        }
      }
    }
  } else {
    iter_each(_hA, A) {
    iter_each(_wA, A) {
    iter_each(_oc3, xc->oc3) {
      bool last_ic4 = _ic4 == xc->ic4 - 1;
      int ic3 = last_ic4 ? xc->ic3 - 1 : xc->ic3;
      iter_each(_ic3, ic3) {
        int attr = _ic3 == 0 ? set_attr(attr_, r_output_idx) : attr_;
        ker_gemm(*xc,
            &md6(atoutput, _hA, _wA, _oc3, 0, 0, 0),
            &md6(atinput, _hA, _wA, _ic3, 0, 0, 0),
            &md5(atweights, _oc3, _ic3, _hA, _wA, 0),
            nullptr, attr);
      }
      if (last_ic4) {
        auto attr = xc->ic3 == 1 ? set_attr(attr_, r_output_idx) : attr_;
        if (xc->Ir != V)
          attr = set_attr(attr, has_Ir_idx);
        ker_gemm(*xc,
            &md6(atoutput, _hA, _wA, _oc3, 0, 0, 0),
            &md6(atinput, _hA, _wA, xc->ic3 - 1, 0, 0, 0),
            &md5(atweights, _oc3, xc->ic3 - 1, _hA, _wA, 0),
            nullptr, attr);
      }
    }}}
  }
}

// tweights: oc3, ic3, A, A, O2, I2, V, V
// tinputs:  t2, A, A, ic3, I2, T, V
// toutput:  t2, A, A, oc3, O2, T, V
template <typename GarrayTypes, const int A, const int V, const int I>
void elx_conv_wino_gemm_t<GarrayTypes, A, V, I>::execute(
    ToutputType *toutput, TinputType *tinput, TweightsType *tweights, int _ic4)
{
  int ithr = el_get_thread_num();
  THREAD_FOR2(5, 4, mthr_, ithr, [&](int _hA, int _wA,
                                     int _oc3, int _t2, int _ic3) {
    MD2(TinputType, atinput2, tinput, xc->t2, A * A * xc->T * xc->ic3 * xc->I2 * V);
    MD2(ToutputType, atoutput2, toutput, xc->t2, A * A * xc->T * xc->oc3 * xc->O2 * V);
    MD5(TweightsType, atweights, tweights, xc->oc3, xc->ic3, A, A, xc->O2 * xc->I2 * V * V);

    int Tz = _t2 == (xc->t2 - 1) ? xc->Tr : xc->T;
    auto ker_gemm = (_t2 == xc->t2 - 1) ? ker_gemm0_ : ker_gemm_;
    MD6(TinputType, atinput6, &md2(atinput2, _t2, 0), A, A, xc->ic3, xc->I2, Tz, V);
    MD6(ToutputType, atoutput6, &md2(atoutput2, _t2, 0), A, A, xc->oc3, xc->O2, Tz, V);

    int attr = attr_;
    if (_ic3 == 0 && _ic4 == 0)
      attr = set_attr(attr, r_output_idx);
    if (xc->Ir != V && _ic4 == xc->ic4 - 1 && _ic3 == xc->ic3 - 1)
      attr = set_attr(attr, has_Ir_idx);

    ker_gemm(*xc, &md6(atoutput6, _hA, _wA, _oc3, 0, 0, 0),
             &md6(atinput6, _hA, _wA, _ic3, 0, 0, 0),
             &md5(atweights, _oc3, _ic3, _hA, _wA, 0), nullptr, attr);
  }, A, A, xc->oc3, xc->t2, xc->ic3);
}

template <typename GarrayTypes, const int A, const int V, const int I>
void elx_conv_wino_gemm_t<GarrayTypes, A, V, I>::execute_na(
    ToutputType *toutput, TinputType *tinput, TweightsType *tweights, int _ic4)
{
  int ithr = el_get_thread_num();
  THREAD_FOR2(5, 4, mthr_, ithr, [&](int _hA, int _wA,
                                     int _oc3, int _t2, int _ic3) {
    MD2(TinputType, atinput2, tinput, xc->t2, A * A * xc->T * xc->ic3 * xc->I2 * V);
    MD2(ToutputType, atoutput2, toutput, xc->t2, A * A * xc->T * xc->oc3 * xc->O2 * V);
    MD5(TweightsType, atweights, tweights, xc->oc3, xc->ic3, A, A, xc->O2 * xc->I2 * V * V);

    int Tz = _t2 == (xc->t2 - 1) ? xc->Tr : xc->T;
    auto ker_gemm = (_t2 == xc->t2 - 1) ? ker_gemm0_ : ker_gemm_;
    MD6(TinputType, atinput6, &md2(atinput2, _t2, 0), A, A, xc->ic3, xc->I2, Tz, V);
    MD6(ToutputType, atoutput6, &md2(atoutput2, _t2, 0), A, A, xc->oc3, xc->O2, Tz, V);

    int attr = attr_;
    if (_ic3 == 0)
      attr = set_attr(attr, r_output_idx);
    if (xc->Ir != V && _ic4 == xc->ic4 - 1 && _ic3 == xc->ic3 - 1)
      attr = set_attr(attr, has_Ir_idx);

    ker_gemm(*xc, &md6(atoutput6, _hA, _wA, _oc3, 0, 0, 0),
             &md6(atinput6, _hA, _wA, _ic3, 0, 0, 0),
             &md5(atweights, _oc3, _ic3, _hA, _wA, 0), nullptr, attr);
  }, A, A, xc->oc3, xc->t2, xc->ic3);
}

} // namespace euler
