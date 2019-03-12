#include "elx_conv_wino_u8s8_gemm.hpp"
#include "el_parallel.hpp"

namespace euler {

template <typename GarrayTypes, const int A, const int V, const int I>
void elx_conv_wino_u8s8_gemm_t<GarrayTypes, A, V, I>::setup(elx_conv_params_t *conv_xc)
{
  attr_ = 0x0;
  xc    = conv_xc;
  mthr_ = xc->nthreads;

  bind_kernel_functions();
}

template <typename GarrayTypes, const int A, const int V, const int I>
void elx_conv_wino_u8s8_gemm_t<GarrayTypes, A, V, I>::bind_kernel_functions()
{
  u8s8_gemm_kernel_binder::bind<GarrayTypes,
      V, 4, I, 1, GKF_CCC>(xc->O, xc->T, &ker_u8s8_gemm_);
  u8s8_gemm_kernel_binder::bind<GarrayTypes,
      V, 4, I, 1, GKF_CCC>(xc->O, xc->Tr, &ker_u8s8_gemm0_);
}

// tweights:      oc4 | oc3, ic3, A, A, O2, I2, V1, V, Vx
// tinputs:        t2 | A, A, ic3, I2, T, V1, Vx
// toutput:   t2, oc4 | A, A, oc3, O2, T, V
// weights_scale  oc4 | oc3, O2, V
// facotr:        oc4 | oc3, A, A, O2, V
template <typename GarrayTypes, const int A, const int V, const int I>
void elx_conv_wino_u8s8_gemm_t<GarrayTypes, A, V, I>::execute(
    ToutputType *toutput, uint8_t *tinput, int8_t *tweights,
    TscaleType *src_scale, TscaleType *weights_scale, TscaleType *weights_factor,
    int _t2, int Tz, int _ic4)
{
  auto ker_gemm = (_t2 == xc->t2 - 1) ? ker_u8s8_gemm0_ : ker_u8s8_gemm_;

  MD6(uint8_t, atinput, tinput, A, A, xc->ic3, xc->I2, Tz, V);
  MD6(ToutputType, atoutput, toutput, A, A, xc->oc3, xc->O2, Tz, V);
  MD5(int8_t, atweights, tweights, xc->oc3, xc->ic3, A, A,
      xc->O2 * xc->I2 * V * V);
  MD6(TscaleType, aweights_scale, weights_scale, xc->oc3, xc->ic3, A, A, xc->O2, V);
  MD6(TscaleType, aweights_factor, weights_factor, xc->oc3, xc->ic3, A, A, xc->O2, V);
  MD5(TscaleType, asrc_scale, src_scale, xc->ic3,  A, A, 2, xc->T);

  iter_each (_hA, A) {
  iter_each (_wA, A) {
  iter_each (_ic3, xc->ic3) {
  iter_each (_oc3, xc->oc3) {
    int attr = _ic3 == 0 && _ic4 == 0 ?  set_attr(attr_, r_output_idx) : attr_;
    attr = set_attr(attr, l_output_idx);
    attr = set_attr(attr, c_output_idx);
    if (xc->Ir != V && _ic4 == xc->ic4 - 1 && _ic3 == xc->ic3 - 1)
      attr = set_attr(attr, has_Ir_idx);

    TscaleType *asrc_s, *asrc_z;
    if (xc->sampling_kind == COARSE || xc->sampling_kind == CALIBRATED) {
      asrc_s = &md5(asrc_scale, 0, 0, 0, 0, 0);
      asrc_z = &md5(asrc_scale, 0, 0, 0, 1, 0);
    } else {
      asrc_s = &md5(asrc_scale, _ic3, _hA, _wA, 0, 0);
      asrc_z = &md5(asrc_scale, _ic3, _hA, _wA, 1, 0);
    }

    ker_gemm(*(elx_conv_params_t *)xc,
        &md6(atoutput, _hA, _wA, _oc3, 0, 0, 0),
        &md6(atinput, _hA, _wA, _ic3, 0, 0, 0),
        &md5(atweights, _oc3, _ic3, _hA, _wA, 0),
        nullptr, attr, asrc_s, asrc_z,
        &md6(aweights_scale, _oc3, _ic3, _hA, _wA, 0, 0),
        &md6(aweights_factor, _oc3, _ic3, _hA, _wA, 0, 0));
  }}}}
}

template <typename GarrayTypes, const int A, const int V, const int I>
void elx_conv_wino_u8s8_gemm_t<GarrayTypes, A, V, I>::execute_na(
    ToutputType *toutput, uint8_t *tinput, int8_t *tweights,
    TscaleType *src_scale, TscaleType *weights_scale, TscaleType *weights_factor,
    int _t2, int Tz, int _ic4)
{
  auto ker_gemm = (_t2 == xc->t2 - 1) ? ker_u8s8_gemm0_ : ker_u8s8_gemm_;

  MD6(uint8_t, atinput, tinput, A, A, xc->ic3, xc->I2, Tz, V);
  MD6(ToutputType, atoutput, toutput, A, A, xc->oc3, xc->O2, Tz, V);
  MD5(int8_t, atweights, tweights, xc->oc3, xc->ic3, A, A,
      xc->O2 * xc->I2 * V * V);
  MD6(TscaleType, aweights_scale, weights_scale, xc->oc3, xc->ic3, A, A, xc->O2, V);
  MD6(TscaleType, aweights_factor, weights_factor, xc->oc3, xc->ic3, A, A, xc->O2, V);
  MD5(TscaleType, asrc_scale, src_scale, xc->ic3, A, A, 2, Tz);

  bool scramble = (xc->T == xc->Tr) || (xc->t2 >= 2 * mthr_);
  if (scramble) {
    int it_start = omp_get_thread_num();
    iter_each(i, A * A) {
      int n = (it_start + i) % (A * A);
      int _wA = n % A;
      int _hA = n / A;
      iter_each(_ic3, xc->ic3) {
      iter_each(_oc3, xc->oc3) {
        auto attr = _ic3 == 0 ? set_attr(attr_, r_output_idx) : attr_;
        attr = set_attr(attr, l_output_idx);
        attr = set_attr(attr, c_output_idx);
        if (xc->Ir != V && _ic4 == xc->ic4 - 1 && _ic3 == xc->ic3 - 1)
          attr = set_attr(attr, has_Ir_idx);

        TscaleType *asrc_s, *asrc_z;
        if (xc->sampling_kind == COARSE || xc->sampling_kind == CALIBRATED) {
          asrc_s = &md5(asrc_scale, 0, 0, 0, 0, 0);
          asrc_z = &md5(asrc_scale, 0, 0, 0, 1, 0);
        } else {
          asrc_s = &md5(asrc_scale, _ic3, _hA, _wA, 0, 0);
          asrc_z = &md5(asrc_scale, _ic3, _hA, _wA, 1, 0);
        }

        ker_gemm(*(elx_conv_params_t *)xc,
            &md6(atoutput, _hA, _wA, _oc3, 0, 0, 0),
            &md6(atinput, _hA, _wA, _ic3, 0, 0, 0),
            &md5(atweights, _oc3, _ic3, _hA, _wA, 0),
            nullptr, attr, asrc_s, asrc_z,
            &md6(aweights_scale, _oc3, _ic3, _hA, _wA, 0, 0),
            &md6(aweights_factor, _oc3, _ic3, _hA, _wA, 0, 0));
      }}
    }
  } else {
    iter_each(_hA, A) {
    iter_each(_wA, A) {
    iter_each(_ic3, xc->ic3) {
    iter_each(_oc3, xc->oc3) {
      auto attr = _ic3 == 0 ? set_attr(attr_, r_output_idx) : attr_;
      attr = set_attr(attr, l_output_idx);
      attr = set_attr(attr, c_output_idx);
      if (xc->Ir != V && _ic4 == xc->ic4 - 1 && _ic3 == xc->ic3 - 1)
        attr = set_attr(attr, has_Ir_idx);

      ker_gemm(*(elx_conv_params_t *)xc,
          &md6(atoutput, _hA, _wA, _oc3, 0, 0, 0),
          &md6(atinput, _hA, _wA, _ic3, 0, 0, 0),
          &md5(atweights, _oc3, _ic3, _hA, _wA, 0),
          nullptr, attr,
          &md5(asrc_scale, _ic3, _hA, _wA, 0, 0),
          &md5(asrc_scale, _ic3, _hA, _wA, 1, 0),
          &md6(aweights_scale, _oc3, _ic3, _hA, _wA, 0, 0),
          &md6(aweights_factor, _oc3, _ic3, _hA, _wA, 0, 0));
    }}}}
  }
}

template <typename GarrayTypes, const int A, const int V, const int I>
void elx_conv_wino_u8s8_gemm_t<GarrayTypes, A, V, I>::execute_na(
    ToutputType *toutput, uint8_t *tinput, int8_t *tweights,
    TscaleType *src_scale, TscaleType *src_factor,
    TscaleType *weights_scale, TscaleType *weights_factor, int _ic4)
{
  int ithr = omp_get_thread_num();
  thread_parallel_for<5, 2>(mthr_, ithr, [&](int _hA, int _wA, int _ic3, int _oc3, int _t2) {
    MD2(uint8_t, atinput2, tinput, xc->t2, A * A * xc->ic3 * xc->I2 * xc->T * V);
    MD2(ToutputType, atoutput2, toutput, xc->t2, A * A * xc->oc3 * xc->O2 * xc->T * V);
    MD5(int8_t, atweights, tweights, xc->oc3, xc->ic3, A, A, xc->O2 * xc->I2 * V * V);
    MD6(TscaleType, aweights_scale, weights_scale, xc->oc3, xc->ic3, A, A, xc->O2, V);
    MD6(TscaleType, aweights_factor, weights_factor, xc->oc3, xc->ic3, A, A, xc->O2, V);
    MD6(TscaleType, asrc_scale, src_scale, xc->t2, A, A, xc->ic3, 2, xc->T);
    int Tz = _t2 == (xc->t2 - 1) ? xc->Tr : xc->T;
    MD6(uint8_t, atinput6, &md2(atinput2, _t2, 0), A, A, xc->ic3, xc->I2, Tz, V);
    MD6(ToutputType, atoutput6, &md2(atoutput2, _t2, 0), A, A, xc->oc3, xc->O2, Tz, V);
    auto ker_gemm = (_t2 == xc->t2 - 1) ? ker_u8s8_gemm0_ : ker_u8s8_gemm_;

    int attr = _ic3 == 0 ?  set_attr(attr_, r_output_idx) : attr_;
    attr = set_attr(attr, l_output_idx);
    attr = set_attr(attr, c_output_idx);
    if (xc->Ir != V && _ic4 == xc->ic4 - 1 && _ic3 == xc->ic3 - 1)
      attr = set_attr(attr, has_Ir_idx);

    TscaleType *asrc_s, *asrc_z;
    if (xc->sampling_kind == COARSE || xc->sampling_kind == CALIBRATED) {
      asrc_s = &md6(asrc_scale, 0, 0, 0, 0, 0, 0);
      asrc_z = &md6(asrc_scale, 0, 0, 0, 0, 1, 0);
    } else {
      asrc_s = &md6(asrc_scale, _t2, _hA, _wA, _ic3, 0, 0);
      asrc_z = &md6(asrc_scale, _t2, _hA, _wA, _ic3, 1, 0);
    }

    ker_gemm(*xc, &md6(atoutput6, _hA, _wA, _oc3, 0, 0, 0),
        &md6(atinput6, _hA, _wA, _ic3, 0, 0, 0),
        &md5(atweights, _oc3, _ic3, _hA, _wA, 0),
        nullptr, attr, asrc_s, asrc_z,
        &md6(aweights_scale, _oc3, _ic3, _hA, _wA, 0, 0),
        &md6(aweights_factor, _oc3, _ic3, _hA, _wA, 0, 0));
  }, A, A, xc->ic3, xc->oc3, xc->t2);
}

} // namespace euler
