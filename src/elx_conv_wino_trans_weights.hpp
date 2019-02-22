#pragma once

#include "euler.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "kernel/elk_conv_wino.hpp"

#include "kernel/elk_conv_wino_2x2_3x3_weights.hxx"
#include "kernel/elk_conv_wino_3x3_3x3_weights.hxx"
#include "kernel/elk_conv_wino_4x4_3x3_weights.hxx"
#include "kernel/elk_conv_wino_5x5_3x3_weights.hxx"

namespace euler {

template <typename TweightsType, typename WeightsType, int I, int A, int K, int V>
class elx_conv_wino_trans_weights_base {
public:
  using op_type = float;

  elx_conv_wino_trans_weights_base() {}
  virtual ~elx_conv_wino_trans_weights_base() {}
  void setup(elx_conv_params_t *xc) {
    this->xc = xc;
    mthr_ = xc->nthreads;
    stream_wei_ = xc->streaming_weights
        ? (xc->streaming_weights == STORE_STREAMING)
        : !(xc->execution_mode & FUS_MSK) ? true : false;

    weights_is_bfmt_ = xc->weights_fmt == OIhw16i16o;
    weights_as_bfmt_ = xc->input_fmt == oihw && xc->weights_as_blocked;

    bind_kernel_functions();
  }

protected:
  void bind_kernel_functions() {
    ker_trans_weights_ = elk_conv_wino_trans_weights<op_type,
                       WeightsType, I, A, K, V>::execute;
  }

  elx_conv_params_t *xc = nullptr;

  decltype(elk_conv_wino_trans_weights<
      op_type, WeightsType, I, A, K, V>::execute) *ker_trans_weights_;

  bool stream_wei_, weights_is_bfmt_, weights_as_bfmt_;
  int mthr_;
};

template <typename TweightsType, typename WeightsType, int I, int A, int K, int V>
class elx_conv_wino_trans_weights_t :
  public elx_conv_wino_trans_weights_base<TweightsType, WeightsType, I, A, K, V> {
public:
  using super
    = elx_conv_wino_trans_weights_base<TweightsType, WeightsType, I, A, K, V>;
  using op_type = typename super::op_type;

public:
  elx_conv_wino_trans_weights_t() {}
  virtual ~elx_conv_wino_trans_weights_t() {}

  void execute(TweightsType *__restrict tweights,
      WeightsType *__restrict weights, int oc4 = 1);
  void execute(TweightsType *__restrict tweights,
      WeightsType *__restrict weights, int _ic4, int _oc4);

  void operator () (TweightsType *__restrict tweights,
      WeightsType *__restrict weights, int oc4 = 1) {
    execute(tweights, weights, oc4);
  }
  void operator() (TweightsType *__restrict tweights,
      WeightsType *__restrict weights, int _ic4, int _oc4) {
    execute(tweights, weights, _ic4, _oc4);
  }

protected:
  using super::xc;
  using super::ker_trans_weights_;
  using super::weights_is_bfmt_;
  using super::weights_as_bfmt_;
  using super::mthr_;

  inline void __execute_blocked(TweightsType *__restrict tweights,
      WeightsType *__restrict weights, int oc4);

  inline void __execute_blocked(TweightsType *__restrict tweights,
      WeightsType *__restrict weights, int _ic4, int _oc4);

  inline void __execute_post(TweightsType * __restrict tweights,
      op_type at[A][A][V][V],
      int _oc4, int _ic4, int _oc3, int _ic3,
      int _O1, int _I2, int _O);

  inline void __execute_oihw(TweightsType *__restrict tweights,
      WeightsType *__restrict weights, int oc4);
  inline void __execute_hwio(TweightsType *__restrict tweights,
      WeightsType *__restrict weights, int oc4);

  inline void __execute_oihw(TweightsType *__restrict tweights,
      WeightsType *__restrict weights, int _ic4, int _oc4);
  inline void __execute_hwio(TweightsType *__restrict tweights,
      WeightsType *__restrict weights, int _ic4, int _oc4);

  using super::stream_wei_;
};

template <typename WeightsType, int I, int A, int K, int V>
class elx_conv_wino_trans_weights_t<int8_t, WeightsType, I, A, K, V>
 : public elx_conv_wino_trans_weights_base<float, WeightsType, I, A, K, V> {
public:
  using TweightsType = float;
  using TscaleType = float;
  using super = elx_conv_wino_trans_weights_base<float, WeightsType, I, A, K, V>;

protected:
  using super::xc;
  using op_type = typename super::op_type;
  using super::stream_wei_;
  using super::weights_is_bfmt_;
  using super::weights_as_bfmt_;
  using super::mthr_;

public:
  elx_conv_wino_trans_weights_t() {}
  virtual ~elx_conv_wino_trans_weights_t() {}

  void execute(TscaleType *__restrict tweights_quant_scale,
      TscaleType *__restrict tweights_factor,
      int8_t *__restrict t_input_s8,
      TweightsType *__restrict t_input,
      WeightsType *__restrict input, int oc4);

  void operator () (TscaleType *__restrict tweights_quant_scale,
      TscaleType *__restrict tweights_factor,
      int8_t *__restrict t_input_s8,
      TweightsType *__restrict t_input,
      WeightsType *__restrict input, int oc4) {
    execute(tweights_quant_scale, tweights_factor,
        t_input_s8, t_input, input, oc4);
  }
protected:
  void __execute_blocked(TscaleType *__restrict tweights_quant_scale,
      TscaleType *__restrict tweights_factor,
      int8_t *__restrict t_input_s8,
      TweightsType *__restrict t_input,
      WeightsType *__restrict input, int oc4);

  using super::ker_trans_weights_;
};

template class elx_conv_wino_trans_weights_t<float, float, ISA_SKX_AVX512, 4, 3, 16>;
template class elx_conv_wino_trans_weights_t<float, float, ISA_SKX_AVX512, 5, 3, 16>;
template class elx_conv_wino_trans_weights_t<float, float, ISA_SKX_AVX512, 6, 3, 16>;
template class elx_conv_wino_trans_weights_t<float, float, ISA_SKX_AVX512, 7, 3, 16>;

template class elx_conv_wino_trans_weights_t<short, float, ISA_SKX_AVX512, 4, 3, 16>;
template class elx_conv_wino_trans_weights_t<short, float, ISA_SKX_AVX512, 5, 3, 16>;
template class elx_conv_wino_trans_weights_t<short, float, ISA_SKX_AVX512, 6, 3, 16>;
template class elx_conv_wino_trans_weights_t<short, float, ISA_SKX_AVX512, 7, 3, 16>;

template class elx_conv_wino_trans_weights_t<int8_t, float, ISA_SKX_AVX512, 4, 3, 16>;
template class elx_conv_wino_trans_weights_t<int8_t, float, ISA_SKX_AVX512, 5, 3, 16>;
template class elx_conv_wino_trans_weights_t<int8_t, float, ISA_SKX_AVX512, 6, 3, 16>;
template class elx_conv_wino_trans_weights_t<int8_t, float, ISA_SKX_AVX512, 7, 3, 16>;

#ifdef ENABLE_USER_FP16
template class elx_conv_wino_trans_weights_t<float, short, ISA_SKX_AVX512, 4, 3, 16>;
template class elx_conv_wino_trans_weights_t<float, short, ISA_SKX_AVX512, 5, 3, 16>;
template class elx_conv_wino_trans_weights_t<float, short, ISA_SKX_AVX512, 6, 3, 16>;
template class elx_conv_wino_trans_weights_t<float, short, ISA_SKX_AVX512, 7, 3, 16>;

template class elx_conv_wino_trans_weights_t<short, short, ISA_SKX_AVX512, 4, 3, 16>;
template class elx_conv_wino_trans_weights_t<short, short, ISA_SKX_AVX512, 5, 3, 16>;
template class elx_conv_wino_trans_weights_t<short, short, ISA_SKX_AVX512, 6, 3, 16>;
template class elx_conv_wino_trans_weights_t<short, short, ISA_SKX_AVX512, 7, 3, 16>;

template class elx_conv_wino_trans_weights_t<int8_t, short, ISA_SKX_AVX512, 4, 3, 16>;
template class elx_conv_wino_trans_weights_t<int8_t, short, ISA_SKX_AVX512, 5, 3, 16>;
template class elx_conv_wino_trans_weights_t<int8_t, short, ISA_SKX_AVX512, 6, 3, 16>;
template class elx_conv_wino_trans_weights_t<int8_t, short, ISA_SKX_AVX512, 7, 3, 16>;
#endif

}  // namespace euler
