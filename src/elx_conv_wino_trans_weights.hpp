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
  void setup(elx_param_t *conv_ep) {
    ep = conv_ep;
    mthr_ = ep->nthreads;

    weights_is_bfmt_ = ep->weights_fmt == OIhw16i16o;
    weights_as_bfmt_ = ep->input_fmt == oihw && ep->weights_as_blocked;

    bind_kernel_functions();
  }

protected:
  void bind_kernel_functions() {
    ker_trans_weights_ = elk_conv_wino_trans_weights<op_type,
                       WeightsType, I, A, K, V>::execute;
  }

  elx_param_t *ep = nullptr;

  decltype(elk_conv_wino_trans_weights<
      op_type, WeightsType, I, A, K, V>::execute) *ker_trans_weights_;

  inline void __execute_blocked(TweightsType *__restrict tweights,
      WeightsType *__restrict weights, int O4);
  inline void __execute_oihw(TweightsType *__restrict tweights,
      WeightsType *__restrict weights, int O4);
  inline void __execute_hwio(TweightsType *__restrict tweights,
      WeightsType *__restrict weights, int O4);

  inline void __execute_blocked(TweightsType *__restrict tweights,
      WeightsType *__restrict weights, int _I4, int _O4);
  inline void __execute_oihw(TweightsType *__restrict tweights,
      WeightsType *__restrict weights, int _I4, int _O4);
  inline void __execute_hwio(TweightsType *__restrict tweights,
      WeightsType *__restrict weights, int _I4, int _O4);

  inline void __execute_post(TweightsType * __restrict tweights,
      op_type at[A][A][V][V],
      int _O4, int _I4, int _O3, int _I3,
      int _O1, int _I2, int _O);

  bool weights_is_bfmt_, weights_as_bfmt_;
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
      WeightsType *__restrict weights, int O4 = 1);
  void execute(TweightsType *__restrict tweights,
      WeightsType *__restrict weights, int _I4, int _O4);

  void operator () (TweightsType *__restrict tweights,
      WeightsType *__restrict weights, int O4 = 1) {
    execute(tweights, weights, O4);
  }
  void operator() (TweightsType *__restrict tweights,
      WeightsType *__restrict weights, int _I4, int _O4) {
    execute(tweights, weights, _I4, _O4);
  }

protected:
  using super::ep;
  using super::ker_trans_weights_;
  using super::weights_is_bfmt_;
  using super::weights_as_bfmt_;
  using super::mthr_;
};

template <typename WeightsType, int I, int A, int K, int V>
class elx_conv_wino_trans_weights_t<int8_t, WeightsType, I, A, K, V>
 : public elx_conv_wino_trans_weights_base<float, WeightsType, I, A, K, V> {
public:
  using TweightsType = float;
  using super = elx_conv_wino_trans_weights_base<float, WeightsType, I, A, K, V>;
  using op_type = typename super::op_type;

public:
  elx_conv_wino_trans_weights_t() {}
  virtual ~elx_conv_wino_trans_weights_t() {}

  void execute(float *__restrict tweights_scale,
      float *__restrict tweights_shift,
      int8_t *__restrict t_input_s8,
      TweightsType *__restrict t_input,
      WeightsType *__restrict input, int O4);

  void operator () (float *__restrict tweights_scale,
      float *__restrict tweights_shift,
      int8_t *__restrict t_input_s8,
      TweightsType *__restrict t_input,
      WeightsType *__restrict input, int O4) {
    execute(tweights_scale, tweights_shift,
        t_input_s8, t_input, input, O4);
  }

  inline void quantization(float *__restrict tweights_scale,
    float *__restrict tweights_shift,
    int8_t *__restrict tweights_s8,
    TweightsType *__restrict tweights, int O4);

protected:
  using super::ep;
  using super::ker_trans_weights_;
  using super::weights_is_bfmt_;
  using super::weights_as_bfmt_;
  using super::mthr_;
};

}  // namespace euler
