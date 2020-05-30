#pragma once

#include "euler.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "kernel/elk_conv_wino.hpp"

#include "kernel/elk_conv_wino_2x2_3x3_output.hxx"
#include "kernel/elk_conv_wino_3x3_3x3_output.hxx"
#include "kernel/elk_conv_wino_4x4_3x3_output.hxx"
#include "kernel/elk_conv_wino_5x5_3x3_output.hxx"

namespace euler {

template <typename OutputType, typename BiasType, typename ToutputType, int I,
    int A, int K, int V>
class elx_conv_wino_trans_output_t {
  public:
  // TODO: deduce TrOpType from OutputType/ToutputType
  using TrOpType = float;

  constexpr static bool is_border = true;
  constexpr static bool has_bias = true;
  constexpr static bool has_relu = true;
  constexpr static bool has_sum = true;
  constexpr static bool no = false;

public:
  elx_conv_wino_trans_output_t() {}
  virtual ~elx_conv_wino_trans_output_t() {}
  void setup(elx_param_t *ep);

  void execute(OutputType *output, ToutputType *toutput, BiasType *bias, int Tz,
      int _t2, int _O4, int _I4);
  void execute(OutputType *output, ToutputType *toutput, BiasType *bias,
      int _O4, int _I4);

  void operator() (OutputType *output, ToutputType *toutput, BiasType *bias,
      int Tz, int _t2, int _O4, int _I4) {
    execute(output, toutput, bias, Tz, _t2, _O4, _I4);
  }
  void operator() (OutputType *output, ToutputType *toutput, BiasType *bias,
      int _O4, int _I4) {
    execute(output, toutput, bias, _O4, _I4);
  }

  private:
  inline void __execute_nhwc(OutputType *output, ToutputType *toutput,
      BiasType *bias, int Tz, int _t2, int _O4, int _I4);
  inline void __execute_nchw(OutputType *output, ToutputType *toutput,
      BiasType *bias, int Tz, int _t2, int _O4, int _I4);
  inline void __execute_blocked(OutputType *output, ToutputType *toutput,
      BiasType *bias, int Tz, int _t2, int _O4, int _I4);
  inline void __execute_nhwc(OutputType *output, ToutputType *toutput,
      BiasType *bias, int _O4, int _I4);
  inline void __execute_nchw(OutputType *output, ToutputType *toutput,
      BiasType *bias, int _O4, int _I4);
  inline void __execute_blocked(OutputType *output, ToutputType *toutput,
      BiasType *bias, int _O4, int _I4);

  void bind_kernel_functions();

  decltype(elk_conv_wino_trans_output<TrOpType, OutputType, BiasType, 0,
      false, false, false, false, I, A, K, V>::execute) *ker_trans_output_;
  decltype(elk_conv_wino_trans_output<TrOpType, OutputType, BiasType, 0,
      false, false, false, false, I, A, K, V>::execute) *ker_trans_output0_;
  decltype(elk_conv_wino_trans_output<TrOpType, OutputType, BiasType, 0,
      false, false, false, false, I, A, K, V>::execute) *ker_trans_output_acc_;
  decltype(elk_conv_wino_trans_output<TrOpType, OutputType, BiasType, 0,
      false, false, false, false, I, A, K, V>::execute) *ker_trans_output0_acc_;

  elx_param_t *ep = nullptr;
  bool stream_out_;
  bool output_is_bfmt_;
  bool output_as_bfmt_;

  int hOA_end_;
  int wOA_end_;
  int hA_end_;
  int wA_end_;

  int mthr_;
};

template <int A, int K>
class output_tile_iter {
  constexpr static int output_line = A - K +1;
public:
  output_tile_iter(int n_init, int t_init, int ht, int wt, int oh, int ow)
    : ht_(ht), wt_(wt),
    h_end_(oh - (ht -1) * output_line -1),
    w_end_(ow - (wt -1) * output_line -1),
    tile_h_(t_init / wt),
    tile_w_(t_init % wt),
    n_(n_init),
    t_(tile_h_ * output_line),
    l_(tile_w_ * output_line),
    d_ (tile_h_ < ht -1 ? A -K : h_end_),
    r_ (tile_w_ < wt -1 ? A -K : w_end_) {}

  inline output_tile_iter &operator ++() {
    if ( ++ tile_w_ < wt_) {
      l_ += output_line;
    } else {
      tile_w_ = 0;
      l_ = 0;
      if ( ++ tile_h_ < ht_ )
        t_ += output_line;
      else {
        tile_h_ = 0;
        t_ = 0;
        n_ += 1;
      }
      d_ = tile_h_ < ht_ - 1 ? A -K : h_end_;
    }

    r_ = tile_w_ < wt_ - 1 ? A -K : w_end_;
    return *this;
  }

  inline void reset(int t = 0) {
    auto res = std::div(t, wt_);
    tile_h_ = res.quot;
    tile_w_ = res.rem;

    t_ = tile_h_ * output_line;
    l_ = tile_w_ * output_line;

    d_ = tile_h_ < ht_ -1 ? A -K : h_end_;
    r_ = tile_w_ < wt_ -1 ? A -K : w_end_;
  }

  inline bool is_border() const {
    return r_ < A - K || d_ < A - K;
  }

protected:
  int ht_, wt_;

public:
  int h_end_, w_end_;
  int tile_h_, tile_w_;
  int n_, t_, l_, d_, r_;
};

}  // namespace euler
