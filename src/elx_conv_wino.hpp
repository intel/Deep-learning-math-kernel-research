#ifndef __ELX_CONV_WINO_GEMM_HPP__
#define __ELX_CONV_WINO_GEMM_HPP__

#include <tuple>

#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "euler.hpp"
#include "elk_conv_wino.hpp"
#include "kernel/elk_conv_wino_gemm.hxx"
#include "kernel/elk_conv_wino_3x3_3x3_input.hxx"
#include "kernel/elk_conv_wino_3x3_3x3_output.hxx"
#include "kernel/elk_conv_wino_3x3_3x3_weights.hxx"

namespace euler {

template <typename Type, const int A, const int K, const int V, const int I>
class elx_conv_wino_t : public elx_conv_t<Type> {
  // Configurable parameters
  using elx_conv_t<Type>::IC;
  using elx_conv_t<Type>::OC;
  using elx_conv_t<Type>::T;
  using elx_conv_t<Type>::I2;
  using elx_conv_t<Type>::O2;
  using elx_conv_t<Type>::oc4;
  using elx_conv_t<Type>::ic3;
  using elx_conv_t<Type>::oc3;
  constexpr static size_t elem_sz_ = sizeof(Type);
  constexpr static bool border = true;
  constexpr static bool not_border = false;
  constexpr static bool with_bias = true;
  constexpr static bool without_bias = false;
  constexpr static bool with_relu = true;
  constexpr static bool without_relu = false;
  constexpr static bool with_sum = true;
  constexpr static bool without_sum = false;
public:
  elx_conv_wino_t(eld_conv_t<Type> &dc);
  virtual ~elx_conv_wino_t();

  virtual void execute(Type *output, Type *input, Type *weights, Type *bias);

protected:
  // Configurator
  inline std::tuple<int, int> tile_blocking_oc4(int num_cpu) const {
    constexpr int reg_max = 32;
    constexpr int reg_min = 15;
    auto part_o_max = 8;

    auto t = this->t;
    auto tb = (t - 1)/ num_cpu + 1;
    auto part_o = 1;

    while (tb < reg_min && part_o < part_o_max) {
      num_cpu /= 2;
      part_o *= 2;
      tb = (t - 1) / num_cpu + 1;
    }

    while (tb > reg_max) {
      tb /= 2;
    }

    return std::make_tuple(tb, part_o);
  }

  inline std::size_t input_unit() const {
    return elem_sz_ * T * V;
  }

  inline std::size_t weights_unit() const {
    return elem_sz_ * V * V;
  }

  inline std::size_t output_unit() const {
    return elem_sz_ * T * V;
  }

  // Return eligible I2 number, I2 iteration prefer L1 reside
  inline std::size_t I2_num(std::size_t cache_sz, std::size_t t) const {
    auto ic2 = this->ic2;
    auto cache_l = cache_sz - output_unit();

    while((input_unit() + weights_unit()) * ic2 > cache_l) {
      if ( (ic2 & 0x1) == 0 )
        ic2 /= 2;
    }
    return ic2;
  }

  // Return eligible O2 number, O2 iteration prefer L2 upper reside
  inline std::size_t O2_num(std::size_t cache_sz, std::size_t i2, std::size_t oc4)
    const {
    auto oc2 = this->oc2/oc4;

    auto cache_l = cache_sz - input_unit() * i2;
    auto wo_unit = weights_unit() + output_unit();

    while(wo_unit * oc2 > cache_l) {
      if ((oc2 & 0x1) == 0)
        oc2 /= 2;
    }
    return oc2;
  }

  // Checkers
  inline std::size_t gemmker_input_footprint() const {
    return input_unit() * I2;
  }

  inline std::size_t gemmker_weights_footprint() const {
    return weights_unit() * I2;
  }

  inline std::size_t gemmker_output_footprint() const {
    return output_unit();
  }

  inline std::size_t gemm_input_reuse_set() const {
    return gemmker_input_footprint() + 
      gemmker_weights_footprint() + gemmker_output_footprint();
  }

  inline std::size_t gemm_output_reuse_set() const {
    return gemmker_input_footprint() +
      gemmker_weights_footprint() * O2 + gemmker_output_footprint() * O2;
  }

  // a061 currently
  inline std::size_t gemm_weights_reuse_set() const {
    auto wtile_sz = elem_sz_ * A * A * IC * OC;
    return wtile_sz / oc4 + (gemmker_input_footprint() * ic3 +
      gemmker_output_footprint() * oc3) * A * A;
  }

private:

  void __execute_a000(Type *output, Type *input, Type *weights, Type *bias);
  void __execute_a040(Type *output, Type *input, Type *weights, Type *bias);
  void __execute_a060(Type *output, Type *input, Type *weights, Type *bias);
  void __execute_a061(Type *output, Type *input, Type *weights, Type *bias);
  void __execute_a0e1(Type *output, Type *input, Type *weights, Type *bias);
  void __execute_a0e0(Type *output, Type *input, Type *weights, Type *bias);
  void __execute_a073(Type *output, Type *input, Type *weights, Type *bias);
  void __execute_a201(Type *output, Type *input, Type *weights, Type *bias);
  void __execute_a241(Type *output, Type *input, Type *weights, Type *bias);
  void __execute_a448(Type *output, Type *input, Type *weights, Type *bias);

  inline void __trans_input_plain(Type *tinput, Type *input, int _t2, int Tz);
  inline void __trans_input_blocked(Type *tinput, Type *input, int _t2, int Tz);
  void trans_input(Type *tinput, Type *input, int _t2, int Tz);

  inline void __trans_input_plain(Type *tinput, Type *input);
  inline void __trans_input_blocked(Type *tinput, Type *input);
  void trans_input(Type *tinput, Type *input);

  inline void __trans_inputa_plain(Type *tinput, Type *input, int _t2, int _wA, int Tz);
  inline void __trans_inputa_blocked(Type *tinput, Type *input, int _t2, int _wA, int Tz);
  void trans_inputa(Type *tinput, Type *input, int _t2, int _wA, int Tz);

  inline void __trans_output_plain(Type *output, Type *toutput, Type *bias, int _t2, int Tz);
  inline void __trans_output_blocked(Type *output, Type *toutput, Type *bias, int _t2, int Tz);
  void trans_output(Type *output, Type *toutput, Type *bias, int _t2, int Tz);

  inline void __trans_output_plain(Type *output, Type *toutput, Type *bias);
  inline void __trans_output_blocked(Type *output, Type *toutput, Type *bias);
  void trans_output(Type *output, Type *toutput, Type *bias);

  inline void __trans_outputa_bh_plain(Type *output, Type *toutputa, Type *bias);
  inline void __trans_outputa_bh_blocked(Type *output, Type *toutputa, Type *bias);
  void trans_outputa_bh(Type *output, Type *toutputa, Type *bias);

  inline void __trans_output_plain(Type *res, Type *output, Type *toutput,
      Type *bias, int _t2, int Tz, int ic4, int oc4, bool inline_reduce);
  inline void __trans_output_blocked(Type *res, Type *output, Type *toutput,
      Type *bias, int _t2, int Tz, int ic4, int oc4, bool inline_reduce);
  void trans_output(Type *res, Type *output, Type *toutput, Type *bias,
      int _t2, int Tz, int ic4, int oc4, bool inline_reduce);

  void trans_outputa_th(Type *toutputa, Type *toutput, int Tz);

  inline void __trans_weights_plain(Type *tweights, Type *weights, int oc4);
  inline void __trans_weights_blocked(Type *tweights, Type *weights, int oc4);
  void trans_weights(Type *tweights, Type *weights, int oc4 = 1);

  inline void __trans_weightsa_plain(Type *tweights, Type *weights);
  inline void __trans_weightsa_blocked(Type *tweights, Type *weights);
  void trans_weightsa(Type *tweights, Type *weights);

  void gemm(Type *toutput, Type *tinput, Type *tweights, int _t2, int Tz);
  void gemm(Type *toutput, Type *tinput, Type *tweights);
  void gemma(Type *toutput, Type *tinput, Type *tweights, int _t2, int Tz);

  int prepare_execute_opt();
  void bind_execute_functions();

  decltype(
      convolution_gemm<Type, I, V, 1>::gemm) *ker_gemm_;
  decltype(
      convolution_gemm<Type, I, V, 1>::gemm) *ker_gemm0_;
  decltype(
      convolution_gemm<Type, I, V, 1>::gemm) *ker_gemm_tail_;
  decltype(
      convolution_gemm<Type, I, V, 1>::gemm) *ker_gemm0_tail_;
  decltype(
      convolution_winograd_kernel<
        Type, I, V, A, K>::template trans_input<not_border>) *ker_trans_input_;
  decltype(
      convolution_winograd_kernel<
        Type, I, V, A, K>::template trans_input<border>) *ker_trans_input0_;
  decltype(
      convolution_winograd_kernel<
        Type, I, V, A, K>::template trans_inputa<not_border>) *ker_trans_inputa_;
  decltype(
      convolution_winograd_kernel<
        Type, I, V, A, K>::template trans_inputa<border>) *ker_trans_inputa0_;
  decltype(
      convolution_winograd_kernel<
        Type, I, V, A, K>::trans_weights) *ker_trans_weights_;
  decltype(
      convolution_winograd_kernel<
        Type, I, V, A, K>::template
        trans_output<false, false, false, false>) *ker_trans_output_;
  decltype(
      convolution_winograd_kernel<
        Type, I, V, A, K>::template
        trans_output<false, false, false, false>) *ker_trans_output0_;
  decltype(
      convolution_winograd_kernel<
      Type, I, V, A, K>::template
      trans_output<false, false, false, false>) *ker_trans_output_nobias_;
  decltype(
      convolution_winograd_kernel<
      Type, I, V, A, K>::template
      trans_output<false, false, false, false>) *ker_trans_output0_nobias_;
  decltype(
      convolution_winograd_kernel<
      Type, I, V, A, K>::template trans_outputa_th<
      false, false, false, false>) *ker_trans_outputa_th_;
  decltype(
      convolution_winograd_kernel<
      Type, I, V, A, K>::template
      trans_outputa_bh<false, false, false, false>) *ker_trans_outputa_bh_;
  decltype(
      convolution_winograd_kernel<
      Type, I, V, A, K>::template
      trans_outputa_bh<false, false, false, false>) *ker_trans_outputa0_bh_;

  void (elx_conv_wino_t::*execute_opt_)(Type *, Type *, Type *, Type *);

  unsigned int xopt_;
  bool is_first_run_;
  bool inference_acc_;
  bool stream_in_;
  bool stream_out_;
  bool stream_wei_;
  bool is_bfmt_;
  bool input_is_bfmt_;
  bool weights_is_bfmt_;
  bool output_is_bfmt_;
  bool input_as_bfmt_;
  bool weights_as_bfmt_;
  bool output_as_bfmt_;
  int mthr_;
  Type *tweights_;
  Type *tinput_;
  Type *toutput_;
  Type *routput_; // reduce output
  Type *toutputa_;
  Type *binput_; // blocked input
  Type *bweights_;
  Type *boutput_;
  unsigned char *routput_cntr_;

  int hOA_end_;
  int wOA_end_;
  int hA_end_;
  int wA_end_;

#define MAX_THREAD_TEAMS (8)
  // tasks allocation per thread team
  struct { int start; int end; } ttm_[MAX_THREAD_TEAMS];
};

template class elx_conv_wino_t<float, 4, 3, 16, ISA_GENERIC>;
template class elx_conv_wino_t<float, 4, 3, 16, ISA_SKX_AVX512>;
template class elx_conv_wino_t<float, 5, 3, 16, ISA_GENERIC>;
template class elx_conv_wino_t<float, 5, 3, 16, ISA_SKX_AVX512>;
template class elx_conv_wino_t<float, 7, 3, 16, ISA_GENERIC>;
template class elx_conv_wino_t<float, 7, 3, 16, ISA_SKX_AVX512>;

}  // namespace euler
#endif  // __ELX_CONV_WINO_GEMM_HPP__
