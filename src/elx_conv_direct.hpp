#ifndef __ELX_CONV_DIRECT_HPP__
#define __ELX_CONV_DIRECT_HPP__

#include "euler.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "kernel/elk_gemm_otj_binder.hxx"
#include "kernel/elk_conv_otj_binder.hxx"

namespace euler {

#define Template_elx_conv_direct_t                                             \
  template <typename UserTypes, typename TarrayTypes, const int V, const int I>

#define Instance_elx_conv_direct_t                                             \
  elx_conv_direct_t<UserTypes, TarrayTypes, V, I>

Template_elx_conv_direct_t class elx_conv_direct_t : public elx_conv_t {
  using InputType = typename UserTypes::InputType;
  using WeightsType = typename UserTypes::WeightsType;
  using OutputType = typename UserTypes::OutputType;
  using BiasType = typename UserTypes::BiasType;

  // t-buffer type
  using TinputType = typename TarrayTypes::InputType;
  using TweightsType = typename TarrayTypes::WeightsType;
  using ToutputType = typename TarrayTypes::OutputType;

  public:
  elx_conv_direct_t(eld_conv_t &dc);
  virtual ~elx_conv_direct_t();

  virtual void execute(void *output, void *input, void *weights, void *bias);

  private:
  void __execute_a060(OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias);
  void __execute_d060(OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias);

  void trans_weights_to_compact(TweightsType *tweights, WeightsType *weights);

  void conv_a060(OutputType *output, InputType *input, TweightsType *weights,
      BiasType *bias, int _ic4, int _oc4, int _ht, int _wt);
  void gemm_d060(OutputType *toutput, InputType *tinput, TweightsType *tweights,
      BiasType *bias, int _ic4, int _oc4, int _ht, int _wt);

  int prepare_execute_opt();
  void bind_execute_functions();

  template <class F> inline void md_loop(F func)
  {
#pragma omp parallel num_threads(mthr_)
    {
      int task_start, task_end;
      int ithr = omp_get_thread_num();
      int _t3_s, _t3_e, _oc4_s, _oc4_e, _ht_s, _ht_e, _wt_s, _wt_e;

      alloc_thread_task(nb_task_, mthr_, ithr, task_start, task_end);
      md_loop_iterator<4> start(
          task_start, this->t3, this->oc4, this->ht, this->wt);
      start.get(_t3_s, _oc4_s, _ht_s, _wt_s);

      md_loop_iterator<4> end(
          task_end, this->t3, this->oc4, this->ht, this->wt);
      end.get(_t3_e, _oc4_e, _ht_e, _wt_e);

      for (int _t3 = _t3_s; _t3 <= _t3_e; ++_t3) {
        int oc4_s = _t3 == _t3_s ? _oc4_s : 0;
        int oc4_e = _t3 == _t3_e ? _oc4_e : this->oc4 - 1;
        for (int _oc4 = oc4_s; _oc4 <= oc4_e; ++_oc4) {
          int ht_s = _oc4 == _oc4_s ? _ht_s : 0;
          int ht_e = _oc4 == _oc4_e ? _ht_e : this->ht - 1;
          for (int _ic4 = 0; _ic4 < this->ic4; ++_ic4) {
            for (int _ht = ht_s; _ht <= ht_e; ++_ht) {
              int wt_s = _ht == _ht_s ? _wt_s : 0;
              int wt_e = _ht == _ht_e ? _wt_e : this->wt - 1;
              for (int _wt = wt_s; _wt <= wt_e; ++_wt) {
                func(_t3, _oc4, _ic4, _ht, _wt);
              }
            }
          }
        }
      }
    }
  }

  // TODO: optimize it
  gemm_kernel_binder::kgemm<TarrayTypes> *ker_gemm_[64][8];
  conv_kernel_binder::kconv<TarrayTypes> *ker_conv_;
  conv_kernel_binder::kconv<TarrayTypes> *ker_conv_Tr_;

  void (elx_conv_direct_t::*execute_opt_)(
      OutputType *, InputType *, WeightsType *, BiasType *);

  bool is_first_run_;
  bool inference_acc_;

  size_t tweights_size_;
  TweightsType *tweights_;
  unsigned int xopt_;
  int attr_;
  int mthr_;
  void *scratch_;
  void *workspace_;
  int nb_task_;
};

// fp32-f32f32f32
template class elx_conv_direct_t<conv::FP32, conv_impl::FP32, 16, ISA_SKX_AVX512>;

#ifdef ENABLE_USER_FP16
// fp16o-f32f32f16
template class elx_conv_direct_t<conv::FP16O, conv_impl::FP32_F16o, 16, ISA_SKX_AVX512>;
#endif

} // namespace euler
#endif // __ELX_CONV_DIRECT_HPP__
