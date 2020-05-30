#ifndef __ELX_CONV_DIRECT_VMG_HPP__
#define __ELX_CONV_DIRECT_VMG_HPP__

#include "euler.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "el_allocator.hpp"
#include "elx_conv.hpp"
#include "kernel/elk_vmg_conv_binder.hxx"

namespace euler {

#define Template_elx_conv_direct_vmg_t                                         \
  template <typename UserTypes, typename TarrayTypes, const int V, const int I>

#define Instance_elx_conv_direct_vmg_t                                         \
  elx_conv_direct_vmg_t<UserTypes, TarrayTypes, V, I>

Template_elx_conv_direct_vmg_t class elx_conv_direct_vmg_t : public elx_conv_t {
  using InputType = typename UserTypes::InputType;
  using WeightsType = typename UserTypes::WeightsType;
  using OutputType = typename UserTypes::OutputType;
  using BiasType = typename UserTypes::BiasType;

  // t-buffer type
  using TinputType = typename TarrayTypes::InputType;
  using TweightsType = typename TarrayTypes::WeightsType;
  using ToutputType = typename TarrayTypes::OutputType;

  public:
  elx_conv_direct_vmg_t(eld_conv_t &dc);
  virtual ~elx_conv_direct_vmg_t();

  virtual void execute(void *output, void *input, void *weights, void *bias);

  private:
  void __execute_c060(OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias);

  void trans_weights_to_compact(TweightsType *tweights, WeightsType *weights);

  void conv_c060(OutputType *output, InputType *input, TweightsType *weights,
      BiasType *bias, int _I4, int _O4, int _ht, int _wt);

  void set_scratch_buffers(void *base);
  void set_workspace_buffers(void *base);
  int prepare_execute_opt();
  void bind_execute_functions();

  // TODO: optimize it
  vmg_conv_kernel_binder::kconv<TarrayTypes> *ker_conv_;
  vmg_conv_kernel_binder::kconv<TarrayTypes> *ker_conv_Tr_;

  void (elx_conv_direct_vmg_t::*execute_opt_)(
      OutputType *, InputType *, WeightsType *, BiasType *);

  int C;
  bool is_first_run_;
  bool inference_acc_;

  size_t tweights_size_;
  TweightsType *tweights_;
  size_t toutput_size_;
  ToutputType *toutput_;
  unsigned int xopt_;
  int attr_;
  int mthr_;
};

} // namespace euler
#endif // __ELX_CONV_DIRECT_VMG_HPP__
