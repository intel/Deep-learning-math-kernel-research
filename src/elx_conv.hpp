#pragma once

#include <mutex>
#include "euler.hpp"
#include "el_def.hpp"
#include "el_allocator.hpp"

namespace euler {

// nChw16c input     : n, ic2, ih, iw, V
// nChw16c output    : n, oc2, oh, ow, V
// OIhw16i16o weights: oc2, ic2, kh, kw, V, V
// wino-gemm tinput  : t2, A*A, I3, I2, T, V
// wino-gemm toutput : t2, A*A, oc3, O2, T, V
// wino-gemm tweights: oc3, I3, A*A, O2, I2, V, V

struct alignas(64) elx_param_t {
  // dimensions
  int g, ic, oc, ih, iw, oh, ow, n, kh, kw;
  // padding, stride, dilation
  int lp, rp, tp, bp, hs, ws, hd, wd;

  // V: vector pack size, Vx: dot-product width, V = V1 * Vx
  int /* V, */ V1, Vx;
  // padded (align-up to V) g/ic/oc
  int G, IC, OC;
  // 2nd/3rd/4-th level blocking unit to g, ic, oc
  int G2, G3, I2, I3, I4, O2, O3, O4;
  // O2 kernel blocking: O2 = O * O1, O/O1: kernel inner/outer loop
  int O, O1;
  // tailing
  int Ir, Or, Tr, O2r, O3r;
  // g23 = G2 * G3, ic234 = I2 * I3 * I4, ic123 = V * I2 * I3, ...
  int g23, ic234, oc234, ic34, oc34, ic123, oc123, ic23, oc23;
  // heler alias
  int &g2 = g23, &ic2 = ic234, &oc2 = oc234, &ic3 = ic34, &oc3 = oc34;
  // saved group number, ic, oc per group, kernel multi-group number
  int grp, icg, ocg, vmg;

  // winograd tile-size
  /* int A; */
  // number of tiles per (image, row, column, tensor)
  int nt, ht, wt, t;

  // spatial blocking unit
  int T;
  // 2nd level blocking to spatial (row, column, tensor)
  int ih2, iw2, oh2, ow2, t2;

  // Or masks
  unsigned int ormask;

  // formats
  int input_fmt;
  int weights_fmt;
  int output_fmt;

  // propagation kind
  int prop_kind;

  // relu, bias, sum
  bool with_relu, with_bias, with_ip_sum, with_op_sum, with_argmax, f16c_opt;

  // streaming hint
  int streaming_input;
  int streaming_output;

  // Use blocked format internally
  bool input_as_blocked;
  bool weights_as_blocked;
  bool output_as_blocked;
  bool use_scratch_pad;

  // threading
  int nthreads;
  int execution_mode;
  int algorithm;
  uint8_t input_data_type, weights_data_type, output_data_type, bias_data_type;


  // calibration coefficients
  float input_quant_S;
  float input_quant_z;
  float input_quant_repS;
  float tinput_quant_S;
  float tinput_quant_z;
  float tinput_quant_repS;
  float output_quant_S;
  float output_quant_z;
  float output_quant_repS;
  float sum_quant_S;
  float sum_quant_z;
  float relu_bound_lower;
  float relu_bound_upper;
  sampling_kind_t sampling_kind;

  bool eager_mode;
  bool stream_sync;

  std::string name;
  std::string shared_workspace_key;
  bool shared_workspace_enabled;

  void *scratch_pad;

  // Redundant data for performance
  alignas(64) float relu_bound_lower_vec[16];
  alignas(64) float relu_bound_upper_vec[16];
  alignas(64) float sum_quant_S_vec[16];
};


const int ELX_EVENT_NORMAL = 0;
const int ELX_EVENT_TEARDOWN = 1;
const int ELX_EVENT_EXIT = 2;

struct alignas(64) elx_conv_t {
public:
  elx_conv_t(eld_conv_t &dc);

  void set_user_buffers(void *output, void *input, void *weights, void *bias);
  void set_scratch_buffers();
  void set_workspace_buffers();

  void execute_verbose(void *output, void *input, void *weights, void *bias);
  virtual void execute(void *output, void *input, void *weights, void *bias) = 0;
  virtual ~elx_conv_t();
  void teardown();
  int on_destroy() { return on_destroy_; }
  template <typename F> void setup_workspace(F func) {
    if (ep.prop_kind == forward_inference && ep.shared_workspace_enabled) {
      const char *key = ep.shared_workspace_key.c_str();
      process_singleton_t process_singleton(key);
      {
        set_workspace_buffers();
        if (!shwalloc::is_setup_done(workspace_)) {
          func();
          shwalloc::set_setup_done(workspace_);
        }
      }
    } else {
      set_workspace_buffers();
      func();
    }
  }

  elx_param_t ep;

  void *workspace_, *output_, *input_, *weights_, *bias_;
  size_t scratch_size_, workspace_size_;
  int on_destroy_;
  bool has_scratch_;
  std::mutex mu_;

  inline bool last_I2(int _I2, int _I3, int _I4) {
    return _I4 == ep.I4 - 1 && _I3 == ep.I3 - 1 && _I2 == ep.I2 - 1;
  }

private:
  virtual void set_workspace_buffers(void *base) = 0;
  virtual void set_scratch_buffers(void *base) = 0;
};

}  // namespace euler
