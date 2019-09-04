#pragma once

#include <mutex>
#include "euler.hpp"
#include "el_def.hpp"
#include "el_allocator.hpp"

namespace euler {

// nChw16c input     : n, ic2, ih, iw, V
// nChw16c output    : n, oc2, oh, ow, V
// OIhw16i16o weights: oc2, ic2, kh, kw, V, V
// wino-gemm tinput  : t2, A*A, ic3, I2, T, V
// wino-gemm toutput : t2, A*A, oc3, O2, T, V
// wino-gemm tweights: oc3, ic3, A*A, O2, I2, V, V

struct elx_conv_params_t {
  // dimensions
  int g, ic, oc, ih, iw, oh, ow, n, t, kh, kw;
  // dimensions in packed unit
  int ic2, oc2, ih2, iw2, oh2, ow2, t2;
  // dimensions in pack-in-pack unit
  int ic3, oc3, ih3, iw3, oh3, ow3, t3;
  // dimensions in tripple level packed unit
  int ic4, oc4;
  // redundant dim size
  int ic234, ic34, oc234, oc34;
  // dimensions in tiles: tiles per (image, line, column)
  int nt, ht, wt;
  // pack size
  // int V;
  // int8 gemm
  int V1, Vx;
  // tile-size
  // int A;
  // register working set
  int T;
  // padding (IC/OC) & tailing dimensions: Ir, Or, Tr
  int IC, OC, Ir, Or, Tr, O2r, oc3r;
  // 2nd/r3d level cache blocking unit(in pack) to ic, oc
  int I2, O, O1, O2, I3, O3;
  // padding
  int lp, rp, tp, bp;
  // stride
  int hs, ws;
  // dilation
  int hd, wd;
  // saved group number, ic, oc per group, kernel multi-group number
  int grp, icg, ocg, vmg;
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
  sampling_kind_t sampling_kind;

  bool verbose;
  bool eager_mode;
  bool stream_sync;

  std::string name;
  std::string shared_workspace_key;
  bool shared_workspace_enabled;

  void *scratch_pad;
  void *output_ptr, *input_ptr, *weights_ptr, *bias_ptr;
  std::mutex mu;
};

struct elx_conv_t : elx_conv_params_t {
public:
  elx_conv_t(eld_conv_t &dc);

  void set_data(void *output, void *input, void *weights, void *bias);
  void timed_execute(void *output, void *input, void *weights, void *bias);

  virtual void execute(
      void *output, void *input, void *weights, void *bias) = 0;
  virtual ~elx_conv_t() {}
};

}  // namespace euler
