#include <assert.h>
#include <string.h>
#include <chrono>
#include "euler.hpp"
#include "el_stl.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "el_init.hpp"
#include "el_parallel.hpp"
#include "elx_conv.hpp"
#include "elx_stream.hpp"

namespace euler {

elx_conv_t::elx_conv_t(eld_conv_t &dc)
{
  ep.n = dc.dims.n;
  ep.g = dc.dims.g;
  ep.ic = dc.dims.ic;
  ep.oc = dc.dims.oc;
  ep.ih = dc.dims.ih;
  ep.iw = dc.dims.iw;
  ep.oh = dc.dims.oh;
  ep.ow = dc.dims.ow;
  ep.kh = dc.dims.kh;
  ep.kw = dc.dims.kw;
  ep.lp = dc.pads.l;
  ep.tp = dc.pads.t;
  ep.hs = dc.strides.h;
  ep.ws = dc.strides.w;
  ep.hd = dc.dilations.h;
  ep.wd = dc.dilations.w;
  // Fix user padding
  // rp = dc.pads.r;
  // bp = dc.pads.b;
  ep.rp = estl::max(0, ep.ws * (ep.ow - 1) + ep.kw - ep.iw - ep.lp);
  ep.bp = estl::max(0, ep.hs * (ep.oh - 1) + ep.kh - ep.ih - ep.tp);

  ep.input_fmt = dc.formats.input;
  ep.weights_fmt = dc.formats.weights;
  ep.output_fmt = dc.formats.output;
  ep.with_relu = dc.with_relu;
  ep.with_bias = dc.with_bias;
  ep.with_ip_sum = dc.with_ip_sum;
  ep.with_op_sum = dc.with_op_sum;
  ep.with_argmax = dc.with_argmax;
  ep.f16c_opt = dc.f16c_opt;
  ep.use_scratch_pad = dc.use_scratch_pad;
  ep.relu_bound_lower = dc.relu_bound.lower;
  ep.relu_bound_upper = dc.relu_bound.upper;

  // redundant
  for (auto i = 0; i < estl::size(ep.relu_bound_lower_vec); ++i)
    ep.relu_bound_lower_vec[i] = ep.relu_bound_lower;
  for (auto i = 0; i < estl::size(ep.relu_bound_upper_vec); ++i)
    ep.relu_bound_upper_vec[i] = ep.relu_bound_upper;

  ep.scratch_pad = dc.scratch_pad;

  ep.prop_kind = dc.prop_kind;

  ep.nthreads = dc.nthreads;
  auto mthr = estl::max_concurrency();
  if (ep.nthreads == 0 || ep.nthreads > mthr) {
    ep.nthreads = mthr;
  }

  ep.algorithm = dc.algorithm;
  ep.input_data_type = dc.data_type.input;
  ep.weights_data_type = dc.data_type.weights;
  ep.output_data_type = dc.data_type.output;
  ep.bias_data_type = dc.data_type.bias;
  ep.execution_mode = dc.execution_mode;

  /* Automatical parameters */
  ep.O = dc.flatting.o;
  ep.T = dc.flatting.t;

  ep.I2 = dc.blocking.i;
  ep.O1 = dc.blocking.o;

  ep.I4 = dc.partition.i;
  ep.O4 = dc.partition.o;
  ep.G3 = dc.partition.g;

  ep.streaming_input = dc.streaming_hint.input;
  ep.streaming_output = dc.streaming_hint.output;

  ep.input_as_blocked = dc.format_as_blocked.input;
  ep.weights_as_blocked = dc.format_as_blocked.weights;
  ep.output_as_blocked = dc.format_as_blocked.output;

  ep.input_quant_S = dc.input_quant.scale;
  ep.input_quant_z = dc.input_quant.z;
  ep.output_quant_S = dc.output_quant.scale;
  ep.output_quant_z = dc.output_quant.z;
  ep.sum_quant_S = dc.sum_quant.scale;
  for (auto i = 0; i < estl::size(ep.sum_quant_S_vec); ++i)
    ep.sum_quant_S_vec[i] = ep.sum_quant_S;
  ep.sum_quant_z = dc.sum_quant.z;
  ep.sampling_kind = dc.sampling_kind;

  ep.ormask = (unsigned int)-1;
  ep.eager_mode = dc.eager_mode;
  ep.stream_sync = dc.stream_sync;
  ep.name = dc.name;
 
  auto env_numa_node = getenv("EULER_NUMA_NODE");
  auto env_shared_workspace = getenv("EULER_SHARED_WORKSPACE");
  ep.shared_workspace_enabled = false;
  ep.shared_workspace_key = "na";
  if (env_shared_workspace != nullptr
      && (env_shared_workspace[0] == '1' || env_shared_workspace[0] == '2')) {
    if (env_numa_node != nullptr) {
      ep.shared_workspace_enabled = true;
      ep.shared_workspace_key = std::string(".euler_key_") + env_numa_node;
    }
  }

  output_ = nullptr;
  input_ = nullptr;
  weights_ = nullptr;
  bias_ = nullptr;
  workspace_ = nullptr;

  scratch_size_ = 0;
  workspace_size_ = 0;
  has_scratch_ = false;
  on_destroy_ = ELX_EVENT_NORMAL;
}

void elx_conv_t::set_workspace_buffers() {
  if (workspace_size_ != 0 && workspace_ == nullptr) {
    if (ep.shared_workspace_enabled) {
      const char *key = ep.shared_workspace_key.c_str();
      workspace_ = shwalloc::acquire(workspace_size_, key);
    } else {
      workspace_ = walloc::acquire(workspace_size_);
    }
  }
  if (workspace_ != nullptr)
    set_workspace_buffers(workspace_);
}

void elx_conv_t::set_scratch_buffers() {
  if (scratch_size_ != 0 && !has_scratch_) {
    void *scratch = scratch = galloc::acquire(scratch_size_);
    if (scratch != nullptr)
      has_scratch_ = true;
  }
  set_scratch_buffers(galloc::get());
}

void elx_conv_t::teardown() {
  if (workspace_ != nullptr && !ep.shared_workspace_enabled) {
    walloc::release(workspace_);
    workspace_ = nullptr;
  } else {
    const char *key = ep.shared_workspace_key.c_str();
    shwalloc::release(workspace_, key);
    workspace_ = nullptr;
  }

  if (scratch_size_ > 0 && has_scratch_)
    galloc::release();
}

elx_conv_t::~elx_conv_t() {
  if (ep.eager_mode) {
    teardown();
  } else {
    // submit an end-of-life request
    ep.stream_sync = true;
    if (on_destroy_ == ELX_EVENT_NORMAL)
      on_destroy_ = ELX_EVENT_TEARDOWN;
    global_stream.submit(this);
    global_stream.wait(this);
  }
}

void elx_conv_t::set_user_buffers(
    void *output, void *input, void *weights, void *bias)
{
  output_ = output;
  input_ = input;
  weights_ = weights;
  bias_ = bias;
}

void elx_conv_t::execute_verbose(void *output, void *input, void *weights,
                                 void *bias) {
  typedef std::chrono::high_resolution_clock hrc;
  typedef std::chrono::duration<float, std::milli> hrc_duration;

  hrc::time_point start_ts;
  start_ts = hrc::now();

  execute(output, input, weights, bias);

  el_log(__PERF_TRACE, "%s,ih:%d;oh:%d;ic:%d;oc:%d,"\
         "%s;%x,src:%s;wei:%s;dst:%s,src:%s;wei:%s;dst:%s;b:%s, %lf",
         ep.name.c_str(), ep.ih, ep.oh, ep.ic, ep.oc,
         algorithm_to_string(ep.algorithm), ep.execution_mode,
         format_to_string(ep.input_fmt), format_to_string(ep.weights_fmt),
         format_to_string(ep.output_fmt),
         datatype_to_string(ep.input_data_type),
         datatype_to_string(ep.weights_data_type),
         datatype_to_string(ep.output_data_type),
         datatype_to_string(ep.bias_data_type),
         hrc_duration(hrc::now() - start_ts).count());
}

int elx_conv(eld_conv_t &desc, void *output, void *input, void *weights, void *bias)
{
  elx_conv_t *xc = desc.xc;

  // Sanity check
  if (input == nullptr || weights == nullptr || output == nullptr
      || (desc.with_bias && bias == nullptr)) {
    el_error("Parameter error. Invalid input data!");
    return ELX_GENERAL_ERROR;
  }

  xc->set_scratch_buffers();

  if (xc->ep.eager_mode) {
    if (ego.verbose)
      xc->execute_verbose(output, input, weights, bias);
    else
      xc->execute(output, input, weights, bias);
  } else {
    xc->set_user_buffers(output, input, weights, bias);
    global_stream.submit(xc);
    if (xc->ep.stream_sync)
      global_stream.wait(xc);
    return ELX_OK;
  }
  
  return ELX_OK;
}

}  // namespace euler
