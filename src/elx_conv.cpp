#include <assert.h>
#include <string.h>
#include <chrono>
#include "euler.hpp"
#include "el_stl.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#if __ICC_COMPILER
#include "xmmintrin.h"
#include "pmmintrin.h"
#endif

namespace euler {

elx_conv_t::elx_conv_t(eld_conv_t &dc)
{
  this->n = dc.dims.n;
  this->g = dc.dims.g;
  this->ic = dc.dims.ic;
  this->oc = dc.dims.oc;
  this->ih = dc.dims.ih;
  this->iw = dc.dims.iw;
  this->oh = dc.dims.oh;
  this->ow = dc.dims.ow;
  this->kh = dc.dims.kh;
  this->kw = dc.dims.kw;
  this->lp = dc.pads.l;
  this->tp = dc.pads.t;
  this->hs = dc.strides.h;
  this->ws = dc.strides.w;
  this->hd = dc.dilations.h;
  this->wd = dc.dilations.w;
  // Fix user padding
  // this->rp = dc.pads.r;
  // this->bp = dc.pads.b;
  this->rp =
      estl::max(0, this->ws * (this->ow - 1) + this->kw - this->iw - this->lp);
  this->bp =
      estl::max(0, this->hs * (this->oh - 1) + this->kh - this->ih - this->tp);

  this->input_fmt = dc.formats.input;
  this->weights_fmt = dc.formats.weights;
  this->output_fmt = dc.formats.output;
  this->with_relu = dc.with_relu;
  this->with_bias = dc.with_bias;
  this->with_ip_sum = dc.with_ip_sum;
  this->with_op_sum = dc.with_op_sum;
  this->with_argmax = dc.with_argmax;
  this->f16c_opt = dc.f16c_opt;
  this->use_scratch_pad = dc.use_scratch_pad;

  this->scratch_pad = dc.scratch_pad;

  this->prop_kind = dc.prop_kind;

  this->nthreads = dc.nthreads;
  this->execution_mode = dc.execution_mode;

  /* Automatical parameters */
  this->O = dc.flatting.o;
  this->T = dc.flatting.t;

  this->I2 = dc.blocking.i;
  this->O1 = dc.blocking.o;

  this->ic4 = dc.partition.i;
  this->oc4 = dc.partition.o;

  this->streaming_input = dc.streaming_hint.input;
  this->streaming_output = dc.streaming_hint.output;

  this->input_as_blocked = dc.format_as_blocked.input;
  this->weights_as_blocked = dc.format_as_blocked.weights;
  this->output_as_blocked = dc.format_as_blocked.output;

  this->input_quant_S = dc.input_quant.scale;
  this->input_quant_z = dc.input_quant.z;
  this->output_quant_S = dc.output_quant.scale;
  this->output_quant_z = dc.output_quant.z;
  this->sum_quant_S = dc.sum_quant.scale;
  this->sum_quant_z = dc.sum_quant.z;
  this->sampling_kind = dc.sampling_kind;

  this->ormask = (unsigned int)-1;

  this->verbose = false;
  auto env_verbose = getenv("EULER_VERBOSE");
  if (env_verbose != nullptr && env_verbose[0] == '1')
    this->verbose = true;
  
  auto env_numa_node = getenv("EULER_NUMA_NODE");
  auto env_shared_workspace = getenv("EULER_SHARED_WORKSPACE");
  if (env_shared_workspace != nullptr && env_shared_workspace[0] == '1') {
    this->shared_workspace_enabled = true;
    this->shared_workspace_key = ".euler_key_" + dc.shared_workspace_key;
    if (env_numa_node != nullptr)
      this->shared_workspace_key += env_numa_node;
      std::replace(this->shared_workspace_key.begin(),
                   this->shared_workspace_key.end(), '/', '_');
  } else {
    this->shared_workspace_enabled = false;
    this->shared_workspace_key = "";
  }
  this->shared_workspace_mgr = nullptr;

  // TODO: move it to euler cpu global init
#if __ICC_COMPILER
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
}

int elx_conv(eld_conv_t &desc, void *output, void *input, void *weights, void *bias)
{
  elx_conv_t &xc = *desc.xc;

  // Sanity check
  if (input == nullptr || weights == nullptr || output == nullptr
      || (desc.with_bias && bias == nullptr)) {
    el_error("Parameter error. Invalid input data!");
    return ELX_GENERAL_ERROR;
  }

  typedef std::chrono::high_resolution_clock hrc;
  typedef std::chrono::duration<float, std::milli> hrc_duration;
  hrc::time_point start_ts;
  if (xc.verbose) start_ts = hrc::now();
  xc.execute(output, input, weights, bias);
  if (xc.verbose) {
    printf("Verbose: Euler kernel execution Duration: %lf(ms), kh=%d, ih=%d, oh=%d, ic=%d, oc=%d\n",
        hrc_duration(hrc::now() - start_ts).count(), xc.kh, xc.ih, xc.oh, xc.ic, xc.oc);
  }
  return ELX_OK;
}

}  // namespace euler
