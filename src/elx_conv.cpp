#include <assert.h>
#include "euler.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"

namespace euler {

elx_conv_t::elx_conv_t(eld_conv_t &dc)
{
  this->n = dc.dims.input.n;
  this->ic = dc.dims.input.c;
  this->oc = dc.dims.output.c;
  this->ih = dc.dims.input.h;
  this->iw = dc.dims.input.w;
  this->oh = dc.dims.output.h;
  this->ow = dc.dims.output.w;
  this->kh = dc.dims.weights.h;
  this->kw = dc.dims.weights.w;
  this->lp = dc.pads.l;
  this->rp = dc.pads.r;
  this->tp = dc.pads.t;
  this->bp = dc.pads.b;
  this->hs = dc.strides.h;
  this->ws = dc.strides.w;
  this->hd = dc.dilations.h;
  this->wd = dc.dilations.w;

  this->input_fmt = dc.formats.input;
  this->weights_fmt = dc.formats.weights;
  this->output_fmt = dc.formats.output;
  this->with_relu = dc.with_relu;
  this->with_bias = dc.with_bias;
  this->with_ip_sum = dc.with_ip_sum;
  this->with_op_sum = dc.with_op_sum;
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
  this->sampling_kind = dc.sampling_kind;

  this->ormask = (unsigned int)-1;
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

  xc.execute(output, input, weights, bias);
  return ELX_OK;
}

}  // namespace euler
