#include "gflags/gflags.h"

#ifdef GFLAGS_NAMESPACE
#define gflags_namespace GFLAGS_NAMESPACE
#else
#define gflags_namespace gflags
#endif

using namespace gflags_namespace;

DECLARE_int32(mb);
DECLARE_int32(g);
DECLARE_int32(ic);
DECLARE_int32(oc);
DECLARE_int32(ih);
DECLARE_int32(iw);
DECLARE_int32(oh);
DECLARE_int32(ow);
DECLARE_int32(kh);
DECLARE_int32(kw);
DECLARE_int32(ph);
DECLARE_int32(pw);
DECLARE_int32(sh);
DECLARE_int32(sw);
DECLARE_int32(dh);
DECLARE_int32(dw);
DECLARE_bool(validate_results);
DECLARE_bool(with_bias);
DECLARE_bool(with_relu);
DECLARE_bool(with_argmax);
DECLARE_int32(repeated_layer);
DECLARE_bool(dbuffering);
DECLARE_bool(output_as_input);
DECLARE_string(alg);
DECLARE_int32(tile_size);
DECLARE_int32(nthreads);
DECLARE_string(execution_mode);
DECLARE_int32(flt_o);
DECLARE_int32(flt_t);
DECLARE_int32(blk_i);
DECLARE_int32(blk_o);
DECLARE_int32(pat_i);
DECLARE_int32(pat_o);
DECLARE_int32(pat_g);
DECLARE_int32(streaming_input);
DECLARE_int32(streaming_output);
DECLARE_string(input_format);
DECLARE_string(weights_format);
DECLARE_string(output_format);
DECLARE_bool(input_as_blocked);
DECLARE_bool(weights_as_blocked);
DECLARE_bool(output_as_blocked);
DECLARE_bool(f16c_opt);
DECLARE_string(data_type_cfg);
DECLARE_bool(with_ip_sum);
DECLARE_int32(sampling_kind);
DECLARE_double(tinput_cali_s);
DECLARE_double(tinput_cali_z);
DECLARE_string(input_data_file);
DECLARE_string(weights_data_file);
DECLARE_string(bias_data_file);
DECLARE_bool(disable_autoparam);
