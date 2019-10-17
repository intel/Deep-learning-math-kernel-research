#include "elt_gflag.hpp"

DEFINE_int32(mb, 0, "Batch size");
DEFINE_int32(g,  1, "Groups size");
DEFINE_int32(ic, 0, "Input channel size");
DEFINE_int32(oc, 0, "Output channel size");
DEFINE_int32(ih, 0, "Input height");
DEFINE_int32(iw, 0, "Input width");
DEFINE_int32(oh, 0, "Output height");
DEFINE_int32(ow, 0, "Output width");
DEFINE_int32(kh, 3, "Kernel height. Default: 3");
DEFINE_int32(kw, 3, "Kernel width: Default: 3");
DEFINE_int32(ph, 1, "Padding along height. Default: 1");
DEFINE_int32(pw, 1, "Padding along width. Default: 1");
DEFINE_int32(sh, 1, "Stride along height. Default: 1");
DEFINE_int32(sw, 1, "Stride along width. Default: 1");
DEFINE_int32(dh, 1, "Dilation along height. Default: 1");
DEFINE_int32(dw, 1, "Dilation along width. Default: 1");
DEFINE_bool(validate_results, false,
            "on|off. Validate correctness. Default: off");
DEFINE_bool(with_bias, true, "on|off. With bias. Default: on");
DEFINE_bool(with_relu, false, "on|off. With relu. Default: off");
DEFINE_bool(with_argmax, false, "on|off. With argmax. Default: off");
DEFINE_int32(repeated_layer, 1, "Number of repeated layers. Default: 1");
DEFINE_bool(dbuffering, false, "Double buffering. Default: off");
DEFINE_bool(output_as_input, false,
            "Output of layer n used as input of layer n+1. Default: off");
DEFINE_string(alg, "wino",
              "deconv|auto|wino|direct|direct_1x1. Algorithm. Default: wino");
DEFINE_int32(tile_size, 0, "Winograd tile size: 0");
DEFINE_int32(nthreads, 0, "Number of threads per team");
DEFINE_string(execution_mode, "0x0", "Execution mode");
DEFINE_int32(flt_o, 1, "OC flatting");
DEFINE_int32(flt_t, 1, "Tile flatting");
DEFINE_int32(blk_i, 1, "IC blocking");
DEFINE_int32(blk_o, 1, "OC blocking");
DEFINE_int32(pat_i, 1, "Partition on ic");
DEFINE_int32(pat_o, 1, "Partition on oc");
DEFINE_int32(pat_g, 1, "Partition on g");
DEFINE_int32(streaming_input, 0,
             "Streaming hint for winograd transformed input");
DEFINE_int32(streaming_output, 0,
             "Streaming hint for winograd transformed output");
DEFINE_string(input_format, "nChw16c",
              "nchw|nhwc|nChw16c. Input data format. Default: nChw16c");
DEFINE_string(weights_format, "OIhw16i16o",
              "oihw|hwio|OIhw16i16o|goihw|ghwio|gOIhw16i16o. Weights data format. Default: OIhw16i16o");
DEFINE_string(output_format, "nChw16c",
              "nchw|nhwc|nChw16c. Output data format. Default: nChw16c");
DEFINE_bool(input_as_blocked, false,
            "on|off. Format input as blocked. Default: off");
DEFINE_bool(weights_as_blocked, false,
            "on|off. Format weighs as blocked. Default: off");
DEFINE_bool(output_as_blocked, false,
            "on|off. Format output as blocked. Default: off");
DEFINE_bool(f16c_opt, false, "on|off. With half-precision opt, Default: off");
DEFINE_string(data_type_cfg, "FP32", "UserTypes, Default: FP32");
DEFINE_bool(with_ip_sum, false, "on|off. With inplace sum, Default: off");
DEFINE_int32(sampling_kind, 2,
             "sampling kind 0: FINE, 1: COARSE, 2: CALIBRATED, Default: 2");
DEFINE_double(tinput_cali_s, 0.0,
              "calibration scale for tinput quantization, Default: 0");
DEFINE_double(tinput_cali_z, 0.0,
              "calibration zero for tinput quantization, Default: 0");
DEFINE_string(input_data_file, "", "Input data file(nchw)");
DEFINE_string(weights_data_file, "", "Weights data file(oihw)");
DEFINE_string(bias_data_file, "", "Bias data file");
DEFINE_string(name, "ioi", "Name of layer");
DEFINE_bool(disable_autoparam, true, "Disable autoparam");

