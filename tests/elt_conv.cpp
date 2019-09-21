#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string>
#include <string.h>
#include <sstream>
#include "elt_utils.hpp"
#include "elt_conv_utils.hpp"
#include "euler.hpp"
#include <iostream>
#include <unordered_map>
#include "elt_gflag.hpp"


using namespace euler;

// Covolution options
int mb = 0, g = 1, ic = 0, ih = 0, iw = 0, oc = 0, oh = 0, ow = 0, kh = 3, kw = 3;
int ph = 1, pw = 1, sh = 1, sw = 1, dh = 1, dw = 1;
bool with_bias = true, with_relu = false, with_ip_sum = false,
     with_argmax = false, f16c_opt = false, disable_autoparam = true;
int data_type_cfg = 0;
int prop_kind = forward_inference, alg = CONV_AUTO;
int input_format = nChw16c, weights_format = OIhw16i16o,
    output_format = nChw16c;
int nthreads = 0;
int execution_mode = 0;
int flt_o = 1, flt_t = 1;
int blk_i = 1, blk_o = 1;
int pat_i = 1, pat_o = 1, pat_g = 1;
int tile_size = 0;
int streaming_input = 0, streaming_output = 0;
bool input_as_blocked = false, weights_as_blocked = false,
     output_as_blocked = false;
const char *input_file = nullptr, *weights_file = nullptr, *bias_file = nullptr;

bool validate_results = false;
int repeated_layer = 1;
bool double_buffering = false;
bool output_as_input = false;

bool is_int8_lp = false;
bool with_real_data = false;

euler::sampling_kind_t sampling_kind = euler::CALIBRATED;
float tinput_cali_s = FLT_MAX;
float tinput_cali_z = FLT_MAX;

int parse_cmd_options(int argc, char **argv) {

  gflags_namespace::SetUsageMessage("Euler convolution benchmark test");
  gflags_namespace::SetVersionString("0.0.1");

  gflags_namespace::ParseCommandLineFlags(&argc, &argv, true);
  mb = FLAGS_mb;
  g  = FLAGS_g;
  ic = FLAGS_ic;
  oc = FLAGS_oc;
  ih = FLAGS_ih;
  iw = FLAGS_iw;
  oh = FLAGS_oh;
  ow = FLAGS_ow;
  kh = FLAGS_kh;
  kw = FLAGS_kw;
  ph = FLAGS_ph;
  pw = FLAGS_pw;
  sh = FLAGS_sh;
  sw = FLAGS_sw;
  dh = FLAGS_dh;
  dw = FLAGS_dw;
  validate_results = FLAGS_validate_results;
  with_bias = FLAGS_with_bias;
  with_relu = FLAGS_with_relu;
  with_argmax = FLAGS_with_argmax;
  repeated_layer = FLAGS_repeated_layer;
  double_buffering = FLAGS_dbuffering;
  output_as_input = FLAGS_output_as_input;
  tile_size = FLAGS_tile_size;
  nthreads = FLAGS_nthreads;
  flt_o = FLAGS_flt_o;
  flt_t = FLAGS_flt_t;
  blk_i = FLAGS_blk_i;
  blk_o = FLAGS_blk_o;
  pat_i = FLAGS_pat_i;
  pat_o = FLAGS_pat_o;
  pat_g = FLAGS_pat_g;
  streaming_input = FLAGS_streaming_input;
  streaming_output = FLAGS_streaming_output;
  input_as_blocked = FLAGS_input_as_blocked;
  weights_as_blocked = FLAGS_weights_as_blocked;
  output_as_blocked = FLAGS_output_as_blocked;
  f16c_opt = FLAGS_f16c_opt;
  with_ip_sum = FLAGS_with_ip_sum;
  sampling_kind = (sampling_kind_t)FLAGS_sampling_kind;
  tinput_cali_s = FLAGS_tinput_cali_s;
  tinput_cali_z = FLAGS_tinput_cali_z;
  disable_autoparam = FLAGS_disable_autoparam;

  std::transform(FLAGS_alg.begin(), FLAGS_alg.end(), FLAGS_alg.begin(),
                 ::toupper);
  if (FLAGS_alg == "DECONV")
    alg = DECONV_DIRECT;
  else if (FLAGS_alg == "AUTO")
    alg = CONV_AUTO;
  else if (FLAGS_alg == "WINO")
    alg = CONV_WINOGRAD;
  else if (FLAGS_alg == "DIRECT")
    alg = CONV_DIRECT;
  else if (FLAGS_alg == "DIRECT_VMG")
    alg = CONV_DIRECT_VMG;
  else if (FLAGS_alg == "DIRECT_1X1")
    alg = CONV_DIRECT_1X1;
  else {
    printf("Error: convolution options: alg should be "
           "deconv|auto|wino|direct|direct_vmg|direct_1x1\n");
    return -1;
  }

  std::stringstream interpreter;
  interpreter << std::hex << FLAGS_execution_mode;
  interpreter >> execution_mode;
  // TODO: improve LP semantics
  if (alg == CONV_WINOGRAD &&
      (execution_mode == 0xa161 || execution_mode == 0xa133 ||
       execution_mode == 0xa173)) {
    is_int8_lp = true;
  } else if (alg == CONV_DIRECT_1X1 && (execution_mode == 0xc160 || execution_mode == 0xb161)) {
    is_int8_lp = true;
  } else if (alg == CONV_DIRECT && (execution_mode == 0xd160 || execution_mode == 0xa160)) {
    is_int8_lp = true;
  }

  if (FLAGS_input_format == "nchw")
    input_format = nchw;
  else if (FLAGS_input_format == "nhwc")
    input_format = nhwc;
  else if (FLAGS_input_format == "nChw16c")
    input_format = nChw16c;
  else {
    printf("Error: convolution options: input-format should be "
           "nchw|nhwc|nChw16c\n");
    return -1;
  }

  if (FLAGS_weights_format == "oihw")
    weights_format = oihw;
  else if (FLAGS_weights_format == "goihw")
    weights_format = goihw;
  else if (FLAGS_weights_format == "hwio")
    weights_format = hwio;
  else if (FLAGS_weights_format == "ghwio")
    weights_format = ghwio;
  else if (FLAGS_weights_format == "OIhw16i16o")
    weights_format = OIhw16i16o;
  else if (FLAGS_weights_format == "gOIhw16i16o")
    weights_format = gOIhw16i16o;
  else {
    printf("Error: convolution options: weights-format should be "
           "oihw|hwio|OIhw16i16o|goihw|ghwio|gOIhw16i16o\n");
    return -1;
  }

  if (FLAGS_output_format == "nchw")
    output_format = nchw;
  else if (FLAGS_output_format == "nhwc")
    output_format = nhwc;
  else if (FLAGS_output_format == "nChw16c")
    output_format = nChw16c;
  else {
    printf("Error: convolution options: output-format should be "
           "nchw|nhwc|nChw16c\n");
    return -1;
  }

  if (FLAGS_data_type_cfg == "FP32")
    data_type_cfg = euler::test::FP32;
  else if (FLAGS_data_type_cfg == "FP16")
    data_type_cfg = euler::test::FP16;
  else if (FLAGS_data_type_cfg == "FP16O")
    data_type_cfg = euler::test::FP16O;
  else if (FLAGS_data_type_cfg == "U8F32U8F32")
    data_type_cfg = euler::test::U8F32U8F32;
  else if (FLAGS_data_type_cfg == "U8F32S8F32")
    data_type_cfg = euler::test::U8F32S8F32;
  else if (FLAGS_data_type_cfg == "U8F32F32F32")
    data_type_cfg = euler::test::U8F32F32F32;
  else if (FLAGS_data_type_cfg == "U8F32U8F32z")
    data_type_cfg = euler::test::U8F32U8F32z;
  else if (FLAGS_data_type_cfg == "U8F32S8F32z")
    data_type_cfg = euler::test::U8F32S8F32z;
  else if (FLAGS_data_type_cfg == "U8F32F32F32z")
    data_type_cfg = euler::test::U8F32F32F32z;
  else {
    data_type_cfg = euler::test::FP32;
  }

  if (FLAGS_input_data_file != "") {
    const char *t = FLAGS_input_data_file.c_str();
    input_file = t == nullptr ? nullptr : strdup(t);
  }
  if (FLAGS_weights_data_file != "") {
    const char *t = FLAGS_weights_data_file.c_str();
    weights_file = t == nullptr ? nullptr : strdup(t);
  }
  if (FLAGS_bias_data_file != "") {
    const char *t = FLAGS_bias_data_file.c_str();
    bias_file = t == nullptr ? nullptr : strdup(t);
  }
  with_real_data = (input_file != nullptr) && (weights_file != nullptr);

  if (output_as_input && double_buffering) {
    printf("Error: convolution options: output-as-input is exclusive with "
           "double-buffering\n");
    return -1;
  }

  iw = iw == 0 ? ih : iw;
  ow = ow == 0 ? oh : ow;

  printf("Convolution options:\n"
         "mb:%d, g:%d, ic:%d, ih:%d, iw:%d, oc:%d, oh:%d, ow:%d, kh:%d, kw:%d, "
         "ph:%d, pw:%d, sh:%d, sw:%d, dh:%d, dw:%d\n"
         "with_bias:%d, with_relu:%d, with_ip_sum:%d, with_argmax:%d, "
         "f16c_opt=%d, data_type_cfg=%d, validate_results:%d\n"
         "flt_o:%d, flt_t:%d, blk_i:%d, blk_o:%d, pat_i:%d, pat_o:%d, pat_g:%d\n"
         "streaming-hint:%d, %d\n"
         "nthreads:%d\n"
         "execution-mode:%x\n",
         mb, g, ic, ih, iw, oc, oh, ow, kh, kw, ph, pw, sh, sw, dh, dw,
         with_bias, with_relu, with_ip_sum, with_argmax,
         f16c_opt, data_type_cfg, validate_results,
         flt_o, flt_t, blk_i, blk_o, pat_i, pat_o, pat_g, streaming_input,
         streaming_output, nthreads, execution_mode);

  std::unordered_map<int, const char *> prop_kind_str{
      {forward_training, "forward_training"},
      {forward_inference, "forward_inference"},
      {backward_data, "backward_data"},
      {backward_weights, "backward_weights"}};
  printf("prop_kind:%s\n", prop_kind_str[prop_kind]);

  std::unordered_map<int, const char *> alg_str{
      {CONV_AUTO, "CONV_AUTO"},
      {CONV_WINOGRAD, "CONV_WINOGRAD"},
      {CONV_DIRECT, "CONV_DIRECT"},
      {CONV_DIRECT_1X1, "CONV_DIRECT_1X1"}};
  printf("alg:%s", alg_str[alg]);
  if (alg == CONV_WINOGRAD)
    printf(", tile-size=%d", tile_size);
  printf("\n");

  std::unordered_map<int, const char *> fmt_str{
      {nchw, "nchw"}, {nhwc, "nhwc"},       {oihw, "oihw"},
      {hwio, "hwio"}, {nChw16c, "nChw16c"}, {OIhw16i16o, "OIhw16i16o"},
      {goihw, "goihw"}, {ghwio, "ghwio"},   {gOIhw16i16o, "gOIhw16i16o"}};
  printf("input-fmt:%s, weights-fmt:%s, output-fmt:%s\n", fmt_str[input_format],
         fmt_str[weights_format], fmt_str[output_format]);
  printf("input-as-blocked:%d, weights_as_blocked:%d, output_as_blocked:%d\n",
         input_as_blocked, weights_as_blocked, output_as_blocked);
  printf("double_buffering: %d, output_as_input=%d\n", double_buffering,
         output_as_input);

  // TODO: support tinput quantization only so far
  if (sampling_kind == euler::CALIBRATED && tinput_cali_s == 0.0 &&
      tinput_cali_z == 0.0) {
    tinput_cali_s = 1.0;
    tinput_cali_z = 1.0;
    printf("sampling-kind<CALIBRATED> tinput calibration scale: %f <dummy> "
           "zero: %f <dummy>\n",
           tinput_cali_s, tinput_cali_z);
  } else if (sampling_kind == euler::CALIBRATED) {
    printf("sampling-kind<CALIBRATED> tinput calibration scale: %f zero: %f\n",
           tinput_cali_s, tinput_cali_z);
  } else if (sampling_kind == euler::COARSE) {
    printf("sampling-kind<COARSES>\n");
  } else if (sampling_kind == euler::FINE) {
    printf("sampling-kind<FINE>\n");
  }

  if (mb <= 0 || g <= 0 || ic <= 0 || ih <= 0 || iw <= 0 || oc <= 0 ||
      oh <= 0 || ow <= 0 || kh <= 0 || kw <= 0) {
    printf("Error: convolution options: mb|g|ic|ih|iw|oc|oh|ow|kh|kw should "
           "greater than 0\n");
    return -1;
  }

  return 0;
}

static inline eld_conv_t &create_conv_desc(eld_conv_t &desc,
                                           int _data_type_cfg) {
  if (_data_type_cfg == euler::test::FP32) {
    desc.data_type = {euler::f32, euler::f32, euler::f32, euler::f32};
  } else if (_data_type_cfg == euler::test::FP16) {
    desc.data_type = {euler::f16, euler::f16, euler::f16, euler::f16};
  } else if (_data_type_cfg == euler::test::FP16O) {
    desc.data_type = {euler::f32, euler::f32, euler::f16, euler::f32};
  } else if (_data_type_cfg == euler::test::U8F32U8F32 ||
             _data_type_cfg == euler::test::U8F32U8F32z) {
    desc.data_type = {euler::u8, euler::f32, euler::u8, euler::f32};
  } else if (_data_type_cfg == euler::test::U8F32S8F32 ||
             _data_type_cfg == euler::test::U8F32S8F32z) {
    desc.data_type = {euler::u8, euler::f32, euler::s8, euler::f32};
  } else if (_data_type_cfg == euler::test::U8F32F32F32 ||
             _data_type_cfg == euler::test::U8F32F32F32z) {
    desc.data_type = {euler::u8, euler::f32, euler::f32, euler::f32};
  } else {
    test::error("Fail: Unsupported user data type ...\n");
    exit(1);
  }

  desc.dims = {mb, g, ic, oc, ih, iw, oh, ow, kh, kw};
  desc.formats = {input_format, weights_format, output_format};
  desc.pads = {pw, pw, ph, ph};
  desc.strides = {sh, sw};
  desc.with_bias = with_bias;
  desc.with_argmax = with_argmax;
  desc.with_ip_sum = with_ip_sum;
  desc.with_relu = with_relu;
  desc.f16c_opt = f16c_opt;
  desc.algorithm = alg;
  desc.tile_size = tile_size;
  desc.prop_kind = prop_kind;
  desc.nthreads = nthreads;
  desc.execution_mode = execution_mode;
  desc.flatting = {flt_o, flt_t};
  desc.blocking = {blk_i, blk_o};
  desc.partition = {pat_i, pat_o, pat_g};
  desc.streaming_hint = {streaming_input, streaming_output};
  desc.format_as_blocked = {input_as_blocked, weights_as_blocked,
                            output_as_blocked};
  desc.sampling_kind = sampling_kind;
  desc.use_scratch_pad = false;
  desc.disable_autoparam = disable_autoparam;
  return desc;
}

static inline int conv_ref_setup(eld_conv_t &desc) {
  auto int8_user_interface = [](int _data_type_cfg) -> bool {
    switch (_data_type_cfg) {
    case euler::test::U8F32U8F32:
    case euler::test::U8F32S8F32:
    case euler::test::U8F32F32F32:
    case euler::test::U8F32U8F32z:
    case euler::test::U8F32S8F32z:
    case euler::test::U8F32F32F32z:
      return true;
    default:
      return false;
    };
  };

  bool fully_setup = false;
  if (int8_user_interface(data_type_cfg) && desc.algorithm == CONV_WINOGRAD) {
    size_t t = (desc.dims.oh + desc.tile_size - 3) /
               (desc.tile_size - 3 + 1) *
               (desc.dims.ow + desc.tile_size - 3) /
               (desc.tile_size - 3 + 1) * desc.dims.n;
    size_t A = desc.tile_size;
    size_t K = 3;
    size_t IC = ALIGNUP(desc.dims.ic, 16);
    size_t tinput_byte_size = A * A * IC * t * sizeof(float);

    MEMALIGN64(&desc.scratch_pad, tinput_byte_size);
    desc.use_scratch_pad = true;
    desc.execution_mode = 0xa033;
    fully_setup = true;
  } else if (int8_user_interface(data_type_cfg) &&
             desc.algorithm == CONV_DIRECT_1X1) {
    if (sh == 1 && sw == 1)
      desc.execution_mode = 0xc060;
    else
      desc.execution_mode = 0xb061;
    desc.blocking.o = 1;
  } else if (int8_user_interface(data_type_cfg) &&
             desc.algorithm == CONV_DIRECT) {
    desc.execution_mode = 0xd060;
  }

  return desc.setup(fully_setup);
}

static inline void conv_execute(eld_conv_t convs[], void **input,
                                void **weights, void **output, void **bias,
                                int C) {
  for (auto c = 0; c < C; ++c) {
    eld_conv_t &_convs = convs[c];
    void *_weights = weights[c], *_bias = bias[c], *_input = input[c],
         *_output = output[c];

    if (double_buffering) {
      if (c % 2 == 0) {
        _input = input[0];
        _output = output[0];
      } else {
        _input = output[0];
        _output = input[0];
      }
    } else if (output_as_input) {
      if (c > 0)
        _input = output[c - 1];
    }

    if (ELX_OK != elx_conv(_convs, _output, _input, _weights, _bias)) {
      test::error("Fail: Convolution execution error!\n");
    }
  }
}

static inline void conv_bench(eld_conv_t convs[], eld_conv_t &conv_ref,
                              void **input, void **weights, void **output,
                              void **bias, int C) {
  auto num_ops = test::cal_ops(conv_ref);
  auto iter = test::cal_iterations(num_ops);
  auto N = iter < C ? C : iter;

  test::timer timer;
  for (auto n = 0; n < N / C; ++n) {
    for (auto c = 0; c < C; ++c) {
      eld_conv_t &_convs = convs[c];
      void *_weights = weights[c], *_bias = bias[c], *_input = input[c],
           *_output = output[c];

      if (double_buffering) {
        if (c % 2 == 0) {
          _input = input[0];
          _output = output[0];
        } else {
          _input = output[0];
          _output = input[0];
        }
      } else if (output_as_input) {
        if (c > 0)
          _input = output[c - 1];
      }

      timer.start();
      if (ELX_OK != elx_conv(_convs, _output, _input, _weights, _bias))
        test::error("Fail: Convolution execution error!\n");
      timer.stop();
    }
  }

  timer.report_tflops("conv", C * (N / C), num_ops);
}

#define RL_MAX 128
int main(int argc, char **argv) {
  if (parse_cmd_options(argc, argv))
    return 0;

  // 1, create convolution desc
  //    setup convolution
  eld_conv_t convs[RL_MAX];
  eld_conv_t conv_ref;
  create_conv_desc(conv_ref, euler::test::FP32);
  if (conv_ref_setup(conv_ref) != ELD_OK) {
    printf("Fail: Convolution setup error!\n");
    return 0;
  }

  const auto C =
      validate_results ? 1 : repeated_layer <= RL_MAX ? repeated_layer : RL_MAX;

  void *input[RL_MAX], *weights[RL_MAX], *output[RL_MAX], *bias[RL_MAX];
  float *input_ref, *weights_ref, *output_ref, *bias_ref;

  bool reuse_inout = double_buffering || output_as_input;

  MEMALIGN64(&input_ref, conv_ref.byte_sizes.input);
  MEMALIGN64(&output_ref, conv_ref.byte_sizes.output);
  MEMALIGN64(&weights_ref, conv_ref.byte_sizes.weights);
  MEMALIGN64(&bias_ref, conv_ref.byte_sizes.bias);

#define _prepare_conv_data(itype, wtype, otype, btype)                         \
  do {                                                                         \
    for (auto c = 0; c < C; ++c) {                                             \
      create_conv_desc(convs[c], data_type_cfg);                               \
      input[c] = nullptr;                                                      \
      output[c] = nullptr;                                                     \
      itype **in = (itype **)&input[c];                                        \
      wtype **wei = (wtype **)&weights[c];                                     \
      otype **out = (otype **)&output[c];                                      \
      btype **b = (btype **)&bias[c];                                          \
      test::prepare_conv_data<itype, wtype, otype, btype>(                     \
          conv_ref, convs[c], input_ref, weights_ref, output_ref, bias_ref,    \
          in, wei, out, b, input_file, weights_file, bias_file, input_format,  \
          weights_format, reuse_inout, data_type_cfg, f16c_opt,                \
          validate_results);                                                   \
                                                                               \
      if (convs[c].setup() != ELD_OK) {                                        \
        printf("Fail: Convolution setup error!\n");                            \
        return 0;                                                              \
      }                                                                        \
    }                                                                          \
  } while (0)

  if (data_type_cfg == euler::test::FP32) {
    _prepare_conv_data(float, float, float, float);
  } else if (data_type_cfg == euler::test::U8F32U8F32 ||
             data_type_cfg == euler::test::U8F32U8F32z) {
    _prepare_conv_data(uint8_t, float, uint8_t, float);
  } else if (data_type_cfg == euler::test::U8F32S8F32 ||
             data_type_cfg == euler::test::U8F32S8F32z) {
    _prepare_conv_data(uint8_t, float, int8_t, float);
  } else if (data_type_cfg == euler::test::U8F32F32F32 ||
             data_type_cfg == euler::test::U8F32F32F32z) {
    _prepare_conv_data(uint8_t, float, float, float);
  }
#ifdef ENABLE_USER_FP16
  else if (data_type_cfg == euler::test::FP16) {
    _prepare_conv_data(uint16_t, uint16_t, uint16_t, uint16_t);
  } else if (data_type_cfg == euler::test::FP16O) {
    _prepare_conv_data(float, float, float, float);
  }
#endif
  else {
    test::error("unsupported UserTypes\n");
  }

  // 2. execute convolution
  conv_execute(convs, input, weights, output, bias, C);

  if (validate_results) {
    // 3. validate results
    eld_conv_t &conv_val = convs[C - 1];
    void *output_val = output[C - 1];

    printf("Validation: ");
    if (test::ref_conv_deconv_2d<float>(conv_ref, output_ref, input_ref,
                                        weights_ref, bias_ref)) {
      printf("Fail: Convolution ref execution error!\n");
    } else {
      float *_output;
      MEMALIGN64(&_output, conv_ref.byte_sizes.output);
      test::post_process_conv_results(_output, conv_val, output_val,
                                      data_type_cfg);

      if (test::compare_conv_results(conv_ref, _output, output_ref,
                                     data_type_cfg, is_int8_lp, with_real_data))
        printf("Fail: Convolution results not correct!\n");
      else
        printf("Convolution Pass!\n");
      free(_output);
    }
  } else {
    // 4. bench
    conv_bench(convs, conv_ref, input, weights, output, bias, C);
  }

  // 5. setdown
  free(input_ref);
  free(output_ref);
  free(weights_ref);
  free(bias_ref);
  for (auto c = 0; c < C; ++c) {
    free(input[c]);
    free(weights[c]);
    free(output[c]);
    free(bias[c]);
  }

  return 0;
}
