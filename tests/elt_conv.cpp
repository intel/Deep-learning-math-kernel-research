#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <boost/program_options.hpp>
#include "elt_utils.hpp"
#include "elt_conv_utils.hpp"
#include "euler.hpp"
#include <iostream>
#include <unordered_map>

using namespace euler;
namespace po = boost::program_options;

// Covolution options
int mb = 0, ic = 0, ih = 0, iw = 0, oc = 0, oh = 0, ow = 0, kh = 3, kw = 3;
int ph = 1, pw = 1, sh = 1, sw = 1, dh = 1, dw = 1;
bool with_bias = true, with_relu = false, with_ip_sum = false, f16c_opt = false;
int data_type_cfg = 0;
int prop_kind = forward_inference, alg = CONV_WINOGRAD;
int input_format = nChw16c, weights_format = OIhw16i16o, output_format = nChw16c;
int nthreads = 0;
int execution_mode = 0;
int flt_o = 1, flt_t = 1;
int blk_i = 1, blk_o = 1;
int pat_i = 1, pat_o = 1;
int tile_size = 7;
int streaming_input = 0, streaming_output = 0;
bool input_as_blocked = false, weights_as_blocked = false, output_as_blocked = false;
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
  po::options_description desc{"Options"};
  desc.add_options()
    ("help", "Convolution Options:")
    ("mb,n", po::value<int>(&mb), "Batch size")
    ("ic,i", po::value<int>(&ic), "Input channel size")
    ("oc,o", po::value<int>(&oc), "Output channel size")
    ("ih,h", po::value<int>(&ih), "Input height")
    ("iw,w", po::value<int>(&iw), "Input width")
    ("oh,H", po::value<int>(&oh), "Output height")
    ("ow,W", po::value<int>(&ow), "Output width")
    ("kh,k", po::value<int>(&kh), "Kernel height. Default: 3")
    ("kw,K", po::value<int>(&kw), "Kernel width: Default: 3")
    ("ph,p", po::value<int>(&ph), "Padding along height. Default: 1")
    ("pw,P", po::value<int>(&pw), "Padding along width. Default: 1")
    ("sh,s", po::value<int>(&sh), "Stride along height. Default: 1")
    ("sw,S", po::value<int>(&sw), "Stride along width. Default: 1")
    ("dh,d", po::value<int>(&dh), "Dilation along height. Default: 1")
    ("dw,D", po::value<int>(&dw), "Dilation along width. Default: 1")
    ("validate-results,v", po::value<bool>(&validate_results), "on|off. Validate correctness. Default: off")
    ("with-bias,b", po::value<bool>(&with_bias), "on|off. With bias. Default: on")
    ("with-relu,r", po::value<bool>(&with_relu), "on|off. With relu. Default: off")
    ("repeated-layer,l", po::value<int>(&repeated_layer), "Number of repeated layers. Default: 16")
    ("double-buffering,B", po::value<bool>(&double_buffering), "Double buffering. Default: off")
    ("output-as-input,A", po::value<bool>(&output_as_input), "Output of layer n used as input of layer n+1. Default: off")
    ("alg,a", po::value<std::string>(), "auto|wino|direct|direct_1x1. Algorithm. Default: wino")
    ("tile-size", po::value<int>(&tile_size), "Winograd tile size: 5")
    ("nthreads", po::value<int>(&nthreads), "Number of threads per team")
    ("execution-mode", po::value<std::string>(), "Execution mode")
    ("flt-o", po::value<int>(&flt_o), "OC flatting")
    ("flt-t", po::value<int>(&flt_t), "Tile flatting")
    ("blk-i", po::value<int>(&blk_i), "IC blocking")
    ("blk-o", po::value<int>(&blk_o), "OC blocking")
    ("pat-i", po::value<int>(&pat_i), "Partition on ic")
    ("pat-o", po::value<int>(&pat_o), "Partition on oc")
    ("streaming-input", po::value<int>(&streaming_input), "Streaming hint for winograd transformed input")
    ("streaming-output", po::value<int>(&streaming_output), "Streaming hint for winograd transformed output")
    ("input-format", po::value<std::string>(), "nchw|nhwc|nChw16c. Input data format. Default: nChw16c")
    ("weights-format", po::value<std::string>(), "oihw|hwio|OIhw16i16o. Weights data format. Default: OIhw16i16o")
    ("output-format", po::value<std::string>(), "nchw|nhwc|nChw16c. Output data format. Default: nChw16c")
    ("input-as-blocked", po::value<bool>(&input_as_blocked), "on|off. Format input as blocked. Default: off")
    ("weights-as-blocked", po::value<bool>(&weights_as_blocked), "on|off. Format weighs as blocked. Default: off")
    ("output-as-blocked", po::value<bool>(&output_as_blocked), "on|off. Format output as blocked. Default: off")
    ("f16c-opt", po::value<bool>(&f16c_opt), "on|off. With half-precision opt, Default: off")
    ("data-type-cfg", po::value<std::string>(), "UserTypes, Default: FP32")
    ("with-ip-sum", po::value<bool>(&with_ip_sum), "on|off. With inplace sum, Default: off")
    ("sampling-kind", po::value<int>((int *)&sampling_kind), "sampling kind 0: FINE, 1: COARSE, 2: CALIBRATED, Default: 2")
    ("tinput-cali-s", po::value<float>(&tinput_cali_s), "calibration scale for tinput quantization, Default: 0")
    ("tinput-cali-z", po::value<float>(&tinput_cali_z), "calibration zero for tinput quantization, Default: 0")
    ("input-data-file", po::value<std::string>(), "Input data file(nchw)")
    ("weights-data-file", po::value<std::string>(), "Weights data file(oihw)")
    ("bias-data-file", po::value<std::string>(), "Bias data file");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return -1;
  }
  if (vm.count("alg")) {
    std::string alg_str = vm["alg"].as<std::string>();
    std::transform(
        alg_str.begin(), alg_str.end(), alg_str.begin(), ::toupper);
    if (alg_str == "AUTO")
      alg = CONV_AUTO;
    else if (alg_str == "WINO")
      alg = CONV_WINOGRAD;
    else if (alg_str == "DIRECT")
      alg = CONV_DIRECT;
    else if (alg_str == "DIRECT_1X1")
      alg = CONV_DIRECT_1X1;
    else {
      printf("Error: convolution options: alg should be auto|wino|direct|direct_1x1\n");
      return -1;
    }
  }
  if (vm.count("execution-mode")) {
    std::stringstream interpreter;
    interpreter << std::hex << vm["execution-mode"].as<std::string>();
    interpreter >> execution_mode;
    // TODO: improve LP semantics
    if (alg == CONV_WINOGRAD &&
        (execution_mode == 0xa161 || execution_mode == 0xa133 ||
         execution_mode == 0xa173)) {
      is_int8_lp = true;
    }
  }
  if (vm.count("input-format")) {
    std::string fmt_str = vm["input-format"].as<std::string>();
    if (fmt_str == "nchw")
      input_format = nchw;
    else if (fmt_str == "nhwc")
      input_format = nhwc;
    else if (fmt_str == "nChw16c")
      input_format = nChw16c;
    else {
      printf("Error: convolution options: input-format should be "
             "nchw|nhwc|nChw16c\n");
      return -1;
    }
  }
  if (vm.count("weights-format")) {
    std::string fmt_str = vm["weights-format"].as<std::string>();
    if (fmt_str == "oihw")
      weights_format = oihw;
    else if (fmt_str == "hwio")
      weights_format = hwio;
    else if (fmt_str == "OIhw16i16o")
      weights_format = OIhw16i16o;
    else {
      printf("Error: convolution options: weights-format should be "
             "oihw|hwio|OIhw16i16o\n");
      return -1;
    }
  }
  if (vm.count("output-format")) {
    std::string fmt_str = vm["output-format"].as<std::string>();
    if (fmt_str == "nchw")
      output_format = nchw;
    else if (fmt_str == "nhwc")
      output_format = nhwc;
    else if (fmt_str == "nChw16c")
      output_format = nChw16c;
    else {
      printf("Error: convolution options: output-format should be "
             "nchw|nhwc|nChw16c\n");
      return -1;
    }
  }
  if (vm.count("data-type-cfg")) {
    std::string fmt_str = vm["data-type-cfg"].as<std::string>();
    if (fmt_str == "FP32")
      data_type_cfg = euler::test::FP32;
    else if (fmt_str == "FP16")
      data_type_cfg = euler::test::FP16;
    else if (fmt_str == "FP16O")
      data_type_cfg = euler::test::FP16O;
    else if (fmt_str == "U8F32U8F32")
      data_type_cfg = euler::test::U8F32U8F32;
    else if (fmt_str == "U8F32F32F32")
      data_type_cfg = euler::test::U8F32F32F32;
    else {
      data_type_cfg = euler::test::FP32;
    }
  }
  if (vm.count("input-data-file")) {
    input_file = strdup(vm["input-data-file"].as<std::string>().c_str());
    with_real_data = true;
  }
  if (vm.count("weights-data-file")) {
    weights_file = strdup(vm["weights-data-file"].as<std::string>().c_str());
    with_real_data = true;
  }
  if (vm.count("bias-data-file")) {
    bias_file = strdup(vm["bias-data-file"].as<std::string>().c_str());
  }
  printf("input-data-file: %s\n", input_file);
  printf("weights-data-file: %s\n", weights_file);
  printf("bias-data-file: %s\n", bias_file);

  if (output_as_input && double_buffering) {
    printf("Error: convolution options: output-as-input is exclusive with "
           "double-buffering\n");
    return -1;
  }

  iw = iw == 0 ? ih : iw;
  ow = ow == 0 ? oh : ow;

  printf("Convolution options:\n"
         "mb:%d, ic:%d, ih:%d, iw:%d, oc:%d, oh:%d, ow:%d, kh:%d, kw:%d, "
         "ph:%d, pw:%d, sh:%d, sw:%d, dh:%d, dw:%d\n"
         "with_bias:%d, with_relu:%d, with_ip_sum:%d, f16c_opt=%d, "
         "data_type_cfg=%d, validate_results:%d\n"
         "flt_o:%d, flt_t:%d, blk_i:%d, blk_o:%d, pat_i:%d, pat_o:%d\n"
         "streaming-hint:%d, %d\n"
         "nthreads:%d\n"
         "execution-mode:%x\n",
      mb, ic, ih, iw, oc, oh, ow, kh, kw, ph, pw, sh, sw, dh, dw, with_bias,
      with_relu, with_ip_sum, f16c_opt, data_type_cfg, validate_results, flt_o,
      flt_t, blk_i, blk_o, pat_i, pat_o, streaming_input,
      streaming_output, nthreads, execution_mode);

  std::unordered_map<int, const char *>prop_kind_str {
    { forward_training, "forward_training"},
    { forward_inference, "forward_inference"},
    { backward_data, "backward_data"},
    { backward_weights, "backward_weights"}
  };
  printf("prop_kind:%s\n", prop_kind_str[prop_kind]);

  std::unordered_map<int, const char *> alg_str {
    {CONV_AUTO, "CONV_AUTO"},
    {CONV_WINOGRAD, "CONV_WINOGRAD"},
    {CONV_DIRECT, "CONV_DIRECT"},
    {CONV_DIRECT_1X1, "CONV_DIRECT_1X1"}
  };
  printf("alg:%s", alg_str[alg]);
  if (alg == CONV_WINOGRAD)
    printf(", tile-size=%d", tile_size);
  printf("\n");

  std::unordered_map<int, const char *> fmt_str { {nchw, "nchw"}, {nhwc, "nhwc"},
    {oihw, "oihw"}, {hwio, "hwio"}, {nChw16c, "nChw16c"}, {OIhw16i16o, "OIhw16i16o"}
  };
  printf("input-fmt:%s, weights-fmt:%s, output-fmt:%s\n", fmt_str[input_format],
      fmt_str[weights_format], fmt_str[output_format]);
  printf("input-as-blocked:%d, weights_as_blocked:%d, output_as_blocked:%d\n",
      input_as_blocked, weights_as_blocked, output_as_blocked);
  printf("double_buffering: %d, output_as_input=%d\n", double_buffering, output_as_input);

  // TODO: support tinput quantization only so far
  if (sampling_kind == euler::CALIBRATED &&
      tinput_cali_s == 0.0 && tinput_cali_z == 0.0) {
    tinput_cali_s = 1.0;
    tinput_cali_z = 1.0;
    printf("sampling-kind<CALIBRATED> tinput calibration scale: %f <dummy> zero: %f <dummy>\n", tinput_cali_s, tinput_cali_z);
  } else if (sampling_kind == euler::CALIBRATED) {
    printf("sampling-kind<CALIBRATED> tinput calibration scale: %f zero: %f\n", tinput_cali_s, tinput_cali_z);
  } else if (sampling_kind == euler::COARSE) {
    printf("sampling-kind<COARSES>\n");
  } else if (sampling_kind == euler::FINE) {
    printf("sampling-kind<FINE>\n");
  }

  if (mb <= 0 || ic <= 0 || ih <= 0 || iw <= 0 || oc <= 0 || oh <= 0
      || ow <= 0 || kh <= 0 || kw <= 0) {
    printf("Error: convolution options: mb|ic|ih|iw|oc|oh|ow|kh|kw should "
           "greater than 0\n");
    return -1;
  }

  return 0;
}

static inline eld_conv_t create_conv_desc(int _data_type_cfg) {
  eld_conv_t desc;
  if (_data_type_cfg == euler::test::FP32) {
    desc.data_type = {
        euler::euler_f32, euler::euler_f32, euler::euler_f32, euler::euler_f32 };
  } else if (_data_type_cfg == euler::test::FP16) {
    desc.data_type = {
        euler::euler_f16, euler::euler_f16, euler::euler_f16, euler::euler_f16 };
  } else if (_data_type_cfg == euler::test::FP16O) {
    desc.data_type = {
        euler::euler_f32, euler::euler_f32, euler::euler_f16, euler::euler_f32 };
  } else if (_data_type_cfg == euler::test::U8F32U8F32) {
    desc.data_type = {
        euler::euler_u8, euler::euler_f32, euler::euler_u8, euler::euler_f32 };
  } else if (_data_type_cfg == euler::test::U8F32F32F32) {
    desc.data_type = {
        euler::euler_u8, euler::euler_f32, euler::euler_f32, euler::euler_f32 };
  } else {
    test::error("Fail: Unsupported user data type ...\n");
    exit(1);
  }
  desc.dims = {{ mb, ic, ih, iw },
               { oc, ic, kh, kw },
               { mb, oc, oh, ow },
               { oc } };
  desc.formats = {
    input_format, weights_format, output_format
  };
  desc.pads = { pw, pw, ph, ph };
  desc.strides   = { sh, sw };
  desc.with_bias = with_bias;
  desc.with_relu = with_relu;
  desc.with_ip_sum = with_ip_sum;
  desc.f16c_opt = f16c_opt;
  desc.algorithm = alg;
  desc.tile_size = tile_size;
  desc.prop_kind = prop_kind;
  desc.nthreads = nthreads;
  desc.execution_mode = execution_mode;
  desc.flatting = { flt_o, flt_t };
  desc.blocking = { blk_i, blk_o };
  desc.partition = { pat_i, pat_o };
  desc.streaming_hint
      = { streaming_input, streaming_output };
  desc.format_as_blocked
      = { input_as_blocked, weights_as_blocked, output_as_blocked };
  desc.sampling_kind = sampling_kind;
  desc.use_scratch_pad = false;
  return desc;
}

static inline int conv_ref_setup(eld_conv_t &desc) {
  auto int8_user_interface = [](int _data_type_cfg) -> bool {
    switch (_data_type_cfg) {
    case euler::test::U8F32U8F32:
    case euler::test::U8F32F32F32:
      return true;
    default:
      return false;
    };
  };

  if (int8_user_interface(data_type_cfg) && desc.algorithm == CONV_WINOGRAD) {
    size_t t = (desc.dims.output.h + desc.tile_size - 3)
        / (desc.tile_size - 3 + 1) * (desc.dims.output.w + desc.tile_size - 3)
        / (desc.tile_size - 3 + 1) * desc.dims.output.n;
    size_t A = desc.tile_size;
    size_t K = 3;
    size_t IC = desc.dims.input.c;
    size_t tinput_byte_size = A * A * IC * t * sizeof(float);

    MEMALIGN64(&desc.scratch_pad, tinput_byte_size);
    desc.use_scratch_pad = true;
    desc.execution_mode = 0xa033;
  }

  return desc.setup();
}

static inline void conv_execute(eld_conv_t convs[],
    void **input, void **weights, void **output, void **bias, int C) {
  for (auto c = 0; c < C; ++c) {
    eld_conv_t &_convs = convs[c];
    void *_weights = weights[c], *_bias = bias[c],
         *_input = input[c], *_output = output[c];

    if (double_buffering) {
      if (c % 2 == 0) {
        _input = input[0];
        _output = output[0];
      } else {
        _input = output[0];
        _output = input[0];
      }
    } else if (output_as_input) {
      if (c > 0) _input = output[c - 1];
    }

    if (ELX_OK != elx_conv(_convs, _output, _input, _weights, _bias)) {
      test::error("Fail: Convolution execution error!\n");
    }
  }
}

static inline void conv_bench(eld_conv_t convs[], eld_conv_t &conv_ref,
    void **input, void **weights, void **output, void **bias, int C) {
  auto num_ops = test::cal_ops(conv_ref);
  auto N = validate_results ? 1 : test::cal_iterations(num_ops);

  test::timer timer;
  for (auto n = 0; n < N / C; ++n) {
    for (auto c = 0; c < C; ++c) {
      eld_conv_t &_convs = convs[c];
      void *_weights = weights[c],
           *_bias = bias[c],
           *_input = input[c],
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
int main(int argc, char **argv)
{
  if (parse_cmd_options(argc, argv))
    return 0;

  // 1, create convolution desc
  auto desc = create_conv_desc(data_type_cfg);
  auto desc_ref = create_conv_desc(euler::test::FP32);

  // 2. setup convolution
  eld_conv_t convs[RL_MAX];
  eld_conv_t conv_ref = desc_ref;
  if (conv_ref_setup(conv_ref) != ELD_OK) {
    printf("Fail: Convolution setup error!\n");
    return 0;
  }

  const auto C = validate_results ?
      1 : repeated_layer <= RL_MAX ? repeated_layer : RL_MAX;

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
      convs[c] = desc;                                                         \
      input[c] = nullptr;                                                      \
      output[c] = nullptr;                                                     \
      itype **in = (itype **)&input[c];                                        \
      wtype **wei = (wtype **)&weights[c];                                     \
      otype **out = (otype **)&output[c];                                      \
      btype **b = (btype **)&bias[c];                                          \
      test::prepare_conv_data<itype, wtype, otype, btype>(                     \
          conv_ref, convs[c],                                                  \
          input_ref, weights_ref, output_ref, bias_ref,                        \
          in, wei, out, b, input_file, weights_file, bias_file,                \
          reuse_inout, data_type_cfg, f16c_opt, validate_results);             \
                                                                               \
      if (convs[c].setup() != ELD_OK) {                                        \
        printf("Fail: Convolution setup error!\n");                            \
        return 0;                                                              \
      }                                                                        \
    }                                                                          \
  } while (0)

  if (data_type_cfg == euler::test::FP32) {
    _prepare_conv_data(float, float, float, float);
  } else if (data_type_cfg == euler::test::U8F32U8F32) {
    _prepare_conv_data(uint8_t, float, uint8_t, float);
  } else if (data_type_cfg == euler::test::U8F32F32F32) {
    _prepare_conv_data(uint8_t, float, float, float);
  }
#ifdef ENABLE_USER_FP16
  else if (data_type_cfg == euler::test::FP16){
    _prepare_conv_data(uint16_t, uint16_t, uint16_t, uint16_t);
  } else if (data_type_cfg == euler::test::FP16O) {
    _prepare_conv_data(float, float, float, float);
  }
#endif
  else {
    printf("unsupported UserTypes\n");
    return 0;
  }

  // 3. execute convolution
  conv_execute(convs, input, weights, output, bias, C);

  if (validate_results) {
    // 4. validate results
    eld_conv_t &conv_val = convs[C - 1];
    void *output_val = output[C - 1];

    printf("Validation: ");
    if (test::ref_convolution2d<float>(
        conv_ref, output_ref, input_ref, weights_ref, bias_ref)) {
      printf("Fail: Convolution ref execution error!\n");
    } else {
      float *_output;
      MEMALIGN64(&_output, conv_ref.byte_sizes.output);
      test::post_process_conv_results(
          _output, conv_val, output_val, data_type_cfg);

      if (test::compare_conv_results(conv_ref, _output,
          output_ref, data_type_cfg, is_int8_lp, with_real_data))
        printf("Fail: Convolution results not correct!\n");
      else
        printf("Convolution Pass!\n");
      free(_output);
    }
  } else {
    // 5. bench
    conv_bench(convs, conv_ref, input, weights, output, bias, C);
  }

  // 6. setdown
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

