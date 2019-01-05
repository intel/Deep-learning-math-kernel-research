#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <boost/program_options.hpp>
#include "elt_utils.hpp"
#include "elt_conv_utils.hpp"
#include "euler.hpp"
#include <iostream>
#include <unordered_map>

using namespace euler;
namespace po = boost::program_options;
int parse_cmd_options(int, char **);

// Covolution options
int mb = 0, ic = 0, ih = 0, iw = 0, oc = 0, oh = 0, ow = 0, kh = 3, kw = 3;
int ph = 1, pw = 1, sh = 1, sw = 1, dh = 1, dw = 1;
bool with_bias = true, with_relu = false, with_ip_sum = false, f16c_opt = false;
int fp_mode = 0;
int prop_kind = forward_inference, alg = CONV_WINOGRAD;
int input_format = nChw16c, weights_format = OIhw16i16o, output_format = nChw16c;
int nthreads = 0;
int execution_mode = 0;
int flt_o = 1, flt_t = 1;
int blk_i = 1, blk_o = 1;
int pat_i = 1, pat_o = 1;
int tile_size = 7;
int streaming_weights = 0, streaming_input = 0, streaming_output = 0;
bool input_as_blocked = false, weights_as_blocked = false, output_as_blocked = false;

bool validate_results = false;
int repeated_layer = 1;
bool double_buffering = false;
bool output_as_input = false;
bool tweights_preprocess = false;

template <typename ConvType>
static inline ConvType create_conv_desc(void) {
  ConvType desc;
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
  desc.fp_mode = fp_mode;
  desc.algorithm = alg;
  desc.tile_size = tile_size;
  desc.prop_kind = prop_kind;
  desc.nthreads = nthreads;
  desc.execution_mode = execution_mode;
  desc.flatting = { flt_o, flt_t };
  desc.blocking = { blk_i, blk_o };
  desc.partition = { pat_i, pat_o };
  desc.streaming_hint
      = { streaming_weights, streaming_input, streaming_output };
  desc.format_as_blocked
      = { input_as_blocked, weights_as_blocked, output_as_blocked };
  return desc;
}

template <typename ConvType, typename T, typename O>
static inline void conv_execute(eld_conv_t<ConvType> convs[],
    T **input, T **weights, O **output, T **bias, int C) {
  for (auto c = 0; c < C; ++c) {
    eld_conv_t<ConvType> &_convs = convs[c];
    T *_weights = weights[c], *_bias = bias[c], *_input = input[c];
    O *_output = output[c];

    if (std::is_same<T, O>::value) {
      if (double_buffering) {
        if (c % 2 == 0) {
          _input = input[0];
          _output = output[0];
        } else {
          _input = (T *)output[0];
          _output = (O *)input[0];
        }
      } else if (output_as_input) {
        if (c > 0) _input = (T *)output[c - 1];
      }
    }

    if (ELX_OK != elx_conv<ConvType>(_convs, _output, _input, _weights, _bias)) {
      test::error("Fail: Convolution execution error!\n");
    }
  }
}

template <typename ConvType, typename T, typename O>
static inline void conv_bench(eld_conv_t<ConvType> convs[],
    eld_conv_t<conv::FP32> &desc0, T **input, T **weights,
    O **output, T **bias, int C) {
  auto num_ops = test::cal_ops(desc0);
  auto N = validate_results ? 1 : test::cal_iterations(num_ops);

  test::timer timer;
  for (auto n = 0; n < N / C; ++n) {
    for (auto c = 0; c < C; ++c) {
      eld_conv_t<ConvType> &_convs = convs[c];
      T *_weights = weights[c], *_bias = bias[c], *_input = input[c];
      O *_output = output[c];

      if (std::is_same<T, O>::value) {
        if (double_buffering) {
          if (c % 2 == 0) {
            _input = input[0];
            _output = output[0];
          } else {
            _input = (T *)output[0];
            _output = (O *)input[0];
          }
        } else if (output_as_input) {
          if (c > 0)
            _input = (T *)output[c - 1];
        }
      }

      timer.start();
      if (ELX_OK
          != elx_conv<ConvType>(_convs, _output, _input, _weights, _bias)) {
        test::error("Fail: Convolution execution error!\n");
      }
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
  auto desc0 = create_conv_desc<eld_conv_t<conv::FP32>>();
  auto desc1 = create_conv_desc<eld_conv_t<conv::FP16>>();
  auto desc2 = create_conv_desc<eld_conv_t<conv::FP16O>>();

  // 2. setup convolution
  eld_conv_t<conv::FP32> convs0[RL_MAX];
  eld_conv_t<conv::FP16> convs1[RL_MAX];
  eld_conv_t<conv::FP16O> convs2[RL_MAX];

  const auto C = validate_results ?
      1 : repeated_layer <= RL_MAX ? repeated_layer : RL_MAX;

  float *input[RL_MAX], *weights[RL_MAX], *output[RL_MAX], *bias[RL_MAX],
      *ref_output;
  short *input1[RL_MAX], *weights1[RL_MAX], *output1[RL_MAX], *bias1[RL_MAX];

  bool reuse_inout = double_buffering || output_as_input;

  if (fp_mode == euler::FP32) {
    for (auto c = 0; c < C; ++c) {
      convs0[c] = desc0;
      if (convs0[c].setup() != ELD_OK) {
        printf("Fail: Convolution setup error!\n");
        return 0;
      }
      input[c] = nullptr;
      output[c] = nullptr;
      float **in = &input[c], **out = &output[c];
      if (double_buffering && (c > 0)) {
        in = nullptr;
        out = nullptr;
      } else if (output_as_input && (c > 0)) {
        in = nullptr;
      }
      test::prepare_conv_data<float, float, float, float>(
          convs0[c], in, &weights[c], out, &bias[c],
          &input1[c], &weights1[c], &output1[c], &bias1[c],
          reuse_inout, fp_mode, f16c_opt, validate_results);
    }

    if (validate_results) {
      ref_output = (float *)malloc(convs0[0].byte_sizes.output);
      if (desc0.with_ip_sum)
        memcpy(ref_output, output[0], convs0[0].byte_sizes.output);
    }
  } else if (fp_mode == euler::FP16){
    for (auto c = 0; c < C; ++c) {
      convs1[c] = desc1;
      if (convs1[c].setup() != ELD_OK) {
        printf("Fail: Convolution setup error!\n");
        return 0;
      }
      input1[c] = nullptr;
      output1[c] = nullptr;
      short **in = &input1[c], **out = &output1[c];
      if (double_buffering && (c > 0)) {
        in = nullptr;
        out = nullptr;
      } else if (output_as_input && (c > 0)) {
        in = nullptr;
      }

      if (c == 0) {
        convs0[0] = desc0;
        if (convs0[0].setup() != ELD_OK) {
          printf("Fail: Convolution setup error!\n");
          return 0;
        }
      }

      test::prepare_conv_data<float, float, float, float>(
          convs0[0], &input[c], &weights[c], &output[c], &bias[c],
          in, &weights1[c], out, &bias1[c], reuse_inout, fp_mode,
          f16c_opt, validate_results);
    }

    if (validate_results) {
      ref_output = (float *)malloc(convs0[0].byte_sizes.output);
      if (desc1.with_ip_sum)
        memcpy(ref_output, output[0], convs0[0].byte_sizes.output);
    }
  } else if (fp_mode == euler::FP16O) {
    for (auto c = 0; c < C; ++c) {
      convs2[c] = desc2;
      if (convs2[c].setup() != ELD_OK) {
        printf("Fail: Convolution setup error!\n");
        return 0;
      }

      if (c == 0) {
        convs0[0] = desc0;
        if (convs0[0].setup() != ELD_OK) {
          printf("Fail: Convolution setup error!\n");
          return 0;
        }
      }

      test::prepare_conv_data<float, float, float, float>(
          convs0[0], &input[c], &weights[c], &output[c], &bias[c],
          nullptr, nullptr, &output1[c], nullptr,
          reuse_inout, fp_mode, f16c_opt, validate_results);
    }

    if (validate_results) {
      ref_output = (float *)malloc(convs0[0].byte_sizes.output);
    }
  } else {
    printf("unsupported UserTypes\n");
    return 0;
  }

  // 3. execute convolution
  if (fp_mode == euler::FP32)
    conv_execute(convs0, input, weights, output, bias, C);
  else if (fp_mode == euler::FP16)
    conv_execute(convs1, input1, weights1, output1, bias1, C);
  else if (fp_mode == euler::FP16O)
    conv_execute(convs2, input, weights, output1, bias, C);

  if (validate_results) {
    // 4. validate results
    printf("Validation: ");
    if (test::ref_convolution2d<float>(
            convs0[0], ref_output, input[0], weights[0], bias[0]))
      printf("Fail: Convolution ref execution error!\n");
    if (fp_mode == euler::FP32) {
      if (test::compare_conv_results(convs0[0], output[0], ref_output, fp_mode))
        printf("Fail: Convolution results not correct!\n");
      else
        printf("Convolution Pass!\n");
    } else {
      if (test::compare_conv_results(convs0[0], output1[0], ref_output, fp_mode))
        printf("Fail: Convolution results not correct!\n");
      else
        printf("Convolution Pass!\n");
    }
    free(ref_output);
  } else {
    // 5. bench
    if (fp_mode == euler::FP32)
      conv_bench(convs0, desc0, input, weights, output, bias, C);
    else if (fp_mode == euler::FP16)
      conv_bench(convs1, desc0, input1, weights1, output1, bias1, C);
    else if (fp_mode == euler::FP16O)
      conv_bench(convs2, desc0, input, weights, output1, bias, C);
  }

  // 6. setdown
  for (auto c = 0; c < C; ++c) {
    test::teardown_conv_data(input[c], weights[c], output[c], bias[c],
        input1[c], weights1[c], output1[c], bias1[c], fp_mode, validate_results);
  }

  return 0;
}

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
    ("tweights-preprocess,T", po::value<bool>(&tweights_preprocess), "Preprocess tweights. Default: off")
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
    ("streaming-weights", po::value<int>(&streaming_weights), "Streaming hint for winograd transformed weights")
    ("streaming-input", po::value<int>(&streaming_input), "Streaming hint for winograd transformed input")
    ("streaming-output", po::value<int>(&streaming_output), "Streaming hint for winograd transformed output")
    ("input-format", po::value<std::string>(), "nchw|nhwc|nChw16c. Input data format. Default: nChw16c")
    ("weights-format", po::value<std::string>(), "oihw|OIhw16i16o. Weights data format. Default: OIhw16i16o")
    ("output-format", po::value<std::string>(), "nchw|nhwc|nChw16c. Output data format. Default: nChw16c")
    ("input-as-blocked", po::value<bool>(&input_as_blocked), "on|off. Format input as blocked. Default: off")
    ("weights-as-blocked", po::value<bool>(&weights_as_blocked), "on|off. Format weighs as blocked. Default: off")
    ("output-as-blocked", po::value<bool>(&output_as_blocked), "on|off. Format output as blocked. Default: off")
    ("f16c-opt", po::value<bool>(&f16c_opt), "on|off. With half-precision opt, Default: off")
    ("fp-mode", po::value<int>(&fp_mode), "fp16 UserTypes, Default: FP32")
    ("with-ip-sum", po::value<bool>(&with_ip_sum), "on|off. With inplace sum, Default: off");

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
    else if (fmt_str == "OIhw16i16o")
      weights_format = OIhw16i16o;
    else {
      printf("Error: convolution options: weights-format should be "
             "oihw|OIhw16i16o\n");
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
         "fp_mode=%d, validate_results:%d\n"
         "flt_o:%d, flt_t:%d, blk_i:%d, blk_o:%d, pat_i:%d, pat_o:%d\n"
         "streaming-hint:%d, %d, %d\n"
         "nthreads:%d\n"
         "execution-mode:%x\n",
      mb, ic, ih, iw, oc, oh, ow, kh, kw, ph, pw, sh, sw, dh, dw, with_bias,
      with_relu, with_ip_sum, f16c_opt, fp_mode, validate_results, flt_o,
      flt_t, blk_i, blk_o, pat_i, pat_o, streaming_weights, streaming_input,
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
    {oihw, "oihw"}, {nChw16c, "nChw16c"}, {OIhw16i16o, "OIhw16i16o"}
  };
  printf("input-fmt:%s, weights-fmt:%s, output-fmt:%s\n", fmt_str[input_format],
      fmt_str[weights_format], fmt_str[output_format]);
  printf("input-as-blocked:%d, weights_as_blocked:%d, output_as_blocked:%d\n",
      input_as_blocked, weights_as_blocked, output_as_blocked);
  printf("double_buffering: %d, output_as_input=%d\n", double_buffering, output_as_input);
  printf("tweights_preprocess: %d\n", tweights_preprocess);

  if (mb <= 0 || ic <= 0 || ih <= 0 || iw <= 0 || oc <= 0 || oh <= 0
      || ow <= 0 || kh <= 0 || kw <= 0) {
    printf("Error: convolution options: mb|ic|ih|iw|oc|oh|ow|kh|kw should "
           "greater than 0\n");
    return -1;
  }

  return 0;
}
