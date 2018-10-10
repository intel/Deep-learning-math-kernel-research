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
bool with_bias = true, with_relu = false, with_ip_sum = false;
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
bool flush_cache = false;
int repeated_layer = 1;
bool double_buffering = false;

#define RL_MAX 128
int main(int argc, char **argv)
{
  if (parse_cmd_options(argc, argv))
    return 0;

  // 1, create convolution desc
  eld_conv_t<float> desc;
  desc.dims = {{ mb, ic, ih, iw },
               { oc, ic, kh, kw },
               { mb, oc, oh, ow },
               { oc } };
  desc.formats = {
    input_format, weights_format, output_format
  };
  desc.pads = { ph, ph, pw, pw };
  desc.strides   = { sh, sw };
  desc.with_bias = with_bias;
  desc.with_relu = with_relu;
  desc.with_ip_sum = with_ip_sum;
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

  // 2. setup convolution
  eld_conv_t<float> convs[RL_MAX];
  float *input[RL_MAX], *weights[RL_MAX], *output[RL_MAX], *bias[RL_MAX],
      *ref_output, *dbuffer[2];
  const auto C = validate_results ?
      1 : repeated_layer <= RL_MAX ? repeated_layer : RL_MAX;

  for (auto c = 0; c < C; ++c) {
    convs[c] = desc;
    if (convs[c].setup() != ELD_OK) {
      printf("Fail: Convolution setup error!\n");
      return -1;
    }
    test::prepare_conv_data<float>(
        convs[c], &input[c], &weights[c], &output[c], &bias[c], double_buffering);
  }

  if (validate_results) {
    ref_output = (float *)malloc(convs[0].byte_sizes.output);
    if (desc.with_ip_sum)
      memcpy(ref_output, output[0], convs[0].byte_sizes.output);
  }

  // 3. execute convolution
  for (auto c = 0; c < C; ++c) {
    if (ELX_OK != elx_conv<float>(
               convs[c], output[c], input[c], weights[c], bias[c])) {
      printf("Fail: Convolution execution error!\n");
      return -1;
    }
  }

  // 4. validate results
  if (validate_results) {
    printf("Validation: ");
    if (test::ref_convolution2d<float>(
            convs[0], ref_output, input[0], weights[0], bias[0]))
      printf("Fail: Convolution ref execution error!\n");
    else if (test::compare_conv_results(convs[0], output[0], ref_output))
      printf("Fail: Convolution results not correct!\n");
    else
      printf("Convolution Pass!\n");

    free(ref_output);
  } else {
    // 5. bench
    auto num_ops = test::cal_ops(desc);
    auto N = validate_results ? 1 : test::cal_iterations(num_ops);

    test::timer timer;
    for (auto n = 0; n < N / C; ++n) {
      for (auto c = 0; c < C; ++c) {
        eld_conv_t<float> &_convs = convs[c];
        float *_weights = weights[c], *_bias = bias[c],
            *_input = input[c], *_output = output[c];
        if (double_buffering) {
          if (c % 2 == 0) {
            _input = input[0];
            _output = output[0];
          } else {
            _input = output[0];
            _output = input[0];
          }
        } else if (flush_cache) {
          test::flush_all_memory(
              _convs, _input, _weights, _output, _bias);
        }

        timer.start();
        if (ELX_OK != elx_conv<float>(
            _convs, _output, _input, _weights, _bias)) {
          test::error("Fail: Convolution execution error!\n");
        }
        timer.stop();
      }
    }

    timer.report_tflops("conv", C * (N / C), num_ops);
  }

  // 6. setdown
  for (auto c = 0; c < C; ++c) {
    test::teardown_conv_data(input[c], weights[c], output[c], bias[c]);
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
    ("flush-cache,f", po::value<bool>(&flush_cache), "on|off. Fush cache. Default: off")
    ("repeated-layer,l", po::value<int>(&repeated_layer), "Number of repeated layers. Default: 16")
    ("double-buffering,B", po::value<bool>(&double_buffering), "Double buffering. Default: off")
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
    ("input-format", po::value<std::string>(), "nchw|nChw16c. Input data format. Default: nChw16c")
    ("weights-format", po::value<std::string>(), "oihw|OIhw16i16o. Weights data format. Default: OIhw16i16o")
    ("output-format", po::value<std::string>(), "nchw|nChw16c. Output data format. Default: nChw16c")
    ("input-as-blocked", po::value<bool>(&input_as_blocked), "on|off. Format input as blocked. Default: off")
    ("weights-as-blocked", po::value<bool>(&weights_as_blocked), "on|off. Format weighs as blocked. Default: off")
    ("output-as-blocked", po::value<bool>(&output_as_blocked), "on|off. Format output as blocked. Default: off")
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
    else if (fmt_str == "nChw16c")
      input_format = nChw16c;
    else {
      printf("Error: convolution options: input-format should be "
             "nchw|nChw16c\n");
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
    else if (fmt_str == "nChw16c")
      output_format = nChw16c;
    else {
      printf("Error: convolution options: output-format should be "
             "nchw|nChw16c\n");
      return -1;
    }
  }

  iw = iw == 0 ? ih : iw;
  ow = ow == 0 ? oh : ow;

  printf("Convolution options:\n"
         "mb:%d, ic:%d, ih:%d, iw:%d, oc:%d, oh:%d, ow:%d, kh:%d, kw:%d, "
         "ph:%d, pw:%d, sh:%d, sw:%d, dh:%d, dw:%d\n"
         "with_bias:%d, with_relu:%d, with_ip_sum:%d, validate_results:%d\n"
         "flt_o:%d, flt_t:%d, blk_i:%d, blk_o:%d, pat_i:%d, pat_o:%d\n"
         "streaming-hint:%d, %d, %d\n"
         "nthreads:%d\n"
         "execution-mode:%x\n",
      mb, ic, ih, iw, oc, oh, ow, kh, kw, ph, pw, sh, sw, dh, dw,
      with_bias, with_relu, with_ip_sum, validate_results,
      flt_o, flt_t, blk_i, blk_o, pat_i, pat_o,
      streaming_weights, streaming_input, streaming_output,
      nthreads, execution_mode);

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

  std::unordered_map<int, const char *> fmt_str { {nchw, "nchw"},
    {oihw, "oihw"}, {nChw16c, "nChw16c"}, {OIhw16i16o, "OIhw16i16o"}
  };
  printf("input-fmt:%s, weights-fmt:%s, output-fmt:%s\n", fmt_str[input_format],
      fmt_str[weights_format], fmt_str[output_format]);
  printf("input-as-blocked:%d, weights_as_blocked:%d, output_as_blocked:%d\n",
      input_as_blocked, weights_as_blocked, output_as_blocked);
  printf("flush_cache: %d, repeated_layer: %d\n", flush_cache, repeated_layer);
  printf("double_buffering: %d\n", double_buffering);

  if (mb <= 0 || ic <= 0 || ih <= 0 || iw <= 0 || oc <= 0 || oh <= 0
      || ow <= 0 || kh <= 0 || kw <= 0) {
    printf("Error: convolution options: mb|ic|ih|iw|oc|oh|ow|kh|kw should "
           "greater than 0\n");
    return -1;
  }

  return 0;
}
