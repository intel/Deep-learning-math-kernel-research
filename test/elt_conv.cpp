#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <boost/program_options.hpp>
#include "elt_utils.hpp"
#include "elt_conv_utils.hpp"
#include "euler.hpp"
#include <iostream>

using namespace euler;
namespace po = boost::program_options;
int parse_cmd_options(int, char **);

// Covolution options
int mb = 0, ic = 0, ih = 0, iw = 0, oc = 0, oh = 0, ow = 0, kh = 3, kw = 3;
int ph = 1, pw = 1, sh = 1, sw = 1, dh = 1, dw = 1;
bool with_bias = true, with_relu = false;
int prop_kind = forward_inference, alg = CONV_WINOGRAD;
int input_fmt = nChw16c, weights_fmt = OIhw16i16o, output_fmt = nChw16c;
int nteams = 0, nthreads = 0;
int execution_mode = 0;
int blk_i = 0, blk_o = 0, blk_t = 0;
int pat_i = 1, pat_o = 1;
int tile_size = 5;
int streaming_weights = 0, streaming_input = 0, streaming_output = 0;

bool validate_results = false;

int main(int argc, char **argv)
{
  if (parse_cmd_options(argc, argv))
    return 0;

  // 1, create convolution desc
  eld_conv_t<float> desc;
  desc.dims = { .input   = { mb, ic, ih, iw },
                .weights = { oc, ic, kh, kw },
                .output  = { mb, oc, oh, ow },
                .bias    = { oc } };
  desc.formats = { .input = input_fmt, .weights = weights_fmt, .output = output_fmt };
  desc.pads = { ph, ph, pw, pw };
  desc.with_bias = with_bias;
  desc.with_relu = with_relu;
  desc.algorithm = alg;
  desc.tile_size = tile_size;
  desc.prop_kind = prop_kind;
  desc.threading = { nteams, nthreads };
  desc.execution_mode = execution_mode;
  desc.blocking = { blk_i, blk_o, blk_t };
  desc.partition = { pat_i, pat_o };
  desc.streaming_hint = { streaming_weights, streaming_input, streaming_output };

  if (desc.setup() != ELD_OK) {
    printf("Fail: Convolution setup error!\n");
    return -1;
  }

  // 2. prepare data
  float *input, *weights, *output, *bias;
  test::prepare_conv_data<float>(desc, &input, &weights, &output, &bias);

  // 3. execute convolution
  int iterations = validate_results ? 1: 6400 / mb;
  size_t num_ops = test::cal_ops(desc);
  time_start(conv);
  for (int n = 0; n < iterations; ++n) {
    if (ELX_OK != elx_conv<float>(desc, output, input, weights, bias)) {
      printf("Fail: Convolution execlution error!\n");
      test::teardown_conv_data(input, weights, output, bias);
      return -1;
    }
  }
  time_end(conv, iterations, num_ops);

  // 4. cosim, setdown
  if (validate_results) {
    printf("Validation: ");
    float *ref_output = (float *)memalign(64, desc.byte_sizes.output);
    if (input_fmt == nChw16c) {
      if (test::ref_convolution2d_block16<float>(
              desc, ref_output, input, weights, bias))
        printf("Fail: Convolution ref execution error!\n");
      else if (test::compare_conv_results_block16(desc, output, ref_output))
        printf("Fail: Convolution results not correct!\n");
      else
        printf("Convolution Pass!\n");
    } else if (input_fmt == nchw) {
      if (test::ref_convolution2d<float>(
              desc, ref_output, input, weights, bias))
        printf("Fail: Convolution ref execution error!\n");
      else if (test::compare_conv_results(desc, output, ref_output))
        printf("Fail: Convolution results not correct!\n");
      else
        printf("Convolution Pass!\n");

    }
    free(ref_output);
  }
  test::teardown_conv_data(input, weights, output, bias);

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
    ("alg,a", po::value<std::string>(), "wino|direct. Algorithm. Default: wino")
    ("tile-size", po::value<int>(&tile_size), "Winograd tile size: 5")
    ("nteams", po::value<int>(&nteams), "Number of thread team")
    ("nthreads", po::value<int>(&nthreads), "Number of threads per team")
    ("execution-mode", po::value<std::string>(), "Execution mode")
    ("blk-i", po::value<int>(&blk_i), "IC blocking")
    ("blk-o", po::value<int>(&blk_o), "OC blocking")
    ("blk-t", po::value<int>(&blk_t), "Tile blocking")
    ("pat-i", po::value<int>(&pat_i), "Partition on ic")
    ("pat-o", po::value<int>(&pat_o), "Partition on oc")
    ("streaming-weights", po::value<int>(&streaming_weights), "Streaming hint for winograd transformed weights")
    ("streaming-input", po::value<int>(&streaming_input), "Streaming hint for winograd transformed input")
    ("streaming-output", po::value<int>(&streaming_output), "Streaming hint for winograd transformed output")
    ("data-fmt", po::value<std::string>(), "Data format. plain | block16. Default: block16");


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
    if (alg_str == "WINO")
      alg = CONV_WINOGRAD;
    else if (alg_str == "DIRECT")
      alg = CONV_DIRECT;
    else {
      printf("Error: convolution options: alg should be wino|direct\n");
      return -1;
    }
  }
  if (vm.count("execution-mode")) {
    std::stringstream interpreter;
    interpreter << std::hex << vm["execution-mode"].as<std::string>();
    interpreter >> execution_mode;
  }
  if (vm.count("data-fmt")) {
    std::string fmt_str = vm["data-fmt"].as<std::string>();
    if (fmt_str == "plain") {
      input_fmt = nchw;
      weights_fmt = oihw;
      output_fmt = nchw;
    } else if (fmt_str == "block16"){
      input_fmt = nChw16c;
      weights_fmt = OIhw16i16o;
      output_fmt = nChw16c;
    } else {
      printf("Error: convolution options: data-format should be plain | block16\n");
      return -1;
    }
  }

  iw = iw == 0 ? ih : iw;
  ow = ow == 0 ? oh : ow;

  printf("Convolution options:\n"
         "mb:%d, ic:%d, ih:%d, iw:%d, oc:%d, oh:%d, ow:%d, kh:%d, kw:%d, "
         "ph:%d, pw:%d, sh:%d, sw:%d, dh:%d, dw:%d\n"
         "with_bias:%d, with_relu:%d, validate_results:%d\n"
         "blk_i:%d, blk_o:%d, blk_t:%d, pat_i:%d, pat_o:%d\n"
         "streaming-hint:%d, %d, %d\n"
         "nteams:%d, nthreads:%d\n"
         "execution-mode:%x\n",
      mb, ic, ih, iw, oc, oh, ow, kh, kw, ph, pw, sh, sw, dh, dw,
      with_bias, with_relu, validate_results, blk_i, blk_o, blk_t, pat_i, pat_o,
      streaming_weights, streaming_input, streaming_output,
      nteams, nthreads, execution_mode);

  const char *prop_kind_str[] = {
    [forward_inference] = "forward_inference",
    [forward_training] = "forward_training",
    [backward_data] = "backward_data",
    [backward_weights] = "backward_weights"
  };
  printf("prop_kind:%s\n", prop_kind_str[prop_kind]);

  const char *alg_str[] = {
    [CONV_DIRECT] = "CONV_DIRECT",
    [CONV_WINOGRAD] = "CONV_WINOGRAD"
  };
  printf("alg:%s, tile-size=%d\n", alg_str[alg], tile_size);

  const char *fmt_str[] = { [nchw] = "nchw",
    [oihw] = "oihw",
    [nChw16c] = "nChw16c",
    [OIhw16i16o] = "OIhw16i16o"
  };
  printf("input-fmt:%s, weights-fmt:%s, output-fmt:%s\n", fmt_str[input_fmt],
      fmt_str[weights_fmt], fmt_str[output_fmt]);

  if (mb <= 0 || ic <= 0 || ih <= 0 || iw <= 0 || oc <= 0 || oh <= 0
      || ow <= 0 || kh <= 0 || kw <= 0) {
    printf("Error: convolution options: mb|ic|ih|iw|oc|oh|ow|kh|kw should "
           "greater than 0\n");
    return -1;
  }

  return 0;
}
