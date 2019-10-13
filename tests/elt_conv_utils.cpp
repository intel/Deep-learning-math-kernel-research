#include <algorithm>
#include <math.h>
#include <omp.h>
#include <memory.h>
#include <random>
#include <fstream>
#include "elt_conv_utils.hpp"
#include "el_intrin.hpp"
#include "el_parallel.hpp"
#include "euler.hpp"
#include "euler_reorder.hpp"

#define MAX_PRINT_ERRORS (20)

namespace euler {
namespace test {
void read_blob_data(void *buffer, const char *filename, size_t size) {
  std::ifstream input_file(filename, std::ios::in | std::ios::binary);
  if (input_file) {
    input_file.read((char *)buffer, size);
    input_file.close();
  }
}
template <typename InputType, typename WeightsType, typename OutputType,
          typename BiasType>
void load_conv_data(eld_conv_t &desc, InputType *input, WeightsType *weights,
                    BiasType *bias, const char *input_file,
                    const char *weights_file, const char *bias_file,
                    int input_format, int weights_format) {
  InputType *nchw_input;
  WeightsType *oihw_weights;
  memalign64(&nchw_input, desc.byte_sizes.input);
  memalign64(&oihw_weights, desc.byte_sizes.weights);

  auto dims = desc.dims;
  auto input_sz = dims.n * dims.ic * dims.ih * dims.iw;
  auto weights_sz =
      dims.g * dims.oc/dims.g * dims.ic/dims.g * dims.kh * dims.kw;

  if (dims.g != 1) {
    error("groups not support in load real data");
  }
  if (input_file != nullptr && input != nullptr) {
    if (nChw16c == input_format) {
      read_blob_data(nchw_input, input_file, input_sz * sizeof(InputType));
      reorder<InputType, nChw16c, nchw>(input, nchw_input, dims.n,
                                        dims.ic, dims.ih, dims.iw);
    } else if (nhwc == input_format) {
      read_blob_data(nchw_input, input_file, input_sz * sizeof(InputType));
      reorder<InputType, nhwc, nchw>(input, nchw_input, dims.n,
                                     dims.ic, dims.ih, dims.iw);
    } else if (nchw == input_format) {
      read_blob_data(input, input_file, input_sz * sizeof(InputType));
    } else {
      printf("Error: unsupported input format \n");
      exit(1);
    }

  }
  free(nchw_input);

  if (weights_file != nullptr && weights != nullptr) {

    if (OIhw16i16o == weights_format) {
      read_blob_data(oihw_weights, weights_file,
                     weights_sz * sizeof(WeightsType));
      reorder<WeightsType, OIhw16i16o, oihw>(weights, oihw_weights,
                                             dims.oc, dims.ic,
                                             dims.kh, dims.kw);
    } else if (hwio == weights_format) {
      read_blob_data(oihw_weights, weights_file,
                     weights_sz * sizeof(WeightsType));
      reorder<WeightsType, hwio, oihw>(weights, oihw_weights, dims.oc,
                                       dims.ic, dims.ih, dims.ow);
    } else if (oihw == weights_format) {
      read_blob_data(weights, weights_file, weights_sz * sizeof(WeightsType));
    } else {
      printf("Error: unsupported weights format \n");
      exit(1);
    }

  }
  free(oihw_weights);
}

__thread unsigned int seed;
template <typename InputType, typename WeightsType, typename OutputType,
          typename BiasType>
void prepare_conv_data(eld_conv_t &desc_ref, eld_conv_t &desc, float *input_ref,
                       float *weights_ref, float *output_ref, float *bias_ref,
                       InputType **input, WeightsType **weights,
                       OutputType **output, BiasType **bias,
                       const char *input_file, const char *weights_file,
                       const char *bias_file, int input_format, int weights_format,
                       bool reuse_inout, int data_type_cfg, bool f16c_opt,
                       bool validate_results) {
  seed = time(nullptr);

  // reference data initialization
  if (input_file != nullptr && weights_file != nullptr) {
    load_conv_data<float, float, float>(
        desc_ref, input_ref, weights_ref, bias_ref, input_file, weights_file,
        bias_file, input_format, weights_format);
  } else if (input_file == nullptr && weights_file == nullptr) {
#define RAND() rand_r(&seed)
    std::default_random_engine gen;
    // std::normal_distribution<float> dInput(-4.0, 20.0);
    std::normal_distribution<float> dWeights(-1.0, 1.0);
    std::normal_distribution<float> dInput_mu_0_sigma_3(0.0, 3.0);
    std::normal_distribution<float> dInput_mu_15_sigma_3(15.0, 3.0);
    std::normal_distribution<float> dWeights_mu_0_sigma_0_1(0.0, 0.1);

    // ref input
    for (size_t i = 0; i < desc_ref.sizes.input; i++) {
      if (data_type_cfg == euler::test::FP32 || data_type_cfg == euler::test::FP16) {
        input_ref[i] = RAND() % 20 - 12; // (-12, 8)
      } else if (data_type_cfg == euler::test::U8F32U8F32 ||
                 data_type_cfg == euler::test::U8F32S8F32 ||
                 data_type_cfg == euler::test::U8F32F32F32) {
        if (validate_results)
          input_ref[i] = dInput_mu_15_sigma_3(gen);
        else
          input_ref[i] = RAND() % 19;
        if (input_ref[i] < 0)
          input_ref[i] = 0;
      } else if (data_type_cfg == euler::test::U8F32U8F32z ||
                 data_type_cfg == euler::test::U8F32S8F32z ||
                 data_type_cfg == euler::test::U8F32F32F32z) {
        if (validate_results)
          input_ref[i] = dInput_mu_0_sigma_3(gen);
        else
          input_ref[i] = RAND() % 19;
      }
    }

    // ref weights
    for (size_t i = 0; i < desc_ref.sizes.weights; i++) {
      if (data_type_cfg == euler::test::FP32 || data_type_cfg == euler::test::FP16) {
        weights_ref[i] = dWeights(gen);
      } else if (data_type_cfg == euler::test::U8F32U8F32 ||
                 data_type_cfg == euler::test::U8F32S8F32 ||
                 data_type_cfg == euler::test::U8F32F32F32 ||
                 data_type_cfg == euler::test::U8F32U8F32z ||
                 data_type_cfg == euler::test::U8F32S8F32z ||
                 data_type_cfg == euler::test::U8F32F32F32z) {
        if (validate_results)
          weights_ref[i] = dWeights_mu_0_sigma_0_1(gen);
        else
          weights_ref[i] = RAND() % 19;
      }
      if (desc_ref.with_relu && i % 7 == 1)
        weights_ref[i] = -weights_ref[i];
    }

    // ref bias
    estl::parallel_for<1>([&](size_t i) {
      bias_ref[i] = RAND() % 100;
    }, desc_ref.sizes.bias);

    // ref ouput
    if (desc_ref.with_ip_sum) {
      estl::parallel_for<1>([&](size_t i) {
        output_ref[i] = RAND() % 10;
      }, desc_ref.sizes.output);
    }
  } else {
    printf("Invalid real data file ...\n");
    exit(1);
  }

  // real data initialization
  size_t input_size = desc_ref.byte_sizes.input;
  size_t output_size = desc_ref.byte_sizes.output;
  if (reuse_inout) {
    input_size = output_size =
        std::max(desc_ref.byte_sizes.input, desc_ref.byte_sizes.output);
  }

  if (data_type_cfg == euler::test::FP32) {
    memalign64(input, input_size);
    memalign64(output, output_size);
    memalign64(weights, desc_ref.byte_sizes.weights);
    memalign64(bias, desc_ref.byte_sizes.bias);
  } else if (data_type_cfg == euler::test::FP16) {
    memalign64(input, input_size / 2);
    memalign64(output, output_size / 2);
    memalign64(weights, desc_ref.byte_sizes.weights / 2);
    memalign64(bias, desc_ref.byte_sizes.bias / 2);
  } else if (data_type_cfg == euler::test::FP16O) {
    memalign64(input, input_size);
    memalign64(weights, desc_ref.byte_sizes.weights);
    memalign64(bias, desc_ref.byte_sizes.bias);
    memalign64(output, output_size / 2);
  } else if (data_type_cfg == euler::test::U8F32U8F32 ||
             data_type_cfg == euler::test::U8F32U8F32z) {
    memalign64(input, input_size / 4);
    memalign64(weights, desc_ref.byte_sizes.weights);
    memalign64(bias, desc_ref.byte_sizes.bias);
    memalign64(output, output_size / 4);
  } else if (data_type_cfg == euler::test::U8F32S8F32 ||
             data_type_cfg == euler::test::U8F32S8F32z) {
    memalign64(input, input_size / 4);
    memalign64(weights, desc_ref.byte_sizes.weights);
    memalign64(bias, desc_ref.byte_sizes.bias);
    memalign64(output, output_size / 4);
  } else if (data_type_cfg == euler::test::U8F32F32F32 ||
             data_type_cfg == euler::test::U8F32F32F32z) {
    memalign64(input, input_size / 4);
    memalign64(weights, desc_ref.byte_sizes.weights);
    memalign64(bias, desc_ref.byte_sizes.bias);
    memalign64(output, output_size);
  }

  // scale initialization
  desc.input_quant.scale = 1.0;
  desc.input_quant.z = 0.0;
  desc.output_quant.scale = 1.0;
  desc.output_quant.z = 0.0;

#define PRECISION_REPRESENTATION_8B 255
#define PRECISION_REPRESENTATION_7B 127
  auto trans_input_scale = [&]() {
    float *_output_ref;
    if (desc_ref.with_ip_sum) {
      memalign64(&_output_ref, desc_ref.byte_sizes.output);
      estl::parallel_for<1>([&](size_t i) {
        _output_ref[i] = output_ref[i];
      }, desc_ref.sizes.output);
    } else {
      _output_ref = output_ref;
    }

    if (ELX_OK !=
        elx_conv(desc_ref, _output_ref, input_ref, weights_ref, bias_ref)) {
      test::error("Fail: Convolution execution error!\n");
    }

    float *tinput = (float *)desc_ref.scratch_pad;
    float min = tinput[0], max = tinput[0];
    size_t t = (desc.dims.oh + desc.tile_size - 3) /
               (desc.tile_size - 3 + 1) *
               (desc.dims.ow + desc.tile_size - 3) /
               (desc.tile_size - 3 + 1) * desc.dims.n;
    size_t A = desc.tile_size;
    size_t K = 3;
    size_t IC = desc.dims.ic;
    size_t tinput_size = A * A * IC * t;
    for (size_t i = 1; i < tinput_size; i++) {
      min = tinput[i] < min ? tinput[i] : min;
      max = tinput[i] > max ? tinput[i] : max;
    }

    auto diff = max - min + 0.000001;
    desc.wino_tinput_quant.scale = diff / PRECISION_REPRESENTATION_7B;
    desc.wino_tinput_quant.z = -min * PRECISION_REPRESENTATION_7B / diff;

    printf("tinput max %f min %f scale %f z %f\n", max, min,
           desc.wino_tinput_quant.scale, desc.wino_tinput_quant.z);

    if (desc_ref.with_ip_sum)
      free(_output_ref);
  };

  auto input_scale = [&](float &iscale) {
    float abs_max = input_ref[0] > 0 ? input_ref[0] : -input_ref[0];
    for (size_t i = 1; i < desc_ref.sizes.input; i++) {
      auto abs_cur = input_ref[i] > 0 ? input_ref[i] : -input_ref[i];
      abs_max = abs_cur > abs_max ? abs_cur : abs_max;
    }
    // U8
    if (data_type_cfg == euler::test::U8F32U8F32 ||
        data_type_cfg == euler::test::U8F32S8F32 ||
        data_type_cfg == euler::test::U8F32F32F32) {
      iscale = abs_max / PRECISION_REPRESENTATION_7B;
      desc.input_quant.z = 0;
    } else if (data_type_cfg == euler::test::U8F32U8F32z ||
        data_type_cfg == euler::test::U8F32S8F32z ||
        data_type_cfg == euler::test::U8F32F32F32z) {
      iscale = abs_max / PRECISION_REPRESENTATION_7B;
      desc.input_quant.z = 128.0;
    } else {
      // S8
      iscale = abs_max / PRECISION_REPRESENTATION_7B;
      desc.input_quant.z = 0;
    }
    desc.input_quant.scale = iscale;
    printf("input abs_max %f scale %f\n", abs_max, iscale);
  };

  auto output_scale = [&](float &oscale, float &opscale, float &oz) {
    float *_output_ref;
    if (desc_ref.with_ip_sum) {
      memalign64(&_output_ref, desc_ref.byte_sizes.output);
      estl::parallel_for<1>([&](size_t i) {
        _output_ref[i] = output_ref[i];
      }, desc_ref.sizes.output);
    } else {
      _output_ref = output_ref;
    }

    if (test::ref_convolution2d<float>(desc_ref, _output_ref, input_ref,
                                       weights_ref, bias_ref)) {
      printf("Fail: scale initialization. Convolution ref execution error!\n");
      exit(1);
    }

    float min = _output_ref[0], max = _output_ref[0];
    for (size_t i = 1; i < desc_ref.sizes.output; i++) {
      min = _output_ref[i] < min ? _output_ref[i] : min;
      max = _output_ref[i] > max ? _output_ref[i] : max;
    }
    float abs_cur = min > 0 ? min : -min;
    float abs_max = max > abs_cur ? max : abs_cur;

    if (desc_ref.with_relu) {
      oscale = abs_max / PRECISION_REPRESENTATION_7B;
      oz = 0.0;
      desc.output_quant.scale = oscale;
      desc.output_quant.z = oz;
    } else if (data_type_cfg == euler::test::U8F32S8F32 ||
               data_type_cfg == euler::test::U8F32S8F32z) {
      oscale = abs_max / PRECISION_REPRESENTATION_7B;
      oz = 0.0;
      desc.output_quant.scale = oscale;
      desc.output_quant.z = oz;
    } else {
      auto diff = max - min + 0.000001;
      desc.output_quant.scale = diff / PRECISION_REPRESENTATION_7B;
      desc.output_quant.z = -min * PRECISION_REPRESENTATION_7B / diff;
    }

    printf("output abs_max %f scale %f\n", abs_max, desc.output_quant.scale);

    min = output_ref[0];
    max = output_ref[0];
    if (desc_ref.with_ip_sum) {
      for (size_t i = 1; i < desc_ref.sizes.output; i++) {
        min = output_ref[i] < min ? output_ref[i] : min;
        max = output_ref[i] > max ? output_ref[i] : max;
      }
      abs_cur = min > 0 ? min : -min;
      abs_max = max > abs_cur ? max : abs_cur;
      opscale = abs_max / PRECISION_REPRESENTATION_7B;
      auto sum_operand_scale = abs_max / PRECISION_REPRESENTATION_7B;
      desc.sum_quant.scale = sum_operand_scale / desc.output_quant.scale;

      printf("sum operand abs_max %f sum scale %f\n", abs_max, desc.sum_quant.scale);
    }

    if (desc_ref.with_ip_sum)
      free(_output_ref);
  };

  auto rounding_to_nearest_even = [](float f32) -> int32_t {
    int32_t i32 = (int32_t)f32;
    if (i32 >= 0) {
      if (i32 % 2) {
        return f32 - (float)i32 <= 0.5 ? i32 : i32 + 1;
      } else {
        return f32 - (float)i32 >= 0.5 ? i32 + 1 : i32;
      }
    } else {
      if (i32 % 2) {
        return (float)i32 - f32 <= 0.5 ? i32 : i32 - 1;
      } else {
        return (float)i32 - f32 >= 0.5 ? i32 - 1 : i32;
      }
    }
  };

  // input
  float iscale = 1.0;
  if (validate_results) {
    if (data_type_cfg == euler::test::U8F32U8F32 ||
        data_type_cfg == euler::test::U8F32S8F32 ||
        data_type_cfg == euler::test::U8F32F32F32 ||
        data_type_cfg == euler::test::U8F32U8F32z ||
        data_type_cfg == euler::test::U8F32S8F32z ||
        data_type_cfg == euler::test::U8F32F32F32z) {
      input_scale(iscale);
      if (desc.algorithm == CONV_WINOGRAD) {
        trans_input_scale();
      }
    }
  }
  estl::parallel_for<1>([&](size_t i) {
    if (data_type_cfg == euler::test::FP32)
      (*input)[i] = input_ref[i];
    else if (data_type_cfg == euler::test::FP16)
      (*input)[i] = float_2_half(input_ref[i]);
    else if (data_type_cfg == euler::test::FP16O)
      (*input)[i] = input_ref[i];
    else if (data_type_cfg == euler::test::U8F32U8F32 ||
             data_type_cfg == euler::test::U8F32S8F32 ||
             data_type_cfg == euler::test::U8F32F32F32 ||
             data_type_cfg == euler::test::U8F32U8F32z ||
             data_type_cfg == euler::test::U8F32S8F32z ||
             data_type_cfg == euler::test::U8F32F32F32z)
      (*input)[i] = (uint8_t)rounding_to_nearest_even(input_ref[i] / iscale +
                                                      desc.input_quant.z);
  }, desc_ref.sizes.input);

  // weights
  estl::parallel_for<1>([&](size_t i) {
    if (data_type_cfg == euler::test::FP32)
      (*weights)[i] = weights_ref[i];
    else if (data_type_cfg == euler::test::FP16)
      (*weights)[i] = float_2_half(weights_ref[i]);
    else if (data_type_cfg == euler::test::FP16O)
      (*weights)[i] = weights_ref[i];
    else if (data_type_cfg == euler::test::U8F32U8F32 ||
             data_type_cfg == euler::test::U8F32S8F32 ||
             data_type_cfg == euler::test::U8F32F32F32||
             data_type_cfg == euler::test::U8F32U8F32z ||
             data_type_cfg == euler::test::U8F32S8F32z ||
             data_type_cfg == euler::test::U8F32F32F32z)
      (*weights)[i] = weights_ref[i];
  }, desc_ref.sizes.weights);

  // bias
  estl::parallel_for<1>([&](size_t i) {
    if (data_type_cfg == euler::test::FP32)
      (*bias)[i] = bias_ref[i];
    else if (data_type_cfg == euler::test::FP16)
      (*bias)[i] = float_2_half(bias_ref[i]);
    else if (data_type_cfg == euler::test::FP16O)
      (*bias)[i] = bias_ref[i];
    else if (data_type_cfg == euler::test::U8F32U8F32 ||
             data_type_cfg == euler::test::U8F32S8F32 ||
             data_type_cfg == euler::test::U8F32F32F32 ||
             data_type_cfg == euler::test::U8F32U8F32z ||
             data_type_cfg == euler::test::U8F32S8F32z ||
             data_type_cfg == euler::test::U8F32F32F32z)
      (*bias)[i] = bias_ref[i];
  }, desc_ref.sizes.bias);

  // output
  float oscale = 1.0, opscale = 1.0, oz = 0.0;
  if (validate_results) {
    if (data_type_cfg == euler::test::U8F32U8F32 ||
        data_type_cfg == euler::test::U8F32S8F32 ||
        data_type_cfg == euler::test::U8F32U8F32z ||
        data_type_cfg == euler::test::U8F32S8F32z)
      output_scale(oscale, opscale, oz);
  }
  if (desc.with_ip_sum) {
    estl::parallel_for<1>([&](size_t i) {
      if (data_type_cfg == euler::test::FP32)
        (*output)[i] = output_ref[i];
      else if (data_type_cfg == euler::test::FP16)
        (*output)[i] = float_2_half(output_ref[i]);
      else if (data_type_cfg == euler::test::FP16O)
        (*output)[i] = output_ref[i];
      else if (data_type_cfg == euler::test::U8F32U8F32 ||
               data_type_cfg == euler::test::U8F32S8F32 ||
               data_type_cfg == euler::test::U8F32U8F32z ||
               data_type_cfg == euler::test::U8F32S8F32z) {
        (*output)[i] =
            (int8_t)rounding_to_nearest_even(
            output_ref[i] / opscale);
      }
    }, desc_ref.sizes.output);
  }
}

int compare_conv_results(eld_conv_t &desc, float *out, float *ref,
                         int data_type_cfg, bool is_int8_lp,
                         bool with_real_data) {
  double acc = is_int8_lp ? (with_real_data ? 1e-1 : 1e-2) : 1e-5;

  if (desc.formats.output == nhwc) {
    acc = desc.with_relu ? 1.0 : acc;
    return __compare_conv_results_nhwc(desc, out, ref, data_type_cfg, acc);
  } else if (desc.formats.output == nchw) {
    return __compare_conv_results_nchw(desc, out, ref, data_type_cfg, acc);
  } else {
    return __compare_conv_results_blocked(desc, out, ref, data_type_cfg, acc);
  }
}

int __compare_conv_results_blocked(eld_conv_t &desc, float *out, float *ref,
                                   int data_type_cfg, double acc) {
  const int V = 16;
  auto dims = desc.dims;
  int C = ALIGNUP(dims.oc, V) / V;
  int Or = dims.oc % V ? dims.oc % V : V;

  MD5(float, aout, out, dims.n, C, dims.oh, dims.ow, V);
  MD5(float, aref, ref, dims.n, C, dims.oh, dims.ow, V);

  size_t errors = 0;

  iter_each(_n, dims.n) {
    iter_each(_C, C) {
      iter_each(_h, dims.oh) {
        iter_each(_w, dims.ow) {
          int v = _C == C - 1 ? Or : V;
          iter_each(_v, v) {
            auto real = md5(aout, _n, _C, _h, _w, _v);
            double delta = fabs(real - md5(aref, _n, _C, _h, _w, _v));
            if (md5(aref, _n, _C, _h, _w, _v) == 0 || real == 0) {
              if (delta > acc) {
                if (errors < MAX_PRINT_ERRORS) {
                  printf("Not equal!: [%d][%d][%d][%d][%d]: %f != %f (ref), "
                         "delta=%g, acc=%g\n",
                         _n, _C, _h, _w, _v, real,
                         md5(aref, _n, _C, _h, _w, _v), delta, acc);
                }
                errors++;
              }
            } else {
              double rel_diff = delta / fabs(md5(aref, _n, _C, _h, _w, _v));
              if (rel_diff > acc) {
                if (errors < MAX_PRINT_ERRORS) {
                  printf("Not equal!: [%d][%d][%d][%d][%d]: %f != %f (ref), "
                         "delta=%g, rel_diff=%g\n",
                         _n, _C, _h, _w, _v, real,
                         md5(aref, _n, _C, _h, _w, _v), delta, rel_diff);
                }
                errors++;
              }
            }
          }
        }
      }
    }
  }

  if (errors > 0) {
    printf("%s: Error: number of errors: %ld/%ld, percentage: %f%%\n",
           desc.name.c_str(), errors, desc.sizes.output,
           ((errors * 1.0) / desc.sizes.output) * 100.0);
    return -1;
  }
  return 0;
}

int __compare_conv_results_nchw(eld_conv_t &desc, float *out, float *ref,
                                int data_type_cfg, double acc) {
  auto dims = desc.dims;
  MD4(float, aout, out, dims.n, dims.oc, dims.oh, dims.ow);
  MD4(float, aref, ref, dims.n, dims.oc, dims.oh, dims.ow);

  size_t errors = 0;

  iter_each(_n, dims.n) {
    iter_each(_c, dims.oc) {
      iter_each(_h, dims.oh) {
        iter_each(_w, dims.ow) {
          auto real = md4(aout, _n, _c, _h, _w);
          double delta = fabs(real - md4(aref, _n, _c, _h, _w));
          if (real == 0 || md4(aref, _n, _c, _h, _w) == 0) {
            if (delta > acc) {
              if (errors < MAX_PRINT_ERRORS) {
                printf("Not equal!: [%d][%d][%d][%d]: %f != %f (ref), "
                       "delta=%g, acc=%g\n",
                       _n, _c, _h, _w, real, md4(aref, _n, _c, _h, _w), delta,
                       acc);
              }
              errors++;
            }
          } else {
            double rel_diff = delta / fabs(md4(aref, _n, _c, _h, _w));
            if (rel_diff > acc) {
              if (errors < MAX_PRINT_ERRORS) {
                printf("Not equal!: [%d][%d][%d][%d]: %f != %f (ref), "
                       "delta=%g, rel_diff=%g\n",
                       _n, _c, _h, _w, real, md4(aref, _n, _c, _h, _w), delta,
                       rel_diff);
              }
              errors++;
            }
          }
        }
      }
    }
  }

  if (errors > 0) {
    printf("%s: Error: number of errors: %ld/%ld, percentage: %f%%\n",
           desc.name.c_str(), errors, desc.sizes.output,
           ((errors * 1.0) / desc.sizes.output) * 100.0);
    return -1;
  }
  return 0;
}

int __compare_conv_results_nhwc(eld_conv_t &desc, float *out, float *ref,
                                int data_type_cfg, double acc) {
  auto dims = desc.dims;
  MD4(float, aout, out, dims.n, dims.oh, dims.ow, dims.oc);
  MD4(float, aref, ref, dims.n, dims.oh, dims.ow, dims.oc);

  size_t errors = 0;

  iter_each(_n, dims.n) {
    iter_each(_c, dims.oc) {
      iter_each(_h, dims.oh) {
        iter_each(_w, dims.ow) {
          auto real = md4(aout, _n, _h, _w, _c);
          double delta = fabs(real - md4(aref, _n, _h, _w, _c));
          if (real == 0 || md4(aref, _n, _h, _w, _c) == 0) {
            if (delta > acc) {
              if (errors < MAX_PRINT_ERRORS) {
                printf("Not equal!: [%d][%d][%d][%d]: %f != %f (ref), "
                       "delta=%g, acc=%g\n",
                       _n, _c, _h, _w, real, md4(aref, _n, _h, _w, _c), delta,
                       acc);
              }
              errors++;
            }
          } else {
            double rel_diff = delta / fabs(md4(aref, _n, _h, _w, _c));
            if (rel_diff > acc) {
              if (errors < MAX_PRINT_ERRORS) {
                printf("Not equal!: [%d][%d][%d][%d]: %f != %f (ref), "
                       "delta=%g, rel_diff=%g\n",
                       _n, _c, _h, _w, real, md4(aref, _n, _h, _w, _c), delta,
                       rel_diff);
              }
              errors++;
            }
          }
        }
      }
    }
  }

  if (errors > 0) {
    printf("%s: Error: number of errors: %ld/%ld, percentage: %f%%\n",
           desc.name.c_str(), errors, desc.sizes.output,
           ((errors * 1.0) / desc.sizes.output) * 100.0);
    return -1;
  }
  return 0;
}

size_t cal_ops(eld_conv_t &desc) {
  size_t num_ops = 0;

  iter_each(_oh, desc.dims.oh) {
    iter_each(_ow, desc.dims.ow) {
      iter_each(_kh, desc.dims.kh) {
        int _ih = _oh * desc.strides.h - desc.pads.t + _kh * desc.dilations.h;
        if (_ih < 0 || _ih >= desc.dims.ih)
          continue;
        iter_each(_kw, desc.dims.kw) {
          int _iw = _ow * desc.strides.w - desc.pads.l + _kw * desc.dilations.w;
          if (_iw < 0 || _iw >= desc.dims.iw)
            continue;
          num_ops += 1;
        }
      }
    }
  }
  int ic = desc.dims.ic / desc.dims.g;
  int oc = desc.dims.oc / desc.dims.g;
  return num_ops * desc.dims.n * desc.dims.g * ic * oc * 2;
}

int cal_iterations(size_t num_ops) {
  float iter = 5e12 / num_ops;
  return std::min(1024, std::max((int)iter, 64));
}

template <typename InputType, typename WeightsType, typename OutputType,
          typename BiasType>
int ref_convolution2d(eld_conv_t &desc, OutputType *output, InputType *input,
                      WeightsType *weights, BiasType *bias) {
  int n = desc.dims.n;
  int g = desc.dims.g;
  int ic = desc.dims.ic / g;
  int oc = desc.dims.oc / g;
  int ih = desc.dims.ih;
  int iw = desc.dims.iw;
  int oh = desc.dims.oh;
  int ow = desc.dims.ow;
  int kh = desc.dims.kh;
  int kw = desc.dims.kw;
  int sh = desc.strides.h;
  int sw = desc.strides.w;
  int pt = desc.pads.t;
  int pl = desc.pads.l;
  int dh = desc.dilations.h;
  int dw = desc.dilations.w;

  InputType *tinput = nullptr, *tweights = nullptr, *toutput = nullptr;
  if (desc.formats.input == nChw16c) {
    tinput = (InputType *)malloc(desc.byte_sizes.input);
    reorder<InputType, nchw, nChw16c>(tinput, input, n, g * ic, ih, iw);
  } else if (desc.formats.input == nhwc) {
    tinput = (InputType *)malloc(desc.byte_sizes.input);
    reorder<InputType, nchw, nhwc>(tinput, input, n, g * ic, ih, iw);
  }
  if (desc.formats.weights == OIhw16i16o || desc.formats.weights == gOIhw16i16o) {
    tweights = (WeightsType *)malloc(desc.byte_sizes.weights);
    reorder<WeightsType, goihw, gOIhw16i16o>(tweights, weights, g, oc, ic, kh, kw);
  } else if (desc.formats.weights == hwio || desc.formats.weights == ghwio) {
    tweights = (WeightsType *)malloc(desc.byte_sizes.weights);
    reorder<WeightsType, goihw, ghwio>(tweights, weights, g, oc, ic, kh, kw);
  }
  if (desc.formats.output == nChw16c) {
    toutput = (OutputType *)malloc(desc.byte_sizes.output);
    reorder<OutputType, nchw, nChw16c>(toutput, output, n, g * oc, oh, ow);
  } else if (desc.formats.output == nhwc) {
    toutput = (OutputType *)malloc(desc.byte_sizes.output);
    reorder<OutputType, nchw, nhwc>(toutput, output, n, g * oc, oh, ow);
  }

  estl::parallel_for<5>([&](int _n, int _g, int _oc, int _oh, int _ow) {
    MD5(InputType, ainput, desc.formats.input == nchw ? input : tinput,
        n, g, ic, ih, iw);
    MD5(WeightsType, aweights,
        (desc.formats.weights == oihw || desc.formats.weights == goihw)
            ? weights
            : tweights,
        g, oc, ic, kh, kw);
    MD5(OutputType, atoutput, desc.formats.output == nchw ? output : toutput,
        n, g, oc, oh, ow);
    MD2(BiasType, abias, bias, g, oc);

    if (desc.with_ip_sum)
      md5(atoutput, _n, _g, _oc, _oh, _ow) +=
          desc.with_bias ? md2(abias, _g, _oc) : 0.0f;
    else
      md5(atoutput, _n, _g, _oc, _oh, _ow) =
          desc.with_bias ? md2(abias, _g, _oc) : 0.0f;

    iter_each(_ic, ic) {
      iter_each(_kh, kh) {
        int _ih = _oh * sh - pt + _kh * dh;
        if (_ih < 0 || _ih >= ih)
          continue;
        iter_each(_kw, kw) {
          int _iw = _ow * sw - pl + _kw * dw;
          if (_iw < 0 || _iw >= iw)
            continue;
          md5(atoutput, _n, _g, _oc, _oh, _ow) +=
              md5(ainput, _n, _g, _ic, _ih, _iw) *
              md5(aweights, _g, _oc, _ic, _kh, _kw);
        }
      }
    }
    md5(atoutput, _n, _g, _oc, _oh, _ow) =
        desc.with_relu && md5(atoutput, _n, _g, _oc, _oh, _ow) < 0.0f
            ? 0.0f
            : md5(atoutput, _n, _g, _oc, _oh, _ow);
  }, n, g, oc, oh, ow);

  if (desc.formats.output == nChw16c) {
    reorder<OutputType, nChw16c, nchw>(output, toutput, n, g * oc, oh, ow);
  } else if (desc.formats.output == nhwc) {
    reorder<OutputType, nhwc, nchw>(output, toutput, n, g * oc, oh, ow);
  }

  if (tinput != nullptr)
    free(tinput);
  if (tweights != nullptr)
    free(tweights);
  if (toutput != nullptr)
    free(toutput);

  return 0;
}

template <typename InputType, typename WeightsType, typename OutputType,
          typename BiasType>
int ref_deconvolution2d(eld_conv_t &desc, OutputType *output, InputType *input,
                      WeightsType *weights, BiasType *bias) {
  int n = desc.dims.n;
  int ic = desc.dims.ic;
  int oc = desc.dims.oc;
  int ih = desc.dims.ih;
  int iw = desc.dims.iw;
  int oh = desc.dims.oh;
  int ow = desc.dims.ow;
  int kh = desc.dims.kh;
  int kw = desc.dims.kw;
  int sh = desc.strides.h;
  int sw = desc.strides.w;
  int pt = desc.pads.t;
  int pl = desc.pads.l;
  int dh = desc.dilations.h;
  int dw = desc.dilations.w;

  InputType *tinput = nullptr, *tweights = nullptr, *toutput = nullptr;
  if (desc.formats.input == nChw16c) {
    tinput = (InputType *)malloc(desc.byte_sizes.input);
    reorder<InputType, nchw, nChw16c>(tinput, input, n, ic, ih, iw);
  } else if (desc.formats.input == nhwc) {
    tinput = (InputType *)malloc(desc.byte_sizes.input);
    reorder<InputType, nchw, nhwc>(tinput, input, n, ic, ih, iw);
  }
  if (desc.formats.weights == OIhw16i16o) {
    tweights = (WeightsType *)malloc(desc.byte_sizes.weights);
    reorder<WeightsType, oihw, OIhw16i16o>(tweights, weights, oc, ic, kh, kw);
  } else if (desc.formats.weights == hwio) {
    tweights = (WeightsType *)malloc(desc.byte_sizes.weights);
    reorder<WeightsType, oihw, hwio>(tweights, weights, oc, ic, kh, kw);
  }
  if (desc.formats.output == nChw16c) {
    toutput = (OutputType *)malloc(desc.byte_sizes.output);
    reorder<OutputType, nchw, nChw16c>(toutput, output, n, oc, oh, ow);
  } else if (desc.formats.output == nhwc) {
    toutput = (OutputType *)malloc(desc.byte_sizes.output);
    reorder<OutputType, nchw, nhwc>(toutput, output, n, oc, oh, ow);
  }

  estl::parallel_for<4>([&](int _n, int _oc, int _oh, int _ow) {
    MD4(InputType, ainput, desc.formats.input == nchw ? input : tinput,
        n, ic, ih, iw);
    MD4(WeightsType, aweights, desc.formats.weights == oihw ? weights : tweights,
        oc, ic, kh, kw);
    MD4(OutputType, atoutput, desc.formats.output == nchw ? output : toutput,
        n, oc, oh, ow);

    md4(atoutput, _n, _oc, _oh, _ow) = desc.with_bias ? bias[_oc] : 0.0f;

    iter_each(_ic, ic) {
      iter_each(_kh, kh) {
        int _ih = (_oh + pt - _kh) / sh;
        if (_ih < 0 || _ih >= ih)
          continue;
        iter_each(_kw, kw) {
          int _iw = (_ow + pl - _kw) / sw;
          if (_iw < 0 || _iw >= iw)
            continue;
          md4(atoutput, _n, _oc, _oh, _ow) +=
              md4(ainput, _n, _ic, _ih, _iw) *
              md4(aweights, _oc, _ic, _kh, _kw);
        }
      }
    }
    md4(atoutput, _n, _oc, _oh, _ow) =
        desc.with_relu && md4(atoutput, _n, _oc, _oh, _ow) < 0.0f
            ? 0.0f
            : md4(atoutput, _n, _oc, _oh, _ow);
  }, n, oc, oh, ow);

  if (desc.formats.output == nChw16c) {
    reorder<OutputType, nChw16c, nchw>(output, toutput, n, oc, oh, ow);
  } else if (desc.formats.output == nhwc) {
    reorder<OutputType, nhwc, nchw>(output, toutput, n, oc, oh, ow);
  }

  if (tinput != nullptr)
    free(tinput);
  if (tweights != nullptr)
    free(tweights);
  if (toutput != nullptr)
    free(toutput);

  return 0;
}

template <typename InputType, typename WeightsType, typename OutputType,
          typename BiasType>
int ref_conv_deconv_2d(eld_conv_t &desc, OutputType *output, InputType *input,
                      WeightsType *weights, BiasType *bias) {
  if (desc.algorithm == DECONV_DIRECT) {
    return ref_deconvolution2d<InputType, WeightsType, OutputType, BiasType>(
        desc, output, input, weights, bias);
  } else {
    return ref_convolution2d<InputType, WeightsType, OutputType, BiasType>(
        desc, output, input, weights, bias);
  }
}

void post_process_conv_results(float *output_ref, eld_conv_t &desc,
                               void *output_res, int data_type_cfg) {
  if (data_type_cfg == euler::test::FP32 ||
      data_type_cfg == euler::test::FP16O ||
      data_type_cfg == euler::test::U8F32F32F32) {
    float *_output_res = (float *)output_res;
    estl::parallel_for<1>([&](size_t i) {
      output_ref[i] = _output_res[i];
    }, desc.sizes.output);
  } else if (data_type_cfg == euler::test::FP16) {
    uint16_t *_output_res = (uint16_t *)output_res;
    estl::parallel_for<1>([&](size_t i) {
      output_ref[i] = half_2_float(_output_res[i]);
    }, desc.sizes.output);
  } else if (data_type_cfg == euler::test::U8F32U8F32 ||
             data_type_cfg == euler::test::U8F32U8F32z) {
    uint8_t *_output_res = (uint8_t *)output_res;
    estl::parallel_for<1>([&](size_t i) {
      float resu8 = (float)_output_res[i];
      output_ref[i] = (resu8 - desc.output_quant.z) * desc.output_quant.scale;
    }, desc.sizes.output);
  } else if (data_type_cfg == euler::test::U8F32S8F32 ||
             data_type_cfg == euler::test::U8F32S8F32z) {
    int8_t *_output_res = (int8_t *)output_res;
    estl::parallel_for<1>([&](size_t i) {
      float resi8 = (float)_output_res[i];
      output_ref[i] = (resi8 - desc.output_quant.z) * desc.output_quant.scale;
    }, desc.sizes.output);
  }
}

template int ref_conv_deconv_2d<float, float, float, float>(eld_conv_t &,
                                                            float *, float *,
                                                            float *, float *);

template int ref_convolution2d<float, float, float, float>(eld_conv_t &,
                                                           float *, float *,
                                                           float *, float *);

template void prepare_conv_data<float, float, float, float>(
    eld_conv_t &, eld_conv_t &, float *, float *, float *, float *, float **,
    float **, float **, float **, const char *, const char *, const char *,
    int, int, bool, int, bool, bool);

template void prepare_conv_data<uint16_t, uint16_t, uint16_t, uint16_t>(
    eld_conv_t &, eld_conv_t &, float *, float *, float *, float *, uint16_t **,
    uint16_t **, uint16_t **, uint16_t **, const char *, const char *,
    const char *, int, int, bool, int, bool, bool);

template void prepare_conv_data<uint8_t, float, uint8_t, float>(
    eld_conv_t &, eld_conv_t &, float *, float *, float *, float *, uint8_t **,
    float **, uint8_t **, float **, const char *, const char *, const char *,
    int, int, bool, int, bool, bool);

template void prepare_conv_data<uint8_t, float, int8_t, float>(
    eld_conv_t &, eld_conv_t &, float *, float *, float *, float *, uint8_t **,
    float **, int8_t **, float **, const char *, const char *, const char *,
    int, int, bool, int, bool, bool);

template void prepare_conv_data<uint8_t, float, float, float>(
    eld_conv_t &, eld_conv_t &, float *, float *, float *, float *, uint8_t **,
    float **, float **, float **, const char *, const char *, const char *,
    int, int, bool, int, bool, bool);
}
}
