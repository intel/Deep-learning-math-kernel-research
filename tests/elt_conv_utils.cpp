#include <algorithm>
#include <math.h>
#include <omp.h>
#include <memory.h>
#include <random>
#include <fstream>
#include "elt_conv_utils.hpp"
#include "el_intrin.hpp"

namespace euler {
namespace test {
  void read_blob_data(void *buffer, const char *filename, size_t size) {
    std::ifstream input_file (filename, std::ios::in | std::ios::binary);
    if (input_file) {
      input_file.read((char *)buffer, size);
      input_file.close();
    }
  }

  template <typename InputType, typename WeightsType, typename OutputType,
      typename BiasType>
  void load_conv_data(eld_conv_t &desc, InputType *input,
      WeightsType *weights, BiasType *bias, const char *input_file,
      const char *weights_file, const char *bias_file)
  {
    InputType *nchw_input;
    WeightsType *oihw_weights;
    MEMALIGN64(&nchw_input, desc.byte_sizes.input);
    MEMALIGN64(&oihw_weights, desc.byte_sizes.weights);

    auto input_dims = desc.dims.input;
    auto weights_dims = desc.dims.weights;
    auto input_sz = input_dims.n * input_dims.c * input_dims.h * input_dims.w;
    auto weights_sz = weights_dims.o * weights_dims.i * weights_dims.h * weights_dims.w;

    if (input_file != nullptr && input != nullptr) {
      read_blob_data(nchw_input, input_file, input_sz * sizeof(InputType));
      reorder<InputType, nChw16c, nchw>(input, nchw_input, input_dims.n,
          input_dims.c, input_dims.h, input_dims.w);
      printf("nchw_input=%g,... %g\n", nchw_input[0], nchw_input[input_sz - 1]);
      free(nchw_input);
    }

    if (weights_file != nullptr && weights != nullptr) {
      read_blob_data(
          oihw_weights, weights_file, weights_sz * sizeof(WeightsType));
      reorder<WeightsType, OIhw16i16o, oihw>(weights, oihw_weights,
          weights_dims.o, weights_dims.i, weights_dims.h, weights_dims.w);
      printf("oihw_weights=%g,... %g\n", oihw_weights[0], oihw_weights[weights_sz - 1]);
      free(oihw_weights);
    }

    if (bias_file != nullptr && bias != nullptr) {
      read_blob_data(bias, bias_file, weights_dims.o * sizeof(BiasType));
    }
  }

  __thread unsigned int seed;
  template <typename InputType, typename WeightsType,
            typename OutputType, typename BiasType>
  void prepare_conv_data(eld_conv_t &desc_ref, eld_conv_t &desc,
      float *input_ref, float *weights_ref, float *output_ref, float *bias_ref,
      InputType **input, WeightsType **weights, OutputType **output, BiasType **bias,
      const char *input_file, const char *weights_file, const char *bias_file,
      bool reuse_inout, int data_type_cfg, bool f16c_opt, bool validate_results)
  {
    seed = time(nullptr);

    // reference data initialization
    if (input_file != nullptr && weights_file != nullptr) {
      load_conv_data<float, float, float>(desc_ref, input_ref, weights_ref,
          bias_ref, input_file, weights_file, bias_file);
    } else if (input_file == nullptr && weights_file == nullptr) {
#define RAND() rand_r(&seed)
      std::default_random_engine gen;
      std::normal_distribution<float> dInput(-4.0, 20.0);
      std::normal_distribution<float> dWeights(-1.0, 1.0);
      std::normal_distribution<float> dInput_mu_15_sigma_3(15.0, 3.0);
      std::normal_distribution<float> dWeights_mu_0_sigma_0_1(0.0, 0.1);

      // ref input
#pragma omp parallel for
      for (size_t i = 0; i < desc_ref.sizes.input; i++) {
        if (data_type_cfg == euler::test::FP32) {
          input_ref[i] = RAND() % 20 - 4;
        } else if (data_type_cfg == euler::test::FP16 || f16c_opt) {
          input_ref[i] = dInput(gen);
        } else if (data_type_cfg == euler::test::U8F32U8F32
            || data_type_cfg == euler::test::U8F32F32F32) {
          input_ref[i] = dInput_mu_15_sigma_3(gen);
          if (input_ref[i] < 0)
            input_ref[i] = 0;
        }
      }

      // ref weights
#pragma omp parallel for
      for (size_t i = 0; i < desc_ref.sizes.weights; i++) {
        if (data_type_cfg == euler::test::FP32) {
          weights_ref[i] = -RAND() % 32;
        } else if (data_type_cfg == euler::test::FP16 || f16c_opt) {
          weights_ref[i] = dWeights(gen);
        } else if (data_type_cfg == euler::test::U8F32U8F32
            || data_type_cfg == euler::test::U8F32F32F32) {
          weights_ref[i] = dWeights_mu_0_sigma_0_1(gen);
        }
        if (desc_ref.with_relu && i % 3 == 1)
          weights_ref[i] = -weights_ref[i];
      }

      // ref bias
#pragma omp parallel for
      for (size_t i = 0; i < desc_ref.sizes.bias; i++)
        bias_ref[i] = RAND() % 100;

      // ref ouput
      if (desc_ref.with_ip_sum) {
#pragma omp parallel for
        for (size_t i = 0; i < desc_ref.sizes.output; i++)
          output_ref[i] = RAND() % 10;
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
      MEMALIGN64(input, input_size);
      MEMALIGN64(output, output_size);
      MEMALIGN64(weights, desc_ref.byte_sizes.weights);
      MEMALIGN64(bias, desc_ref.byte_sizes.bias);
    } else if (data_type_cfg == euler::test::FP16) {
      MEMALIGN64(input, input_size / 2);
      MEMALIGN64(output, output_size / 2);
      MEMALIGN64(weights, desc_ref.byte_sizes.weights / 2);
      MEMALIGN64(bias, desc_ref.byte_sizes.bias / 2);
    } else if (data_type_cfg == euler::test::FP16O){
      MEMALIGN64(input, input_size);
      MEMALIGN64(weights, desc_ref.byte_sizes.weights);
      MEMALIGN64(bias, desc_ref.byte_sizes.bias);
      MEMALIGN64(output, output_size / 2);
    } else if (data_type_cfg == euler::test::U8F32U8F32){
      MEMALIGN64(input, input_size / 4);
      MEMALIGN64(weights, desc_ref.byte_sizes.weights);
      MEMALIGN64(bias, desc_ref.byte_sizes.bias);
      MEMALIGN64(output, output_size / 4);
    } else if (data_type_cfg == euler::test::U8F32F32F32){
      MEMALIGN64(input, input_size / 4);
      MEMALIGN64(weights, desc_ref.byte_sizes.weights);
      MEMALIGN64(bias, desc_ref.byte_sizes.bias);
      MEMALIGN64(output, output_size);
    }

    // scale initialization
    desc.input_quant.scale = 1.0;
    desc.input_quant.z = 0.0;
    desc.output_quant.scale = 1.0;
    desc.output_quant.z = 0.0;

    // for Winograd INT8 cases
    #define PRECISION_REPRESENTATION_8B 255
    #define PRECISION_REPRESENTATION_7B 127
    auto trans_input_scale = [&] () {
      float *_output_ref;
      if (desc_ref.with_ip_sum) {
        MEMALIGN64(&_output_ref, desc_ref.byte_sizes.output);
#pragma omp parallel for
        for (size_t i = 0; i < desc_ref.sizes.output; i++)
          _output_ref[i] = output_ref[i];
      } else {
        _output_ref = output_ref;
      }

      if (ELX_OK != elx_conv(
          desc_ref, _output_ref, input_ref, weights_ref, bias_ref)) {
        test::error("Fail: Convolution execution error!\n");
      }

      float *tinput = (float *)desc_ref.scratch_pad;
      float min = tinput[0], max = tinput[0];
      size_t t = (desc.dims.output.h + desc.tile_size - 3)
          / (desc.tile_size - 3 + 1) * (desc.dims.output.w + desc.tile_size - 3)
          / (desc.tile_size - 3 + 1) * desc.dims.output.n;
      size_t A = desc.tile_size;
      size_t K = 3;
      size_t IC = desc.dims.input.c;
      size_t tinput_size = A * A * IC * t;
      for (size_t i = 1; i < tinput_size; i++) {
        min = tinput[i] < min ? tinput[i] : min;
        max = tinput[i] > max ? tinput[i] : max;
      }

      auto diff = max - min + 0.000001;
      desc.wino_tinput_quant.scale = diff / PRECISION_REPRESENTATION_8B;
      desc.wino_tinput_quant.z = -min * PRECISION_REPRESENTATION_8B / diff;

      printf("tinput max %f min %f scale %f z %f\n",
          max, min, desc.wino_tinput_quant.scale, desc.wino_tinput_quant.z);

      if (desc_ref.with_ip_sum)
        free(_output_ref);
    };

    auto input_scale = [&] (float &iscale) {
      float abs_max = input_ref[0] > 0 ? input_ref[0] : -input_ref[0];
      for (size_t i = 1; i < desc_ref.sizes.input; i++) {
        auto abs_cur = input_ref[i] > 0 ? input_ref[i] : -input_ref[i];
        abs_max = abs_cur > abs_max ? abs_cur : abs_max;
      }
      // U8
      if (data_type_cfg == euler::test::U8F32U8F32
          || data_type_cfg == euler::test::U8F32F32F32)
        iscale = abs_max / PRECISION_REPRESENTATION_8B;
      // S8
      else
        iscale = abs_max / PRECISION_REPRESENTATION_7B;
      desc.input_quant.scale = iscale;
      desc.input_quant.z = 0;
      printf("input abs_max %f scale %f\n", abs_max, iscale);
    };

    auto output_scale = [&] (float &oscale) {
      float *_output_ref;
      if (desc_ref.with_ip_sum) {
        MEMALIGN64(&_output_ref, desc_ref.byte_sizes.output);
#pragma omp parallel for
        for (size_t i = 0; i < desc_ref.sizes.output; i++)
          _output_ref[i] = output_ref[i];
      } else {
        _output_ref = output_ref;
      }

      if (test::ref_convolution2d<float>(
          desc_ref, _output_ref, input_ref, weights_ref, bias_ref)) {
        printf("Fail: scale initialization. Convolution ref execution error!\n");
        exit(1);
      }

      float abs_max = _output_ref[0] > 0 ? _output_ref[0] : -_output_ref[0];
      for (size_t i = 1; i < desc_ref.sizes.output; i++) {
        auto abs_cur = _output_ref[i] > 0 ? _output_ref[i] : -_output_ref[i];
        abs_max = abs_cur > abs_max ? abs_cur : abs_max;
      }

      if (desc_ref.with_relu)
        oscale = PRECISION_REPRESENTATION_8B / abs_max;
      else
        oscale = PRECISION_REPRESENTATION_7B / abs_max;

      desc.output_quant.scale = 1.0 / oscale;
      desc.output_quant.z = 0.0;

      printf("output abs_max %f scale %f\n", abs_max, desc.output_quant.scale);

      if (desc_ref.with_ip_sum)
        free(_output_ref);
    };

    auto rounding_to_nearest_even = [] (float f32) -> int32_t {
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
    float iscale;
    if (data_type_cfg == euler::test::U8F32U8F32
        || data_type_cfg == euler::test::U8F32F32F32) {
      input_scale(iscale);
      trans_input_scale();
    }
#pragma omp parallel for
    for (size_t i = 0; i < desc_ref.sizes.input; i++) {
      if (data_type_cfg == euler::test::FP32)
        (*input)[i] = input_ref[i];
      else if (data_type_cfg == euler::test::FP16)
        (*input)[i] = float_2_half(input_ref[i]);
      else if (data_type_cfg == euler::test::FP16O)
        (*input)[i] = input_ref[i];
      else if (data_type_cfg == euler::test::U8F32U8F32
          || data_type_cfg == euler::test::U8F32F32F32)
        (*input)[i] = (uint8_t)rounding_to_nearest_even(input_ref[i] / iscale);
    }

    // weights
#pragma omp parallel for
    for (size_t i = 0; i < desc_ref.sizes.weights; i++) {
      if (data_type_cfg == euler::test::FP32)
        (*weights)[i] = weights_ref[i];
      else if (data_type_cfg == euler::test::FP16)
        (*weights)[i] = float_2_half(weights_ref[i]);
      else if (data_type_cfg == euler::test::FP16O)
        (*weights)[i] = weights_ref[i];
      else if (data_type_cfg == euler::test::U8F32U8F32
          || data_type_cfg == euler::test::U8F32F32F32)
        (*weights)[i] = weights_ref[i];
    }

    // bias
#pragma omp parallel for
    for (size_t i = 0; i < desc_ref.sizes.bias; i++) {
      if (data_type_cfg == euler::test::FP32)
        (*bias)[i] = bias_ref[i];
      else if (data_type_cfg == euler::test::FP16)
        (*bias)[i] = float_2_half(bias_ref[i]);
      else if (data_type_cfg == euler::test::FP16O)
        (*bias)[i] = bias_ref[i];
      else if (data_type_cfg == euler::test::U8F32U8F32
          || data_type_cfg == euler::test::U8F32F32F32)
        (*bias)[i] = bias_ref[i];
    }

    // output
    float oscale;
    if (data_type_cfg == euler::test::U8F32U8F32)
      output_scale(oscale);
    if (desc.with_ip_sum) {
#pragma omp parallel for
      for (size_t i = 0; i < desc_ref.sizes.output; i++) {
        if (data_type_cfg == euler::test::FP32)
          (*output)[i] = output_ref[i];
        else if (data_type_cfg == euler::test::FP16)
          (*output)[i] = float_2_half(output_ref[i]);
        else if (data_type_cfg == euler::test::FP16O)
          (*output)[i] = output_ref[i];
        else if (data_type_cfg == euler::test::U8F32U8F32) {
          (*output)[i] = (uint8_t)rounding_to_nearest_even(output_ref[i] * oscale);
        }
      }
    }
  }

  int compare_conv_results(eld_conv_t &desc, float *out,
      float *ref, int data_type_cfg, bool is_int8_lp, bool with_real_data)
  {
    double acc = is_int8_lp ? (with_real_data ? 1e-1 : 1e-2) : 1e-5;

    if (desc.formats.output == nhwc) {
      acc = desc.with_relu ? 1.0 : 1e-5;
      return __compare_conv_results_nhwc(desc, out, ref, data_type_cfg, acc);
    } else if (desc.formats.output == nchw) {
      return __compare_conv_results_nchw(desc, out, ref, data_type_cfg, acc);
    } else {
      return __compare_conv_results_blocked(desc, out, ref, data_type_cfg, acc);
    }
  }

  int __compare_conv_results_blocked(
      eld_conv_t &desc, float *out, float *ref, int data_type_cfg, double acc)
  {
    const int V = 16;
    auto dims = desc.dims.output;
    int C = ALIGNUP(dims.c, V) / V;
    int Or = dims.c % V ? dims.c % V: V;

    MD5(float, aout, out, dims.n, C, dims.h, dims.w, V);
    MD5(float, aref, ref, dims.n, C, dims.h, dims.w, V);

#define MAX_PRINT_ERRORS (20)
    size_t errors = 0;

#pragma omp parallel for collapse(3)
    iter_each (_n, dims.n) {
      iter_each (_C, C) {
        iter_each (_h, dims.h) {
          iter_each (_w, dims.w) {
            int v = _C == C - 1 ? Or : V;
            iter_each (_v, v) {
              auto real = md5(aout, _n, _C, _h, _w, _v);
              double delta = fabs(real - md5(aref, _n, _C, _h, _w, _v));
              if (md5(aref, _n, _C, _h, _w, _v) == 0 || real == 0) {
                if (delta < acc)
                  continue;
                else if (errors < MAX_PRINT_ERRORS) {
                  printf("Not equal!: [%d][%d][%d][%d][%d]: %f != %f (ref), "
                         "delta=%g, acc=%g\n",
                      _n, _C, _h, _w, _v, real, md5(aref, _n, _C, _h, _w, _v),
                      delta, acc);
                }
#pragma omp atomic
                  errors++;
              } else {
                double rel_diff = delta / fabs(md5(aref, _n, _C, _h, _w, _v));
                if (rel_diff > acc) {
                  if (errors < MAX_PRINT_ERRORS) {
                    printf("Not equal!: [%d][%d][%d][%d][%d]: %f != %f (ref), "
                           "delta=%g, rel_diff=%g\n",
                        _n, _C, _h, _w, _v, real, md5(aref, _n, _C, _h, _w, _v),
                        delta, rel_diff);
                  }
#pragma omp atomic
                  errors++;
                }
              }
            }
          }
        }
      }
    }

    if (errors > 0) {
      printf("Error: number of errors: %ld/%ld, percentage: %f%%\n", errors,
          desc.sizes.output, ((errors * 1.0) / desc.sizes.output) * 100.0);
      return -1;
    }
    return 0;
  }

  int __compare_conv_results_nchw(
      eld_conv_t &desc, float *out, float *ref, int data_type_cfg, double acc)
  {
    auto dims = desc.dims.output;
    MD4(float, aout, out, dims.n, dims.c, dims.h, dims.w);
    MD4(float, aref, ref, dims.n, dims.c, dims.h, dims.w);

#define MAX_PRINT_ERRORS (20)
    size_t errors = 0;

#pragma omp parallel for collapse(3)
    iter_each (_n, dims.n) {
      iter_each (_c, dims.c) {
        iter_each (_h, dims.h) {
          iter_each (_w, dims.w) {
            auto real = md4(aout, _n, _c, _h, _w);
            double delta = fabs(real - md4(aref, _n, _c, _h, _w));
            if (real == 0 || md4(aref, _n, _c, _h, _w) == 0) {
              if (delta < acc)
                continue;
              else if (errors < MAX_PRINT_ERRORS) {
                printf("Not equal!: [%d][%d][%d][%d]: %f != %f (ref), "
                       "delta=%g, acc=%g\n",
                    _n, _c, _h, _w, real, md4(aref, _n, _c, _h, _w), delta, acc);
#pragma omp atomic
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
#pragma omp atomic
                errors++;
              }
            }
          }
        }
      }
    }

    if (errors > 0) {
      printf("Error: number of errors: %ld/%ld, percentage: %f%%\n", errors,
          desc.sizes.output, ((errors * 1.0) / desc.sizes.output) * 100.0);
      return -1;
    }
    return 0;
  }

  int __compare_conv_results_nhwc(
      eld_conv_t &desc, float *out, float *ref, int data_type_cfg, double acc)
  {
    auto dims = desc.dims.output;
    MD4(float, aout, out, dims.n, dims.h, dims.w, dims.c);
    MD4(float, aref, ref, dims.n, dims.h, dims.w, dims.c);

#define MAX_PRINT_ERRORS (20)
    size_t errors = 0;

#pragma omp parallel for collapse(3)
    iter_each (_n, dims.n) {
      iter_each (_c, dims.c) {
        iter_each (_h, dims.h) {
          iter_each (_w, dims.w) {
            auto real = md4(aout, _n, _h, _w, _c);
            double delta = fabs(real - md4(aref, _n, _h, _w, _c));
            if (real == 0 || md4(aref, _n, _h, _w, _c) == 0) {
              if (delta < acc)
                continue;
              else if (errors < MAX_PRINT_ERRORS) {
                printf("Not equal!: [%d][%d][%d][%d]: %f != %f (ref), "
                       "delta=%g, acc=%g\n",
                    _n, _c, _h, _w, real, md4(aref, _n, _h, _w, _c), delta, acc);
#pragma omp atomic
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
#pragma omp atomic
                errors++;
              }
            }
          }
        }
      }
    }

    if (errors > 0) {
      printf("Error: number of errors: %ld/%ld, percentage: %f%%\n", errors,
          desc.sizes.output, ((errors * 1.0) / desc.sizes.output) * 100.0);
      return -1;
    }
    return 0;
  }

  size_t cal_ops(eld_conv_t &desc)
  {
    size_t num_ops = 0;

    iter_each (_oh, desc.dims.output.h) {
      iter_each (_ow, desc.dims.output.w) {
        iter_each (_kh, desc.dims.weights.h) {
          int _ih
              = _oh * desc.strides.h - desc.pads.t + _kh * desc.dilations.h;
          if (_ih < 0 || _ih >= desc.dims.input.h)
            continue;
          iter_each (_kw, desc.dims.weights.w) {
            int _iw
                = _ow * desc.strides.w - desc.pads.l + _kw * desc.dilations.w;
            if (_iw < 0 || _iw >= desc.dims.input.w)
              continue;
            num_ops += 1;
          }
        }
      }
    }
    return num_ops * desc.dims.input.n * desc.dims.weights.i
        * desc.dims.weights.o * 2;
  }

  int cal_iterations(size_t num_ops)
  {
    float iter = 5e12 / num_ops;
    return std::max((int)iter, 64);
  }

  template <typename Type>
  reorder<Type, nchw, nhwc>::reorder(
      Type *dst, Type *src, int n, int c, int h, int w)
  {
    MD4(Type, asrc, src, n, h, w, c);
    MD4(Type, adst, dst, n, c, h, w);

#pragma omp parallel for collapse(3)
    iter_each (_n, n) {
      iter_each (_c, c) {
        iter_each (_h, h) {
          iter_each (_w, w) {
            md4(adst, _n, _c, _h, _w)
              = md4(asrc, _n, _h, _w, _c);
          }
        }
      }
    }
  }

  template <typename Type>
  reorder<Type, nhwc, nchw>::reorder(
      Type *dst, Type *src, int n, int c, int h, int w)
  {
    MD4(Type, asrc, src, n, c, h, w);
    MD4(Type, adst, dst, n, h, w, c);

#pragma omp parallel for collapse(3)
    iter_each (_n, n) {
      iter_each (_c, c) {
        iter_each (_h, h) {
          iter_each (_w, w) {
            md4(adst, _n, _h, _w, _c)
              = md4(asrc, _n, _c, _h, _w);
          }
        }
      }
    }
  }

  template <typename Type>
  reorder<Type, nchw, nChw16c>::reorder(
      Type *dst, Type *src, int n, int c, int h, int w)
  {
    int C = ALIGNUP(c, 16) / 16; // padding
    int Vr = c % 16 ? c % 16 : 16;

    MD5(Type, asrc, src, n, C, h, w, 16);
    MD4(Type, adst, dst, n, c, h, w);

#pragma omp parallel for collapse(3)
    iter_each (_n, n) {
      iter_each (_C, C) {
        iter_each (_h, h) {
          iter_each (_w, w) {
            int v = (_C == C - 1) ? Vr : 16;
            iter_each (_v, v) {
              md4(adst, _n, _C * 16 + _v, _h, _w)
                  = md5(asrc, _n, _C, _h, _w, _v);
            }
          }
        }
      }
    }
  }

  template <typename Type>
  reorder<Type, nChw16c, nchw>::reorder(
      Type *dst, Type *src, int n, int c, int h, int w)
  {
    int C = ALIGNUP(c, 16) / 16; // padding
    int Vr = c % 16 ? c % 16 : 16;

    MD4(Type, asrc, src, n, c, h, w);
    MD5(Type, adst, dst, n, C, h, w, 16);

#pragma omp parallel for collapse(3)
    iter_each (_n, n) {
      iter_each (_C, C) {
        iter_each (_h, h) {
          iter_each (_w, w) {
            int v = (_C == C - 1) ? Vr : 16;
            iter_each (_v, 16) {
              if (_v < v)
                md5(adst, _n, _C, _h, _w, _v)
                    = md4(asrc, _n, _C * 16 + _v, _h, _w);
              else
                md5(adst, _n, _C, _h, _w, _v) = 0;
            }
          }
        }
      }
    }
  }

  template <typename Type>
  reorder<Type, OIhw16i16o, oihw>::reorder(
      Type *dst, Type *src, int o, int i, int h, int w)
  {
    int O = ALIGNUP(o, 16) / 16; // padding
    int I = ALIGNUP(i, 16) / 16; // padding
    int Or = o % 16 ? o % 16 : 16;
    int Ir = i % 16 ? i % 16 : 16;

    MD4(Type, asrc, src, o, i, h, w);
    MD6(Type, adst, dst, O, I, h, w, 16, 16);

#pragma omp parallel for collapse(3)
    iter_each (_O, O) {
      iter_each (_I, I) {
        iter_each (_h, h) {
          iter_each (_w, w) {
            int ov = (_O == O - 1) ? Or : 16;
            int iv = (_I == I - 1) ? Ir : 16;
            iter_each (_iv, 16) {
              iter_each (_ov, 16) {
                if (_iv < iv && _ov < ov)
                  md6(adst, _O, _I, _h, _w, _iv, _ov)
                      = md4(asrc, _O * 16 + _ov, _I * 16 + _iv, _h, _w);
                else
                  md6(adst, _O, _I, _h, _w, _iv, _ov) = 0;
              }
            }
          }
        }
      }
    }
  }

  template <typename Type>
  reorder<Type, oihw, OIhw16i16o>::reorder(
      Type *dst, Type *src, int o, int i, int h, int w)
  {
    int O = ALIGNUP(o, 16) / 16; // padding
    int I = ALIGNUP(i, 16) / 16; // padding
    int Or = o % 16 ? o % 16 : 16;
    int Ir = i % 16 ? i % 16 : 16;

    MD6(Type, asrc, src, O, I, h, w, 16, 16);
    MD4(Type, adst, dst, o, i, h, w);

#pragma omp parallel for collapse(3)
    iter_each (_O, O) {
      iter_each (_I, I) {
        iter_each (_h, h) {
          iter_each (_w, w) {
            int ov = _O == O - 1 ? Or : 16;
            int iv = _I == I - 1 ? Ir : 16;
            iter_each (_iv, iv) {
              iter_each (_ov, ov) {
                md4(adst, _O * 16 + _ov, _I * 16 + _iv, _h, _w)
                    = md6(asrc, _O, _I, _h, _w, _iv, _ov);
              }
            }
          }
        }
      }
    }
  }

  template <typename Type>
  reorder<Type, oihw, hwio>::reorder(
      Type *dst, Type *src, int o, int i, int h, int w)
  {
    MD4(Type, asrc, src, h, w, i, o);
    MD4(Type, adst, dst, o, i, h, w);

#pragma omp parallel for collapse(3)
    iter_each (_o, o) {
      iter_each (_i, i) {
        iter_each (_h, h) {
          iter_each (_w, w)
            md4(adst, _o, _i, _h, _w) = md4(asrc, _h, _w, _i, _o);
        }
      }
    }
  }

  template <typename InputType, typename WeightsType, typename OutputType, typename BiasType>
  int ref_convolution2d(eld_conv_t &desc,
      OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias)
  {
    int n = desc.dims.input.n;
    int ic = desc.dims.input.c;
    int oc = desc.dims.output.c;
    int ih = desc.dims.input.h;
    int iw = desc.dims.input.w;
    int oh = desc.dims.output.h;
    int ow = desc.dims.output.w;
    int kh = desc.dims.weights.h;
    int kw = desc.dims.weights.w;
    int sh = desc.strides.h;
    int sw = desc.strides.w;
    int pt = desc.pads.t;
    int pl = desc.pads.l;
    int dh = desc.dilations.h;
    int dw = desc.dilations.w;

    if (desc.dims.input.n != desc.dims.output.n
        || desc.dims.input.c != desc.dims.weights.i
        || desc.dims.output.c != desc.dims.weights.o
        || desc.dims.output.c != desc.dims.bias.c) {
      printf("Dimension error!");
      return -1;
    }

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

    MD4(InputType, ainput, desc.formats.input == nchw ? input : tinput, n, ic, ih, iw);
    MD4(WeightsType, aweights, desc.formats.weights == oihw ? weights : tweights, oc, ic, kh, kw);
    MD4(OutputType, atoutput, desc.formats.output == nchw ? output : toutput, n, oc, oh, ow);

#pragma omp parallel for collapse(4)
    iter_each (_n, n) {
      iter_each (_oc, oc) {
        iter_each (_oh, oh) {
          iter_each (_ow, ow) {
            if (desc.with_ip_sum)
              md4(atoutput, _n, _oc, _oh, _ow) += desc.with_bias ? bias[_oc] : 0.0f;
            else
              md4(atoutput, _n, _oc, _oh, _ow) = desc.with_bias ? bias[_oc] : 0.0f;

            iter_each (_ic, ic) {
              iter_each (_kh, kh) {
                int _ih = _oh * sh - pt + _kh * dh;
                if (_ih < 0 || _ih >= ih)
                  continue;
                iter_each (_kw, kw) {
                  int _iw = _ow * sw - pl + _kw * dw;
                  if (_iw < 0 || _iw >= iw)
                    continue;
                  md4(atoutput, _n, _oc, _oh, _ow)
                      += md4(ainput, _n, _ic, _ih, _iw)
                      * md4(aweights, _oc, _ic, _kh, _kw);
                }
              }
            }
            md4(atoutput, _n, _oc, _oh, _ow) =
                desc.with_relu && md4(atoutput, _n, _oc, _oh, _ow) < 0.0f ?
                0.0f : md4(atoutput, _n, _oc, _oh, _ow);
          }
        }
      }
    }

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

  template <typename InputType, typename WeightsType, typename OutputType, typename BiasType>
  int ref_convolution2d_block16(eld_conv_t &desc,
      OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
  {
    int n = desc.dims.input.n;
    int IC = ALIGNUP(desc.dims.input.c, 16) / 16;
    int OC = ALIGNUP(desc.dims.output.c, 16) / 16;
    int ih = desc.dims.input.h;
    int iw = desc.dims.input.w;
    int oh = desc.dims.output.h;
    int ow = desc.dims.output.w;
    int kh = desc.dims.weights.h;
    int kw = desc.dims.weights.w;
    int sh = desc.strides.h;
    int sw = desc.strides.w;
    int pt = desc.pads.t;
    int pl = desc.pads.l;
    int dh = desc.dilations.h;
    int dw = desc.dilations.w;

    MD5(InputType, ainput, input, n, IC, ih, iw, 16);
    MD6(WeightsType, aweights, weights, OC, IC, kh, kw, 16, 16);
    MD5(OutputType, aoutput, output, n, OC, oh, ow, 16);

    if (desc.dims.input.n != desc.dims.output.n
        || desc.dims.input.c != desc.dims.weights.i
        || desc.dims.output.c != desc.dims.weights.o
        || desc.dims.output.c != desc.dims.bias.c) {
      printf("Dimension error!");
      return -1;
    }

    if (desc.formats.input != nChw16c || desc.formats.weights != OIhw16i16o
        || desc.formats.output != nChw16c) {
      printf("Format error!");
      return -1;
    }
    int Or = desc.dims.output.c % 16 ?  desc.dims.output.c % 16 : 16;
    int Ir = desc.dims.input.c % 16 ? desc.dims.input.c % 16 : 16;

#pragma omp parallel for collapse(4)
    iter_each (_n, n) {
      iter_each (_OC, OC) {
        iter_each (_oh, oh) {
          iter_each (_ow, ow) {
            int ov = _OC == OC - 1 ? Or : 16;
            iter_each (_ov, ov) {
              if (desc.with_ip_sum)
                md5(aoutput, _n, _OC, _oh, _ow, _ov)
                    += desc.with_bias ? bias[_OC * 16 + _ov] : 0.0f;
              else
                md5(aoutput, _n, _OC, _oh, _ow, _ov)
                    = desc.with_bias ? bias[_OC * 16 + _ov] : 0.0f;
              iter_each (_IC, IC) {
                int iv = _IC == IC - 1 ? Ir : 16;
                iter_each (_iv, iv) {
                  iter_each (_kh, kh) {
                    int _ih = _oh * sh - pt + _kh * dh;
                    if (_ih < 0 || _ih >= ih)
                      continue;
                    iter_each (_kw, kw) {
                      int _iw = _ow * sw - pl + _kw * dw;
                      if (_iw < 0 || _iw >= iw)
                        continue;
                      md5(aoutput, _n, _OC, _oh, _ow, _ov)
                          += md5(ainput, _n, _IC, _ih, _iw, _iv)
                          * md6(aweights, _OC, _IC, _kh, _kw, _iv, _ov);
                    }
                  }
                }
              }
              md5(aoutput, _n, _OC, _oh, _ow, _ov) =
                  desc.with_relu &&
                  md5(aoutput, _n, _OC, _oh, _ow, _ov) < 0.0f ?
                  0.0f : md5(aoutput, _n, _OC, _oh, _ow, _ov);
            }
          }
        }
      }
    }

    return 0;
  }

  void post_process_conv_results(
      float *output_ref, eld_conv_t &desc, void *output_res, int data_type_cfg) {
    if (data_type_cfg == euler::test::FP32
        || data_type_cfg == euler::test::FP16O
        || data_type_cfg == euler::test::U8F32F32F32) {
      float *_output_res = (float *)output_res;
#pragma omp parallel for
      for (size_t i = 0; i < desc.sizes.output; i++)
        output_ref[i] = _output_res[i];
    } else if (data_type_cfg == euler::test::FP16) {
      uint16_t *_output_res = (uint16_t *)output_res;
#pragma omp parallel for
      for (size_t i = 0; i < desc.sizes.output; i++)
        output_ref[i] = half_2_float(_output_res[i]);
    } else if (data_type_cfg == euler::test::U8F32U8F32) {
      uint8_t *_output_res = (uint8_t *)output_res;
#pragma omp parallel for
      for (size_t i = 0; i < desc.sizes.output; i++) {
        float resu8 = (float)_output_res[i];
        output_ref[i] = (resu8 - desc.output_quant.z) * desc.output_quant.scale;
      }
    }
  }

  template int ref_convolution2d<float, float, float, float>(
      eld_conv_t &, float *, float *, float *, float *);

  template int ref_convolution2d_block16<float, float, float, float>(
      eld_conv_t &, float *, float *, float *, float *);

  template void prepare_conv_data<float, float, float, float>(
      eld_conv_t &, eld_conv_t &,
      float *, float *, float *, float *, float **, float **, float **,
      float **, const char *, const char *, const char *, bool, int, bool, bool);

  template void prepare_conv_data<uint16_t, uint16_t, uint16_t, uint16_t>(
      eld_conv_t &, eld_conv_t &,
      float *, float *, float *, float *, uint16_t **, uint16_t **, uint16_t **,
      uint16_t **, const char *, const char *, const char *, bool, int, bool, bool);

  template void prepare_conv_data<uint8_t, float, uint8_t, float>(
      eld_conv_t &, eld_conv_t &,
      float *, float *, float *, float *, uint8_t **, float **, uint8_t **,
      float **, const char *, const char *, const char *, bool, int, bool, bool);

  template void prepare_conv_data<uint8_t, float, float, float>(
      eld_conv_t &, eld_conv_t &,
      float *, float *, float *, float *, uint8_t **, float **, float **,
      float **, const char *, const char *, const char *, bool, int, bool, bool);

}
}
