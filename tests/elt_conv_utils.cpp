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
  template <typename InputType, typename WeightsType, typename OutputType, typename BiasType>
  void prepare_conv_data(
      eld_conv_t<ConvTypes<InputType, WeightsType, OutputType, BiasType>> &,
      InputType **, WeightsType **, OutputType **, BiasType **,
      short **, short **, short **, short **, bool, bool)
  {
  }

  void read_blob_data(void *buffer, const char *filename, size_t size) {
    std::ifstream input_file (filename, std::ios::in | std::ios::binary);
    if (input_file) {
      input_file.read((char *)buffer, size);
      input_file.close();
    }
  }

  template <typename InputType, typename WeightsType, typename OutputType,
      typename BiasType>
  void load_conv_data(eld_conv_t<conv::FP32> &desc, InputType *input,
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
  template <>
  void prepare_conv_data<float>(eld_conv_t<conv::FP32> &desc,
      float **input, float **weights, float **output, float **bias, short **input1,
      short **weights1, short **output1, short **bias1,
      const char *input_file, const char *weights_file, const char *bias_file,
      bool reuse_inout, int fp_mode, bool f16c_opt, bool validate_results)
  {
    seed = time(nullptr);
    size_t input_size = desc.byte_sizes.input;
    size_t output_size = desc.byte_sizes.output;
    if (reuse_inout) {
      input_size = std::max(desc.byte_sizes.input, desc.byte_sizes.output);
      output_size = input_size;
    }

    if (fp_mode == euler::FP16) {
      if (input1 != nullptr)
        MEMALIGN64(input1, input_size / 2);
      if (output1 != nullptr)
        MEMALIGN64(output1, output_size / 2);
      if (weights1 != nullptr)
        MEMALIGN64(weights1, desc.byte_sizes.weights / 2);
      if (bias1 != nullptr)
        MEMALIGN64(bias1, desc.byte_sizes.bias / 2);

      if (validate_results) {
        if (input != nullptr)
          MEMALIGN64(input, input_size);
        if (output != nullptr)
          MEMALIGN64(output, output_size);
        if (weights != nullptr)
          MEMALIGN64(weights, desc.byte_sizes.weights);
        if (bias != nullptr)
          MEMALIGN64(bias, desc.byte_sizes.bias);
      }
    } else if (fp_mode == euler::FP16O){
      if (input != nullptr)
        MEMALIGN64(input, input_size);
      if (weights != nullptr)
        MEMALIGN64(weights, desc.byte_sizes.weights);
      if (bias != nullptr)
        MEMALIGN64(bias, desc.byte_sizes.bias);
      if (output1 != nullptr)
        MEMALIGN64(output1, output_size / 2);

      if (validate_results && output != nullptr)
        MEMALIGN64(output, output_size);
    } else {
      if (input != nullptr)
        MEMALIGN64(input, input_size);
      if (output != nullptr)
        MEMALIGN64(output, output_size);
      if (weights != nullptr)
        MEMALIGN64(weights, desc.byte_sizes.weights);
      if (bias != nullptr)
        MEMALIGN64(bias, desc.byte_sizes.bias);
    }

    if (input_file != nullptr && weights_file != nullptr) {
      load_conv_data<float, float, float>(
          desc, *input, *weights, *bias, input_file, weights_file, bias_file);
      return;
    }
#define RAND() rand_r(&seed)

    std::default_random_engine gen;
    std::normal_distribution<float> dInput(-4.0, 20.0);
    std::normal_distribution<float> dWeights(-1.0, 1.0);

    {
      if (fp_mode == euler::FP16  && !validate_results) {
        if (input1 != nullptr) {
#pragma omp parallel for
          for (size_t i = 0; i < desc.sizes.input; i++) {
            (*input1)[i] = float_2_half(dInput(gen));
          }
        }
      } else {
        if (input_file == nullptr && input != nullptr) {
#pragma omp parallel for
          for (size_t i = 0; i < desc.sizes.input; i++) {
            (*input)[i]
                = (fp_mode == euler::FP16 || f16c_opt)
                ? dInput(gen)
                : RAND() % 20 - 4;
          }

          if (fp_mode == euler::FP16 && (input1 != nullptr)) {
#pragma omp parallel for
            for (size_t i = 0; i < desc.sizes.input; i++) {
              (*input1)[i] = float_2_half((*input)[i]);
            }
          }
        }
      }

      if (fp_mode == euler::FP16  && !validate_results) {
        if (weights1 != nullptr) {
          if (desc.with_relu) {
#pragma omp parallel for
            for (size_t i = 0; i < desc.sizes.weights; i++) {
              (*weights1)[i] = float_2_half(dWeights(gen));
              if (i % 3 == 1)
                (*weights1)[i] = -(*weights1)[i];
            }
          } else {
#pragma omp parallel for
            for (size_t i = 0; i < desc.sizes.weights; i++) {
              (*weights1)[i] = float_2_half(dWeights(gen));
            }
          }
        }
      } else {
        if (weights != nullptr) {
          if (desc.with_relu) {
#pragma omp parallel for
            for (size_t i = 0; i < desc.sizes.weights; i++) {
              (*weights)[i]
                  = (fp_mode == euler::FP16 || f16c_opt)
                  ? dWeights(gen)
                  : -RAND() % 32;
              if (i % 3 == 1)
                (*weights)[i] = -(*weights)[i];
            }
          } else {
#pragma omp parallel for
            for (size_t i = 0; i < desc.sizes.weights; i++) {
              (*weights)[i]
                  = (fp_mode == euler::FP16 || f16c_opt)
                  ? dWeights(gen)
                  : RAND() % 32;
            }
          }
          if (fp_mode == euler::FP16 && (weights1 != nullptr)) {
#pragma omp parallel for
            for (size_t i = 0; i < desc.sizes.weights; i++) {
              (*weights1)[i] = float_2_half((*weights)[i]);
            }
          }
        }
      }

      if (fp_mode == euler::FP16  && !validate_results) {
        if (bias1 != nullptr) {
#pragma omp parallel for
          for (size_t i = 0; i < desc.sizes.bias; i++) {
            (*bias1)[i] = float_2_half(RAND() % 100);
          }
        }
      } else {
        if (bias != nullptr) {
#pragma omp parallel for
          for (size_t i = 0; i < desc.sizes.bias; i++) {
            (*bias)[i] = RAND() % 100;
          }
          if (fp_mode == euler::FP16 && (bias1 != nullptr)) {
#pragma omp parallel for
            for (size_t i = 0; i < desc.sizes.bias; i++) {
              (*bias1)[i] = float_2_half((*bias)[i]);
            }
          }
        }
      }

      if (fp_mode == euler::FP16  && !validate_results) {
        if (output1 != nullptr && desc.with_ip_sum) {
#pragma omp parallel for
          for (size_t i = 0; i < desc.sizes.output; i++) {
            (*output1)[i] = float_2_half(RAND() % 10);
          }
        }
      } else {
        if (output != nullptr && desc.with_ip_sum) {
#pragma omp parallel for
          for (size_t i = 0; i < desc.sizes.output; i++) {
            (*output)[i] = RAND() % 10;
          }
          if (fp_mode  == euler::FP16 && (output1 != nullptr)) {
#pragma omp parallel for
            for (size_t i = 0; i < desc.sizes.output; i++) {
              (*output1)[i] = float_2_half((*output)[i]);
            }
          }
        }
      }
    }
  }

  void teardown_conv_data(void *input, void *weights, void *output, void *bias,
      void *input1, void *weights1, void *output1, void *bias1, int fp_mode,
      bool validate_results)
  {
    if (fp_mode == euler::FP32) {
      if (input)
        free(input);
      if (weights)
        free(weights);
      if (output)
        free(output);
      if (bias)
      free(bias);
    } else {
      if (validate_results) {
        if (input)
          free(input);
        if (weights)
          free(weights);
        if (output)
          free(output);
        if (bias)
        free(bias);
      }

      if (fp_mode == euler::FP16) {
        if (input1)
          free(input1);
        if (weights1)
          free(weights1);
        if (output1)
          free(output1);
        if (bias1)
          free(bias1);
      } else if (fp_mode == euler::FP16O){
        if (output1)
          free(output1);
      }
    }
  }

  template <typename OutputType>
  int compare_conv_results(eld_conv_t<conv::FP32> &desc, OutputType *out,
      float *ref, int fp_mode)
  {
    if (desc.formats.output == nchw)
      return __compare_conv_results_nchw(desc, out, ref, fp_mode);
    else if (desc.formats.output == nhwc)
      return __compare_conv_results_nhwc(desc, out, ref, fp_mode);
    else
      return __compare_conv_results_blocked(desc, out, ref, fp_mode);
  }

  template <typename InputType, typename WeightsType, typename OutputType, typename BiasType>
  int __compare_conv_results_nchw(eld_conv_t<ConvTypes<InputType, WeightsType, OutputType, BiasType>> &,
      OutputType *, OutputType *, int fp_mode)
  {
    return -1;
  }

  template <typename InputType, typename WeightsType, typename OutputType, typename BiasType>
  int __compare_conv_results_nhwc(eld_conv_t<ConvTypes<InputType, WeightsType, OutputType, BiasType>> &,
      OutputType *, OutputType *, int fp_mode)
  {
    return -1;
  }

  template <typename OutputType>
  int __compare_conv_results_blocked(
      eld_conv_t<conv::FP32> &desc, OutputType *out, float *ref, int fp_mode)
  {
    const int V = 16;
    auto dims = desc.dims.output;
    int C = ALIGNUP(dims.c, V) / V;
    int Or = dims.c % V ? dims.c % V: V;

    MD5(OutputType, aout, out, dims.n, C, dims.h, dims.w, V);
    MD5(float, aref, ref, dims.n, C, dims.h, dims.w, V);

#define MAX_PRINT_ERRORS (20)
    size_t errors = 0;
    double acc = desc.with_relu ? 1.0 : 1.e-5;

#pragma omp parallel for collapse(3)
    iter_each (_n, dims.n) {
      iter_each (_C, C) {
        iter_each (_h, dims.h) {
          iter_each (_w, dims.w) {
            int v = _C == C - 1 ? Or : V;
            iter_each (_v, v) {
              auto real = fp_mode != 0
                  ? half_2_float(md5(aout, _n, _C, _h, _w, _v))
                  : md5(aout, _n, _C, _h, _w, _v);
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

  template <typename OutputType>
  int __compare_conv_results_nchw(
      eld_conv_t<conv::FP32> &desc, OutputType *out, float *ref, int fp_mode)
  {
    auto dims = desc.dims.output;
    MD4(OutputType, aout, out, dims.n, dims.c, dims.h, dims.w);
    MD4(float, aref, ref, dims.n, dims.c, dims.h, dims.w);

#define MAX_PRINT_ERRORS (20)
    size_t errors = 0;
    double acc = desc.with_relu ? 1.0 : 1e-5;

#pragma omp parallel for collapse(3)
    iter_each (_n, dims.n) {
      iter_each (_c, dims.c) {
        iter_each (_h, dims.h) {
          iter_each (_w, dims.w) {
            auto real = fp_mode != 0
                ? half_2_float(md4(aout, _n, _c, _h, _w))
                : md4(aout, _n, _c, _h, _w);
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

  template <typename OutputType>
  int __compare_conv_results_nhwc(
      eld_conv_t<conv::FP32> &desc, OutputType *out, float *ref, int fp_mode)
  {
    auto dims = desc.dims.output;
    MD4(OutputType, aout, out, dims.n, dims.h, dims.w, dims.c);
    MD4(float, aref, ref, dims.n, dims.h, dims.w, dims.c);

#define MAX_PRINT_ERRORS (20)
    size_t errors = 0;
    double acc = desc.with_relu ? 1.0 : 1e-5;

#pragma omp parallel for collapse(3)
    iter_each (_n, dims.n) {
      iter_each (_c, dims.c) {
        iter_each (_h, dims.h) {
          iter_each (_w, dims.w) {
            auto real = fp_mode != 0
                ? half_2_float(md4(aout, _n, _h, _w, _c))
                : md4(aout, _n, _h, _w, _c);
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

  size_t cal_ops(eld_conv_t<conv::FP32> &desc)
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

  template <typename InputType, typename WeightsType, typename OutputType, typename BiasType>
  int ref_convolution2d(eld_conv_t<ConvTypes<InputType, WeightsType, OutputType, BiasType>> &desc,
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
  int ref_convolution2d_block16(eld_conv_t<ConvTypes<InputType, WeightsType, OutputType, BiasType>> &desc,
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

  template int compare_conv_results<float>(
      eld_conv_t<conv::FP32> &, float *, float *, int);

  template int compare_conv_results<short>(
      eld_conv_t<conv::FP32> &, short *, float *, int);

  template int ref_convolution2d<float, float, float, float>(
      eld_conv_t<conv::FP32> &, float *, float *, float *, float *);

  template int ref_convolution2d_block16<float, float, float, float>(
      eld_conv_t<conv::FP32> &, float *, float *, float *, float *);

}
}
