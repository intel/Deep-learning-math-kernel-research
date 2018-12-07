#include <algorithm>
#include <math.h>
#include <omp.h>
#include <memory.h>
#include "elt_conv_utils.hpp"
#include "el_intrin.hpp"

namespace euler {
namespace test {
  template <typename InputType, typename WeightsType, typename OutputType, typename BiasType>
  void prepare_conv_data(
      eld_conv_t<ConvTypes<InputType, WeightsType, OutputType, BiasType>> &,
      InputType **, WeightsType **, OutputType **, BiasType **, bool)
  {
  }

  __thread unsigned int seed;
  template <>
  void prepare_conv_data<float>(eld_conv_t<conv::FP32> &desc,
      float **input, float **weights, float **output, float **bias, bool reuse_inout)
  {
    seed = time(nullptr);
    size_t input_size = desc.byte_sizes.input;
    size_t output_size = desc.byte_sizes.output;
    if (reuse_inout) {
      input_size = std::max(desc.byte_sizes.input, desc.byte_sizes.output);
      output_size = input_size;
    }
    if (input != nullptr)
      MEMALIGN64(input, input_size);
    if (output != nullptr)
      MEMALIGN64(output, output_size);
    if (weights != nullptr)
      MEMALIGN64(weights, desc.byte_sizes.weights);
    if (bias != nullptr)
      MEMALIGN64(bias, desc.byte_sizes.bias);
#define RAND() rand_r(&seed)
#pragma omp parallel
    {
      if (input != nullptr) {
#pragma omp parallel for
        for (size_t i = 0; i < desc.sizes.input; i++) {
          (*input)[i] = RAND() % 20 - 4;
        }
      }
      if (weights != nullptr) {
        if (desc.with_relu) {
#pragma omp parallel for
          for (size_t i = 0; i < desc.sizes.weights; i++) {
            (*weights)[i] = -RAND() % 32;
            if (i % 3 == 1)
              (*weights)[i] = -(*weights)[i];
          }
        } else {
#pragma omp parallel for
          for (size_t i = 0; i < desc.sizes.weights; i++) {
            (*weights)[i] = RAND() % 32;
          }
        }
      }
      if (bias != nullptr) {
#pragma omp parallel for
        for (size_t i = 0; i < desc.sizes.bias; i++) {
          (*bias)[i] = RAND() % 100;
        }
      }
      if (output != nullptr && desc.with_ip_sum) {
#pragma omp parallel for
        for (size_t i = 0; i < desc.sizes.output; i++) {
          (*output)[i] = RAND() % 10;
        }
      }
    }
  }

  void teardown_conv_data(
      void *input, void *weights, void *output, void *bias)
  {
    if (input)
      free(input);
    if (weights)
      free(weights);
    if (output)
      free(output);
    if (bias)
      free(bias);
  }

  template <typename InputType, typename WeightsType, typename OutputType, typename BiasType>
  int __compare_conv_results_plain(eld_conv_t<ConvTypes<InputType, WeightsType, OutputType, BiasType>> &,
      OutputType *, OutputType *)
  {
    return -1;
  }

  template <typename InputType, typename WeightsType, typename OutputType, typename BiasType>
  int __compare_conv_results_blocked(eld_conv_t<ConvTypes<InputType, WeightsType, OutputType, BiasType>> &,
      OutputType *, OutputType *)
  {
    return -1;
  }

  template <typename InputType, typename WeightsType, typename OutputType, typename BiasType>
  int compare_conv_results(eld_conv_t<ConvTypes<InputType, WeightsType, OutputType, BiasType>> &desc,
      OutputType *out, OutputType *ref)
  {
    if (desc.formats.output == nchw)
      return __compare_conv_results_plain(desc, out, ref);
    else
      return __compare_conv_results_blocked(desc, out, ref);
  }

  template <>
  int __compare_conv_results_blocked<float, float, float, float>(
      eld_conv_t<conv::FP32> &desc, float *out, float *ref)
  {
    const int V = 16;
    auto dims = desc.dims.output;
    int C = ALIGNUP(dims.c, V) / V;
    int Or = dims.c % V ? dims.c % V: V;

    MD5(float, aout, out, dims.n, C, dims.h, dims.w, V);
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
              double delta = fabs(
                  md5(aout, _n, _C, _h, _w, _v) - md5(aref, _n, _C, _h, _w, _v));
              if (md5(aref, _n, _C, _h, _w, _v) == 0 ||
                  md5(aout, _n, _C, _h, _w, _v) == 0) {
                if (delta < acc)
                  continue;
                else if (errors < MAX_PRINT_ERRORS) {
                  printf("Not equal!: [%d][%d][%d][%d][%d]: %f != %f (ref), "
                         "delta=%g, acc=%g\n",
                      _n, _C, _h, _w, _v, md5(aout, _n, _C, _h, _w, _v),
                      md5(aref, _n, _C, _h, _w, _v), delta, acc);
                  errors++;
                }
              } else {
                double rel_diff = delta / fabs(md5(aref, _n, _C, _h, _w, _v));
                if (rel_diff > acc) {
                  if (errors < MAX_PRINT_ERRORS) {
                    printf("Not equal!: [%d][%d][%d][%d][%d]: %f != %f (ref), "
                           "delta=%g, rel_diff=%g\n",
                        _n, _C, _h, _w, _v, md5(aout, _n, _C, _h, _w, _v),
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
      printf("Error: number of errors: %ld/%ld, percentage: %f%%\n", errors,
          desc.sizes.output, ((errors * 1.0) / desc.sizes.output) * 100.0);
      return -1;
    }
    return 0;
  }

  template <>
  int __compare_conv_results_plain<float, float, float, float>(
      eld_conv_t<conv::FP32> &desc, float *out, float *ref)
  {
    auto dims = desc.dims.output;
    MD4(float, aout, out, dims.n, dims.c, dims.h, dims.w);
    MD4(float, aref, ref, dims.n, dims.c, dims.h, dims.w);

#define MAX_PRINT_ERRORS (20)
    size_t errors = 0;
    double acc = desc.with_relu ? 1.0 : 1e-5;

#pragma omp parallel for collapse(3)
    iter_each (_n, dims.n) {
      iter_each (_c, dims.c) {
        iter_each (_h, dims.h) {
          iter_each (_w, dims.w) {
            double delta = fabs(
                md4(aout, _n, _c, _h, _w) - md4(aref, _n, _c, _h, _w));
            if (md4(aout, _n, _c, _h, _w) == 0 ||
                md4(aref, _n, _c, _h, _w) == 0) {
              if (delta < acc)
                continue;
              else if (errors < MAX_PRINT_ERRORS) {
                printf("Not equal!: [%d][%d][%d][%d]: %f != %f (ref), "
                       "delta=%g, acc=%g\n",
                    _n, _c, _h, _w, md4(aout, _n, _c, _h, _w),
                    md4(aref, _n, _c, _h, _w), delta, acc);
                errors++;
              }
            } else {
              double rel_diff = delta / fabs(md4(aref, _n, _c, _h, _w));
              if (rel_diff > acc) {
                if (errors < MAX_PRINT_ERRORS) {
                  printf("Not equal!: [%d][%d][%d][%d]: %f != %f (ref), "
                         "delta=%g, rel_diff=%g\n",
                      _n, _c, _h, _w, md4(aout, _n, _c, _h, _w),
                      md4(aref, _n, _c, _h, _w), delta, rel_diff);
                }
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
    }
    if (desc.formats.weights == OIhw16i16o) {
      tweights = (WeightsType *)malloc(desc.byte_sizes.weights);
      reorder<WeightsType, oihw, OIhw16i16o>(tweights, weights, oc, ic, kh, kw);
    }

    if (desc.formats.output == nChw16c) {
      toutput = (OutputType *)malloc(desc.byte_sizes.output);
      reorder<OutputType, nchw, nChw16c>(toutput, output, n, oc, oh, ow);
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

  template int compare_conv_results<float, float, float, float>(
      eld_conv_t<conv::FP32> &, float *, float *);

  template int ref_convolution2d<float, float, float, float>(
      eld_conv_t<conv::FP32> &, float *, float *, float *, float *);

  template int ref_convolution2d_block16<float, float, float, float>(
      eld_conv_t<conv::FP32> &, float *, float *, float *, float *);

}
}
