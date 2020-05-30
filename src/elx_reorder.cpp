#include <stdlib.h>
#include "euler.hpp"
#include "euler_reorder.hpp"
#include "el_mdarray.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "el_parallel.hpp"

namespace euler {

template <typename Type>
reorder<Type, nchw, nhwc>::reorder(
    Type *dst, Type *src, int n, int c, int h, int w) {
  estl::parallel_for<4>([&](int _n, int _c, int _h, int _w) {
    MD4(Type, asrc, src, n, h, w, c);
    MD4(Type, adst, dst, n, c, h, w);

    md4(adst, _n, _c, _h, _w) = md4(asrc, _n, _h, _w, _c);
  }, n, c, h, w);
}

template <typename Type>
reorder<Type, nhwc, nchw>::reorder(
    Type *dst, Type *src, int n, int c, int h, int w) {
  estl::parallel_for<4>([&](int _n, int _c, int _h, int _w) {
    MD4(Type, asrc, src, n, c, h, w);
    MD4(Type, adst, dst, n, h, w, c);

    md4(adst, _n, _h, _w, _c) = md4(asrc, _n, _c, _h, _w);
  }, n, c, h, w);
}

template <typename Type>
reorder<Type, nchw, nChw16c>::reorder(
    Type *dst, Type *src, int n, int c, int h, int w) {
  int C = ALIGNUP(c, 16) / 16; // padding
  int Vr = c % 16 ? c % 16 : 16;

  estl::parallel_for<4>([&](int _n, int _C, int _h, int _w) {
    MD5(Type, asrc, src, n, C, h, w, 16);
    MD4(Type, adst, dst, n, c, h, w);

    int v = (_C == C - 1) ? Vr : 16;
    iter_each(_v, v) {
      md4(adst, _n, _C * 16 + _v, _h, _w) = md5(asrc, _n, _C, _h, _w, _v);
    }
  }, n, C, h, w);
}

template <typename Type>
reorder<Type, nChw16c, nchw>::reorder(
    Type *dst, Type *src, int n, int c, int h, int w) {
  int C = ALIGNUP(c, 16) / 16; // padding
  int Vr = c % 16 ? c % 16 : 16;

  estl::parallel_for<4>([&](int _n, int _C, int _h, int _w) {
    MD4(Type, asrc, src, n, c, h, w);
    MD5(Type, adst, dst, n, C, h, w, 16);

    int v = (_C == C - 1) ? Vr : 16;
    iter_each(_v, 16) {
      if (_v < v)
        md5(adst, _n, _C, _h, _w, _v) = md4(asrc, _n, _C * 16 + _v, _h, _w);
      else
        md5(adst, _n, _C, _h, _w, _v) = 0;
    }
  }, n, C, h, w);
}

template <typename Type>
reorder<Type, OIhw16i16o, oihw>::reorder(
    Type *dst, Type *src, int o, int i, int h, int w) {
  int O = ALIGNUP(o, 16) / 16; // padding
  int I = ALIGNUP(i, 16) / 16; // padding
  int Or = o % 16 ? o % 16 : 16;
  int Ir = i % 16 ? i % 16 : 16;

  estl::parallel_for<4>([&](int _O, int _I, int _h, int _w) {
    MD4(Type, asrc, src, o, i, h, w);
    MD6(Type, adst, dst, O, I, h, w, 16, 16);

    int ov = (_O == O - 1) ? Or : 16;
    int iv = (_I == I - 1) ? Ir : 16;
    iter_each(_iv, 16) {
      iter_each(_ov, 16) {
        if (_iv < iv && _ov < ov)
          md6(adst, _O, _I, _h, _w, _iv, _ov) =
              md4(asrc, _O * 16 + _ov, _I * 16 + _iv, _h, _w);
        else
          md6(adst, _O, _I, _h, _w, _iv, _ov) = 0;
      }
    }
  }, O, I, h, w);
}

template <typename Type>
reorder<Type, gOIhw16i16o, goihw>::reorder(
    Type *dst, Type *src, int g, int o, int i, int h, int w) {
  int O = ALIGNUP(o, 16) / 16; // padding
  int I = ALIGNUP(i, 16) / 16; // padding
  int Or = o % 16 ? o % 16 : 16;
  int Ir = i % 16 ? i % 16 : 16;

  estl::parallel_for<5>([&](int _g, int _O, int _I, int _h, int _w) {
    MD5(Type, asrc, src, g, o, i, h, w);
    MD7(Type, adst, dst, g, O, I, h, w, 16, 16);

    int ov = (_O == O - 1) ? Or : 16;
    int iv = (_I == I - 1) ? Ir : 16;
    iter_each(_iv, 16) {
      iter_each(_ov, 16) {
        if (_iv < iv && _ov < ov)
          md7(adst, _g, _O, _I, _h, _w, _iv, _ov) =
              md5(asrc, _g, _O * 16 + _ov, _I * 16 + _iv, _h, _w);
        else
          md7(adst, _g, _O, _I, _h, _w, _iv, _ov) = 0;
      }
    }
  }, g, O, I, h, w);
}

template <typename Type>
reorder<Type, oihw, OIhw16i16o>::reorder(
    Type *dst, Type *src, int o, int i, int h, int w) {
  int O = ALIGNUP(o, 16) / 16; // padding
  int I = ALIGNUP(i, 16) / 16; // padding
  int Or = o % 16 ? o % 16 : 16;
  int Ir = i % 16 ? i % 16 : 16;

  estl::parallel_for<4>([&](int _O, int _I, int _h, int _w) {
    MD6(Type, asrc, src, O, I, h, w, 16, 16);
    MD4(Type, adst, dst, o, i, h, w);

    int ov = _O == O - 1 ? Or : 16;
    int iv = _I == I - 1 ? Ir : 16;
    iter_each(_iv, iv) {
      iter_each(_ov, ov) {
        md4(adst, _O * 16 + _ov, _I * 16 + _iv, _h, _w) =
            md6(asrc, _O, _I, _h, _w, _iv, _ov);
      }
    }
  }, O, I, h, w);
}

template <typename Type>
reorder<Type, goihw, gOIhw16i16o>::reorder(
    Type *dst, Type *src, int g, int o, int i, int h, int w) {
  int O = ALIGNUP(o, 16) / 16; // padding
  int I = ALIGNUP(i, 16) / 16; // padding
  int Or = o % 16 ? o % 16 : 16;
  int Ir = i % 16 ? i % 16 : 16;

  estl::parallel_for<5>([&](int _g, int _O, int _I, int _h, int _w) {
    MD7(Type, asrc, src, g, O, I, h, w, 16, 16);
    MD5(Type, adst, dst, g, o, i, h, w);

    int ov = _O == O - 1 ? Or : 16;
    int iv = _I == I - 1 ? Ir : 16;
    iter_each(_iv, iv) {
      iter_each(_ov, ov) {
        md5(adst, _g, _O * 16 + _ov, _I * 16 + _iv, _h, _w) =
            md7(asrc, _g, _O, _I, _h, _w, _iv, _ov);
      }
    }
  }, g, O, I, h, w);
}

template <typename Type>
reorder<Type, oihw, hwio>::reorder(
    Type *dst, Type *src, int o, int i, int h, int w) {
  estl::parallel_for<4>([&](int _o, int _i, int _h, int _w) {
    MD4(Type, asrc, src, h, w, i, o);
    MD4(Type, adst, dst, o, i, h, w);
    md4(adst, _o, _i, _h, _w) = md4(asrc, _h, _w, _i, _o);
  }, o, i, h, w);
}

template <typename Type>
reorder<Type, goihw, ghwio>::reorder(
    Type *dst, Type *src, int g, int o, int i, int h, int w) {

  estl::parallel_for<5>([&](int _g, int _o, int _i, int _h, int _w) {
    MD5(Type, asrc, src, g, h, w, i, o);
    MD5(Type, adst, dst, g, o, i, h, w);
    md5(adst, _g, _o, _i, _h, _w) = md5(asrc, _g, _h, _w, _i, _o);
  }, g, o, i, h, w);
}

template <typename Type>
reorder<Type, hwio, oihw>::reorder(
    Type *dst, Type *src, int o, int i, int h, int w) {
  estl::parallel_for<4>([&](int _h, int _w, int _i, int _o) {
    MD4(Type, asrc, src, o, i, h, w);
    MD4(Type, adst, dst, h, w, i, o);
    md4(adst, _h, _w, _i, _o) = md4(asrc, _o, _i, _h, _w);
  }, h, w, i, o);
}

template <typename Type>
reorder<Type, ghwio, goihw>::reorder(
    Type *dst, Type *src, int g, int o, int i, int h, int w) {
  estl::parallel_for<5>([&](int _g, int _h, int _w, int _i, int _o) {
    MD5(Type, asrc, src, g, o, i, h, w);
    MD5(Type, adst, dst, g, h, w, i, o);
    md5(adst, _g, _h, _w, _i, _o) = md5(asrc, _g, _o, _i, _h, _w);
  }, g, h, w, i, o);
}

template struct reorder<float,   nchw, nhwc>;
template struct reorder<float,   nhwc, nchw>;
template struct reorder<float,   nchw, nChw16c>;
template struct reorder<float,   nChw16c, nchw>;
template struct reorder<float,   oihw, OIhw16i16o>;
template struct reorder<float,   goihw, gOIhw16i16o>;
template struct reorder<float,   OIhw16i16o, oihw>;
template struct reorder<float,   gOIhw16i16o, goihw>;
template struct reorder<float,   oihw, hwio>;
template struct reorder<float,   goihw, ghwio>;
template struct reorder<float,   hwio, oihw>;
template struct reorder<float,   ghwio, goihw>;
template struct reorder<uint8_t, nchw, nhwc>;
template struct reorder<uint8_t, nhwc, nchw>;
template struct reorder<uint8_t, nchw, nChw16c>;
template struct reorder<uint8_t, nChw16c, nchw>;
template struct reorder<uint8_t, oihw, OIhw16i16o>;
template struct reorder<uint8_t, goihw, gOIhw16i16o>;
template struct reorder<uint8_t, OIhw16i16o, oihw>;
template struct reorder<uint8_t, gOIhw16i16o, goihw>;
template struct reorder<uint8_t, oihw, hwio>;
template struct reorder<uint8_t, goihw, ghwio>;
template struct reorder<uint8_t, hwio, oihw>;
template struct reorder<uint8_t, ghwio, goihw>;

} // namespace euler
