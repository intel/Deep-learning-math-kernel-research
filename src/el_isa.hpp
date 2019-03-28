#pragma once

#include <cpuid.h>
#include <immintrin.h>

#define SSE42      0x00100000 // ecx[bit 20]
#define AVX        0x10000000 // ecx[bit 28]
#define AVX2       0x00000020 // ebx[bit 5]
#define AVX512F    0x00010000 // ebx[bit 16]
#define AVX512BW   0x40000000 // ebx[bit 30]
#define AVX512VNNI 0x00000800 // ecx[bit 11]

enum {
  isa_undef = 0,
  sse42,
  avx,
  avx2,
  avx512_common,
  avx512_core,
  avx512_core_vnni
};

struct cpuid_regs {
  unsigned int eax;
  unsigned int ebx;
  unsigned int ecx;
  unsigned int edx;
};

static cpuid_regs regs0, regs1;
static void __attribute__ ((constructor)) get_cpuinfo(void) {
  __cpuid(0x1, regs0.eax, regs0.ebx, regs0.ecx, regs0.edx);
  __cpuid_count(0x7, 0, regs1.eax, regs1.ebx, regs1.ecx, regs1.edx);
}

static inline bool cpu_has(int isa)
{
  switch (isa) {
  case sse42:
    return !!(regs0.ecx & SSE42);
  case avx:
    return !!(regs0.ecx & AVX);
  case avx2:
    return !!(regs1.ebx & AVX2);
  case avx512_common:
    return !!(regs1.ebx & AVX512F);
  case avx512_core:
    return !!(regs1.ebx & AVX512F) && !!(regs1.ebx & AVX512BW);
#if defined(WITH_VNNI)
  case avx512_core_vnni:
    return !!(regs1.ebx & AVX512F) && !!(regs1.ebx & AVX512BW)
        && !!(regs1.ecx & AVX512VNNI);
#endif
  default:
    return false;
  }
  return false;
}

// CPU V length by byte
static inline int cpu_vector_length() {
  if (cpu_has(avx512_common))
    return 64; // zmm
  else if (cpu_has(avx))
    return 32; // ymm
  else if (cpu_has(sse42))
    return 16; // xmm
  else
    return 8;
}
