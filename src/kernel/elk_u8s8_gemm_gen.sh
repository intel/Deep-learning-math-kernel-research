#!/bin/bash

# Build gemm kernel instantiation
#

src_file=$1; dst_dir=$2; cc=$3; enable_user_fp16=$4

if [ ! -f $src_file ] || [ ! -d $dst_dir ]; then
  "Invalid src_file=$src_file or dst_dir=$dst_dir"
  exit -1
fi

__u8s8_kgemm_generate_inst__() {
  ktype=$1; dtype=$2; otype=$3; V=$4; Vx=$5; I=$6; S=$7; F=$8;

  cat <<@ > $dst_dir/elk_${ktype}_${dtype}_${otype}_${V}_${Vx}_${I}_${S}_${F}.cpp
// _generated_kgemm_file_
//
#include "$src_file"

using namespace euler;

namespace euler {

#undef E
#define E(O, T) \\
  ${ktype}_kernel_binder::gemm_ker_cls<conv_impl::$dtype, \\
      $otype, $V, $Vx, $I, $S, $F, O, T, 1>::gemm
  ${ktype}_kernel_binder::kgemm<conv_impl::$dtype, $otype>
      *${ktype}_kernel_binder::kgemm_${dtype}_${otype}_${V}_${Vx}_${I}_${S}_${F}[8][32] =
  { // 8
    { // 32
      E(1, 1),
      E(1, 2),
      E(1, 3),
      E(1, 4),
      E(1, 5),
      E(1, 6),
      E(1, 7),
      E(1, 8),
      E(1, 9),
      E(1, 10),
      E(1, 11),
      E(1, 12),
      E(1, 13),
      E(1, 14),
      E(1, 15),
      E(1, 16),
      E(1, 17),
      E(1, 18),
      E(1, 19),
      E(1, 20),
      E(1, 21),
      E(1, 22),
      E(1, 23),
      E(1, 24),
      E(1, 25),
      E(1, 26),
      E(1, 27),
      E(1, 28),
      E(1, 29),
      E(1, 30),
      E(1, 31),
    },
    { // 32
      E(2, 1),
      E(2, 2),
      E(2, 3),
      E(2, 4),
      E(2, 5),
      E(2, 6),
      E(2, 7),
      E(2, 8),
      E(2, 9),
      E(2, 10),
      E(2, 11),
      E(2, 12),
      E(2, 13),
      E(2, 14),
    },
    { // 32
      E(3, 1),
      E(3, 2),
      E(3, 3),
      E(3, 4),
      E(3, 5),
      E(3, 6),
      E(3, 7),
      E(3, 8),
      E(3, 9),
      E(3, 10),
      E(3, 11),
      E(3, 12),
      E(3, 13),
      E(3, 14),
    },
    { // 32
      E(4, 1),
      E(4, 2),
      E(4, 3),
      E(4, 4),
      E(4, 5),
      E(4, 6),
      E(4, 7),
      E(4, 8),
      E(4, 9),
      E(4, 10),
      E(4, 11),
      E(4, 12),
      E(4, 13),
      E(4, 14),
    },
    { // 32
      E(5, 1),
      E(5, 2),
      E(5, 3),
      E(5, 4),
      E(5, 5),
    },
    { // 32
      E(6, 1),
      E(6, 2),
      E(6, 3),
      E(6, 4),
    },
    { // 32
      E(7, 1),
      E(7, 2),
      E(7, 3),
    },
    { // 32
      E(8, 1),
      E(8, 2),
      E(8, 3),
      E(8, 4),
      E(8, 5),
      E(8, 6),
      E(8, 7),
      E(8, 8),
    },
  };

} // namespace
@
}

if [ $enable_user_fp16 == "ON" ]; then
  eval $($cc -DENABLE_USER_FP16 -DBUILD_OTJ_TBL -E $src_file 2>&1 | grep _generate_inst_)
else
  eval $($cc -DBUILD_OTJ_TBL -E $src_file 2>&1 | grep _generate_inst_)
fi
