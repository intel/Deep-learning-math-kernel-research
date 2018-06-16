#!/bin/bash

ROOT_DIR="$(dirname $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd))"

echo Root dir: $ROOT_DIR
echo

nsockets=
ncores_per_socket=
nthreads_per_core=1
nthreads=

nsockets=${nsockets:=$( lscpu | grep 'Socket(s)' | cut -d: -f2 )}
ncores_per_socket=${ncores_per_socket:=$( lscpu | grep 'Core(s) per socket' | cut -d: -f2 )}
nthreads=${nthreads:=$(( nsockets  * ncores_per_socket * nthreads_per_core ))}

OMP_ENV="OMP_NUM_THREADS=$(( nthreads )) \
  KMP_HW_SUBSET=$(( nsockets ))s,$(( ncores_per_socket ))c,$(( nthreads_per_core ))t \
  KMP_AFFINITY=compact,granularity=fine \
  KMP_BLOCKTIME=infinite"
echo OMP Environment: $OMP_ENV

function build() {
  cd "$ROOT_DIR" && make distclean && make -j all && cd -
}

function conv_test() {
  # Default
  n=1; i=0; o=0; h=0; w=0; H=0; W=0; k=3; K=3; p=1; P=1; s=1; S=1
  b=1; r=0; v=1; a=wino; blk_i=0; blk_o=0; blk_t=0; pat_i=1; pat_o=1
  tile_size=5; nteams=0; nthreads=0; execution_mode=0

  OPTIND=1
  while getopts ":n:i:o:h:w:H:W:k:K:p:P:s:S:b:r:v:a:-:" opt; do
    case "$opt" in
      n) n=$OPTARG ;;
      i) i=$OPTARG ;;
      o) o=$OPTARG ;;
      h) h=$OPTARG ;;
      w) w=$OPTARG ;;
      H) H=$OPTARG ;;
      W) W=$OPTARG ;;
      k) k=$OPTARG ;;
      K) K=$OPTARG ;;
      p) p=$OPTARG ;;
      P) P=$OPTARG ;;
      s) s=$OPTARG ;;
      S) S=$OPTARG ;;
      b) b=$OPTARG ;;
      r) r=$OPTARG ;;
      v) v=$OPTARG ;;
      a) a=$OPTARG ;;
      -)
        case "${OPTARG}" in
          nteams) nteams="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          nteams=*) nteams=${OPTARG#*=}
            ;;
          nthreads) nthreads="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          nthreads=*) nthreads=${OPTARG#*=}
            ;;
          execution-mode) execution_mode="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          execution-mode=*) execution_mode=${OPTARG#*=}
            ;;
          blk-i) blk_i="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          blk-i=*) blk_i=${OPTARG#*=}
            ;;
          blk-o) blk_o="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          blk-o=*) blk_o=${OPTARG#*=}
            ;;
          blk-t) blk_t="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          blk-t=*) blk_t=${OPTARG#*=}
            ;;
          pat-i) pat_i="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          pat-i=*) pat_i=${OPTARG#*=}
            ;;
          pat-o) pat_o="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          pat-o=*) pat_o=${OPTARG#*=}
            ;;
          tile-size) tile_size="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          tile-size=*) tile_size=${OPTARG#*=}
            ;;

       esac
       ;;
    esac
  done
  shift $((OPTIND-1))
  eval $OMP_ENV $ROOT_DIR/build/release/bin/elt_conv \
    -n$n -i$i -o$o -h$h -w$w -H$H -W$W -k$k -K$K -p$p -P$P -s$s -S$S \
    -b$b -r$r -v$v -a$a --blk-i=$blk_i --blk-o=$blk_o --blk-t=$blk_t \
    --pat-i=$pat_i --pat-o=$pat_o --tile-size=$tile_size \
    --nteams=$nteams --nthreads=$nthreads --execution-mode=$execution_mode
}

function unit_test() {
  eval $OMP_ENV $ROOT_DIR/build/release/bin/elt_unitests -t
}

function show_help() {
cat <<@

Euler test script:
  -h        display this help and exit.
  -b        rebuild
  -c        convolution test
  -u        unit test

@
}

OPTIND=1
while getopts ":hbuc" opt; do
  case "$opt" in
    h)
      show_help
      exit 0
      ;;
    b)
      echo Release Build...
      build
      ;;
    c)
      echo Run convolution test...
      shift
      conv_test $@
      exit 0
      ;;
    u)
      echo Run unit tests...
      unit_test
      ;;
    \?)
      show_help
      exit 0
      ;;
  esac
done
shift $((OPTIND-1))
