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
  n=1; i=64; o=64; h=224; w=224; H=224; W=224; k=3; K=3; p=1; P=1; s=1; S=1
  b=1; r=0; v=1; a=wino; nteams=0; nthreads=0

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
       esac
       ;;
    esac
  done
  shift $((OPTIND-1))
  eval $OMP_ENV $ROOT_DIR/build/release/bin/elt_conv \
    -n$n -i$i -o$o -h$h -w$w -H$H -W$W -k$k -K$K -p$p -P$P -s$s -S$S \
    -b$b -r$r -v$v -a$a --nteams=$nteams --nthreads=$nthreads
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
