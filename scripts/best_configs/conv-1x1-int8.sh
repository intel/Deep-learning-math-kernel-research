#!/bin/bash

source ./scripts/best_configs/common.sh $@
COMMON="$COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 "

# clx low bin, 28c

# direct 1x1, s=1
NSOCKETS=1 ./scripts/run.sh -c -i1024 -h28 -w28 -o2048 -H28 -W28 -k1 -K1 -s1 -S1 -p0 -P0 -n1 -adirect_1x1 --execution-mode=0xa160 --blk-i=16 --blk-o=2 --pat-o=1 --flt-o=2 --flt-t=14 --pat-i=4   $COMMON
# 10.8 tflops, ref (7.6)
NSOCKETS=1 ./scripts/run.sh -c -i1024 -h28 -w28 -o2048 -H28 -W28 -k1 -K1 -s1 -S1 -p0 -P0 -n64 -adirect_1x1 --execution-mode=0xa160 --blk-i=64 --blk-o=16 --pat-o=1 --flt-o=2 --flt-t=12 --pat-i=1   $COMMON
# 10.9 tflops(12)

# direct, s=1
NSOCKETS=1 ./scripts/run.sh -c -i1024 -h28 -w28 -o2048 -H28 -W28 -k1 -K1 -s1 -S1 -p0 -P0 -n1 -adirect --execution-mode=0xa160 --blk-i=16 --blk-o=1 --pat-o=2 --flt-o=2 --flt-t=14 --pat-i=4   $COMMON
# 10.0, ref (7.6tflops)
NSOCKETS=1 ./scripts/run.sh -c -i1024 -h28 -w28 -o2048 -H28 -W28 -k1 -K1 -s1 -S1 -p0 -P0 -n64 -adirect --execution-mode=0xa160 --blk-i=64 --blk-o=4 --pat-o=2 --flt-o=2 --flt-t=14 --pat-i=1   $COMMON
# 9.22 tflops, ref(12 tflops)

# direct 1x1, s=2
NSOCKETS=1 ./scripts/run.sh -c -i1024 -h28 -w28 -o2048 -H14 -W14 -k1 -K1 -s2 -S2 -p0 -P0 -n1 -adirect_1x1 --execution-mode=0xa160 --blk-i=32 --blk-o=2 --pat-o=1 --flt-o=2 --flt-t=14 --pat-i=2   $COMMON
# 7.0 tflops, ref (7.7 tflops)
NSOCKETS=1 ./scripts/run.sh -c -i1024 -h28 -w28 -o2048 -H14 -W14 -k1 -K1 -s2 -S2 -p0 -P0 -n64 -adirect_1x1 --execution-mode=0xa160 --blk-i=64 --blk-o=2 --pat-o=1 --flt-o=2 --flt-t=14 --pat-i=1   $COMMON
# 10.03 tflops. ref (9.7 tflops)

# direct s=2
NSOCKETS=1 ./scripts/run.sh -c -i1024 -h28 -w28 -o2048 -H14 -W14 -k1 -K1 -s2 -S2 -p0 -P0 -n1 -adirect --execution-mode=0xa160 --blk-i=16 --blk-o=2 --pat-o=1 --flt-o=2 --flt-t=14 --pat-i=4   $COMMON
# 6.64 (7.7)
NSOCKETS=1 ./scripts/run.sh -c -i1024 -h28 -w28 -o2048 -H14 -W14 -k1 -K1 -s2 -S2 -p0 -P0 -n64 -adirect --execution-mode=0xa160 --blk-i=64 --blk-o=2 --pat-o=1 --flt-o=2 --flt-t=14 --pat-i=1   $COMMON
# 8.9 (9.7)


