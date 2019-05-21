#!/bin/bash

source ./scripts/best_configs/common.sh $@
COMMON="$COMMON --f16c-opt=1"

# nhwc
NSOCKETS=1 ./scripts/run.sh -c -n1 -i1024 -o3072 -h1 -w42 -H1 -W42 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect --blk-i=16 --pat-i=4 --blk-o=1 --flt-o=2 --flt-t=7 --execution-mode=0xd060 --pat-o=1 $COMMON --input-format=nhwc --output-format=nhwc
NSOCKETS=1 ./scripts/run.sh -c -n1 -i1024 -o1024 -h1 -w42 -H1 -W42 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect --blk-i=32 --pat-i=2 --blk-o=1 --flt-o=2 --flt-t=14 --execution-mode=0xd060 --pat-o=1 $COMMON --input-format=nhwc --output-format=nhwc
NSOCKETS=1 ./scripts/run.sh -c -n1 -i1024 -o4096 -h1 -w42 -H1 -W42 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect --blk-i=16 --pat-i=4 --blk-o=1 --flt-o=2 --flt-t=7 --execution-mode=0xd060 --pat-o=1 $COMMON --input-format=nhwc --output-format=nhwc
NSOCKETS=1 ./scripts/run.sh -c -n1 -i4096 -o1024 -h1 -w42 -H1 -W42 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect --blk-i=16 --pat-i=16 --blk-o=1 --flt-o=1 --flt-t=7 --execution-mode=0xd060 --pat-o=1 $COMMON --input-format=nhwc --output-format=nhwc

# blocked
NSOCKETS=1 ./scripts/run.sh -c -n1 -i1024 -o3072 -h1 -w42 -H1 -W42 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect --blk-i=32 --pat-i=2 --blk-o=1 --flt-o=1 --flt-t=28 --execution-mode=0xd060 --pat-o=1 $COMMON
NSOCKETS=1 ./scripts/run.sh -c -n1 -i1024 -o1024 -h1 -w42 -H1 -W42 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect --blk-i=32 --pat-i=2 --blk-o=1 --flt-o=1 --flt-t=28 --execution-mode=0xd060 --pat-o=1 $COMMON
NSOCKETS=1 ./scripts/run.sh -c -n1 -i1024 -o4096 -h1 -w42 -H1 -W42 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect --blk-i=32 --pat-i=2  --blk-o=1 --flt-o=1 --flt-t=28 --execution-mode=0xd060 --pat-o=1 $COMMON
NSOCKETS=1 ./scripts/run.sh -c -n1 -i4096 -o1024 -h1 -w42 -H1 -W42 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect --blk-i=16 --pat-i=16 --blk-o=1 --flt-o=1 --flt-t=28 --execution-mode=0xd060 --pat-o=1 $COMMON
