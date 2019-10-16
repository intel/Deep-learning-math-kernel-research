#!/bin/bash

source ./scripts/best_configs/common.sh $@
COMMON="$COMMON "

# conv: 3x3
# Conv__224
NSOCKETS=1 ./scripts/run.sh -c --name=Conv__224 -i3 -h224 -o32 -H112 -k3 -K3 -s2 -S2 -p0 -P0 -n1 -adirect --execution-mode=0xc160 --blk-i=1 --blk-o=1 --flt-o=2 --flt-t=14 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32z --sampling-kind=2 --input-format=nhwc --output-format=nChw16c --weights-format=OIhw16i16o

# conv 1x1
# Conv__226
NSOCKETS=1 ./scripts/run.sh -c --name=Conv__226 -i32 -h112 -o64 -H112 -k1 -K1 -s1 -S1 -p0 -P0 -n1 -adirect_1x1 --execution-mode=0xa160 --blk-i=2 --blk-o=2 --flt-o=2 --flt-t=12 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=OIhw16i16o

# Conv__228
NSOCKETS=1 ./scripts/run.sh -c --name=Conv__228 -i64 -h56 -o128 -H56 -k1 -K1 -s1 -S1 -p0 -P0 -n1 -adirect_1x1 --execution-mode=0xa160 --blk-i=4 --blk-o=4 --flt-o=2 --flt-t=12 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=OIhw16i16o

# Conv__230
NSOCKETS=1 ./scripts/run.sh -c --name=Conv__230 -i128 -h56 -o128 -H56 -k1 -K1 -s1 -S1 -p0 -P0 -n1 -adirect_1x1 --execution-mode=0xa160 --blk-i=8 --blk-o=4 --flt-o=2 --flt-t=12 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=OIhw16i16o

# Conv__232
NSOCKETS=1 ./scripts/run.sh -c --name=Conv__232 -i128 -h28 -o256 -H28 -k1 -K1 -s1 -S1 -p0 -P0 -n1 -adirect_1x1 --execution-mode=0xa160 --blk-i=8 --blk-o=4 --flt-o=2 --flt-t=12 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=OIhw16i16o

# Conv__234
NSOCKETS=1 ./scripts/run.sh -c --name=Conv__234 -i256 -h28 -o256 -H28 -k1 -K1 -s1 -S1 -p0 -P0 -n1 -adirect_1x1 --execution-mode=0xa160 --blk-i=16 --blk-o=4 --flt-o=2 --flt-t=12 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=OIhw16i16o

# Conv__236
NSOCKETS=1 ./scripts/run.sh -c --name=Conv__236 -i256 -h14 -o512 -H14 -k1 -K1 -s1 -S1 -p0 -P0 -n1 -adirect_1x1 --execution-mode=0xa160 --blk-i=16 --blk-o=2 --flt-o=2 --flt-t=12 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=OIhw16i16o

# Conv__238, Conv__240, Conv__242, Conv__244, Conv__246
NSOCKETS=1 ./scripts/run.sh -c --name=Conv__238 -i512 -h14 -o512 -H14 -k1 -K1 -s1 -S1 -p0 -P0 -n1 -adirect_1x1 --execution-mode=0xa160 --blk-i=32 --blk-o=1 --flt-o=2 --flt-t=12 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=OIhw16i16o

# Conv__248
NSOCKETS=1 ./scripts/run.sh -c --name=Conv__248 -i512 -h7 -o1024 -H7 -k1 -K1 -s1 -S1 -p0 -P0 -n1 -adirect_1x1 --execution-mode=0xa160 --blk-i=32 --blk-o=2 --flt-o=2 --flt-t=13 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=OIhw16i16o

# Conv__250
NSOCKETS=1 ./scripts/run.sh -c --name=Conv__250 -i1024 -h7 -o1024 -H7 -k1 -K1 -s1 -S1 -p0 -P0 -n1 -adirect_1x1 --execution-mode=0xa160 --blk-i=64 --blk-o=2 --flt-o=2 --flt-t=13 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nhwc --weights-format=OIhw16i16o

# Conv__252
# FC
NSOCKETS=1 ./scripts/run.sh -c --name=Conv__252 -i1024 -h1 -o1001 -H1 -k1 -K1 -s1 -S1 -p0 -P0 -n1 -adirect --execution-mode=0xa160 --blk-i=64 --blk-o=1 --flt-o=3 --flt-t=1 --pat-i=1 --pat-o=1 $COMMON --data-type-cfg=U8F32F32F32 --sampling-kind=2 --input-format=nhwc --output-format=nhwc


# group conv
# Conv2d_1_depthwise
NSOCKETS=1 ./scripts/run.sh -c --name=Conv2d_1_depthwise -g32 -i32 -h112 -o32 -H112 -k3 -K3 -s1 -S1 -p1 -P1 -n1 -adirect --execution-mode=0xc160 --blk-i=1 --blk-o=1 --flt-o=1 --flt-t=28 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=goihw

# Conv2d_2_depthwise
NSOCKETS=1 ./scripts/run.sh -c --name=Conv2d_2_depthwise -g64 -i64 -h112 -o64 -H56 -k3 -K3 -s2 -S2 -p0 -P0 -n1 -adirect --execution-mode=0xc160 --blk-i=1 --blk-o=1 --flt-o=1 --flt-t=28 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=goihw 

# Conv2d_3_depthwise
NSOCKETS=1 ./scripts/run.sh -c --name=Conv2d_3_depthwise -g128 -i128 -h56 -o128 -H56 -k3 -K3 -s1 -S1 -p1 -P1 -n1 -adirect --execution-mode=0xc160 --blk-i=1 --blk-o=1 --flt-o=1 --flt-t=14 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=goihw 

# Conv2d_4_depthwise
NSOCKETS=1 ./scripts/run.sh -c --name=Conv2d_4_depthwise -g128 -i128 -h56 -o128 -H28 -k3 -K3 -s2 -S2 -p0 -P0 -n1 -adirect --execution-mode=0xc160 --blk-i=1 --blk-o=1 --flt-o=1 --flt-t=28 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=goihw 

# Conv2d_5_depthwise
NSOCKETS=1 ./scripts/run.sh -c --name=Conv2d_5_depthwise -g256 -i256 -h28 -o256 -H28 -k3 -K3 -s1 -S1 -p1 -P1 -n1 -adirect --execution-mode=0xc160 --blk-i=1 --blk-o=1 --flt-o=1 --flt-t=14 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=goihw 

# Conv2d_6_depthwise
NSOCKETS=1 ./scripts/run.sh -c --name=Conv2d_6_depthwise -g256 -i256 -h28 -o256 -H14 -k3 -K3 -s2 -S2 -p0 -P0 -n1 -adirect --execution-mode=0xc160 --blk-i=1 --blk-o=1 --flt-o=1 --flt-t=14 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=goihw 

# Conv2d_7_depthwise, Conv2d_8_depthwise, Conv2d_9_depthwise, Conv2d_10_depthwise, Conv2d_11_depthwise
NSOCKETS=1 ./scripts/run.sh -c --name=Conv2d_7_depthwise -g512 -i512 -h14 -o512 -H14 -k3 -K3 -s1 -S1 -p1 -P1 -n1 -adirect --execution-mode=0xc160 --blk-i=1 --blk-o=1 --flt-o=1 --flt-t=14 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=goihw 

# Conv2d_12_depthwise
NSOCKETS=1 ./scripts/run.sh -c --name=Conv2d_12_depthwise -g512 -i512 -h14 -o512 -H7 -k3 -K3 -s2 -S2 -p0 -P0 -n1 -adirect --execution-mode=0xc160 --blk-i=1 --blk-o=1 --flt-o=1 --flt-t=7 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=goihw 

# Conv2d_13_depthwise
NSOCKETS=1 ./scripts/run.sh -c --name=Conv2d_13_depthwise -g1024 -i1024 -h7 -o1024 -H7 -k3 -K3 -s1 -S1 -p1 -P1 -n1 -adirect --execution-mode=0xc160 --blk-i=1 --blk-o=1 --flt-o=1 --flt-t=7 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=goihw 
