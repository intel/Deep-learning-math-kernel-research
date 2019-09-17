#!/bin/bash

source ./scripts/best_configs/common.sh $@
COMMON="$COMMON "

echo "Compute conv: 3x3"
# mb1_ic3oc32_ih224oh112kh3sh2dh0ph0_iw224ow112kw3sw2dw0pw0
NSOCKETS=1 ./scripts/run.sh -c -i3 -h224 -o32 -H112 -k3 -K3 -s2 -S2 -p1 -P1 -n64 -adirect --execution-mode=0xa160 --blk-i=1 --blk-o=1 --flt-o=2 --flt-t=14 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32z --sampling-kind=2 --input-format=nhwc --output-format=nChw16c --weights-format=OIhw16i16o

echo "Compute conv 1x1"
# Conv__226: mb1_ic32oc64_ih112oh112kh1sh1dh0ph0_iw112ow112kw1sw1dw0pw0 
NSOCKETS=1 ./scripts/run.sh -c -i32 -h112 -o64 -H112 -k1 -K1 -s1 -S1 -p0 -P0 -n64 -adirect_1x1 --execution-mode=0xc160 --blk-i=2 --blk-o=2 --flt-o=2 --flt-t=12 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=OIhw16i16o

# Conv__228: mb1_ic64oc128_ih56oh56kh1sh1dh0ph0_iw56ow56kw1sw1dw0pw0
NSOCKETS=1 ./scripts/run.sh -c -i64 -h56 -o128 -H56 -k1 -K1 -s1 -S1 -p0 -P0 -n64 -adirect_1x1 --execution-mode=0xc160 --blk-i=4 --blk-o=4 --flt-o=2 --flt-t=12 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=OIhw16i16o

# Conv__230: mb1_ic128oc128_ih56oh56kh1sh1dh0ph0_iw56ow56kw1sw1dw0pw0
NSOCKETS=1 ./scripts/run.sh -c -i128 -h56 -o128 -H56 -k1 -K1 -s1 -S1 -p0 -P0 -n64 -adirect_1x1 --execution-mode=0xc160 --blk-i=8 --blk-o=4 --flt-o=2 --flt-t=12 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=OIhw16i16o

# Conv__232: mb1_ic128oc256_ih28oh28kh1sh1dh0ph0_iw28ow28kw1sw1dw0pw0
NSOCKETS=1 ./scripts/run.sh -c -i128 -h28 -o256 -H28 -k1 -K1 -s1 -S1 -p0 -P0 -n64 -adirect_1x1 --execution-mode=0xc160 --blk-i=8 --blk-o=8 --flt-o=2 --flt-t=12 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=OIhw16i16o

# Conv__234: mb1_ic256oc256_ih28oh28kh1sh1dh0ph0_iw28ow28kw1sw1dw0pw0
NSOCKETS=1 ./scripts/run.sh -c -i256 -h28 -o256 -H28 -k1 -K1 -s1 -S1 -p0 -P0 -n64 -adirect_1x1 --execution-mode=0xc160 --blk-i=16 --blk-o=8 --flt-o=2 --flt-t=12 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=OIhw16i16o

# Conv__236: mb1_ic256oc512_ih14oh14kh1sh1dh0ph0_iw14ow14kw1sw1dw0pw0
NSOCKETS=1 ./scripts/run.sh -c -i256 -h14 -o512 -H14 -k1 -K1 -s1 -S1 -p0 -P0 -n64 -adirect_1x1 --execution-mode=0xc160 --blk-i=16 --blk-o=8 --flt-o=2 --flt-t=12 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=OIhw16i16o

# Conv__238, Conv__240, Conv__242, Conv__244, Conv__246: mb1_ic512oc512_ih14oh14kh1sh1dh0ph0_iw14ow14kw1sw1dw0pw0
NSOCKETS=1 ./scripts/run.sh -c -i512 -h14 -o512 -H14 -k1 -K1 -s1 -S1 -p0 -P0 -n64 -adirect_1x1 --execution-mode=0xc160 --blk-i=32 --blk-o=16 --flt-o=2 --flt-t=12 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=OIhw16i16o

# Conv__248: mb1_ic512oc1024_ih7oh7kh1sh1dh0ph0_iw7ow7kw1sw1dw0pw0
NSOCKETS=1 ./scripts/run.sh -c -i512 -h7 -o1024 -H7 -k1 -K1 -s1 -S1 -p0 -P0 -n64 -adirect_1x1 --execution-mode=0xc160 --blk-i=32 --blk-o=8 --flt-o=4 --flt-t=6 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=OIhw16i16o

# Conv__250: mb1_ic1024oc1024_ih7oh7kh1sh1dh0ph0_iw7ow7kw1sw1dw0pw0
NSOCKETS=1 ./scripts/run.sh -c -i1024 -h7 -o1024 -H7 -k1 -K1 -s1 -S1 -p0 -P0 -n64 -adirect_1x1 --execution-mode=0xc160 --blk-i=64 --blk-o=4 --flt-o=4 --flt-t=6 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nhwc --weights-format=OIhw16i16o

# Conv__252: mb1_ic1024oc1001_ih1oh1kh1sh1dh0ph0_iw1ow1kw1sw1dw0pw0
# FC
NSOCKETS=1 ./scripts/run.sh -c -i1024 -h8 -o1001 -H8 -k1 -K1 -s1 -S1 -p0 -P0 -n1 -adirect --execution-mode=0xd160 --blk-i=64 --blk-o=1 --flt-o=3 --flt-t=4 --pat-i=1 --pat-o=1 $COMMON --data-type-cfg=U8F32F32F32 --sampling-kind=2 --input-format=nhwc --output-format=nhwc


echo "Compute group conv"
# Conv2d_1_depthwise: mb1_g32ic32oc32_ih112oh112kh3sh1dh0ph1_iw112ow112kw3sw1dw0pw1
NSOCKETS=1 ./scripts/run.sh -c -g32 -i32 -h112 -o32 -H112 -k3 -K3 -s1 -S1 -p1 -P1 -n64 -adirect --execution-mode=0xa160 --blk-i=1 --blk-o=1 --flt-o=1 --flt-t=28 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=goihw

# Conv2d_2_depthwise: mb1_g64ic64oc64_ih112oh56kh3sh2dh0ph0_iw112ow56kw3sw2dw0pw0
NSOCKETS=1 ./scripts/run.sh -c -g64 -i64 -h112 -o64 -H56 -k3 -K3 -s2 -S2 -p1 -P1 -n64 -adirect --execution-mode=0xa160 --blk-i=1 --blk-o=1 --flt-o=1 --flt-t=28 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=goihw 

# Conv2d_3_depthwise: mb1_g128ic128oc128_ih56oh56kh3sh1dh0ph1_iw56ow56kw3sw1dw0pw1
NSOCKETS=1 ./scripts/run.sh -c -g128 -i128 -h56 -o128 -H56 -k3 -K3 -s1 -S1 -p1 -P1 -n64 -adirect --execution-mode=0xa160 --blk-i=1 --blk-o=1 --flt-o=1 --flt-t=14 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=goihw 

# Conv2d_4_depthwise: mb1_g128ic128oc128_ih56oh28kh3sh2dh0ph0_iw56ow28kw3sw2dw0pw0
NSOCKETS=1 ./scripts/run.sh -c -g128 -i128 -h56 -o128 -H28 -k3 -K3 -s2 -S2 -p1 -P1 -n64 -adirect --execution-mode=0xa160 --blk-i=1 --blk-o=1 --flt-o=1 --flt-t=28 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=goihw 

# Conv2d_5_depthwise: mb1_g256ic256oc256_ih28oh28kh3sh1dh0ph1_iw28ow28kw3sw1dw0pw1
NSOCKETS=1 ./scripts/run.sh -c -g256 -i256 -h28 -o256 -H28 -k3 -K3 -s1 -S1 -p1 -P1 -n64 -adirect --execution-mode=0xa160 --blk-i=1 --blk-o=1 --flt-o=1 --flt-t=28 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=goihw 

# Conv2d_6_depthwise: mb1_g256ic256oc256_ih28oh14kh3sh2dh0ph0_iw28ow14kw3sw2dw0pw0
NSOCKETS=1 ./scripts/run.sh -c -g256 -i256 -h28 -o256 -H14 -k3 -K3 -s2 -S2 -p1 -P1 -n64 -adirect --execution-mode=0xa160 --blk-i=1 --blk-o=1 --flt-o=1 --flt-t=14 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=goihw 

# Conv2d_7_depthwise, Conv2d_8_depthwise, Conv2d_9_depthwise, Conv2d_10_depthwise, Conv2d_11_depthwise
# mb1_g512ic512oc512_ih14oh14kh3sh1dh0ph1_iw14ow14kw3sw1dw0pw1
NSOCKETS=1 ./scripts/run.sh -c -g512 -i512 -h14 -o512 -H14 -k3 -K3 -s1 -S1 -p1 -P1 -n64 -adirect --execution-mode=0xa160 --blk-i=1 --blk-o=1 --flt-o=1 --flt-t=14 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=goihw 

# Conv2d_12_depthwise: mb1_g512ic512oc512_ih14oh7kh3sh2dh0ph0_iw14ow7kw3sw2dw0pw0
NSOCKETS=1 ./scripts/run.sh -c -g512 -i512 -h14 -o512 -H7 -k3 -K3 -s2 -S2 -p1 -P1 -n64 -adirect --execution-mode=0xa160 --blk-i=1 --blk-o=1 --flt-o=1 --flt-t=7 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=goihw 

# Conv2d_13_depthwise: mb1_g1024ic1024oc1024_ih7oh7kh3sh1dh0ph1_iw7ow7kw3sw1dw0pw1
NSOCKETS=1 ./scripts/run.sh -c -g1024 -i1024 -h7 -o1024 -H7 -k3 -K3 -s1 -S1 -p1 -P1 -n64 -adirect --execution-mode=0xa160 --blk-i=1 --blk-o=1 --flt-o=1 --flt-t=7 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2 --input-format=nChw16c --output-format=nChw16c --weights-format=goihw 
