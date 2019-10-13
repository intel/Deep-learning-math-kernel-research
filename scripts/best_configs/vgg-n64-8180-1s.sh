#/bin/bash

# VGG
# batch-size: 64
# SKX 8180 2S

source ./scripts/best_configs/common.sh $@
COMMON="$COMMON --f16c-opt=1"

# vgg19_conv1_1
NSOCKETS=1 ./scripts/run.sh -c --name=vgg19_conv1_1 -i3 -h224 -o64 -H224 -n64 --blk-i=1 --blk-o=2 --flt-o=2 --flt-t=14 --tile-size=6 --execution-mode=0xa061 --pat-o=1 --output-as-blocked=true $COMMON --streaming-output=2
sleep 1
# vgg19_conv1_2
NSOCKETS=1 ./scripts/run.sh -c --name=vgg19_conv1_2 -i64 -h224 -o64 -H224 -n64 --blk-i=4 --blk-o=2 --flt-o=2 --flt-t=14 --tile-size=6 --execution-mode=0xa061 --pat-o=1 --output-as-blocked=true $COMMON
sleep 1
# vgg19_conv2_1
NSOCKETS=1 ./scripts/run.sh -c --name=vgg19_conv2_1 -i64 -h112 -o128 -H112 -n64 --tile-size=6 --blk-i=4 --blk-o=2 --flt-o=1 --flt-t=28 --execution-mode=0xa061 --pat-o=4 $COMMON
sleep 1
# vgg19_conv2_2
NSOCKETS=1 ./scripts/run.sh -c --name=vgg19_conv2_2 -i128 -h112 -o128 -H112 -n64 --tile-size=6 --blk-i=8 --blk-o=1 --flt-t=28 --execution-mode=0xa061 --pat-o=8 $COMMON
sleep 1
# vgg19_conv3_1
NSOCKETS=1 ./scripts/run.sh -c --name=vgg19_conv3_1 -i128 -h56 -o256 -H56 -n64 --tile-size=6 --blk-i=8 --blk-o=1 --flt-o=2 --flt-t=14 --pat-i=1 --pat-o=8 --execution-mode=0xa061 $COMMON
sleep 1
# vgg19_conv3_2
NSOCKETS=1 ./scripts/run.sh -c --name=vgg19_conv3_2 -i256 -h56 -o256 -H56 -n64 --tile-size=6 --blk-i=8 --blk-o=2 --flt-o=1 --flt-t=28 --pat-i=1 --pat-o=4 --execution-mode=0xa061 $COMMON
sleep 1
# vgg19_conv4_1
NSOCKETS=1 ./scripts/run.sh -c --name=vgg19_conv4_1 -i256 -h28 -o512 -H28 -n64 --tile-size=6 --blk-i=8 --blk-o=2 --flt-t=28 --execution-mode=0xa061 --pat-i=1 --pat-o=8 $COMMON
sleep 1
# vgg19_conv4_2
NSOCKETS=1 ./scripts/run.sh -c --name=vgg19_conv4_2 -i512 -h28 -o512 -H28 -n64 --tile-size=6 --blk-i=8 --blk-o=2 --flt-t=28 --execution-mode=0xa061 --pat-i=1 --pat-o=8 $COMMON
sleep 1
# vgg19_conv5_1
NSOCKETS=1 ./scripts/run.sh -c --name=vgg19_conv5_1 -i512 -h14 -o512 -H14 -n64 --tile-size=7 --blk-i=16 --blk-o=4 --flt-o=1 --flt-t=21 --execution-mode=0xa061 --pat-i=1 --pat-o=8 $COMMON
