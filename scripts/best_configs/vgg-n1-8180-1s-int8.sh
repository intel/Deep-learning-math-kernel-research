#/bin/bash

# VGG19
# batch-size: 1
# SKX 8180 1S


source ./scripts/best_configs/common.sh $@

# vgg19_conv1_2, 6.73T <acc: F43 sampling in 64i>
NSOCKETS=1 ./scripts/run.sh -c -i64 -h224 -o64 -H224 -n1 --blk-i=4 --blk-o=1 --flt-o=1 --flt-t=28 --tile-size=6 --execution-mode=0xa161 --pat-o=4 --output-as-blocked=true -v0 $COMMON

sleep 1
# vgg19_conv2_1, 8.32T <acc: F43 sampling in 64i>
NSOCKETS=1 ./scripts/run.sh -c -i64 -h112 -o128 -H112 -n1 --blk-i=4 --blk-o=1 --flt-o=1 --flt-t=28 --tile-size=6 --execution-mode=0xa161 --pat-o=8 -v0 $COMMON

sleep 1
# vgg19_conv2_2, 9.53T <acc: F43 sampling in 128i>
NSOCKETS=1 ./scripts/run.sh -c -i128 -h112 -o128 -H112 -n1 --blk-i=8 --blk-o=1 --flt-o=1 --flt-t=28 --tile-size=6 --execution-mode=0xa161 --pat-o=8 -v0 $COMMON

sleep 1
# vgg19_conv3_1, 9.95T <acc: F43 sampling in 128i>
NSOCKETS=1 ./scripts/run.sh -c -i128 -h56 -o256 -H56 -n1 --blk-i=8 --blk-o=1 --flt-o=2 --flt-t=14 --pat-i=1 --pat-o=8 --tile-size=6 --execution-mode=0xa161 -v0 $COMMON

sleep 1
# vgg19_conv3_2, 9.48T <acc: F43 sampling in 256i>
NSOCKETS=1 ./scripts/run.sh -c -i256 -h56 -o256 -H56 -n1 --blk-i=16 --blk-o=8 --flt-o=1 --flt-t=28 --pat-i=1 --pat-o=1 --tile-size=6 --execution-mode=0xa133 -v0 $COMMON

sleep 1
# vgg19_conv4_1, 9.80T <acc: F43 sampling in 256i>
NSOCKETS=1 ./scripts/run.sh -c -i256 -h28 -o512 -H28 -n1 --blk-i=16 --blk-o=1 --flt-o=4 --flt-t=6 --tile-size=6 --execution-mode=0xa133 --streaming-input=1 -v0 $COMMON

sleep 1
# vgg19_conv4_2, 11.32T <acc: F43 sampling in 256i>
NSOCKETS=1 ./scripts/run.sh -c -i512 -h28 -o512 -H28 -n1 --blk-i=16 --blk-o=2 --flt-o=2 --flt-t=13 --tile-size=6 --execution-mode=0xa133 -v0 $COMMON

sleep 1
# vgg19_conv5_1, 7.01T <acc: F33 sampling in 512i>
NSOCKETS=1 ./scripts/run.sh -c -i512 -h14 -o512 -H14 -n1 --blk-i=32 --blk-o=2 --flt-o=2 --flt-t=13 --tile-size=5 --execution-mode=0xa133 -v0 $COMMON
