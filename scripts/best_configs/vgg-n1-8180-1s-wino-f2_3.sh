#/bin/bash

# VGG19
# batch-size: 1
# SKX 8180 1S


source ./scripts/best_configs/common.sh $@

# vgg19_conv1_2, 7.0T  tflops= 5.50285
NSOCKETS=1 ./scripts/run.sh -c -i64 -h224 -o64 -H224 -n1 --blk-i=4 --blk-o=2 --blk-t=30 --tile-size=4 --execution-mode=0xa040 --output-as-blocked=true $COMMON
sleep 1
# vgg19_conv2_1, 7.3T   tflops=5.85272
NSOCKETS=1 ./scripts/run.sh -c -i64 -h112 -o128 -H112 -n1 --blk-i=4 --blk-o=8 --blk-t=28  --tile-size=4 --execution-mode=0xa040 $COMMON
sleep 1
# vgg19_conv2_2, 7.4T   tflops=5.92653
NSOCKETS=1 ./scripts/run.sh -c -i128 -h112 -o128 -H112 -n1 --blk-i=8 --blk-o=2 --blk-t=28 --tile-size=4 --execution-mode=0xa040 $COMMON
sleep 1
# vgg19_conv3_1, 7.3T   tflops=6.08779
NSOCKETS=1 ./scripts/run.sh -c -i128 -h56 -o256 -H56 -n1 --blk-i=8 --blk-o=16 --blk-t=28 --pat-o=1 --tile-size=4 --execution-mode=0xa061 --output-as-blocked=true $COMMON
sleep 1
# vgg19_conv3_2, 7.3T  tflops=6.33066 
NSOCKETS=1 ./scripts/run.sh -c -i256 -h56 -o256 -H56 -n1 --blk-i=8 --blk-o=1 --blk-t=28 --pat-o=16 --tile-size=4 --execution-mode=0xa060 $COMMON
sleep 1
# vgg19_conv4_1, 6.6T  tflops=6.13859
NSOCKETS=1 ./scripts/run.sh -c -i256 -h28 -o512 -H28 -n1 --blk-i=8 --blk-o=16 --blk-t=28 --tile-size=4 --execution-mode=0xa000 --streaming-input=1 $COMMON
sleep 1
# vgg19_conv4_2, 6.4T  tflops=6.11829
NSOCKETS=1 ./scripts/run.sh -c -i512 -h28 -o512 -H28 -n1 --blk-i=8 --blk-o=8 --blk-t=25 --tile-size=4 --execution-mode=0xa000 --streaming-input=1 $COMMON
sleep 1
# vgg19_conv5_1, 5.2T tflops=4.81118
NSOCKETS=1 ./scripts/run.sh -c -i512 -h14 -o512 -H14 -n1 --blk-i=8 --blk-o=8 --blk-t=25 --tile-size=4 --execution-mode=0xa000 $COMMON

