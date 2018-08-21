#/bin/bash

# VGG19
# batch-size: 1
# SKX 8180 1S


source ./scripts/best_configs/common.sh $@

# vgg19_conv1_2, T
NSOCKETS=1 ./scripts/run.sh -c -i64 -h224 -o64 -H224 -n1 --blk-i=4 --blk-o=4 --blk-t=28 --tile-size=6 --execution-mode=0xa040 --output-as-blocked=true $COMMON 
sleep 1
# vgg19_conv2_1, 7.3T
NSOCKETS=1 ./scripts/run.sh -c -i64 -h112 -o128 -H112 -n1 --blk-i=4 --blk-o=8 --blk-t=28 --tile-size=6 --execution-mode=0xa040 $COMMON 
sleep 1
# vgg19_conv2_2, 7.4T
NSOCKETS=1 ./scripts/run.sh -c -i128 -h112 -o128 -H112 -n1 --blk-i=8 --blk-o=8 --blk-t=28 --tile-size=6 --execution-mode=0xa040 $COMMON 
sleep 1
# vgg19_conv3_1, 7.3T
 NSOCKETS=1 ./scripts/run.sh -c -i128 -h56 -o256 -H56 -n1 --blk-i=8 --blk-o=4 --blk-t=28 --pat-o=4 --tile-size=6 --execution-mode=0xa061 $COMMON 
sleep 1
# vgg19_conv3_2, 7.3T
 NSOCKETS=1 ./scripts/run.sh -c -i256 -h56 -o256 -H56 -n1 --blk-i=8 --blk-o=8 --blk-t=14 --pat-o=2 --tile-size=6 --execution-mode=0xa060 $COMMON 
sleep 1
# vgg19_conv4_1, 6.6T
NSOCKETS=1 ./scripts/run.sh -c -i256 -h28 -o512 -H28 -n1 --blk-i=8 --blk-o=4 --blk-t=25 --tile-size=6 --execution-mode=0xa000 --streaming-input=1 $COMMON 
sleep 1
# vgg19_conv4_2, 6.4T
NSOCKETS=1 ./scripts/run.sh -c -i512 -h28 -o512 -H28 -n1 --blk-i=8 --blk-o=4 --blk-t=26 --tile-size=6 --execution-mode=0xa000 --streaming-input=1 $COMMON 
sleep 1
# vgg19_conv5_1, 5.2T
NSOCKETS=1 ./scripts/run.sh -c -i512 -h14 -o512 -H14 -n1 --blk-i=8 --blk-o=4 --blk-t=16 --tile-size=6 --execution-mode=0xa000 $COMMON 
