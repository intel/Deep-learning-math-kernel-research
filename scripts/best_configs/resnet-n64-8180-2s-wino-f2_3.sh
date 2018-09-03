#/bin/bash

#/bin/bash

# Resnet50
# batch size: 64
# SKX 8180 2S

source ./scripts/best_configs/common.sh $@

# resnet50_res2a_branch2b, 11.8T  tflops:9.7289
NSOCKETS=2 ./scripts/run.sh -c -i64 -h56 -o64 -H56 -n64 --tile-size=4 --execution-mode=0xa040 --blk-i=4 --blk-o=2 --blk-t=29 $COMMON
sleep 1
# resnet50_res3a_branch2b, 11.9T  tflops:10.9736
NSOCKETS=2 ./scripts/run.sh -c -i128 -h28 -o128 -H28 -n64 --tile-size=4 --execution-mode=0xa040 --blk-i=8 --blk-o=8 --blk-t=25 $COMMON
sleep 1
# resnet50_res4a_branch2b, 10.0Ti  tflops:10.6828
NSOCKETS=2 ./scripts/run.sh -c -i256 -h14 -o256 -H14 -n64 --tile-size=4 --execution-mode=0xa061 --blk-i=8 --blk-o=8 --blk-t=28 --pat-o=1 $COMMON
sleep 1
# resnet50_res5a_branch2b, 6.7T  tflops:6.97863                                        0xa000
NSOCKETS=2 ./scripts/run.sh -c -i512 -h7 -o512 -H7 -n64 --tile-size=4 --execution-mode=0xa201 --blk-i=8 --blk-o=16 --blk-t=25 $COMMON
# resnet50_res5a_branch2b, 6.7T  tflops:7.06438
#NSOCKETS=2 ./scripts/run.sh -c -i512 -h7 -o512 -H7 -n64 --tile-size=4 --execution-mode=0xa0e1 --blk-i=8 --blk-o=8 --blk-t=30 pat-o=2 $COMMON

