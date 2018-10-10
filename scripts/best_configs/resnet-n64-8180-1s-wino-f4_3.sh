#/bin/bash

#/bin/bash

# Resnet50
# batch size: 64
# SKX 8180 2S

source ./scripts/best_configs/common.sh $@


# resnet50_res2a_branch2b
NSOCKETS=1 ./scripts/run.sh -c -i64 -h56 -o64 -H56 -n64 --tile-size=6 --execution-mode=0xa061 --blk-i=4 --blk-o=4 --flt-t=28 $COMMON
sleep 1
# resnet50_res3a_branch2b
NSOCKETS=1 ./scripts/run.sh -c -i128 -h28 -o128 -H28 -n64 --tile-size=6 --execution-mode=0xa061 --blk-i=8 --blk-o=8 --flt-t=28 $COMMON
sleep 1
# resnet50_res4a_branch2b
NSOCKETS=1 ./scripts/run.sh -c -i256 -h14 -o256 -H14 -n64 --tile-size=6 --execution-mode=0xa060 --blk-i=8 --blk-o=4 --flt-t=25 --pat-o=4 $COMMON
sleep 1
# resnet50_res5a_branch2b
NSOCKETS=1 ./scripts/run.sh -c -i512 -h7 -o512 -H7 -n64 --tile-size=6 --execution-mode=0xa000 --blk-i=8 --blk-o=8 --flt-t=29 $COMMON
