#/bin/bash

# Resnet50
# batch-size: 1
# SKX 8180 1S

source ./scripts/best_configs/common.sh $@

# resnet50_res2a_branch2b, 5.2T
NSOCKETS=1 ./scripts/run.sh -c -i64 -h56 -o64 -H56 -n1 --tile-size=6 --execution-mode=0xa161 --blk-i=1 --blk-o=1 --flt-o=2 --flt-t=7 --pat-o=2 --output-as-blocked=true --input-data-file=tests/data/ResNet50/res2_0_branch2c/op11_input__res2_0_branch2b_bn  --weights-data-file=tests/data/ResNet50/res2_0_branch2c/op11_weights__res2_0_branch2c_w -b0 $COMMON
sleep 1
# resnet50_res3a_branch2b, 7.0 - 7.5T
NSOCKETS=1 ./scripts/run.sh -c -i128 -h28 -o128 -H28 -n1 --tile-size=6 --execution-mode=0xa161 --blk-i=2 --blk-o=1 --flt-o=2 --flt-t=7 --pat-o=4 --input-data-file=tests/data/ResNet50/res3_0_branch2c/op43_input__res3_0_branch2b_bn  --weights-data-file=tests/data/ResNet50/res3_0_branch2c/op43_weights__res3_0_branch2c_w -b0 $COMMON
sleep 1
# resnet50_res4a_branch2b, 4.2T
NSOCKETS=1 ./scripts/run.sh -c -i256 -h14 -o256 -H14 -n1 --tile-size=5 --execution-mode=0xa133 --blk-i=4 --blk-o=2 --flt-o=4 --flt-t=5 --input-data-file=tests/data/ResNet50/res4_0_branch2c/op85_input__res4_0_branch2b_bn  --weights-data-file=tests/data/ResNet50/res4_0_branch2c/op85_weights__res4_0_branch2c_w -b0 $COMMON
sleep 1
# resnet50_res5a_branch2b, 2.5T
NSOCKETS=1 ./scripts/run.sh -c -i512 -h7 -o512 -H7 -n1 --tile-size=4 --execution-mode=0xa133 --blk-i=8 --blk-o=1 --flt-o=1 --flt-t=26 --input-data-file=tests/data/ResNet50/res5_0_branch2c/op147_input__res5_0_branch2b_bn  --weights-data-file=tests/data/ResNet50/res5_0_branch2c/op147_weights__res5_0_branch2c_w -b0 $COMMON
