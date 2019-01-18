#/bin/bash

# VGG19
# batch-size: 1
# SKX 8180 1S


source ./scripts/best_configs/common.sh $@

# vgg19_conv1_2, 6.73T <acc: F43 sampling in 64i>
NSOCKETS=1 ./scripts/run.sh -c --input-data-file=tests/data/VGG19/conv1_2/op3_input_conv1_1 --weights-data-file=tests/data/VGG19/conv1_2/op3_weights_conv1_2_w --bias-data-file=tests/data/VGG19/conv1_2/op3_bias_conv1_2_b -n1 -k3 -p1 -o64 -i64 -h224 -H224 --blk-i=1 --blk-o=1 --flt-o=1 --flt-t=28 --tile-size=6 --execution-mode=0xa161 --pat-o=4 --output-as-blocked=true $COMMON

sleep 1
# vgg19_conv2_1, 8.32T <acc: F43 sampling in 64i>
NSOCKETS=1 ./scripts/run.sh -c --input-data-file=tests/data/VGG19/conv2_1/op6_input_pool1 --weights-data-file=tests/data/VGG19/conv2_1/op6_weights_conv2_1_w --bias-data-file=tests/data/VGG19/conv2_1/op6_bias_conv2_1_b -n1 -k3 -p1 -o128 -i64 -h112 -H112 --blk-i=1 --blk-o=1 --flt-o=1 --flt-t=28 --tile-size=6 --execution-mode=0xa161 --pat-o=8 $COMMON

sleep 1
# vgg19_conv2_2, 9.53T <acc: F43 sampling in 128i>
NSOCKETS=1 ./scripts/run.sh -c --input-data-file=tests/data/VGG19/conv2_2/op8_input_conv2_1 --weights-data-file=tests/data/VGG19/conv2_2/op8_weights_conv2_2_w --bias-data-file=tests/data/VGG19/conv2_2/op8_bias_conv2_2_b -n1 -k3 -p1 -o128 -i128 -h112 -H112 --blk-i=2 --blk-o=1 --flt-o=1 --flt-t=28 --tile-size=6 --execution-mode=0xa161 --pat-o=8 $COMMON

sleep 1
# vgg19_conv3_1, 9.95T <acc: F43 sampling in 128i>
NSOCKETS=1 ./scripts/run.sh -c --input-data-file=tests/data/VGG19/conv3_1/op11_input_pool2 --weights-data-file=tests/data/VGG19/conv3_1/op11_weights_conv3_1_w --bias-data-file=tests/data/VGG19/conv3_1/op11_bias_conv3_1_b -n1 -k3 -p1 -o256 -i128 -h56  -H56 --blk-i=2 --blk-o=1 --flt-o=2 --flt-t=14 --pat-i=1 --pat-o=8 --tile-size=6 --execution-mode=0xa161 $COMMON

sleep 1
# vgg19_conv3_2, 9.48T <acc: F43 sampling in 256i>
NSOCKETS=1 ./scripts/run.sh -c --input-data-file=tests/data/VGG19/conv3_2/op13_input_conv3_1 --weights-data-file=tests/data/VGG19/conv3_2/op13_weights_conv3_2_w --bias-data-file=tests/data/VGG19/conv3_2/op13_bias_conv3_2_b -n1 -k3 -p1 -o256 -i256 -h56 -H56 --blk-i=4 --blk-o=8 --flt-o=1 --flt-t=28 --pat-i=1 --pat-o=1 --tile-size=6 --execution-mode=0xa133 $COMMON

sleep 1
NSOCKETS=1 ./scripts/run.sh -c --input-data-file=tests/data/VGG19/conv3_3/op15_input_conv3_2 --weights-data-file=tests/data/VGG19/conv3_3/op15_weights_conv3_3_w --bias-data-file=tests/data/VGG19/conv3_3/op15_bias_conv3_3_b -n1 -k3 -p1 -o256 -i256 -h56  -H56 --blk-i=4 --blk-o=8 --flt-o=1 --flt-t=28 --pat-i=1 --pat-o=1 --tile-size=6 --execution-mode=0xa133 $COMMON

sleep 1
NSOCKETS=1 ./scripts/run.sh -c --input-data-file=tests/data/VGG19/conv3_4/op17_input_conv3_3 --weights-data-file=tests/data/VGG19/conv3_4/op17_weights_conv3_4_w --bias-data-file=tests/data/VGG19/conv3_4/op17_bias_conv3_4_b -n1 -k3 -p1 -o256 -i256 -h56  -H56 --blk-i=4 --blk-o=8 --flt-o=1 --flt-t=28 --pat-i=1 --pat-o=1 --tile-size=6 --execution-mode=0xa133 $COMMON

sleep 1
# vgg19_conv4_1, 9.80T <acc: F43 sampling in 256i>
NSOCKETS=1 ./scripts/run.sh -c --input-data-file=tests/data/VGG19/conv4_1/op20_input_pool3 --weights-data-file=tests/data/VGG19/conv4_1/op20_weights_conv4_1_w --bias-data-file=tests/data/VGG19/conv4_1/op20_bias_conv4_1_b -n1 -k3 -p1 -o512 -i256 -h28 -H28 --blk-i=4 --blk-o=1 --flt-o=4 --flt-t=6 --tile-size=6 --execution-mode=0xa133 --streaming-input=1 $COMMON

sleep 1
# vgg19_conv4_2, 11.32T <acc: F43 sampling in 256i>
NSOCKETS=1 ./scripts/run.sh -c --input-data-file=tests/data/VGG19/conv4_2/op22_input_conv4_1 --weights-data-file=tests/data/VGG19/conv4_2/op22_weights_conv4_2_w --bias-data-file=tests/data/VGG19/conv4_2/op22_bias_conv4_2_b -n1 -k3 -p1 -o512 -i512 -h28 -H28 --blk-i=4 --blk-o=2 --flt-o=2 --flt-t=13 --tile-size=6 --execution-mode=0xa133 $COMMON

sleep 1
NSOCKETS=1 ./scripts/run.sh -c --input-data-file=tests/data/VGG19/conv4_3/op24_input_conv4_2 --weights-data-file=tests/data/VGG19/conv4_3/op24_weights_conv4_3_w --bias-data-file=tests/data/VGG19/conv4_3/op24_bias_conv4_3_b -n1 -k3 -p1 -o512 -i512 -h28 -H28 --blk-i=4 --blk-o=2 --flt-o=2 --flt-t=13 --tile-size=6 --execution-mode=0xa133 $COMMON

sleep 1
NSOCKETS=1 ./scripts/run.sh -c --input-data-file=tests/data/VGG19/conv4_4/op26_input_conv4_3 --weights-data-file=tests/data/VGG19/conv4_4/op26_weights_conv4_4_w --bias-data-file=tests/data/VGG19/conv4_4/op26_bias_conv4_4_b -n1 -k3 -p1 -o512 -i512 -h28  -H28 --blk-i=4 --blk-o=2 --flt-o=2 --flt-t=13 --tile-size=6 --execution-mode=0xa133 $COMMON

sleep 1
# vgg19_conv5_1, 7.01T <acc: F33 sampling in 512i>
NSOCKETS=1 ./scripts/run.sh -c --input-data-file=tests/data/VGG19/conv5_1/op29_input_pool4 --weights-data-file=tests/data/VGG19/conv5_1/op29_weights_conv5_1_w --bias-data-file=tests/data/VGG19/conv5_1/op29_bias_conv5_1_b -n1 -k3 -p1 -o512 -i512 -h14  -H14 --blk-i=8 --blk-o=2 --flt-o=2 --flt-t=13 --tile-size=5 --execution-mode=0xa133 $COMMON

sleep 1
NSOCKETS=1 ./scripts/run.sh -c --input-data-file=tests/data/VGG19/conv5_2/op31_input_conv5_1 --weights-data-file=tests/data/VGG19/conv5_2/op31_weights_conv5_2_w --bias-data-file=tests/data/VGG19/conv5_2/op31_bias_conv5_2_b -n1 -k3 -p1 -o512 -i512 -h14 -H14 --blk-i=8 --blk-o=2 --flt-o=2 --flt-t=13 --tile-size=5 --execution-mode=0xa133 $COMMON

sleep 1
NSOCKETS=1 ./scripts/run.sh -c --input-data-file=tests/data/VGG19/conv5_3/op33_input_conv5_2 --weights-data-file=tests/data/VGG19/conv5_3/op33_weights_conv5_3_w --bias-data-file=tests/data/VGG19/conv5_3/op33_bias_conv5_3_b -n1 -k3 -p1 -o512 -i512 -h14  -H14 --blk-i=8 --blk-o=2 --flt-o=2 --flt-t=13 --tile-size=5 --execution-mode=0xa133 $COMMON

sleep 1
NSOCKETS=1 ./scripts/run.sh -c --input-data-file=tests/data/VGG19/conv5_4/op35_input_conv5_3 --weights-data-file=tests/data/VGG19/conv5_4/op35_weights_conv5_4_w --bias-data-file=tests/data/VGG19/conv5_4/op35_bias_conv5_4_b -n1 -k3 -p1 -o512 -i512 -h14  -H14 --blk-i=8 --blk-o=2 --flt-o=2 --flt-t=13 --tile-size=5 --execution-mode=0xa133 $COMMON
