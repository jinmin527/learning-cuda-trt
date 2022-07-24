#!/bin/bash

cd mmdetection-2.21.0
export PYTHONPATH=$PYTHONPATH:.

# python tools/deployment/pytorch2onnx.py \
#     configs/yolo/yolov3_d53_320_273e_coco.py \
#     checkpoints/yolov3_d53_320_273e_coco-421362b6.pth \
#     --output-file checkpoints/yolov3_d53_320_273e_coco-421362b6.onnx \
#     --input-img demo/demo.jpg \
#     --test-img tests/data/color.jpg \
#     --shape 608 608 \
#     --show \
#     --verify \
#     --dynamic-export \
#     --cfg-options \
#       model.test_cfg.deploy_nms_pre=-1

if [ ! -s checkpoints/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth ]; then
    mkdir -p checkpoints
    cd checkpoints
    wget https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth
    cd ..
fi

python export.py
mv yolox.onnx ../workspace/yolox-tiny.onnx