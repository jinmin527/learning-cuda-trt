#!/bin/bash

cd AlphaPose-change

export PYTHONPATH=$PYTHONPATH:.

# 256x192_res50_lr1e-3_2x-regression.yaml 其中的CONV_DIM要修改为128，下载时是256
python scripts/demo_inference.py --cfg configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml --checkpoint pretrained_models/multi_domain_fast50_regression_256x192.pth --indir examples/demo/ --save_img

mkdir -p ../workspace/pytorch-vis/
mv examples/res/vis/*.jpg ../workspace/pytorch-vis/