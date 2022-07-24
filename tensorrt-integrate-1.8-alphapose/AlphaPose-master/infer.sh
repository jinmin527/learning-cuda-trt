#!/bin/bash

export PYTHONPATH=.
python scripts/demo_inference.py \
    --cfg=configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml \
    --checkpoint=pretrained_models/multi_domain_fast50_regression_256x192.pth \
    --sp \
    --image=examples/demo/1.jpg \
    --save_img