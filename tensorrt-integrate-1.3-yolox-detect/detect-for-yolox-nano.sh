#!/bin/bash

cd YOLOX-0.2.0
export PYTHONPATH=$PYTHONPATH:.

python tools/demo.py -c yolox_nano.pth -f exps/default/yolox_nano.py --save_result image --path=../workspace/car.jpg -expn=demo

mv YOLOX_outputs/demo/vis_res/car.jpg ../workspace/car-pytorch.jpg
rm -rf YOLOX_outputs