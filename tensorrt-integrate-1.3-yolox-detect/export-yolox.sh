#!/bin/bash

cd YOLOX-0.2.0
export PYTHONPATH=$PYTHONPATH:.

python tools/export_onnx.py -c yolox_nano.pth -f exps/default/yolox_nano.py --output-name=yolox_nano.onnx --dynamic --no-onnxsim

mv yolox_nano.onnx ../workspace/