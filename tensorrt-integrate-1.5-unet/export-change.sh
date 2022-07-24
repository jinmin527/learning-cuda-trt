#!/bin/bash

cd unet-pytorch-change
export PYTHONPATH=$PYTHONPATH:.

python export.py

mv unet.onnx ../workspace/unet.onnx