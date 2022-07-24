#!/bin/bash

cd Pytorch_Retinaface-master
export PYTHONPATH=$PYTHONPATH:.

python convert_to_onnx.py

mv FaceDetector.onnx ../workspace/mb_retinaface.onnx