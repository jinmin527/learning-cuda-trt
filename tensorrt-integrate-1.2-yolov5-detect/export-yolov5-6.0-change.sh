#!/bin/bash

cd yolov5-6.0-change
python export.py --weights=../yolov5s.pt --dynamic --include=onnx --opset=11

mv ../yolov5s.onnx ../workspace/