#!/bin/bash

cd yolov5-6.0

python detect.py --weights=yolov5s.pt --source=../workspace/car.jpg --iou-thres=0.5 --conf-thres=0.25 --project=../workspace/

mv ../workspace/exp/car.jpg ../workspace/car-pytorch.jpg
rm -rf ../workspace/exp