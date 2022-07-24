#!/bin/bash

cd yolov5-6.0

python detect.py --weights=yolov5s.pt --source=../workspace/input-image.jpg --iou-thres=0.5 --conf-thres=0.25 --project=../workspace/

mv ../workspace/exp/input-image.jpg ../workspace/input-image-pytorch.jpg
rm -rf ../workspace/exp