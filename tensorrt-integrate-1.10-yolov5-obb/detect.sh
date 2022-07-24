#!/bin/bash

source env-source.sh
cd yolov5_obb/utils/nms_rotated

sofile=`find . -name *.so`

if [ ! -f "${sofile}" ];
then
    python setup.py build
    sofile=`find . -name *.so`
    mv ${sofile} ./
    rm -rf build
fi

cd ../../
python detect.py --weights=best.pt --source=../workspace/P0032.jpg --iou-thres=0.5 --conf-thres=0.25 --project=../workspace/

mv ../workspace/exp/P0032.jpg ../workspace/P0032-pytorch.jpg
rm -rf ../workspace/exp