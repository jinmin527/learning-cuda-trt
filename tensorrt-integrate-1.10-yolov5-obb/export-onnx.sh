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

python export.py --weights=best.pt --dynamic --include=onnx --opset=11

mv best.onnx ../workspace/