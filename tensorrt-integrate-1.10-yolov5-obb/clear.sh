#!/bin/bash

cd yolov5_obb
files=`find . -name __pycache__`

for file in ${files};
do
    echo Remove ${file}
    rm -rf ${file}
done

echo Remove yolov5_obb/utils/nms_rotated/*.so
rm -rf yolov5_obb/utils/nms_rotated/*.so