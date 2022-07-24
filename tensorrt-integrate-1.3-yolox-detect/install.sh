#!/bin/bash

cd YOLOX-0.2.0
export PYTHONPATH=$PYTHONPATH:.

wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth

pip install -r requirements.txt