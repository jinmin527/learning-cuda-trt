#!/bin/bash

cd Bert-Chinese-Text-Classification
export PYTHONPATH=$PYTHONPATH:.

if [ ! -s "THUCNews/saved_dict/bert.ckpt" ]; then
    echo 请先执行训练，得到训练好的模型
    exit
fi

echo Export onnx to ../workspace

python export.py
mv classifier.onnx ../workspace/

echo Done.!