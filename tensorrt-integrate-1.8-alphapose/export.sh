#!/bin/bash

cd AlphaPose-change

export PYTHONPATH=$PYTHONPATH:.

python scripts/export.py

mv alpha-pose-136.onnx ../workspace/