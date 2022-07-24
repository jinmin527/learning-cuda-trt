#!/bin/bash

cd insightface-master
export PYTHONPATH=${PYTHONPATH}:./python-package

python predict.py

mv t1_output.jpg ../workspace/