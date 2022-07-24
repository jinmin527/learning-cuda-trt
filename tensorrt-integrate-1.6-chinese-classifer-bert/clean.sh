#!/bin/bash

cd Bert-Chinese-Text-Classification/bert_pretrain
rm *.json
rm *.gz
rm *.bin

cd ..
rm THUCNews/saved_dict/bert.ckpt
rm -rf __pycache__
rm -rf models/__pycache__
rm -rf pytorch_pretrained/__pycache__