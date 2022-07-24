#!/bin/bash

cd Bert-Chinese-Text-Classification/bert_pretrain

if [ ! -s "bert-base-chinese.tar.gz" ]; then
    wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz
fi

if [ ! -s "pytorch_model.bin" ]; then
    echo Uncompress bert-base-chinese.tar.gz
    tar -zxf bert-base-chinese.tar.gz
fi

cd ..
python run.py --model=bert