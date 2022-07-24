# 知识点

# 运行步骤
1. 训练模型`bash train.sh`
    - 这个步骤需要下载预训练权重，请查看train.sh
2. 导出模型`bash export.sh`到onnx
    - 这个步骤要求训练好的模型
3. 执行编译推理`make run -j64`

# Reference
- https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch