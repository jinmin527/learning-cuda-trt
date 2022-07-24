# 知识点
1. 对图像做warpAffine预处理后，后处理则可以使用过逆变换回来

# 运行步骤
1. 导出onnx模型
    - `bash export.sh`
    - 脚本中会把模型文件移动到workspace/unet.onnx下
2. 运行编译和推理
    - `make run -j64`

# Reference
- https://github.com/shouxieai/unet-pytorch
- 模型的下载地址：https://pan.baidu.com/s/1vEq7OvRQ7re8b6Ri8461jg 提取码: c3c2