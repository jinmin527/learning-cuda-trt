# 知识点
1. 预处理部分，使用了仿射变换，请参照仿射变换原理
    - 使用仿射变换实现letterbox的理由是
        - 1. 便于操作，得到变换矩阵即可
        - 2. 便于逆操作，实则是逆矩阵映射即可
        - 3. 便于cuda加速，cuda版本的加速已经在cuda系列中提到了warpaffine实现
            - 该加速可以允许warpaffine、normalize、除以255、减均值除以标准差、变换RB通道等等在一个核中实现，性能最好
2. 后处理部分，反算到图像坐标，实际上是乘以逆矩阵
    - 而由于逆矩阵实际上有效自由度是3，也就是d2i中只有3个数是不同的，其他都一样。也因此你看到的是d2i[0]、d2i[2]、d2i[5]在作用

# 运行步骤
1. 导出onnx模型
    - `bash export-retinaface.sh`
    - 脚本中会把模型文件移动到workspace/mb_retinaface.onnx下
2. 运行编译和推理
    - `make run -j64`


# 修改过的地方：
```python
# models/retinaface.py第24行，
# return out.view(out.shape[0], -1, 2) 修改为
return out.view(-1, int(out.size(1) * out.size(2) * 2), 2)

# models/retinaface.py第35行，
# return out.view(out.shape[0], -1, 4) 修改为
return out.view(-1, int(out.size(1) * out.size(2) * 2), 4)

# models/retinaface.py第46行，
# return out.view(out.shape[0], -1, 10) 修改为
return out.view(-1, int(out.size(1) * out.size(2) * 2), 10)

# 以下是保证resize节点输出是按照scale而非shape，从而让动态大小和动态batch变为可能
# models/net.py第89行，
# up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest") 修改为
up3 = F.interpolate(output3, scale_factor=2, mode="nearest")

# models/net.py第93行，
# up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest") 修改为
up2 = F.interpolate(output2, scale_factor=2, mode="nearest")

# 以下代码是去掉softmax（某些时候有bug），同时合并输出为一个，简化解码部分代码
# models/retinaface.py第123行
# if self.phase == 'train':
#     output = (bbox_regressions, classifications, ldm_regressions)
# else:
#     output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
# return output
# 修改为，注意这里没有softmax，通过手段去掉softmax节点
output = (bbox_regressions, classifications, ldm_regressions)
return torch.cat(output, dim=-1)

# 添加opset_version=11，使得算子按照预期导出
# torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False,
#     input_names=input_names, output_names=output_names)
torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False, opset_version=11,
    input_names=input_names, output_names=output_names,
    dynamic_axes={
        "input0": {0: "batch"},
        "output0": {0: "batch"}
    }
)
```

# Reference
- https://github.com/shouxieai/tensorRT_Pro