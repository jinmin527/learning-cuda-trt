# 知识点
1. yolox的预处理部分，使用了仿射变换，请参照仿射变换原理
    - 使用仿射变换实现letterbox的理由是
        - 1. 便于操作，得到变换矩阵即可
        - 2. 便于逆操作，实则是逆矩阵映射即可
        - 3. 便于cuda加速，cuda版本的加速已经在cuda系列中提到了warpaffine实现
            - 该加速可以允许warpaffine、normalize、除以255、减均值除以标准差、变换RB通道等等在一个核中实现，性能最好
2. 后处理部分，反算到图像坐标，实际上是乘以逆矩阵
    - 而由于逆矩阵实际上有效自由度是3，也就是d2i中只有3个数是不同的，其他都一样。也因此你看到的是d2i[0]、d2i[2]、d2i[5]在作用


# 是的你没看错
- 只要onnx这部分预处理后处理改好了，推理部分代码与yolov5完全一样
- 也就是后续任何检测器，都可以让他输出box，这样推理代码就不怎么需要改了

# 运行步骤
1. 安装yolox的依赖，执行`bash install.sh`
2. 导出onnx模型
    - `bash export-yolox.sh`
    - 脚本中会把模型文件移动到workspace/yolox_nano.onnx下
3. 运行编译和推理
    - `make run -j64`

# 使用pytorch的yolox进行导出
- 运行`bash detect-for-yolox-nano.sh`

# 修改过的地方：
```python
# line 206 forward fuction in yolox/models/yolo_head.py. Replace the commented code with the uncommented code
# self.hw = [x.shape[-2:] for x in outputs] 
self.hw = [list(map(int, x.shape[-2:])) for x in outputs]


# line 208 forward function in yolox/models/yolo_head.py. Replace the commented code with the uncommented code
# [batch, n_anchors_all, 85]
# outputs = torch.cat(
#     [x.flatten(start_dim=2) for x in outputs], dim=2
# ).permute(0, 2, 1)
proc_view = lambda x: x.view(-1, int(x.size(1)), int(x.size(2) * x.size(3)))
outputs = torch.cat(
    [proc_view(x) for x in outputs], dim=2
).permute(0, 2, 1)


# line 253 decode_output function in yolox/models/yolo_head.py Replace the commented code with the uncommented code
#outputs[..., :2] = (outputs[..., :2] + grids) * strides
#outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
#return outputs
xy = (outputs[..., :2] + grids) * strides
wh = torch.exp(outputs[..., 2:4]) * strides
return torch.cat((xy, wh, outputs[..., 4:]), dim=-1)

# line 77 in tools/export_onnx.py
model.head.decode_in_inference = True
```

# Reference
- https://github.com/shouxieai/tensorRT_Pro