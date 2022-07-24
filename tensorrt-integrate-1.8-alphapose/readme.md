# 运行步骤
1. 导出onnx模型
    - `python export.py`
    - 导出onnx模型alpha-pose-136.onnx到workspace下
        - 此时他会自动下载模型文件到model下
2. 运行编译和推理
    - `make run -j64`

# Reference
- https://github.com/MVIG-SJTU/AlphaPose