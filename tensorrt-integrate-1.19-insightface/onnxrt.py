
# import onnxruntime
# import numpy as np

# providers = ['CPUExecutionProvider']
# session = onnxruntime.InferenceSession("insightface-master/insightface-models/buffalo_l/1k3d68.onnx", providers=providers)

# image = np.zeros((1, 3, 192, 192), dtype=np.float32)
# net_outs = session.run(["fc1"], {"data": image})

# print(net_outs)

# 暂时没有处理，其实通过predict.py调试进去就可以看到他的源代码