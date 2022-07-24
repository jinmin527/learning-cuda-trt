#----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#----------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image
import torch
import torch.onnx
from unet import Unet

if __name__ == "__main__":
    unet = Unet()
    dummy = torch.zeros(1, 3, 512, 512).cuda()
    torch.onnx.export(
        unet.net, (dummy,), "unet.onnx", input_names=["images"], output_names=["output"], opset_version=11,
        dynamic_axes={
            "images":{0: "batch"},
            "output":{0: "batch"}
        }
    )
    print("Done")