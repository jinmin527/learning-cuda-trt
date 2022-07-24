
from unet import unet
import torch
import cv2
import numpy as np


class MyUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = unet(num_classes=21, backbone="vgg")
        state_dict = torch.load("../unet_voc.pth", map_location="cpu")
        self.model.load_state_dict(state_dict)
    
    def forward(self, x):
        y = self.model(x)
        y = y.permute(0, 2, 3, 1).softmax(dim=-1)
        return y
        
device = "cpu"
# model = unet(num_classes=21, backbone="vgg")
# state_dict = torch.load("../unet_voc.pth", map_location="cpu")
# model.load_state_dict(state_dict)
# model.eval().to(device)
model = MyUNet().eval().to(device)

image = cv2.imread("img/street.jpg")
image = cv2.resize(image, (512, 512))

# To RGB
image = image[..., ::-1]  # 是一种toRGB的方法 

#preprocess
image = (image / 255.0).astype(np.float32)

# totensor
image = image.transpose(2, 0, 1)[None]
image = torch.from_numpy(image).to(device)

with torch.no_grad():
    prob = model(image)

    torch.onnx.export(
        model, (image,), "unet.onnx",
        input_names=["image"], output_names=["prob"],
        opset_version=11, dynamic_axes={"image":{0:"batch"}, "prob": {0:"batch"}}
    )

# softmax
# 概率合并  
#prob = predict.permute(0, 2, 3, 1).softmax(dim=-1)  # 1, 512, 512, 21
colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
    (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
    (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
    (128, 64, 12)]
label_map = prob.argmax(dim=-1)

seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(label_map, [-1])], [512, 512, -1])
print(seg_img.shape)
cv2.imwrite("seg_img.jpg", seg_img)