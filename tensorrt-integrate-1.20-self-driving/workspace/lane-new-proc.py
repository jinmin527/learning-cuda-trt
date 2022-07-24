import onnx
import onnxruntime
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.special
import torch

def preprocess(img_original, dshape):

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    width, height = dshape
    img = img_original.copy()
    img = cv2.resize(img, dshape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = (img - mean) / std
    img = img.astype(np.float32)
    tensor = img.reshape(1, height, width, 3).transpose(0, 3, 1, 2)
    return tensor

''' Load model '''
sess = onnxruntime.InferenceSession(
    "new-lane.onnx",
    providers=["CPUExecutionProvider"]
)

''' Run inference '''
img_original = cv2.imread("imgs/img.jpg")
input = preprocess(img_original, (800, 288))
input_name = sess.get_inputs()[0].name
output = sess.run(None, {input_name: input})

''' Show result '''
xs = torch.tensor(output[0])
#griding_num = 200
# col_sample = torch.tensor(np.linspace(0, 800 - 1, griding_num))
# col_sample_w = col_sample[1] - col_sample[0]

img_w, img_h = img_original.shape[1], img_original.shape[0]
# xs = out * col_sample_w * img_w / 800
# print(col_sample_w / 800)
xs = xs * 0.0050 * img_w - 1

#row_anchor = torch.tensor([121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287][::-1])
#cls_num_per_lane = 18

#kk = torch.arange(xs.shape[1], dtype=int)
#ys = (row_anchor[cls_num_per_lane-1-kk]/288)
ys = torch.tensor([0.4201, 0.4549, 0.4896, 0.5208, 0.5556, 0.5903, 0.6250, 0.6562, 0.6910,
        0.7257, 0.7604, 0.7917, 0.8264, 0.8611, 0.8958, 0.9271, 0.9618, 0.9965])
ys = img_h * ys - 1

ys = ys.view(1, -1, 1).repeat(xs.shape[0], 1, 4)
points = torch.cat([xs[..., None], ys[..., None]], dim=-1)

for x, y in points[0].view(-1, 2):
    if x > 0:
        cv2.circle(img_original, (int(x), int(y)),5,(0,255,255),-1)

cv2.imwrite("imgs/ultra-lane-draw.jpg", img_original)