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
    "ultra_fast_lane_detection_culane_288x800.onnx",
    providers=["CPUExecutionProvider"]
)

class Postprocess(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out_j):
        griding_num = 200
        col_sample = torch.tensor(np.linspace(0, 800 - 1, griding_num))
        col_sample_w = col_sample[1] - col_sample[0]

        batch = out_j.size(0)
        #out_j = out_j[:, ::-1, :]
        prob = torch.softmax(out_j[:, :-1], dim=1)
        idx = torch.arange(griding_num) + 1
        idx = idx.view(1, -1, 1, 1)
        loc = torch.sum(prob * idx, dim=1)
        out_j = torch.argmax(out_j, dim=1)
        loc[out_j == griding_num] = 0
        #out_j = loc

        #row_anchor = torch.tensor([121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287][::-1])
        #cls_num_per_lane = 18
        # img_w, img_h = img1.shape[1], img1.shape[0]
        # out_j = out_j * col_sample_w * img_w / 800 - 1

        return loc
        #points = torch.cat([out_j[..., None], yy[..., None]], dim=-1)
        #return points

''' Run inference '''
img1 = cv2.imread("imgs/img.jpg")
img2 = cv2.imread("imgs/dashcam_00.jpg")
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
input1 = preprocess(img1, (800, 288))
input2 = preprocess(img2, (800, 288))
input_name = sess.get_inputs()[0].name
output1 = sess.run(None, {input_name: input1})
output2 = sess.run(None, {input_name: input2})

''' Show result '''
out1 = torch.tensor(output1[0])
out2 = torch.tensor(output2[0])
out = torch.cat([out1, out2], dim=0)
print(out.shape)

model = Postprocess().eval()

xs = model(out)

griding_num = 200
col_sample = torch.tensor(np.linspace(0, 800 - 1, griding_num))
col_sample_w = col_sample[1] - col_sample[0]

img_w, img_h = img1.shape[1], img1.shape[0]
xs = xs * col_sample_w * img_w / 800 - 1

row_anchor = torch.tensor([121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287][::-1])
cls_num_per_lane = 18
img_w, img_h = img1.shape[1], img1.shape[0]

kk = torch.arange(xs.shape[1], dtype=int)
ys = (img_h * (row_anchor[cls_num_per_lane-1-kk]/288) - 1)

torch.onnx.export(model, (out,), "postprocess.onnx", opset_version=11)


ys = ys.view(1, -1, 1).repeat(out.shape[0], 1, 4)
points = torch.cat([xs[..., None], ys[..., None]], dim=-1)
print(points.size(), "======")

# for i in range(out_j.shape[1]):
#     if torch.sum(out_j[:, i] != 0) > 2:
#         for k in range(out_j.shape[0]):
#             if out_j[k, i] > 0:
#                 ppp = (int(out_j[k, i]), int(yy[k, i]) )
#                 cv2.circle(img_original,ppp,5,(0,255,0),-1)
for x, y in points[0].view(-1, 2):
    if x > 0:
        cv2.circle(img1, (int(x), int(y)),5,(0,255,0),-1)

for x, y in points[1].view(-1, 2):
    if x > 0:
        cv2.circle(img2, (int(x), int(y)),5,(0,255,0),-1)

cv2.imwrite("imgs/lane1-draw.jpg", img1)
cv2.imwrite("imgs/lane2-draw.jpg", img2)