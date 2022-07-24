import onnx
import onnxruntime
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.special

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

''' Run inference '''
img_original = cv2.imread("imgs/img.jpg")
input = preprocess(img_original, (800, 288))
input_name = sess.get_inputs()[0].name
output = sess.run(None, {input_name: input})

''' Show result '''
out = np.array(output[0])[0]
griding_num = 200
col_sample = np.linspace(0, 800 - 1, griding_num)
col_sample_w = col_sample[1] - col_sample[0]

out_j = out
out_j = out_j[:, ::-1, :]
prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
idx = np.arange(griding_num) + 1
idx = idx.reshape(-1, 1, 1)
loc = np.sum(prob * idx, axis=0)
out_j = np.argmax(out_j, axis=0)
classj = out_j
loc[out_j == griding_num] = 0
out_j = loc

row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]
cls_num_per_lane = 18
img_w, img_h = img_original.shape[1], img_original.shape[0]

for i in range(out_j.shape[1]):
    if np.sum(out_j[:, i] != 0) > 2:
        for k in range(out_j.shape[0]):
            if out_j[k, i] > 0:
                ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                cv2.circle(img_original,ppp,5,(0,255,0),-1)

cv2.imwrite("imgs/ultra-lane-draw.jpg", img_original)