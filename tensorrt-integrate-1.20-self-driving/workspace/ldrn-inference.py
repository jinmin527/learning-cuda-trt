import onnx
import onnxruntime
import numpy as np
import matplotlib.pyplot as plt
import cv2

def preprocess(img_original, dshape):

    width, height = dshape
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_original.copy()
    img = cv2.resize(img, dshape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.
    img = (img - mean) / std
    img = img.astype(np.float32)
    # tensor = cv2.dnn.blobFromImage(img)
    tensor = img.transpose(2, 0, 1).reshape(1, 3, height, width)
    return tensor

''' Load model '''
sess = onnxruntime.InferenceSession(
    "ldrn_kitti_resnext101_pretrained_data_grad_256x512.onnx",
    providers=["CPUExecutionProvider"]
)

''' Run inference '''
img_original = cv2.imread("imgs/img.jpg")
input = preprocess(img_original, (512, 256))
input_name = sess.get_inputs()[0].name
output = sess.run(None, {input_name: input})

''' Show result '''
out = np.array(output[5])
out = np.squeeze(out, 0)
out = np.squeeze(out, 0)
out = out[int(out.shape[0] * 0.18) : , : ]
#out = (out * 256.0).astype(np.uint8)
# out = out * 256.0
# out = (out/out.max())
out = (out - 5)
print(out.min())
plt.imsave("imgs/ldrn-draw.jpg", out, cmap='plasma_r')
#cv2.imwrite("workspace/ldrn-draw.jpg", out)
print(out.shape, img_original.shape)