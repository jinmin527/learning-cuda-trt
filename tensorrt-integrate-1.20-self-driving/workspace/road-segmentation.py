import onnx
import onnxruntime
import numpy as np
import matplotlib.pyplot as plt
import cv2

def preprocess(img_original, dshape):

    width, height = dshape
    img = img_original.copy()
    img = cv2.resize(img, dshape)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = img / 255.0
    img = img.astype(np.float32)
    tensor = np.ascontiguousarray(img.transpose(2, 0, 1).reshape(1, 3, height, width))
    #tensor = img.reshape(1, height, width, 3)
    return tensor

''' Load model '''
sess = onnxruntime.InferenceSession(
    "road-segmentation-adas.onnx", #"road-segmentation-adas-0001.onnx",
    providers=["CPUExecutionProvider"]
)

''' Run inference '''
img_original = cv2.imread("imgs/img.jpg")
input = preprocess(img_original, (896, 512))
input_name = sess.get_inputs()[0].name
output = sess.run(None, {input_name: input})

''' Show result '''
out = output[0][0].transpose(2, 0, 1)
print(out.max(), out.shape)
# out = np.squeeze(out, 0)
# out = np.squeeze(out, 0)
# out = out[int(out.shape[0] * 0.18) : , : ]
#out = (out * 256.0).astype(np.uint8)
#out = out*255.0
# b = out[0] * 70 + out[1] * 255 + out[2] * 0 + out[3] * 0
# g = out[0] * 70 + out[1] * 0 + out[2] * 255 + out[3] * 0
# r = out[0] * 70 + out[1] * 0 + out[2] * 0 + out[3] * 255
# out = np.concatenate((b[None], g[None], r[None]), axis=0).transpose(1, 2, 0)
mask = out[3]
plt.imsave("imgs/road-draw.jpg", mask, cmap='plasma_r')
#cv2.imwrite("workspace/ldrn-draw.jpg", out)
print(out.shape, img_original.shape)