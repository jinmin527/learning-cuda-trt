
import yolo
import os
import cv2

if not os.path.exists("yolov5s.trtmodel"):
    yolo.compileTRT(
        max_batch_size=1,
        source="yolov5s.onnx",
        output="yolov5s.trtmodel",
        fp16=False,
        device_id=0
    )

infer = yolo.Yolo("yolov5s.trtmodel")
if not infer.valid:
    print("invalid trtmodel")
    exit(0)

image = cv2.imread("rq.jpg")
boxes = infer.commit(image).get()

for box in boxes:
    l, t, r, b = map(int, [box.left, box.top, box.right, box.bottom])
    cv2.rectangle(image, (l, t), (r, b), (0, 255, 0), 2, 16)

cv2.imwrite("detect.jpg", image)