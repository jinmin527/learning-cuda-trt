
import onnxruntime
import cv2
import numpy as np

def preprocess(image, input_w=640, input_h=640):
    scale = min(input_h / image.shape[0], input_w / image.shape[1])
    ox = (-scale * image.shape[1] + input_w + scale  - 1) * 0.5
    oy = (-scale * image.shape[0] + input_h + scale  - 1) * 0.5
    M = np.array([
        [scale, 0, ox],
        [0, scale, oy]
    ], dtype=np.float32)
    IM = cv2.invertAffineTransform(M)

    image_prep = cv2.warpAffine(image, M, (input_w, input_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
    image_prep = (image_prep[..., ::-1] / 255.0).astype(np.float32)
    image_prep = image_prep.transpose(2, 0, 1)[None]
    return image_prep, M, IM

def nms(boxes, threshold=0.5):

    keep = []
    remove_flags = [False] * len(boxes)
    for i in range(len(boxes)):

        if remove_flags[i]:
            continue

        ib = boxes[i]
        keep.append(ib)
        for j in range(len(boxes)):
            if remove_flags[j]:
                continue

            jb = boxes[j]

            # class mismatch or image_id mismatch
            if ib[6] != jb[6] or ib[5] != jb[5]:
                continue

            cleft,  ctop    = max(ib[:2], jb[:2])
            cright, cbottom = min(ib[2:4], jb[2:4])
            cross = max(0, cright - cleft) * max(0, cbottom - ctop)
            union = max(0, ib[2] - ib[0]) * max(0, ib[3] - ib[1]) + max(0, jb[2] - jb[0]) * max(0, jb[3] - jb[1]) - cross
            iou = cross / union
            if iou >= threshold:
                remove_flags[j] = True
    return keep

def post_process(pred, IM, threshold=0.25):

    # b, n, 85
    boxes = []
    for image_id, box_id in zip(*np.where(pred[..., 4] >= threshold)):
        item = pred[image_id, box_id]
        cx, cy, w, h, objness = item[:5]
        label = item[5:].argmax()
        confidence = item[5 + label] * objness
        if confidence < threshold:
            continue

        boxes.append([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5, confidence, image_id, label])

    boxes = np.array(boxes)
    lr = boxes[:, [0, 2]]
    tb = boxes[:, [1, 3]]
    boxes[:, [0, 2]] = lr * IM[0, 0] + IM[0, 2]
    boxes[:, [1, 3]] = tb * IM[1, 1] + IM[1, 2]

    # left, top, right, bottom, confidence, image_id, label
    boxes = sorted(boxes.tolist(), key=lambda x:x[4], reverse=True)
    return nms(boxes)


if __name__ == "__main__":

    session = onnxruntime.InferenceSession("workspace/yolov5s.onnx", providers=["CPUExecutionProvider"])

    image = cv2.imread("workspace/car.jpg")
    image_input, M, IM = preprocess(image)
    pred = session.run(["output"], {"images": image_input})[0]
    boxes = post_process(pred, IM)

    for obj in boxes:
        left, top, right, bottom = map(int, obj[:4])
        confidence = obj[4]
        label = int(obj[6])
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, f"{label}: {confidence:.2f}", (left, top+20), 0, 1, (0, 0, 255), 2, 16)

    cv2.imwrite("workspace/python-ort.jpg", image)