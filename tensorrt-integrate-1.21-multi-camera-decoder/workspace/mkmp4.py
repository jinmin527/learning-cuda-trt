
import cv2
import numpy as np
import sys

if len(sys.argv) != 2:
    print("Unknow prefix argument")
    exit(0)

prefix = sys.argv[1]
print(f"Make {prefix}")

writer = cv2.VideoWriter(
    f"exp/{prefix}.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
    15, (640, 480), True
)

for i in range(100):
    i += 1
    image = np.zeros((480, 640, 3), np.uint8)
    cv2.putText(image, f"{prefix}-{i:03d}", (300, 240), 0, 1, (0, 255, 0), 1, 16)
    writer.write(image)

writer.release()