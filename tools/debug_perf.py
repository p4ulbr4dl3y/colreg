import os
import time

import cv2

from colreg_vision.pipeline import VideoAnalyticsPipeline

os.environ["YOLO_VERBOSE"] = "False"
pipeline = VideoAnalyticsPipeline()
img = cv2.imread("test_images/day/cbd.png")
start = time.time()
for _ in range(3):
    res = pipeline.process(img)
end = time.time()
print(f"3 iterations took {end - start:.2f} seconds")
