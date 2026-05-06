import cv2
import time
import os
from colreg_vision.pipeline import VideoAnalyticsPipeline

# Disable Ultralytics print to keep output clean
os.environ["YOLO_VERBOSE"] = "False"

pipeline = VideoAnalyticsPipeline()
img = cv2.imread("test_images/day/cbd.png")

start = time.time()
for _ in range(3):
    res = pipeline.process(img)
end = time.time()

print(f"3 iterations took {end - start:.2f} seconds")
