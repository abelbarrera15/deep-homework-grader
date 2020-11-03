from MNIST_getter import DownloadMNISTData
import numpy as np
from imageio import imwrite
import cv2
import random

# MNIST_Data_Load = DownloadMNISTData(fullData=False, trainAmt=10, testAmt=10)

# MNIST_Data_Load.loadData()


img = np.zeros([1000, 1000, 3], dtype=np.uint8)
img.fill(255)
# imwrite("Result.jpg", img)

line_loc = random.randint(100, 1000)

print(line_loc)

cv2.line(img=img, pt1=(0, 100), pt2=(
    1000, 100), color=(0, 0, 0), thickness=2)

# above we see we AT A MINIMUM need to give 100 y-diff to be able to draw a circle and put the image

imwrite("Result.jpg", img)
