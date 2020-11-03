from MNIST_getter import DownloadMNISTData
import numpy as np
from imageio import imwrite
import cv2
import random
from PIL import Image

# MNIST_Data_Load = DownloadMNISTData(fullData=False, trainAmt=10, testAmt=10)

# MNIST_Data_Load.loadData()


img = np.zeros([1000, 1000, 3], dtype=np.uint8)
img.fill(255)
# imwrite("Result.jpg", img)

line_loc = random.randint(100, 1000)

cv2.line(img=img, pt1=(0, 100), pt2=(
    1000, 100), color=(0, 0, 0), thickness=2)

latest_location = 100

digit_img = cv2.imread('./data/MNIST/jpg_form/test/00000.jpg')

digit_img = cv2.bitwise_not(digit_img)

digit_img_PIL = Image.fromarray(np.uint8(digit_img)).convert(
    'RGB')  # Image.open(digit_img)

img_PIL = Image.fromarray(np.uint8(img)).convert(
    'RGB')  # Image.open(img)

img_PIL.paste(digit_img_PIL, (50, 20))

# above we see we AT A MINIMUM need to give 100 y-diff to be able to draw a circle and put the image

imwrite("Result.jpg", img_PIL)
