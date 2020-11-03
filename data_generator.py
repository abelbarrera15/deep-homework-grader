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

latest_line_loc = 0
previous_line_loc = 0

img_PIL = None

while latest_line_loc <= 900:

    line_loc = random.randint(100, 1000)

    if 800 <= latest_line_loc <= 1000 or latest_line_loc + line_loc >= 950:
        digit_img = cv2.imread('./data/MNIST/jpg_form/test/00000.jpg')

        digit_img = cv2.bitwise_not(digit_img)

        dig_x, dig_y, dig_channel = digit_img.shape

        dig_holder = np.zeros([dig_x + 20, dig_y + 20, 3], dtype=np.uint8)

        dig_holder.fill(255)

        dig_holder_x, dig_holder_y, dig_holder_channel = dig_holder.shape

        y_offset = round((dig_holder_x-dig_x)/2)
        x_offset = round((dig_holder_y-dig_y)/2)

        dig_final = dig_holder.copy()

        dig_final[y_offset:y_offset+dig_y, x_offset:x_offset+dig_x] = digit_img

        cv2.circle(dig_final, (int(dig_holder_x/2),
                               int(dig_holder_y/2)), 18, (0, 0, 0), 2)

        digit_img_PIL = Image.fromarray(np.uint8(dig_final)).convert(
            'RGB')

        img_PIL = Image.fromarray(np.uint8(img)).convert(
            'RGB')

        img_PIL.paste(
            digit_img_PIL, (random.randint(20, 800), random.randint(latest_line_loc + 10, int(1000*.95))))

        img = np.array(img_PIL)

        break

    elif latest_line_loc + line_loc > 900:
        break

    latest_line_loc += line_loc

    cv2.line(img=img, pt1=(0, latest_line_loc), pt2=(
        1000, latest_line_loc), color=(0, 0, 0), thickness=2)

    digit_img = cv2.imread('./data/MNIST/jpg_form/test/00000.jpg')

    digit_img = cv2.bitwise_not(digit_img)

    dig_x, dig_y, dig_channel = digit_img.shape

    dig_holder = np.zeros([dig_x + 20, dig_y + 20, 3], dtype=np.uint8)

    dig_holder.fill(255)

    dig_holder_x, dig_holder_y, dig_holder_channel = dig_holder.shape

    y_offset = round((dig_holder_x-dig_x)/2)
    x_offset = round((dig_holder_y-dig_y)/2)

    dig_final = dig_holder.copy()

    dig_final[y_offset:y_offset+dig_y, x_offset:x_offset+dig_x] = digit_img

    cv2.circle(dig_final, (int(dig_holder_x/2),
                           int(dig_holder_y/2)), 18, (0, 0, 0), 2)

    digit_img_PIL = Image.fromarray(np.uint8(dig_final)).convert(
        'RGB')

    img_PIL = Image.fromarray(np.uint8(img)).convert(
        'RGB')

    img_PIL.paste(
        digit_img_PIL, (random.randint(20, 800), random.randint(int(previous_line_loc), int(latest_line_loc*.95))))

    img = np.array(img_PIL)

    previous_line_loc = latest_line_loc


# above we see we AT A MINIMUM need to give 100 y-diff to be able to draw a circle and put the image

imwrite("Result.jpg", img_PIL)
