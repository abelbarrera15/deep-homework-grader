from keras.utils import np_utils
from keras.optimizers import Adam
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
from google.colab.patches import cv2_imshow

from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from PIL import Image, ImageDraw

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# train_images[0].reshape(28,28)
#plt.imshow(train_images[0], cmap="gray")
# plt.show()
plt.imshow(train_images[0].reshape(28, 28), cmap="gray")
plt.show()

print(train_labels[0])

print("Shape of X_train: {}".format(train_images.shape))
print("Shape of y_train: {}".format(train_labels.shape))
print("Shape of X_test: {}".format(train_images.shape))
print("Shape of y_test: {}".format(test_labels.shape))


train_images = train_images.reshape(60000, 28, 28, 1)
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                 input_shape=(28, 28, 1), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.summary()


model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, validation_data=(
    test_images, test_labels), epochs=3)

example = train_images[10]
prediction = model.predict_classes(example.reshape(1, 28, 28, 1))
print(prediction)
plt.imshow(example.reshape(28, 28), cmap="gray")
plt.show()

imagem = cv2.imread('00000.jpg', cv2.IMREAD_COLOR)

img = cv2.bitwise_not(imagem)
plt.imshow(img, cmap='gray')
plt.show()


#img = cv2.imread('00000.jpg', cv2.IMREAD_COLOR)
# Convert to gray-scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image to reduce noise
img_blur = cv2.medianBlur(gray, 5)
# Find the edges in the image using canny detector
edges = cv2.Canny(img_blur, 50, 200)
# Detect points that form a line
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                        minLineLength=10, maxLineGap=250)
# Draw lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

shaped = []
cropped = []

# Apply hough transform on the image
circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1,
                           img.shape[0]/64, param1=200, param2=10, minRadius=10, maxRadius=20)
# Draw detected circles
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw outer circle
        cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 0), 10)
        #crop_img = img[i[1]:(i[1]+2*i[2]), i[0]:(i[0]+2*i[2])]
        crop_img = img[i[1] - i[2]:i[1] + i[2], i[0] - i[2]:i[0] + i[2], :]
        #img[i[1] - 9:i[1] + i[2] - 4, i[0] - 9:i[0] + i[2] - 9, :]
        cropped.append(np.array(crop_img))

cv2_imshow(cropped[1])

read = cv2.cvtColor(cropped[0], cv2.COLOR_BGR2GRAY)
# print(read.shape)
dim = (28, 28)
resized = cv2.resize(read, dim)
# print(resized.shape)
imge = np.resize(resized, (28, 28, 1))
arr = np.array(imge)
im2arr = arr.reshape(1, 28, 28, 1)
y_pred = model.predict_classes(im2arr)
print(y_pred)
plt.imshow(digit, cmap='gray')
plt.show()
print("\n\nFinal Output: {}".format(np.argmax(prediction)))

#prediction = model.predict(digit.reshape(1, 28, 28, 1))
#print ("\n\n---------------------------------------\n\n")
#print ("=========PREDICTION============ \n\n")
#

#print ("\nPrediction (Softmax) from the neural network:\n\n {}".format(prediction))

#hard_maxed_prediction = np.zeros(prediction.shape)
#hard_maxed_prediction[0][np.argmax(prediction)] = 1
#print ("\n\nHard-maxed form of the prediction: \n\n {}".format(hard_maxed_prediction))
#print ("\n\n---------------------------------------\n\n")

for digit in cropped:
    read = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
    print(read.shape)
    dim = (28, 28)
    resized = cv2.resize(read, dim)
    print(resized.shape)
    imge = np.resize(resized, (28, 28, 1))
    arr = np.array(imge)
    im2arr = arr.reshape(1, 28, 28, 1)
    y_pred = model.predict_classes(im2arr)
    print("Predicted class is", y_pred)
    plt.imshow(digit, cmap='gray')
    plt.show()
    #print("\n\nFinal Output: {}".format(np.argmax(prediction)))

    #prediction = model.predict(digit.reshape(1, 28, 28, 1))
    #print ("\n\n---------------------------------------\n\n")
    #print ("=========PREDICTION============ \n\n")
    #

    #print ("\nPrediction (Softmax) from the neural network:\n\n {}".format(prediction))

    #hard_maxed_prediction = np.zeros(prediction.shape)
    #hard_maxed_prediction[0][np.argmax(prediction)] = 1
    #print ("\n\nHard-maxed form of the prediction: \n\n {}".format(hard_maxed_prediction))
    #print ("\n\n---------------------------------------\n\n")

#img = cv2.imread('000032.jpg', cv2.IMREAD_COLOR)
# Convert to gray-scale

imagem = cv2.imread('00032.jpg', cv2.IMREAD_COLOR)

img = cv2.bitwise_not(imagem)
plt.imshow(img, cmap='gray')
plt.show()


#img = cv2.imread('00000.jpg', cv2.IMREAD_COLOR)
# Convert to gray-scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image to reduce noise
img_blur = cv2.medianBlur(gray, 5)
# Find the edges in the image using canny detector
edges = cv2.Canny(img_blur, 50, 200)
# Detect points that form a line
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                        minLineLength=10, maxLineGap=250)
# Draw lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

shaped = []
cropped = []

# Apply hough transform on the image
circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1,
                           img.shape[0]/64, param1=200, param2=10, minRadius=10, maxRadius=20)
# Draw detected circles
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw outer circle
        cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 0), 10)
        #crop_img = img[i[1]:(i[1]+2*i[2]), i[0]:(i[0]+2*i[2])]
        crop_img = img[i[1] - i[2]:i[1] + i[2], i[0] - i[2]:i[0] + i[2], :]
        #img[i[1] - 9:i[1] + i[2] - 4, i[0] - 9:i[0] + i[2] - 9, :]
        cropped.append(np.array(crop_img))

cv2_imshow(cropped[1])

for digit in cropped:
    read = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
    print(read.shape)
    dim = (28, 28)
    resized = cv2.resize(read, dim)
    print(resized.shape)
    imge = np.resize(resized, (28, 28, 1))
    arr = np.array(imge)
    im2arr = arr.reshape(1, 28, 28, 1)
    y_pred = model.predict_classes(im2arr)
    print("Predicted class is", y_pred)
    plt.imshow(digit, cmap='gray')
    plt.show()
    #print("\n\nFinal Output: {}".format(np.argmax(prediction)))

    #prediction = model.predict(digit.reshape(1, 28, 28, 1))
    #print ("\n\n---------------------------------------\n\n")
    #print ("=========PREDICTION============ \n\n")
    #

    #print ("\nPrediction (Softmax) from the neural network:\n\n {}".format(prediction))

    #hard_maxed_prediction = np.zeros(prediction.shape)
    #hard_maxed_prediction[0][np.argmax(prediction)] = 1
    #print ("\n\nHard-maxed form of the prediction: \n\n {}".format(hard_maxed_prediction))
    #print ("\n\n---------------------------------------\n\n")
