#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import glob
#from google.colab.patches import cv2_imshow
from keras.datasets import mnist
from keras import models
from keras import layers
from PIL import Image, ImageDraw
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils


# In[4]:


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# In[5]:


example = train_images[100]
plt.imshow(example.reshape(28, 28), cmap="gray")
plt.show()


# In[8]:


print ("Shape of X_train: {}".format(train_images.shape))
print ("Shape of y_train: {}".format(train_labels.shape))
print ("Shape of X_test: {}".format(train_images.shape))
print ("Shape of y_test: {}".format(test_labels.shape))

train_images = train_images.reshape(60000, 28, 28, 1)
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# In[9]:


def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28, 28, 1), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model    


# In[10]:


model=create_model()
model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=2)


# In[13]:


def initial_predict(example):
    prediction = model.predict_classes(example.reshape(1, 28, 28, 1))
    #plt.imshow(example.reshape(28, 28), cmap="gray")
    #plt.show()
    return prediction

plt.imshow(train_images[100].reshape(28, 28), cmap="gray")
plt.show()
init = initial_predict(train_images[100])
print("Predicted class for test dataset image from mnist",init)


# In[91]:


### Final - Final

def cropping(path):
    imagem = cv2.imread(path, cv2.IMREAD_COLOR)
    ### Convert black pixels to white and white to black
    img = cv2.bitwise_not(imagem)
    plt.imshow(img,cmap='gray')
    plt.show()
    # Convert to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image to reduce noise
    img_blur = cv2.medianBlur(gray, 5)
    # Find the edges in the image using canny detector
    edges = cv2.Canny(img_blur, 50, 200)
    # Detect points that form a line
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=10, maxLineGap=250)
    # Draw lines on the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
    cropped = []
    # Apply hough transform on the image
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0]/64, param1=200, param2=10, minRadius=10, maxRadius=20)
    # Draw detected circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        sorted_array = circles[0][np.argsort(circles[0][:, 1])]
        for i in sorted_array:
            # Draw outer circle
            cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 0), 10)
            crop_img = img[i[1] - i[2]:i[1] + i[2], i[0] - i[2]:i[0] + i[2], :]
            cropped.append(crop_img)
    return cropped
    #plt.imshow(cropped[0],cmap='gray')


### Give location of image
cropped_image = cropping('C:/Users/Vishal Patil/Desktop/Data Science/Sem 3/Deep Learning Neural Networks/Project/deep-homework-grader-main/data/homework/train/00065.jpg')
plt.imshow(cropped_image[0],cmap='gray')


# In[92]:


def predict(images):
    digits_stored = []
    for i in images:
        im_gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        thresh = 127
        im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
        dim = (28, 28)
        resized = cv2.resize(im_bw, dim)
        #print(resized.shape)
        imge = np.resize(resized, (28,28,1))
        arr = np.array(imge) / 255
        im2arr = arr.reshape(1,28,28,1)
        y_pred = model.predict_classes(im2arr)
        digits_stored.append(y_pred)
        print("Predicted class is",y_pred)
        plt.imshow(im_bw,cmap='gray')
        plt.show() 
        
    return digits_stored

image_prediction = predict(cropped_image)
#print(image_prediction[0])


# In[94]:


## Matching with existing answer sheet

def matching(image_prediction,standard_array):
    total_marks = 0
    #unmatched = []
    for i in range(len(standard_array)):
        if standard_array[i] == image_prediction[i]:
            print("Your answer is correct for question",i+1,",Your answer is",image_prediction[i])
            total_marks = total_marks + 1
        else:
            print("Your answer is incorrect for question",i+1,"and Your answer is",image_prediction[i],",Expected answer is",standard_array[i])
            
    return total_marks

### Define standard array which contains answers
standard_array = np.array([[5], [6], [5]])
#np.where(standard_array==image_prediction)
#np.intersect1d(image_prediction,standard_array)
match = matching(image_prediction,standard_array)   
print("Total marks",match)


# In[95]:


# Saving model

#model.save('C:/Users/Vishal Patil/Desktop/Data Science/Sem 3/Deep Learning Neural Networks/Project/saved model/model.h5')

