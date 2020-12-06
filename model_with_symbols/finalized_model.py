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


# In[410]:


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# In[411]:


## Insert local drive location where all 5 signs are extracted

def path_directory(path):
    path_name = []
    for filename in os.listdir(path):
        path_name.append(filename)
        
    return path_name


path = path_directory('C:/Users/Vishal Patil/Desktop/Data Science/Sem 3/Deep Learning Neural Networks/Project/Data_symbols/extract_necessary')
print(path)


# In[412]:


def add_symbols(path):
    images = []
    labels = []
    for path in path:
        for img in glob.glob("C:/Users/Vishal Patil/Desktop/Data Science/Sem 3/Deep Learning Neural Networks/Project/Data_symbols/extract_necessary/{}/*.jpg".format(path)):
            #images.append(img)
            labels.append(path)
            img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
            ret,thresh=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
            ctrs,ret=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            cnt=sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
            w=int(28)
            h=int(28)
            maxi=0
            for c in cnt:
                x,y,w,h=cv2.boundingRect(c)
                maxi=max(w*h,maxi)
                if maxi==w*h:
                    x_max=x
                    y_max=y
                    w_max=w
                    h_max=h
            im_crop= thresh[y_max:y_max+h_max+10, x_max:x_max+w_max+10]
            im_resize = cv2.resize(im_crop,(28,28), interpolation=cv2.INTER_AREA)
            im_bw = cv2.bitwise_not(im_resize)
            #dilated_image = cv2.dilate(im_resize, (3, 3))
            im_resize=np.reshape(im_bw,(28,28))
            images.append(im_resize)
            #np.append(train_images,im_resize)
    return images,labels

images,labels = add_symbols(path)


# In[416]:


### dictionary to convert string classes to integer

dict1 = {10:'+',11:'-', 12:'=', 13:'division', 14:'multiplication'}

for i in range(len(labels)):
    for j in dict1:
        if labels[i] == dict1[j]:
            labels[i] = j
print(len(labels))
labels_array = np.array(labels)
train_labels = np.concatenate((train_labels,labels_array))
symbols_array = np.array(images)
train_images = np.concatenate((train_images,symbols_array))


# In[451]:


print("Label is",train_labels[500])
example = train_images[500]
plt.imshow(example.reshape(28, 28),cmap='gray')
plt.show()


# In[427]:


train_labels[0].shape


# In[420]:


print ("Shape of X_train: {}".format(train_images.shape))
print ("Shape of y_train: {}".format(train_labels.shape))
print ("Shape of X_test: {}".format(test_images.shape))
print ("Shape of y_test: {}".format(test_labels.shape))

train_images = train_images.reshape(75913, 28, 28, 1)
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# In[428]:


def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28, 28, 1), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(15, activation='softmax'))

    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model    


# In[429]:


model=create_model()
model.fit(train_images, train_labels, epochs=2)


# In[430]:


def initial_predict(example):
    prediction = model.predict_classes(example.reshape(1, 28, 28, 1))
    #plt.imshow(example.reshape(28, 28), cmap="gray")
    #plt.show()
    return prediction

plt.imshow(test_images[100].reshape(28, 28), cmap="gray")
plt.show()
init = initial_predict(test_images[100])
print("Predicted class for test dataset image from mnist",init)


# In[446]:


### Final - Final

def cropping(path):
    print('i am here')
    imagem = cv2.imread(path, cv2.IMREAD_COLOR)
    # Convert black pixels to white and white to black
    img = cv2.bitwise_not(imagem)
    # Convert to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image to reduce noise
    img_blur = cv2.medianBlur(gray, 5)
    # Find the edges in the image using canny detector
    edges = cv2.Canny(img_blur, 50, 200)
    # Detect points that form a line
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                            threshold=100, minLineLength=10, maxLineGap=250)
    # Draw lines on the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 1)
    cropped = []
    # Apply hough transform on the image
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1,
                               img.shape[0]/64, param1=200, param2=10, minRadius=10, maxRadius=20)
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

### Give location of image
cropped_image = cropping('C:/Users/Vishal Patil/Desktop/Data Science/Sem 3/Deep Learning Neural Networks/Project/deep-homework-grader-main/data/homework/train/00059.jpg')

plt.imshow(cropped_image[0],cmap='gray')


# In[472]:


def predict(images):
    digits_stored = []
    signs = []
    for i in images:
        im_gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        thresh = 127
        im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
        dim = (28, 28)
        resized = cv2.resize(im_bw, dim)
        # print(resized.shape)
        imge = np.resize(resized, (28, 28, 1))
        arr = np.array(imge) / 255
        im2arr = arr.reshape(1, 28, 28, 1)
        y_pred = model.predict_classes(im2arr)
        if y_pred > 9:
            for j in dict1:
                if y_pred == j:
                    signs.append(dict1[j])
        digits_stored.append(y_pred)
        print("Predicted class is", y_pred)

    return digits_stored,signs

image_prediction,signs = predict(cropped_image)


# In[473]:


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
standard_array = np.array([[5], [3], [0], [6]])
#np.where(standard_array==image_prediction)
#np.intersect1d(image_prediction,standard_array)
match = matching(image_prediction,standard_array)   
print("Total marks =",match)


# In[95]:


# Saving model

#model.save('C:/Users/Vishal Patil/Desktop/Data Science/Sem 3/Deep Learning Neural Networks/Project/saved model/model.h5')

