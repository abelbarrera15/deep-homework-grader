# IMPORTANT! The below code works best within an ipynb and is just for seeing
import cv2
import glob
import matplotlib.pyplot as plt


filenames_train = [img for img in glob.glob(
    "C:/Users/Vishal Patil/Desktop/Data Science/Sem 3/Deep Learning Neural Networks/Project/deep-homework-grader-main/data/homework/train/*.jpg")]
filenames_test = [img for img in glob.glob(
    "C:/Users/Vishal Patil/Desktop/Data Science/Sem 3/Deep Learning Neural Networks/Project/deep-homework-grader-main/data/homework/test/*.jpg")]

filenames_train.sort()  # ADD THIS LINE
filenames_test.sort()

images_test = []
images_train = []

for img1 in filenames_train:
    n1 = cv2.imread(img1)
    images_train.append(n1)
    plt.imshow(n1, cmap='gray')  # graph it
#     plt.show()  # display!
#     print(img1)

print("Training Image:\n")
plt.imshow(images_train[0], cmap='gray')  # graph it
plt.show()  # display!

for img2 in filenames_test:
    n2 = cv2.imread(img2)
    images_test.append(n2)
    plt.imshow(n2, cmap='gray')  # graph it
#     plt.show()  # display!
#     print(img)
print("Test Image:\n")
plt.imshow(images_test[0], cmap='gray')  # graph it
plt.show()  # display!
