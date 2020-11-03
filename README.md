#Datasets Used

Our primary dataset will be generated by a combination of code and handwritten images. Since our goal is to be able to extract answers from a handwritten math assignment, we will be focusing on a specific step of accomplishing this. We will be focusing on the answer the student is asserting for a specific problem which we assume/require to be circled.
As such, we will be using this dataset: http://yann.lecun.com/exdb/mnist/ to be our handwritten images of digits. Then we will be using python libraries to insert the image inside of a “circular-like” boundary using cv2 primarily like so: https://note.nkmk.me/en/python-pillow-paste/; https://www.geeksforgeeks.org/python-opencv-cv2-circle-method/
Multiple “images” with circles around them may be found inside of a given file. We will be delineating each “answer” by using lines which will also be created through code like such: https://www.geeksforgeeks.org/python-opencv-cv2-line-method/
Likely in the case of both the lines and circles, we will need to purposely find ways to morph the objects such that they are not computer perfect and more human-like.
This should allow us to generate >= 1,000 files to test/train our data on.
