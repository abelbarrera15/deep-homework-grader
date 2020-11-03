from MNIST_getter import DownloadMNISTData
import numpy as np
from imageio import imwrite
#from scipy.misc import imsave

#MNIST_Data_Load = DownloadMNISTData(fullData=False, trainAmt=10, testAmt=10)

# MNIST_Data_Load.loadData()


img = np.zeros([1000, 1000, 3], dtype=np.uint8)
img.fill(255)
imwrite("Result.jpg", img)
