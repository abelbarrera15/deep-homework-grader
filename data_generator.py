from MNIST_getter import DownloadMNISTData

MNIST_Data_Load = DownloadMNISTData(fullData=False, trainAmt=10, testAmt=10)

MNIST_Data_Load.loadData()
