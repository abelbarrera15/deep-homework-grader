from MNIST_getter import DownloadMNISTData
from homework_maker import Homework

MNIST_Data_Load = DownloadMNISTData(
    fullData=False, trainAmt=4000, testAmt=1000)

MNIST_Data_Load.loadData()

creator = Homework(numberOfHomeworks_Test=1500, numberOfHomeworks_Train=500)

creator.createHomework()
