from MNIST_getter import DownloadMNISTData
from homework_maker import Homework

MNIST_Data_Load = DownloadMNISTData(
    fullData=False, trainAmt=5250, testAmt=750)

MNIST_Data_Load.loadData()

creator = Homework(numberOfHomeworks_Train=1750, numberOfHomeworks_Test=250)

creator.createHomework()
