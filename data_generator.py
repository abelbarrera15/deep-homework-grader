# from where I learned to grab the data:
# https://github.com/zingp/ml-py-lib/blob/c91ab4b8dc93a9a5956db73caf00e71c80f4b8ca/PyTorchCS/ann-mnists.ipynb,
# https://nextjournal.com/gkoehler/pytorch-mnist
# https://pytorch.org/docs/stable/torchvision/datasets.html#mnist
from torchvision import datasets, transforms

# Note: because we want to keep it in image form (not vector) I will not be transforming the raw PIL image, but refer to the above to convert PIL image to vector!!!!


class DownloadMNISTData():
    def __init__(self, fullData: bool, trainAmt: int, testAmt: int, folderLoc: str = './data', jpgConvert: bool = True):
        self.full_data = full_data
        self.train_amt = train_amt
        self.test_amt = test_amt
        self.folderLoc = folderLoc

    def loadData(self):
        # DOWNLOAD MINST DATA
        train_datasets = datasets.MNIST(
            root=self.folderLoc,
            train=True,
            # transform=trans_data, #you can find versions of this in use in the nextjournal link or the github
            download=True)

        test_datasets = datasets.MNIST(
            root=folderLoc,
            train=False,
            # transform=trans_data, #you can find versions of this in use in the nextjournal link or the github
            download=True)

        if self.jpgConvert == True:
            self.__jpgConvert()

    def __jpgConvert(self):
        if self.fullData == True:
            # FORMAT TO JPEG FORM TO BE ABLE TO SUPERIMPOSE ON CREATED IMAGE
            for idx, (img, _) in enumerate(train_datasets):
            img.save(
                folderLoc + '/MNIST/jpg_form/train/{:05d}.jpg'.format(idx))

            for idx, (img, _) in enumerate(test_datasets):
                img.save(
                    folderLoc + '/MNIST/jpg_form/test/{:05d}.jpg'.format(idx))

        else:
            train_count = 0
            for idx, (img, _) in enumerate(train_datasets):
                img.save(
                    folderLoc + '/MNIST/jpg_form/train/{:05d}.jpg'.format(idx))
                train_count += 1
                if train_count == trainAmt:
                    break

            test_count = 0
            for idx, (img, _) in enumerate(test_datasets):
                img.save(
                    folderLoc + '/MNIST/jpg_form/test/{:05d}.jpg'.format(idx))
                test_count += 1
                if test_count == testAmt:
                    break
