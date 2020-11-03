

class Homework():
    def __init__(self, numberOfHomeworks_Test: int, numberOfHomeworks_Train: int):
        self.numberOfHomeworks_Test = numberOfHomeworks_Test
        self.numberOfHomeworks_Train = numberOfHomeworks_Train

    def createHomework(self):
        for index in (range(self.numberOfHomeworks_Test + 1)):
            