

class Homework():
    def __init__(self, numberOfHomeworks: int):
        self.numberOfHomeworks = numberOfHomeworks

    def createHomework(self):
        for index in (range(self.numberOfHomeworks + 1)):
