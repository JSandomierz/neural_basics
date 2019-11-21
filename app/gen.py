import numpy as np

class Gen:

    def __init__(self, numDimensions = 2):
        self.points = [[] for i in range(numDimensions)]

    def addPointGauss(self, numPoints, sigma, center_x, center_y):
        X = np.random.normal(scale=sigma, size=numPoints)
        Y = np.random.normal(scale=sigma, size=numPoints)
        X = np.add(X, center_x)
        Y = np.add(Y, center_y)
        self.points[0] += list(X)
        self.points[1] += list(Y)
        print(self.points)


