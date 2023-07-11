import numpy as np


class LinearRegression():

    def __init__(self, learning_rate=0.01, epochs=10):

        self.X = None
        self.Y = None
        self.weight = 0.0
        self.intercept = 0.0
        self.loss = 0.0

        self.epochs = epochs
        self.learning_rate = learning_rate


    def fit(self, x, y):
        self.X = x
        self.Y = y

        N = len(self.X)

       

        for epoch in range(self.epochs):

            for xi, yi in zip(self.X, self.Y):
                dldm = 0.0

                dldc = 0.0

                dldm = dldm + 2*(yi - (self.weight* xi + self.intercept)) *-xi
                dldc = dldc + 2*(yi - (self.weight* xi + self.intercept)) * -1

                self.weight = self.weight - self.learning_rate * (sum(dldm) * 1/N)
                self.intercept = self.intercept - self.learning_rate * (sum(dldc) * 1/N)
            

    def predict(self, X):

        if self.X is None or self.Y is None:
            raise LinearFitException("Linear Model has not be fitted")

        y_pred = (self.weight * X) + self.intercept

        return y_pred
        


class LinearFitException(Exception):
    pass