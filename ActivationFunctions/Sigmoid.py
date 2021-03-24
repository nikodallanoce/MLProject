from ActivationFunctions.ActFunction import ActivationFunction
import math
from numpy import ndarray
import numpy as np


class Sigmoid(ActivationFunction):

    def __init__(self, a: float):
        super(Sigmoid, self).__init__()
        self.a = a

    def compute(self, x: ndarray) -> ndarray:
        a = self.a
        phi = 1 / (1 + np.exp(-a * x))
        return phi

    def compute_derivative(self, x: ndarray) -> ndarray:
        phi = self.compute(x)
        derivative = phi * (1 - phi)
        return derivative

    def threshold(self) -> float:
        return 0.5
