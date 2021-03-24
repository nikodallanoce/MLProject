from ActivationFunctions.ActFunction import ActivationFunction
from numpy import ndarray
import numpy as np


class LeakyReLU(ActivationFunction):

    def __init__(self):
        super(LeakyReLU, self).__init__()

    def compute(self, x: ndarray) -> ndarray:
        ris = np.where(x >= 0, x, x * 0.01)
        return ris

    def compute_derivative(self, x: ndarray) -> ndarray:
        derivative = np.where(x >= 0, 1, 0.01)
        return derivative

    def threshold(self) -> float:
        raise Exception()
