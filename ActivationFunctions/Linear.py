from ActivationFunctions.ActFunction import ActivationFunction
import numpy as np
from numpy import ndarray


class Linear(ActivationFunction):

    def __init__(self, a: float):
        super(Linear, self).__init__()
        self.a = a

    def compute(self, v: ndarray) -> ndarray:
        ris = self.a * v
        return ris

    def compute_derivative(self, x: ndarray) -> ndarray:
        # r = np.sign(x) * self.a
        r = np.ones(len(x)) * self.a
        return r
