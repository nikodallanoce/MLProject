from ActivationFunctions.ActFunction import ActivationFunction
import numpy as np
from numpy import ndarray


class TanH(ActivationFunction):

    def __init__(self):
        super(TanH, self).__init__()

    def compute(self, v: ndarray) -> ndarray:
        ris = np.tanh(v)
        return ris

    def compute_derivative(self, x: ndarray) -> ndarray:
        return 1 / (np.cosh(x) ** 2)

    def threshold(self) -> float:
        return 0
