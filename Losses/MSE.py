from Losses.LossFunction import LossFunction
import numpy as np


class MSE(LossFunction):
    def __init__(self):
        super(MSE, self).__init__()

    # def compute_error(self, t, o):
    #   return 0.5 * ((t - o) ** 2)

    def compute_error(self, t, o):
        r: np.ndarray = (o - t)
        r = np.power(r, 2)
        r = r.sum()
        return r

    def compute_error_derivative(self, t, o):
        return t - o
