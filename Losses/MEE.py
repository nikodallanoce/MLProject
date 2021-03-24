from Losses.LossFunction import LossFunction
import numpy as np
from numpy import ndarray


class MEE(LossFunction):
    def __init__(self):
        super(MEE, self).__init__()

    def compute_error(self, t, o):
        e: ndarray = (o - t)
        e = np.power(e, 2)
        loss = np.sqrt(e.sum())
        return loss
