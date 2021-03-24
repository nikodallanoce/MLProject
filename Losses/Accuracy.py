from ActivationFunctions import ActFunction
from Losses.LossFunction import LossFunction
import numpy as np


class Accuracy(LossFunction):
    def __init__(self, act_f: ActFunction):
        super(Accuracy, self).__init__()
        self.act_f = act_f

    # def compute_error(self, t, o):
    #   return 0.5 * ((t - o) ** 2)

    def compute_error(self, t, o):
        accuracy = 0
        o = np.where(o >= self.act_f.threshold(), 1, 0)
        if o == t:
            accuracy = 1
        return accuracy
