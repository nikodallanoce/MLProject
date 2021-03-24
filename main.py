from ActivationFunctions.LeakyReLU import LeakyReLU
from ActivationFunctions.Linear import Linear
from ActivationFunctions.Sigmoid import Sigmoid
from ActivationFunctions.TanH import TanH
# from Batch import Batch
from GridSearchCV import GridSearchCV
from Losses.MSE import MSE
from Losses.MEE import MEE
from NeuralNetwork import NeuralNetwork
from utilities import *
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import time

def cup_results():
    o = []
    for pattern in cup_test:
        o.append(nn.feed_forward(pattern))

    return o


if __name__ == '__main__':
    tr_set, ts_set = read_tr_file("CUP/ML-CUP20-TR.csv", 0.2)
    cup_test = read_ts_file("CUP/ML-CUP20-TS.csv")
    #struct = [[10, 10, 10, 2]]
    #hyp = [struct, [0.9], [0.001], [int(len(tr_set) * (3 / 4))], [0.00001]]
    #kf = GridSearchCV(tr_set, 4, MEE(), MEE(), hyp, None)
    #b_hyp = kf.compute(kf.get_all_hyp_comb(hyp), ep=10)

    nn = NeuralNetwork([10, 40, 40, 2], TanH(), Linear(1), ts_set)
    start = time.time()
    tr_err, ts_err = nn.train(tr_set, 0.6, 0.005, 0.00001, 3000, MEE(), len(tr_set), True)
    print(time.time()-start)
    print(ts_err[-1])
    write_report("SushiPizza_ML-CUP20-TS.csv", cup_results())


