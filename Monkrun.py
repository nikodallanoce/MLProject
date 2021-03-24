# import matplotlib.pyplot as plt
# import numpy as np
from ActivationFunctions.LeakyReLU import LeakyReLU
from ActivationFunctions.Sigmoid import Sigmoid
# from ActivationFunctions.TanH import TanH
from GridSearchCV import GridSearchCV
# from Losses.Accuracy import Accuracy
from Losses import *
from NeuralNetwork import NeuralNetwork
from utilities import *
from Losses.MSE import MSE

if __name__ == '__main__':
    mean_tr, mean_ts = 0, 0
    monk = "monks-{}".format(2)
    tr_set = read_monk("monk/{}.train".format(monk), 0, False)
    ts_set = read_monk("monk/{}.test".format(monk), 0, False)
    structure = [len(tr_set[0, 0]), 5, len(tr_set[0, 1])]
    mean_acc_tr = 0
    mean_acc_ts = 0
    r=5
    for _ in range(r):
        nn = NeuralNetwork(structure, LeakyReLU(), Sigmoid(1), ts_set)
        epochs = 800
        alpha = 0.75
        eta = 0.9
        lamb = 0.0
        tr_e, ts_e, tr_a, ts_a = nn.train(tr_set, alpha, eta, lamb, epochs, MSE(), len(tr_set), False)

        GridSearchCV.plot(tr_e, ts_e, "MSE {}".format(monk))
        GridSearchCV.plot(tr_a, ts_a, "Accuracy {}".format(monk))

        mean_acc_tr += tr_a[-1] / r
        mean_acc_ts += ts_a[-1] / r
        mean_tr += tr_e[-1] / r
        mean_ts += ts_e[-1] / r

    print("{0}||{1}".format(mean_tr, mean_ts))
    print("{0}||{1}".format(mean_acc_tr, mean_acc_ts))
