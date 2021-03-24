import multiprocessing
from concurrent import futures
import time

import numpy as np
from numpy import ndarray
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from ActivationFunctions.LeakyReLU import LeakyReLU
from ActivationFunctions.Linear import Linear
from ActivationFunctions.Sigmoid import Sigmoid
from ActivationFunctions.TanH import TanH
from Losses.LossFunction import LossFunction
from NeuralNetwork import NeuralNetwork
from threading import Thread


class GridSearchCV:

    def __init__(self, tr_set: ndarray, k: int, tr_loss: LossFunction, ts_loss: LossFunction, hyp: list, ts_set):
        self.tr_set = tr_set
        self.k = k
        self.tr_loss = tr_loss
        self.ts_loss = ts_loss
        self.hyp = hyp
        self.ts_set = ts_set

    def parallel(self):

        pool = multiprocessing.Pool(multiprocessing.cpu_count() - 2)
        for hyp in self.get_all_hyp_comb(self.hyp)[1:8]:
            # self.compute_parallel(hyp)
            pool.apply_async(func=self.compute_parallel, args=([hyp]))
        pool.close()
        pool.join()

    def compute_parallel(self, hyp: list, ep=1500, pid=-1):
        best_model = None
        best_mean = 9999999.0

        mean_tr = 0
        mean_ts = 0
        structure = hyp[0]
        for tr_i, ts_i in self.kfolds_iter(self.tr_set, self.k):
            train = self.tr_set[tr_i]
            val = self.tr_set[ts_i]
            nn = NeuralNetwork(structure, TanH(), Linear(1), val)

            tr_e, vl_e = nn.train(train, hyp[1], hyp[2], hyp[4], ep, self.tr_loss, int(hyp[3]))

            ris = "node= {0}, a={1}, eta={2}, batch={3}|\n tr_e={4} ts_e={5}".format(hyp[0], hyp[1], hyp[2], hyp[3],
                                                                                     tr_e[-1], vl_e[-1])
            mean_tr += tr_e[-1]
            mean_ts += vl_e[-1]
            print("node= {0}, a={1}, eta={2}, batch={3}| tr_e={4} ts_e={5}".format(hyp[0], hyp[1], hyp[2], hyp[3],
                                                                                   tr_e[-1], vl_e[-1]))
            self.plot(tr_e, vl_e, ris)
        if mean_ts < best_mean:
            best_mean = mean_ts
            best_model = [structure, hyp[1], hyp[2], hyp[4], int(hyp[3])]

        print("mean tr: {0} | mean ts: {1}".format(mean_tr / self.k, mean_ts / self.k))
        print("------------------------------------------------------------------------")

        return best_model

    def compute(self, hyp_comb: list, ep=1500, pid=-1):
        best_model = None
        best_mean = 9999999.0
        for hyp in hyp_comb:
            mean_tr = 0
            mean_ts = 0
            structure = hyp[0]
            for tr_i, ts_i in self.kfolds_iter(self.tr_set, self.k):
                train = self.tr_set[tr_i]
                val = self.tr_set[ts_i]
                nn = NeuralNetwork(structure, TanH(), Linear(1), val)

                tr_e, vl_e = nn.train(train, hyp[1], hyp[2], hyp[4], ep, self.tr_loss, int(hyp[3]), True)

                mean_tr += tr_e[-1]
                mean_ts += vl_e[-1]
                print("node= {0}, a={1}, eta={2}, batch={3}| tr_e={4} ts_e={5}".format(hyp[0], hyp[1], hyp[2], hyp[3],
                                                                                       tr_e[-1], vl_e[-1]))
            if mean_ts < best_mean:
                best_mean = mean_ts
                best_model = [structure, hyp[1], hyp[2], hyp[4], int(hyp[3])]

            print("mean tr: {0} | mean ts: {1}".format(mean_tr / self.k, mean_ts / self.k))
            print("------------------------------------------------------------------------")
        return best_model

    @classmethod
    def test_eval(cls, nn, ts_set, ts_loss: LossFunction):
        e = 0
        for pattern, t in ts_set:
            o = nn.feed_forward(pattern)
            e = e + ts_loss.compute_error(t, o)
        e = e / len(ts_set)
        return e

    def get_all_hyp_comb(self, hyp_l: list):
        comb = []
        all_hyp_tested = False
        indexes = np.zeros(len(hyp_l), int)
        while not all_hyp_tested:
            comb.append(self.get_curr_hyp(hyp_l, indexes))
            all_hyp_tested = self.next_indexes(hyp_l, indexes)
        return comb

    def next_indexes(self, hyp_l: list, indexes: ndarray):
        hl_size = len(hyp_l)
        if hl_size != len(indexes): raise IndexError()
        ris = 1
        for i in range(len(indexes) - 1, -1, -1):
            (ris, rem) = divmod(indexes[i] + ris, len(hyp_l[i]))
            indexes[i] = rem
            if ris == 0:
                break
        overflow = False
        if ris == 1: overflow = True
        return overflow

    def kfolds_iter(self, t_set: ndarray, k: int):
        kf = KFold(k)
        kf.get_n_splits(t_set)
        return kf.split(t_set)

    def get_curr_hyp(self, hyp_l: list, ind: ndarray):
        hyp = []
        for i in range(len(hyp_l)):
            hyp.append(hyp_l[i][ind[i]])
        return hyp

    @classmethod
    def test_accuracy(cls, nn: NeuralNetwork, ts_set: ndarray):
        acc = 0
        for pattern, t in ts_set:
            o = nn.feed_forward(pattern)
            acc = acc + nn.classify(t, o)
        acc = acc / len(ts_set)
        return acc

    @classmethod
    def plot(cls, tr_err, vl_err, title):
        epochs = len(tr_err)
        epoch = np.arange(epochs)
        plt.plot(epoch, tr_err)
        plt.plot(epoch, vl_err, "--")
        plt.legend(['Training', 'Internal Test'])
        plt.grid(color='g', linestyle='--', linewidth=0.5)
        plt.xlabel("Epochs")
        plt.ylabel("MEE")
        plt.title(title)
        plt.show()
