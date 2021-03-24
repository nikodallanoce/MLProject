import numpy as np
from numpy import ndarray
from ActivationFunctions import ActFunction
# from Batch import Batch
from Layer import Layer
from Losses.Accuracy import Accuracy
from Losses.LossFunction import LossFunction


class NeuralNetwork:

    def __init__(self, structure: list, hid_act_funct: ActFunction, out_act_fun: ActFunction, vl_set: ndarray):
        self.layers: [Layer] = self.create_nn(structure, hid_act_funct, out_act_fun)
        self.vl_set = vl_set

    '''def train_class(self, tr_set: ndarray, a: float, eta: float, lamb: float, epochs: int, loss: LossFunction,
                    b_size: int,
                    isRegression: bool):
        train_err = []
        val_err = []
        flag = True
        i = 0
        tr_e = 0
        while flag:
            self.do_epoch(a, b_size, eta, lamb, tr_set)
            tr_e = self.epoch_error(tr_set, loss)
            train_err.append(tr_e)
            val_err.append(self.epoch_error(self.vl_set, loss))
            np.random.shuffle(tr_set)
            i = i + 1
            flag = i < epochs

        return train_err, val_err'''

    def train(self, tr_set: ndarray, alpha: float, eta: float, lamb: float, epochs: int, loss: LossFunction,
              b_size: int, is_regression: bool):
        train_err = []
        val_err = []
        train_acc = []
        val_acc = []

        for _ in range(epochs):
            self.do_epoch(alpha, b_size, eta, lamb, tr_set)
            tr_e = self.epoch_error(tr_set, loss)
            train_err.append(tr_e)
            val_err.append(self.epoch_error(self.vl_set, loss))
            if not is_regression:
                train_acc.append(self.epoch_error(tr_set, Accuracy(self.layers[-1].act_funct)))
                val_acc.append(self.epoch_error(self.vl_set, Accuracy(self.layers[-1].act_funct)))

        if not is_regression:
            return train_err, val_err, train_acc, val_acc
        else:
            return train_err, val_err

    def do_epoch(self, alpha, b_size, eta, lamb, tr_set):
        batches = self.generate_batches(b_size, tr_set)
        for batch in batches:
            grad_acc = [0 for _ in range(len(self.layers))]
            for pattern, t in batch:
                o = self.feed_forward(pattern)
                self.back_propagation(t - o)  # back propagation
                p_grad = self.pattern_grad(pattern)
                grad_acc = [i + j for i, j in zip(grad_acc, p_grad)]

            dw = self.compute_delta_weights(alpha, eta, np.divide(grad_acc, len(batch)))
            self.update_net_weights(lamb, dw)  # update dei pesi

    def adaptive_learn_rate(self, n_0: float, epoch: int, limit: int):
        a = epoch / limit
        if a > 1:
            a = 1
        n_t = n_0 * 0.01
        n_s = (1 - a) * n_0 + a * n_t
        return n_s

    def pattern_grad(self, pattern: ndarray):
        grad_patt = []
        for i in range(len(self.layers) - 1, 0, -1):
            prev_layer_nodes = np.concatenate((np.array([1]), self.layers[i - 1].nodes))
            r = np.dot(self.layers[i].gradients[:, np.newaxis], prev_layer_nodes[:, np.newaxis].T)
            grad_patt.insert(0, r)

        prev_layer_nodes = np.concatenate((np.array([1]), pattern))
        r = np.dot(self.layers[0].gradients[:, np.newaxis], prev_layer_nodes[:, np.newaxis].T)
        grad_patt.insert(0, r)
        return grad_patt

    def compute_delta_weights(self, alpha: float, eta: float, sum_grad_pattern: list):
        layer_d_w = []
        for layer, gp in zip(self.layers, sum_grad_pattern):
            r = eta * gp
            s = alpha * layer.d_weights
            d_w = r + s
            layer_d_w.append(d_w)
        return layer_d_w

    def epoch_error(self, patterns: ndarray, loss_f: LossFunction):
        err = 0
        for pattern, t in patterns:
            o = self.feed_forward(pattern)
            err += loss_f.compute_error(t, o)

        err /= len(patterns)
        return err

    def classify(self, target: ndarray, out):
        accuracy = 0
        o = np.where(out >= self.layers[-1].act_funct.threshold(), 1, 0)
        if o == target:
            accuracy = 1
        return accuracy

    def update_net_weights(self, lamb: float, dw: list):
        for layer, d in zip(self.layers, dw):
            layer.update_weights(lamb, d)
            layer.d_weights = d

    def back_propagation(self, err: ndarray):
        f_der: ActFunction = self.layers[-1].act_funct
        grad = err * f_der.compute_derivative(self.layers[-1].nets)  # out layer gradients
        self.layers[-1].gradients = grad
        w = self.layers[-1].weights[:, 1:]
        for layer in reversed(self.layers[:-1]):
            f_der = layer.act_funct.compute_derivative(layer.nets)
            grad = np.multiply(f_der, np.dot(grad, w))
            layer.gradients = grad
            w = layer.weights[:, 1:]

    def generate_batches(self, batch_size, tr_set):
        batch_index = list(range(batch_size, len(tr_set), batch_size))  # indici per la suddivisione in batch
        batches = np.array_split(tr_set, batch_index)  # batches del tr_set
        return batches

    def feed_forward(self, pattern: ndarray):
        prev_layer_nodes = pattern
        for layer in self.layers:
            w = layer.weights
            layer.nets = np.dot(w[:, 1:], prev_layer_nodes) + w[:, 0]
            layer.nodes = layer.act_funct.compute(layer.nets)
            prev_layer_nodes = layer.nodes

        return prev_layer_nodes

    def create_nn(self, structure, f_h: ActFunction, f_o: ActFunction) -> [Layer]:
        layers: [Layer] = []
        for i in range(1, len(structure) - 1):
            layers.append(Layer(f_h, structure[i], structure[i - 1], 2 / structure[i - 1]))  # hidden_layer

        layers.append(Layer(f_o, structure[-1], structure[-2], 1))  # output_layer
        return layers
