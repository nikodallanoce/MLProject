import numpy as np
from numpy import ndarray
from ActivationFunctions import ActFunction


class Layer:

    def __init__(self, act_funct: ActFunction, curr_layer_dim, prev_layer_dim, m):
        self.nodes: ndarray = np.zeros(curr_layer_dim)
        self.act_funct: ActFunction = act_funct
        self.nets: ndarray = np.zeros(curr_layer_dim)
        self.d_weights = np.zeros((curr_layer_dim, prev_layer_dim+1))
        self.gradients = np.zeros(self.nodes.size)
        self.weights: ndarray = np.random.uniform(-0.7 * m, 0.7 * m, (curr_layer_dim, prev_layer_dim + 1))

    def __str__(self):
        return self.nodes.__str__()

    def update_weights(self, lamb: float, dw: ndarray):
        self.weights = self.weights + dw - lamb * self.weights

    '''def set_delta_weights(self, d_w: ndarray):
        self.d_weights = d_w'''
