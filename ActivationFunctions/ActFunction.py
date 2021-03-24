from numpy import ndarray
from abc import ABC, abstractmethod


class ActivationFunction(ABC):

    @abstractmethod
    def compute(self, v: ndarray) -> ndarray:
        return

    @abstractmethod
    def compute_derivative(self, x: ndarray) -> ndarray:
        return


