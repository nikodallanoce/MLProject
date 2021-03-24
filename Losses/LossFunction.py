from abc import ABC, abstractmethod
import math
import numpy as np


class LossFunction(ABC):
    """
    Classe astratta per le funzioni di perdita
    """

    @abstractmethod
    def compute_error(self, t, o):
        return
