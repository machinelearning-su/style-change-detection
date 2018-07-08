from abc import ABC, abstractmethod
from utils import update_dict
import numpy as np

class BaseEstimator(ABC):
    @abstractmethod
    def fit(self, train_x, train_y, train_positions):
        pass

    @abstractmethod
    def predict(self, test_x, test_y):
        pass

    def get_params(self, deep = True):
        return {}

    def set_params(self, **params):
        for key, value in params.items():
            update_dict(self.params, key.split('__'), value)
        return self

    def get_grid_params(self):
        return {}
