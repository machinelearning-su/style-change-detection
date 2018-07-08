import numpy as np
import random
from models.base_estimator import BaseEstimator

class BaselineRND(BaseEstimator):
    def __init__(self):
        self.params = {
            'model_description': 'BaselineRND',
        }

    def fit(self, train_x, train_y, train_positions):
        pass

    def fit_with_test(self, train_x, train_y, train_positions, test_x):
        pass

    def predict(self, test_x):
        predictions = []
        for x in test_x:
            n = random.randint(0, 10)
            p = random.sample(range(1, len(x) - 1), n)
            print(p)
            predictions.append(p)

        return predictions