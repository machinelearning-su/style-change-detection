import numpy as np
import random
from models.base_estimator import BaseEstimator

class BaselineEQ(BaseEstimator):
    def __init__(self):
        self.params = {
            'model_description': 'BaselineEQ',
        }

    def fit(self, train_x, train_y, train_positions):
        pass

    def fit_with_test(self, train_x, train_y, train_positions, test_x):
        pass

    def predict(self, test_x):
        predictions = []
        for x in test_x:
            n = random.randint(0, 10)
            if n == 0:
                p = []
            else:
                p = range(0, len(x), len(x) // (n + 1))
            print(p)
            predictions.append(p)

        return predictions