from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from models.feature_estimator import FeatureEstimator

class LogReg(FeatureEstimator):
    def __init__(self):
        self.params = {
            'model_description': 'LogReg_GMM',
            'log_params': {
                'penalty': 'l2',
                'C': 1.0,
                'solver': 'liblinear',
                'tol': 0.001,
                'verbose': False,
                'random_state': 42
            },
            'gmm': {
                'pca': False,
                'n_components': 3,
                'covariance_type': 'spherical'
            }
        }

    def fit(self, train_x, train_y, train_positions):
        super(LogReg, self).fit(train_x, train_y)

        self.model = Pipeline([
            ('clf', LogisticRegression(**self.params['log_params']))
        ])

        print('Fitting LogReg model...')

        self.model.fit(self.train_x, train_y)

    def predict(self, test_x):
        return super(LogReg, self).predict(test_x)
