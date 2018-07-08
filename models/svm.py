from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from models.feature_estimator import FeatureEstimator

class SVM(FeatureEstimator):
    def __init__(self):
        self.params = {
            'model_description': 'SVM',
            'svc_params': {
                'C': 1.0,
                'kernel': 'rbf',
                'tol': 0.001,
                'verbose': True,
                'max_iter': -1,
                'random_state': 42
            }
        }

    def fit(self, train_x, train_y, train_positions):
        super(SVM, self).fit(train_x, train_y, train_positions)

        self.model = Pipeline([
            ('clf', SVC(**self.params['svc_params']))
        ])

        print('Fitting SVM model...')
        self.model.fit(self.train_x, train_y)

    def predict(self, test_x):
        return super(SVM, self).predict(test_x)
