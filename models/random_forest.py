from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from models.feature_estimator import FeatureEstimator
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

class RandomForest(FeatureEstimator):
    def __init__(self):
        self.params = {
            'model_description': 'RandomForest',
            'rf_params': {
                'n_estimators': 300,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': 0
            },
            'logistic_regression' : {
                'inverse_strength' : 2.0,
                'solver' : 'sag',
                'minimum_feature_weight' : 0.3,
            }
        }

    def fit(self, train_x, train_y, train_positions):
        super(RandomForest, self).fit(train_x, train_y)

        logreg_model = LogisticRegression(C=self.params['logistic_regression']['inverse_strength'],
                                       solver=self.params['logistic_regression']['solver'])

        self.model = Pipeline([
            ('sfm', SelectFromModel(logreg_model, threshold=self.params['logistic_regression']['minimum_feature_weight'])),
            ('clf', RandomForestClassifier(**self.params['rf_params']))
        ])

        print('Fitting RndForest model...')

        self.model.fit(self.train_x, train_y)

    def predict(self, test_x):
        return super(RandomForest, self).predict(test_x)
