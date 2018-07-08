import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from models.feature_estimator import FeatureEstimator
from features.ngrams import NGrams

class MLP_max_diff(FeatureEstimator):
    def __init__(self):
        self.params = {
            'model_description': 'MLP max_diff',
            'mlp': {
                'fit': {
                    'batch_size': 32,
                    'epochs': 10,
                    'verbose': 1,
                    'validation_split': 0.1
                },
                'predict': {
                    'batch_size': 32,
                    'verbose': 0
                }
            },
            'gmm': {
                'pca': True,
                'n_components': 3,
                'covariance_type': 'full'
            }
        }

    def fit(self, train_x, train_y, train_positions):
        super(MLP_max_diff, self).fit(train_x, train_y)

        num_classes = np.max(train_y) + 1

        train_y = keras.utils.to_categorical(train_y, num_classes=num_classes)

        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(
            self.train_x.shape[1],), activation='sigmoid'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes, activation='softmax'))
        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

        print('Fitting MLP model...')
        self.model.fit(self.train_x, train_y, **self.params['mlp']['fit'])

    def predict(self, test_x):
        pos_tag_x = [NGrams.to_pos_tags(x) for x in test_x]
        test_x, _ = self.pipeline(test_x, pos_tag_x, fit_scalers=False)

        predictions = self.model.predict(
            test_x, **self.params['mlp']['predict'])

        return predictions.argmax(axis=-1)

    def get_grid_params(self):
        return {
            'gmm__covariance_type': ('full', 'tied', 'diag', 'spherical')
        }
