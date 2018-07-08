from models.base_estimator import BaseEstimator
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import  StratifiedKFold # always go #Stratified on Twitter


from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
import gc

TARGET_LABEL = "is_multi_author"
TEXT_AS_KEY = "text"
SUCCESS_MESSAGE = "Done"
SUCCESS_HSTACK = "HStack Done"


class LightGbmWithLogReg(BaseEstimator):
    def __init__(self):
        self.params = {
            'model_description' : 'LightGBM with Logistic Regression',
            'lighgbm' : {
                'learning_rate': 0.1,
                'application': 'binary',
                'num_leaves': 31,
                'verbosity': -1,
                'metric': 'auc', # All our validation and calculation metrics are in terms of Accuracy ! I am adding another point of view with 'auc' for additional monitoring
                'data_random_seed': 2,
                'bagging_fraction': 0.8,
                'feature_fraction': 0.6,
                'nthread': 4,
                'lambda_l1': 1,
                'lambda_l2': 1,
                'min_data_in_leaf' : 40,
               # 'scale_pos_weight': 20
            },
            'logistic_regression' : {
                'inverse_strength' : 2.0,
                'solver' : 'sag',
                'minimum_feature_weight' : 0.1,
            },
            'optimal_rounds' : {
                'is_multi_author' : 140,
            },
            'num_folds' : 5,
            'cv_random_seed' : 777,
            'max_features' : 100000,
        }

    def fit_feature_vectors(self, train_features, train_y):
        skf = StratifiedKFold(n_splits= self.params['num_folds'], random_state= self.params['cv_random_seed'])  # Yeah, hope to be lucky today

        cvAccuracy = 0  # Our total CV Accuracy for Bagging=5 will be stored here

        self.bagged_models = []
        submission = pd.DataFrame() # Hold final predictions

        train_target = np.array(train_y)
        logreg_model = LogisticRegression(C=self.params['logistic_regression']['inverse_strength'],
                                       solver=self.params['logistic_regression']['solver'])
        self.sfm = SelectFromModel(logreg_model)

        train_sparse_matrix = self.sfm.fit_transform(train_features, train_target)

        validation_sparse_matrix = self.sfm.transform(train_features)

        class_name = 'is_multi_author'
        for (f, (train_index, validation_index)) in enumerate(skf.split(train_sparse_matrix, train_target)):
            d_train = lgb.Dataset(train_sparse_matrix[train_index], label=train_target[train_index])
            d_valid = lgb.Dataset(train_sparse_matrix[validation_index], label=train_target[validation_index])

            watchlist = [d_train, d_valid]

            model = lgb.train(self.params['lighgbm'],
                              train_set=d_train,
                              num_boost_round=self.params['optimal_rounds'][class_name],
                              valid_sets=watchlist,
                              verbose_eval=0  # Ten is suitable choice for good monitoring
                              )

            crossValidationPrediction = pd.DataFrame()
            crossValidationPrediction[class_name] = model.predict(validation_sparse_matrix[validation_index])

            validationSubset = pd.DataFrame()
            validationSubset[class_name] = train_target[validation_index]

            crossValidationPrediction = crossValidationPrediction.reset_index(drop=True)
            validationSubset = validationSubset.reset_index(drop=True)

            crossValidationPrediction[crossValidationPrediction[class_name] < 0.5] = 0
            crossValidationPrediction[crossValidationPrediction[class_name] >= 0.5] = 1

            foldAccuracy = (
                    ((crossValidationPrediction[TARGET_LABEL] == validationSubset[TARGET_LABEL]).sum())
                    / len(validationSubset))


            cvAccuracy += foldAccuracy

            self.bagged_models.append(model)

        cvAccuracy /= self.params['num_folds']  # Number of folds = 5 Bagging :)

    def fit(self, train_x, train_y, train_positions):
        train_text = train_x

        self.word_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            ngram_range=(1, 2),
            max_features=self.params['max_features']
        )

        self.word_vectorizer.fit(train_text)
        train_word_features = self.word_vectorizer.transform(train_text)

        self.char_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='char',
            stop_words='english',
            ngram_range=(2, 6),
            max_features=self.params['max_features']
        )

        self.char_vectorizer.fit(train_text)
        train_char_features = self.char_vectorizer.transform(train_text)

        train_features = hstack([train_char_features, train_word_features])

        del train_text
        del train_char_features
        del train_word_features

        gc.collect()

        self.fit_feature_vectors(train_features, train_y)

    def fit_with_test(self, train_x, train_y, train_positions, test_x):
        train_text = train_x
        test_text = test_x

        all_text = train_x + test_x

        self.word_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            ngram_range=(1, 2),
            max_features=self.params['max_features']
        )

        self.word_vectorizer.fit(all_text)
        train_word_features = self.word_vectorizer.transform(train_text)

        self.char_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='char',
            stop_words='english',
            ngram_range=(2, 6),
            max_features=self.params['max_features']
        )

        self.char_vectorizer.fit(all_text)
        train_char_features = self.char_vectorizer.transform(train_text)

        train_features = hstack([train_char_features, train_word_features])

        del train_text
        del test_text
        del all_text
        del train_char_features
        del train_word_features
        gc.collect()

        self.fit_feature_vectors(train_features, train_y)

    def predict(self, test_x, return_probability=False):
        train_text = test_x

        test_word_features = self.word_vectorizer.transform(train_text)
        test_char_features = self.char_vectorizer.transform(train_text)
        test_features = hstack([test_char_features, test_word_features])

        submission = pd.DataFrame()  # Our final prediction will be stored here

        test_sparse_matrix = self.sfm.transform(test_features)

        for cnt in range(len(self.bagged_models)):
            # Bagging 5 predictions on the test :)
            if cnt == 0:
                submission[TARGET_LABEL] = self.bagged_models[cnt].predict(test_sparse_matrix)
            else:
                submission[TARGET_LABEL] += self.bagged_models[cnt].predict(test_sparse_matrix)


        submission[TARGET_LABEL] /= self.params['num_folds']

        if not return_probability:
            submission[submission.is_multi_author < 0.5] = 0
            submission[submission.is_multi_author >= 0.5] = 1

        return submission.is_multi_author

    def predict_proba(self, test_x):
        probabilities = self.predict(test_x, return_probability=True)
        return np.array(map(lambda x: [x], probabilities))
