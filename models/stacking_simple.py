import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from models.base_estimator import BaseEstimator
from models.light_gbm_model import LightGbmWithLogReg
from models.character_cnn import CharacterCNN
from features import lexical
from chunkers import word_chunks
from features.word_frequency import WordFrequency
from transformers import phrase_frequency
from transformers import frequent_words_diff
from transformers import apostrophe_discrepancies
from features import readability
from transformers import quote_discrepancies
from transformers import ascii_discrepancies
from transformers import max_diff
from transformers import text_length
from transformers import split_points
from preprocessing.basic_preprocessor import BasicPreprocessor


preprocessor = BasicPreprocessor()

class StackingSimple(BaseEstimator):
    def __init__(self):
        self.params = {
            'model_description': 'Meta Stacking',
            'meta_learner': True,
            'meta_learner_train_ratio': 0.75,
            'meta_learner_prob': True,
            'meta_params': {
                'penalty': 'l2',
                'C': 1.0,
                'solver': 'liblinear',
                'tol': 0.001,
                'verbose': False,
                'random_state': 42,
            },
            'svm_params': {
                'C': 1.0,
                'kernel': 'rbf',
                'tol': 0.001,
                'verbose': 0,
                'max_iter': -1,
                'random_state': 42,
                'probability': True
            },
            'rf_params': {
                'n_estimators': 300,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': 0
            },
            'ab_params': {
                'n_estimators': 300,
                'random_state': 42,
            },
            'mlp_params': {
                'max_iter': 10000,
                'verbose': 0,
                'tol': 0.01,
                'alpha': 0.01,
                'solver': 'adam'
            },
            'phrase_transformer': {
                'word_gram_sizes': [1, 2, 3, 4],
                'stop_words': False,
                'use_mean': True
            },
            'frequent_words_diff_transformer': {
                'window': 0.33,
                'use_stop_words': True,
                'use_func_words': True,
                'local_diff': False
            },
            'ascii_transformer': {
                'window': 0.33,
                'local_diff': False
            },
            'scaler_params': {
                'with_mean': True,
                'with_std': True
            },
            'word_chunk_params': {
               'chunks': 3,
               'process': False,
               'sliding': True
            },
            'split_points_params': {},
            'readability_params': {
                'indices': [
                    'flesch_reading_ease',
                    'smog_index',
                    'flesch_kincaid_grade',
                    'coleman_liau_index',
                    'automated_readability_index',
                    'dale_chall_readability_score',
                    'difficult_words',
                    'linsear_write_formula',
                    'gunning_fog'
                 ]
            },
            'fit_with_train': True,
            'use_raw_text_models': True,
            'output_probabilities_train': False,
            'output_probabilities_test': False,
            'use_nn' : False,
        }

    def fit_with_test(self, train_x, train_y, train_positions, test_x):
        self.fit(train_x, train_y, train_positions, test_x)

    def fit(self, train_x, train_y, train_positions, test_x= None):
        train_x = [preprocessor.process_text(x) for x in train_x]
        phrase_frequency_func = lambda x: np.array(phrase_frequency(x, **self.params['phrase_transformer']))
        frequent_words_diff_func = lambda x: np.array(frequent_words_diff(x, **self.params['frequent_words_diff_transformer']))
        min_max_lexical_per_segment_func = lambda x: max_diff(lexical(x))
        apostrophe_discrepancies_func = lambda x: np.array(apostrophe_discrepancies(x))
        quote_discrepancies_func = lambda x: np.array(quote_discrepancies(x))
        ascii_discrepancies_func = lambda x: np.array(ascii_discrepancies(x, **self.params['ascii_transformer']))
        rare_richness_func = lambda x: max_diff(WordFrequency().average_word_frequency((x)))
        text_length_func = lambda x: np.array(text_length(x))
        readability_func = lambda x: np.array(max_diff(readability(x)))

        self.stack = [
            (phrase_frequency_func, False, False, StandardScaler(**self.params['scaler_params']), [
                RandomForestClassifier(**self.params['rf_params']),
                MLPClassifier(**self.params['mlp_params']),
                SVC(**self.params['svm_params']),
                AdaBoostClassifier(**self.params['ab_params']),
            ]),

            (frequent_words_diff_func, False, False, StandardScaler(**self.params['scaler_params']), [
                RandomForestClassifier(**self.params['rf_params']),
                MLPClassifier(**self.params['mlp_params']),
                SVC(**self.params['svm_params']),
                AdaBoostClassifier(**self.params['ab_params']),
            ]),

            (readability_func, True, False, StandardScaler(**self.params['scaler_params']), [
                RandomForestClassifier(**self.params['rf_params']),
                MLPClassifier(**self.params['mlp_params']),
                SVC(**self.params['svm_params']),
                AdaBoostClassifier(**self.params['ab_params']),
            ]),

            (apostrophe_discrepancies_func, False, False, StandardScaler(**self.params['scaler_params']), [
                RandomForestClassifier(**self.params['rf_params']),
                MLPClassifier(**self.params['mlp_params']),
                SVC(**self.params['svm_params']),
                AdaBoostClassifier(**self.params['ab_params']),
            ]),

            (quote_discrepancies_func, False, False, StandardScaler(**self.params['scaler_params']), [
                RandomForestClassifier(**self.params['rf_params']),
                MLPClassifier(**self.params['mlp_params']),
                SVC(**self.params['svm_params']),
                AdaBoostClassifier(**self.params['ab_params']),
            ]),

            (rare_richness_func, True, False, StandardScaler(**self.params['scaler_params']), [
                RandomForestClassifier(**self.params['rf_params']),
                MLPClassifier(**self.params['mlp_params']),
                SVC(**self.params['svm_params']),
                AdaBoostClassifier(**self.params['ab_params']),
            ]),

            (min_max_lexical_per_segment_func, True, True, StandardScaler(**self.params['scaler_params']), [
                RandomForestClassifier(**self.params['rf_params']),
                MLPClassifier(**self.params['mlp_params']),
                SVC(**self.params['svm_params']),
                AdaBoostClassifier(**self.params['ab_params']),
            ]),

            # (ascii_discrepancies_func, False, False, StandardScaler(**self.params['scaler_params']), [
            #     RandomForestClassifier(**self.params['rf_params']),
            #     MLPClassifier(**self.params['mlp_params']),
            #     SVC(**self.params['svm_params']),
            #     AdaBoostClassifier(**self.params['ab_params']),
            # ]),

            (text_length_func, False, False, StandardScaler(**self.params['scaler_params']), [
                RandomForestClassifier(**self.params['rf_params']),
                MLPClassifier(**self.params['mlp_params']),
                SVC(**self.params['svm_params']),
                AdaBoostClassifier(**self.params['ab_params']),
            ])
        ]

        self.raw_text_models = [
            LightGbmWithLogReg(),
        ]

        if self.params["use_nn"]:
            self.raw_text_models.append(CharacterCNN())

        if(self.params['meta_learner']):
            self.global_model_weights = []

            train_x_zero, train_y_zero, train_x_meta, train_y_meta = self.split_data(train_x, train_y)

            self.meta_learner = LogisticRegression(**self.params['meta_params'])

            predictions_zero = self.stack_fit_predict(train_x_zero, train_y_zero, train_x_meta, train_y_meta, test_additional=train_x_meta)
            self.meta_learner.fit(self.convert_to_meta_input(predictions_zero), train_y_meta)

        self.stack_fit_predict(train_x, train_y, test_additional=test_x)

    def predict(self, test_x):
        predictions = []

        test_x_preprocessed = [preprocessor.process_text(x) for x in test_x]
        test_word_chunks = word_chunks(test_x, **self.params['word_chunk_params'])
        test_word_chunks_preprocessed = word_chunks(test_x_preprocessed, chunks=3, process=True, sliding=True)

        cnt = 0
        if self.params['use_raw_text_models']:
            for model in self.raw_text_models:
                print("Predicting via LightGBM ...")
                test_probs = model.predict_proba(test_x).tolist()
                predictions.append(test_probs)
                if self.params['output_probabilities_test']:
                    res = pd.DataFrame(np.array(test_probs))
                    res.to_csv('model_{0}_test.csv'.format(cnt))
                    cnt += 1
        else:
            print("No")

        test_word_chunks = word_chunks(test_x, **self.params['word_chunk_params'])
        print("Computed word chunks")

        for (transformer, apply_on_word_chunks, preprocess, scaler, predictors), model_weights in zip(self.stack, self.global_model_weights):
            if preprocess:
                raw_test = test_word_chunks_preprocessed if apply_on_word_chunks else test_x_preprocessed
            else:
                raw_test = test_word_chunks if apply_on_word_chunks else test_x

            data_transformed = scaler.transform(transformer(raw_test))

            local_predictions = [predictor.predict_proba(data_transformed).tolist() for predictor in predictors]
            predictions.append(self.weight_local_predictions(local_predictions, model_weights))

            if self.params['output_probabilities_test']:
                res = pd.DataFrame(np.array(test_probs))
                res.to_csv('model_{0}_test.csv'.format(cnt))
                cnt += 1

        if(self.params['meta_learner']):
            prediction_converted = self.convert_to_meta_input(predictions)

            return self.meta_learner.predict(prediction_converted)
        else:
            predictions_prob = [[sum(y) for y in zip(*x)] for x in zip(*predictions)]

            return [x[0] < x[1] for x in predictions_prob]


    def stack_fit_predict(self, train_x, train_y, test_x = None, test_y = None, test_additional= None):
        predictions = []

        train_x_preprocessed = [preprocessor.process_text(x) for x in train_x]
        train_word_chunks = word_chunks(train_x, **self.params['word_chunk_params'])
        train_word_chunks_preprocessed = word_chunks(train_x_preprocessed, chunks=3, process=True, sliding=True)


        if(test_x): test_size = len(test_x)

        cnt = 0
        if self.params['use_raw_text_models']:
            for model in self.raw_text_models:
                if self.params['fit_with_train'] and test_additional:
                    print("Fitting LightGBM ...")
                    model.fit_with_test(train_x, train_y, [], test_additional)
                else:
                    print("Fitting LightGBM without train")
                    model.fit(train_x, train_y, [])
                if test_x:
                    predictions.append(model.predict_proba(test_x).tolist())
                elif self.params['output_probabilities_train']:
                    res = pd.DataFrame(model.predict_proba(train_x))
                    res.to_csv("model_{0}_train.csv".format(cnt))
                    cnt += 1
        else:
            print("No")

        train_word_chunks = word_chunks(train_x, **self.params['word_chunk_params'])
        test_word_chunks = None
        if test_x:
            test_x_preprocessed = [preprocessor.process_text(x) for x in test_x]
            test_word_chunks = word_chunks(test_x, **self.params['word_chunk_params'])
            test_word_chunks_preprocessed = word_chunks(test_x_preprocessed, chunks=3, process=True, sliding=True)
        print("Computed word chunks")

        for transformer, apply_on_word_chunks, preprocess, scaler, predictors in self.stack:
            if preprocess:
                raw_data = train_word_chunks_preprocessed if apply_on_word_chunks else train_x_preprocessed
            else:
                raw_data = train_word_chunks if apply_on_word_chunks else train_x

            data_transformed_zero = scaler.fit_transform(transformer(raw_data))

            if(test_x):
                if preprocess:
                    raw_test = test_word_chunks_preprocessed if apply_on_word_chunks else test_x_preprocessed
                else:
                    raw_test = test_word_chunks if apply_on_word_chunks else test_x

                data_transformed_meta = scaler.transform(transformer(raw_test))

            local_predictions = []
            for predictor in predictors:
                predictor.fit(data_transformed_zero, train_y)

                if(test_x):
                    local_predictions.append(predictor.predict_proba(data_transformed_meta).tolist())
                if self.params['output_probabilities_train']:
                    res = pd.DataFrame(predictor.predict_proba(data_transformed_zero))
                    res.to_csv("model_{0}_train.csv".format(cnt))
                    cnt += 1

            if(test_x):
                scores = []
                for model_predictions in local_predictions:
                     acc = [(prob[0] < prob[1]) == truth for prob, truth in zip(model_predictions, test_y)].count(True) / test_size
                     scores.append(acc)

                sum_scores = sum(scores)
                model_weights = [x / sum_scores for x in scores]

                print('Model weights: ', model_weights)
                self.global_model_weights.append(model_weights)

                predictions.append(self.weight_local_predictions(local_predictions, model_weights))


        if(test_x): return predictions

    def weight_local_predictions(self, predictions, model_weights):
        weighted_mean_prediction = list(map(
            lambda sample: [sum([x * model_weights[i]  for i, x in enumerate(preds)]) for preds in zip(*sample)],
            list(zip(*predictions))
        ))

        return weighted_mean_prediction


    def convert_to_meta_input(self, predictions):
        return list(map(
            lambda sample: list(map(lambda s: s[0] if self.params['meta_learner_prob'] else s[0] < s[1], sample)),
            list(zip(*predictions))
        ))

    def split_data(self, train_x, train_y):
        split_pos = round(len(train_x) * self.params['meta_learner_train_ratio'])

        train_x_zero = train_x[:split_pos]
        train_x_meta = train_x[split_pos:]

        train_y_zero = train_y[:split_pos]
        train_y_meta = train_y[split_pos:]

        return train_x_zero, train_y_zero, train_x_meta, train_y_meta

    def get_grid_params(self): pass
