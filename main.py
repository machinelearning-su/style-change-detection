import time

from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import accuracy_score

from utils import get_data, get_results, write_results_to_file, get_external_data
from utils import persist_output, get_n_jobs, get_arguments, print_splits, config_local
from models import MLP, SVM, MLP_GMM, GAN, LogReg, MLP_max_diff, Stacking, StackingSimple, LSTM_model, CharacterCNN
from models.light_gbm_model import LightGbmWithLogReg
from nltk import ConfusionMatrix
from sklearn.model_selection import StratifiedKFold
import numpy as np
from metrics.breach_evaluator import evaluate, computeMeasures
from sklearn.metrics import confusion_matrix
from chunkers import get_sentences

TRAINING_DIR = 'data/training'
TRAINING_EXTERNAL_FILE = 'data/feather/external_stackexchange_feather'
VALIDATION_DIR = 'data/validation'

BREACH_DIR = 'data/breach'

input_dir, output_dir = get_arguments()
TEST_DIR = input_dir or 'data/validation'
OUTPUT_DIR = output_dir or 'data/output'

# average text length = 4329
# average tokens count = 863


def main(estimator=MLP,
         cv_split=5,
         with_cross_validation=True,
         with_validation=False,
         with_test=False,
         with_external_data=False,
         validate_on_external=False,
         with_grid_search=False,
         with_full_data_tfidf=False,
         train_with_validation=False,
         predict_breaches=True,
         train_on_breach=True,
         cutoff=None):

    if validate_on_external:
        train_x, validation_x, train_y, validation_y = get_external_data(TRAINING_EXTERNAL_FILE, 3000, 1500)
        train_positions = []
    elif train_on_breach:
        train_x, train_y, train_positions, train_file_names = get_data(
            main_dir=BREACH_DIR,
            external_file=None,
            breach=True
        )
    else:
        train_x, train_y, train_positions, train_file_names = get_data(
            main_dir=TRAINING_DIR,
            external_file=TRAINING_EXTERNAL_FILE if with_external_data else None
        )

        # return print_splits(train_x[:3000], train_positions[:3000])
        
        if train_with_validation:
            validation_x, validation_y, validation_positions, validation_file_names = get_data(
                main_dir=VALIDATION_DIR
            )
            train_x.extend(validation_x)
            train_y.extend(validation_y)

        if cutoff:
            train_x = train_x[:cutoff]
            train_y = train_y[:cutoff]
            train_positions = train_positions[:cutoff]
        

    print("Training on {0} examples".format(len(train_x)))

    clf, cv, val, gs = None, None, None, None

    if estimator:
        clf = estimator()

    if with_cross_validation:
        if with_full_data_tfidf:
            skf = StratifiedKFold(n_splits=cv_split, random_state=42, shuffle=True)
            all_acc = []
            X = np.array(train_x)
            y = np.array(train_y)
            for train_index, test_index in skf.split(X, y):
                y_train, y_test = y[train_index], y[test_index]
                X_train, X_test = X[train_index], X[test_index]
                print(X_train.shape)

                clf.fit_with_test(X_train.tolist(), y_train, train_positions, X_test.tolist())
                predictions = clf.predict(X_test.tolist())
                all_acc.append(accuracy_score(y_test, predictions))

            print("Accuracies:", all_acc)
            print("Mean:", np.mean(all_acc))
            print("Stdev:", np.std(all_acc))

        elif train_on_breach:
            skf = StratifiedKFold(n_splits=cv_split, random_state=42, shuffle=True)
            f_scores = []
            diff=[]
            r=[]
            p=[]
            all_acc = []
            X = np.array(train_x)
            y = np.array(train_y)
            pos = np.array(train_positions)
            for train_index, test_index in skf.split(X, y):
                y_train, y_test = y[train_index], y[test_index]
                X_train, X_test = X[train_index], X[test_index]
                pos_train, pos_test = pos[train_index], pos[test_index]
                print(X_train.shape)

                clf.fit_with_test(X_train.tolist(), y_train, train_positions, X_test.tolist())

                change_predictions = clf.predict(X_test.tolist())
                tn, fp, fn, tp = confusion_matrix(y_test, change_predictions).ravel()
                print('tn: {}, fp: {}, fn: {}, tp: {}'.format(tn, fp, fn, tp))
                all_acc.append(accuracy_score(y_test, change_predictions))
                predictions = get_breach_predictions(clf, X_test.tolist(), change_predictions)
                totalWinDiff, totalWinR, totalWinP, totalWinF, outStr = evaluate(X_test, pos_test, predictions)
                print("%s" % outStr)
                diff.append(totalWinDiff)
                r.append(totalWinR)
                p.append(totalWinP)
                f_scores.append(totalWinF)

            print("Mean diff:", np.mean(diff))
            print("Mean r:", np.mean(r))
            print("Mean p:", np.mean(p))
            print("Mean f:", np.mean(f_scores))


            print("Accuracies:", all_acc)
            print("Mean:", np.mean(all_acc))
            print("Stdev:", np.std(all_acc))

            # for m in all_measures:
            #     print("%s" % m)
            #     print('----------------------------------')

        else:
            cv = cross_validate(estimator=clf, X=train_x, y=train_y, fit_params={'train_positions': train_positions}, cv=cv_split,
                            scoring="accuracy", n_jobs=get_n_jobs(), return_train_score=True)

    if with_grid_search:
        clf, best_score = __grid_search(clf, clf.get_grid_params(), train_x, train_y)
        gs = { 'accuracy': best_score }

    if with_validation:
        if not validate_on_external:
            validation_x, validation_y, validation_positions, validation_file_names = get_data(
                main_dir=VALIDATION_DIR
            )
        
        if cutoff:
            validation_x = validation_x[:cutoff]
            validation_y = validation_y[:cutoff]

        t_start = time.time()
        if with_full_data_tfidf:
            clf.fit_with_test(train_x, train_y, train_positions, validation_x)
        else:
            clf.fit(train_x, train_y, train_positions)

        if predict_breaches:
            predictions = get_breach_predictions(clf, validation_x, validation_y)
        else:
            predictions = clf.predict(validation_x)
        t_end = time.time()

        if predict_breaches:
            persist_output(OUTPUT_DIR, predictions, validation_file_names, breach=predict_breaches)
            print("%s" % evaluate(validation_x, validation_positions, predictions))
        else:
            print(ConfusionMatrix(validation_y, predictions))

            val = {
                'accuracy': accuracy_score(validation_y, predictions),
                'time': t_end - t_start
            }

    if with_test:
        test(clf, train_x, train_y, train_positions, with_full_data_tfidf)

    results = get_results(len(train_x), clf_params=clf.params, cv=cv, val=val, gs=gs)
    print(results)

    if config_local().get('persist_results', False): write_results_to_file(results)

def get_breach_predictions(clf, test_x, change_predictions):
    predictions = []
    for has_change, text in zip(change_predictions, test_x):
        if has_change:
            sentences = get_sentences(text)
            breaches = find_breaches(clf, sentences, 0, len(sentences))
            print('BREACHES: ', breaches)
            predictions.append(breaches)
        else:
            predictions.append([])

    return predictions


def find_breaches(clf, sentences, l, r):
    x = np.expand_dims(' '.join(sentences[l:r]), axis=0)
    has_change = clf.predict(x)[0]

    if not has_change:
        return []

    if r - l <= 10:
        return [len(' '.join(sentences[:(l+r)//2]))]
    else:
        mid = (r-l) // 2
        left = find_breaches(clf, sentences, l, l+mid)
        right = find_breaches(clf, sentences, l+mid, r)
        if len(left) == 0 and len(right) == 0:
            return [len(' '.join(sentences[:l+mid]))]

        return left + right

def test(clf, train_x, train_y, train_positions, with_full_data_tfidf):
    test_x, test_y, test_positions, test_file_names = get_data(main_dir=TEST_DIR)

    if len(test_x) <= 0: return print('Test dataset is empty!')

    t_start = time.time()
    if with_full_data_tfidf:
        clf.fit_with_test(train_x, train_y, train_positions, test_x)
    else:
        clf.fit(train_x, train_y, train_positions)

    predictions = get_breach_predictions(clf, test_x)
    t_end = time.time()

    persist_output(OUTPUT_DIR, predictions, test_file_names, breach=predict_breaches)

def grid_search(clf, params, x, y, positions):
    gs_clf = GridSearchCV(clf, params, n_jobs=get_n_jobs(), scoring='accuracy', verbose=1, cv=2)
    gs_clf.fit(x, y, positions)

    print("Best parameters:")
    print(gs_clf.best_params_)
    print("Best score: %0.3f" % gs_clf.best_score_)

    return gs_clf.best_estimator_, gs_clf.best_score_


if __name__ == '__main__':
    t_start = time.time()
    main()
    t_end = time.time()
    m, s = divmod(t_end - t_start, 60)
    print("All done in %fm and %fs" % (round(m), round(s)))
