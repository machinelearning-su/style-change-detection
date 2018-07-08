import time

from sklearn.metrics import accuracy_score

from utils import get_data
from models import MLP, SVM, MLP_GMM, GAN, LogReg, MLP_max_diff, Stacking, StackingSimple, LSTM_model, CharacterCNN, RandomForest
from models.light_gbm_model import LightGbmWithLogReg
from nltk import ConfusionMatrix
import numpy as np
import pandas as pd

TRAINING_DIR = 'data/training'
VALIDATION_DIR = 'data/validation'


def main(estimators=[LightGbmWithLogReg, Stacking],
         with_full_data_tfidf=True,
         cutoff=None):
    train_x, train_y, train_positions, train_file_names = get_data(
        main_dir=TRAINING_DIR
    )

    validation_x, validation_y, validation_positions, validation_file_names = get_data(
        main_dir=VALIDATION_DIR
    )    
    
    if cutoff:
        train_x = train_x[:cutoff]
        validation_x = validation_x[:cutoff]
        train_y = train_y[:cutoff]
        validation_y = validation_y[:cutoff]
        train_positions = train_positions[:cutoff]

    clfs = [estimator() for estimator in estimators]
    predictions_df = pd.DataFrame()
    predictions_df['text'] = validation_x
    predictions_df['actual'] = validation_y

    for clf in clfs:
        t_start = time.time()
        if with_full_data_tfidf:
            clf.fit_with_test(train_x, train_y, validation_x)
        else:
            clf.fit(train_x, train_y)

        predictions = clf.predict(validation_x)
        t_end = time.time()        
        m, s = divmod(t_end - t_start, 60)

        print(ConfusionMatrix(validation_y, predictions))

        acc = accuracy_score(validation_y, predictions)
        model_name = clf.params['model_description']

        print(model_name)
        print("Done in %fm and %fs" % (round(m), round(s)))
        print("Accuracy:" + str(acc))

        predictions_df[model_name] = predictions
    
    predictions_df.to_csv("data/output/model_predictions.csv", index=False)


if __name__ == '__main__':
    t_start = time.time()
    main()
    t_end = time.time()
    m, s = divmod(t_end - t_start, 60)
    print("All done in %fm and %fs" % (round(m), round(s)))
