import pandas as pd
import numpy as np
import re
import lightgbm as lgb
import warnings

# warnings.filterwarnings(action='ignore', category=DeprecationWarning, module='sklearn')
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import log_loss, matthews_corrcoef, roc_auc_score
from sklearn.linear_model import LogisticRegression

def get_subs():
    oofs = [
    '/Users/atanas/Downloads/model_A_train_prediction.csv',
    # and so on
    ]

    subs = [
        '/Users/atanas/Downloads/model_A_test_prediction.csv',
    # and so on
    ]

    OOFS = np.hstack([np.array(pd.read_csv(oof)[LABELS]) for oof in oofs])

    SUBS = np.hstack(
        [np.array(pd.read_csv(sub)[LABELS]) for sub in subs])

    return SUBS, OOFS

if __name__ == "__main__":

    train = pd.read_csv('train.csv').fillna(' ')
    test = pd.read_csv('test.csv').fillna(' ')
    INPUT_COLUMN = "text"
    LABELS = ["text"]

    subs, oofs = get_subs()

    F_train = (train[INPUT_COLUMN])
    F_test = (test[INPUT_COLUMN])

    X_train = np.hstack([F_train.as_matrix(), oofs])
    X_test = np.hstack([F_test.as_matrix(), subs])

    n_estimators = 200
    stacker = clf = XGBClassifier(n_estimators=n_estimators,
                                  max_depth=4,
                                  objective="binary:logistic",
                                  learning_rate=.1,
                                  subsample=.8,
                                  colsample_bytree=.8,
                                  gamma=1,
                                  reg_alpha=0,
                                  reg_lambda=1,
                                  nthread=2)

    # Fit and submit
    scores = []
    sub = pd.DataFrame()

    for label in LABELS:
        print(label)
        score = cross_val_score(stacker, X_train, train[label], cv=5, scoring='roc_auc')
        print("AUC:", score)
        scores.append(np.mean(score))
        stacker.fit(X_train, train[label])
        sub[label] = stacker.predict_proba(X_test)[:, 1]
    print("CV score:", np.mean(scores))

    sub.to_csv("xgboost_final_ensemble.csv", index=False)
