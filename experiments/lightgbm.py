import numpy as np
import pandas as pd
import lightgbm as lgb  #
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Read data
folder = "/pan_data/"
df_train = pd.read_csv(folder + "train.csv")
df_test = pd.read_csv(folder + "test.csv")
df_sub = pd.read_csv(folder + "sample_submission.csv")
print("input data read")

train_text = df_train['comment_text']
test_text = df_test['comment_text']
all_text = pd.concat([train_text, test_text])
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(all_text)
print("vectorizer ready")
train_word_features = word_vectorizer.transform(train_text)
print("training data transformed")
test_word_features = word_vectorizer.transform(test_text)
print("testing data transformed")

# LGBM Model
class_names = ['is_multi_author']
predictions = {'id': df_test['id']}
for class_name in class_names:
    x_train, x_valid = train_test_split(train_word_features, test_size=0.2, random_state=123)
    y_train, y_valid = train_test_split(df_train[class_name], test_size=0.2, random_state=123)
    d_train = lgb.Dataset(x_train, label=y_train)
    d_valid = lgb.Dataset(x_valid, label=y_valid)

    params = {}
    params['max_bin'] = 20
    params['learning_rate'] = 0.0021
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'binary'
    params['sub_feature'] = 0.75
    params['bagging_fraction'] = 0.85
    params['bagging_freq'] = 40
    params['num_leaves'] = 512
    params['min_data'] = 500
    params['min_hessian'] = 0.05
    params['num_iterations'] = 1000
    params['early_stopping_round'] = 5
    params['verbose'] = -1
    print(class_name + " started")
    model_lgb = lgb.train(params, train_set=d_train, valid_sets=d_valid, verbose_eval=50)
    df_sub[class_name] = model_lgb.predict(test_word_features, num_iteration=model_lgb.best_iteration)
    print(class_name + " predicted")

# Print output to file
df_sub.to_csv('/output/simple_lgb.csv', index=False)